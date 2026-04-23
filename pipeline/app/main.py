"""
Enterprise Document Intelligence API
------------------------------------
FastAPI service that ingests PDFs (e.g., 10-K filings) and returns a JSON
payload of verbatim key-value extractions with strict page/table/text citations.

Design goals (see ARCHITECTURE.md):
  - Zero-hallucination: every returned value is substring-validated against
    the source PDF before being emitted. Values that fail verification are
    dropped (never paraphrased).
  - Scalable: stateless API workers behind a load balancer; heavy parsing
    and LLM calls dispatched to a Celery worker pool backed by Redis.
  - Secure: OAuth2/JWT auth, per-tenant API keys, rate limiting, signed
    upload URLs, and PII-safe logging.

This file is the API surface. Workers live in `app/workers/extractor.py`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.workers.extractor import extract_document_task
from app.workers.celery_app import celery_app
from app.schemas import ExtractionRequest, ExtractionResponse, JobStatus

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
SECRET_KEY = os.environ["JWT_SECRET_KEY"]          # 32+ random bytes, rotated
ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_MIN = 30
MAX_UPLOAD_BYTES = 100 * 1024 * 1024                # 100 MB hard cap
ALLOWED_MIME = {"application/pdf"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("docai.api")

# --------------------------------------------------------------------------- #
# Auth primitives                                                              #
# --------------------------------------------------------------------------- #
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def make_access_token(subject: str, tenant_id: str, scopes: list[str]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "tid": tenant_id,
        "scp": scopes,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_TTL_MIN)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def require_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from e
    return payload


def require_scope(scope: str):
    def _dep(user: Annotated[dict, Depends(require_user)]) -> dict:
        if scope not in user.get("scp", []):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing scope {scope}")
        return user
    return _dep


# --------------------------------------------------------------------------- #
# FastAPI app                                                                  #
# --------------------------------------------------------------------------- #
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app = FastAPI(title="Enterprise Document Intelligence", version="1.0.0")
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)


@app.exception_handler(RateLimitExceeded)
async def _rl_handler(req: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse({"error": "rate_limit_exceeded"}, status_code=429)


# --------------------------------------------------------------------------- #
# Auth routes                                                                  #
# --------------------------------------------------------------------------- #
class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_TTL_MIN * 60


@app.post("/auth/token", response_model=TokenOut)
@limiter.limit("10/minute")
async def issue_token(request: Request, form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenOut:
    """
    Exchange service credentials for a short-lived JWT.
    Real deployment: back this with your IdP (Okta/Entra/Auth0) via OIDC.
    """
    user = _lookup_user(form.username)
    if not user or not pwd_ctx.verify(form.password, user["password_hash"]):
        # Constant-time failure; do not leak which half was wrong.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = make_access_token(subject=user["id"], tenant_id=user["tenant_id"], scopes=user["scopes"])
    return TokenOut(access_token=token)


# --------------------------------------------------------------------------- #
# Extraction routes                                                            #
# --------------------------------------------------------------------------- #
@app.post("/v1/extractions", status_code=202)
@limiter.limit("30/minute")
async def submit_extraction(
    request: Request,
    user: Annotated[dict, Depends(require_scope("extract:write"))],
    file: Annotated[UploadFile, File(description="PDF document to extract from")],
    extraction_schema: Annotated[str, Header(alias="X-Extraction-Schema")] = "amazon_10k_v1",
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict:
    """
    Async submission — returns a job_id immediately. Worker pool picks up the job
    from Redis and performs parsing + extraction + verification.
    """
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "PDF only")

    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    sha256 = hashlib.sha256(body).hexdigest()
    object_key = f"tenants/{user['tid']}/uploads/{sha256}.pdf"
    _s3_put_with_sse(object_key, body)   # server-side AES-256 encryption

    job_id = idempotency_key or str(uuid.uuid4())
    extract_document_task.apply_async(
        kwargs=dict(job_id=job_id, object_key=object_key, schema=extraction_schema, tenant_id=user["tid"]),
        task_id=job_id,
    )
    log.info("submitted", extra={"job_id": job_id, "tenant": user["tid"], "sha256": sha256})
    return {"job_id": job_id, "status": "accepted"}


@app.get("/v1/extractions/{job_id}", response_model=ExtractionResponse | JobStatus)
async def get_extraction(
    job_id: str,
    user: Annotated[dict, Depends(require_scope("extract:read"))],
) -> dict:
    """Poll endpoint. Completed jobs return the ExtractionResponse JSON payload."""
    res = celery_app.AsyncResult(job_id)
    if res.state in ("PENDING", "STARTED", "RETRY"):
        return {"job_id": job_id, "status": res.state.lower()}
    if res.state == "FAILURE":
        raise HTTPException(500, f"extraction_failed: {res.result}")
    payload = res.get(timeout=1.0)
    # Tenant isolation check — never leak another tenant's data.
    if payload.get("tenant_id") != user["tid"]:
        raise HTTPException(404, "not_found")
    return payload


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------- #
# Stubs — replace with real impls backed by Postgres/S3 in production          #
# --------------------------------------------------------------------------- #
def _lookup_user(username: str) -> dict | None:
    # Replace with: SELECT ... FROM users WHERE username = $1
    return None


def _s3_put_with_sse(key: str, body: bytes) -> None:
    # boto3.client('s3').put_object(Bucket=..., Key=key, Body=body,
    #                               ServerSideEncryption='AES256')
    pass

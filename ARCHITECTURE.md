# Enterprise Document Intelligence — Architecture (v2)

**Assignment:** AI Solution Architect — Document AI extraction pipeline for
highly complex, multi-page PDFs (Amazon.com, Inc. Form 10-K).
**Author:** Yash Shukla
**Date:** 23 April 2026

---

## 1. Executive summary

A multi-tenant, horizontally scalable service that ingests PDFs and returns a
strictly-typed JSON payload of verbatim key-value extractions with full page,
table, and surrounding-text citations.

Three commitments that trace directly to the brief:

1. **Zero hallucination** — enforced at the code level by a substring-verification gate after every LLM call. Any answer that cannot be located character-for-character on its cited page is dropped, never paraphrased. ([engine/verify.py](engine/verify.py))
2. **Strict traceability** — every output carries a `citation` block with page number, item/note, table name, row, column, and exact surrounding text.
3. **Production-grade** — API gateway with auth + rate limiting, Redis caching (don't re-embed the same query), message queue for async LLM calls, per-dependency circuit breakers, GPU autoscaling with scale-to-zero, and load balancing across GPU nodes.

A companion **Streamlit application** (`streamlit_app.py`) lets a reviewer run the entire extractor locally against any uploaded PDF, with local embeddings (sentence-transformers) and an optional local LLM (Ollama). The same `engine/` module powers both the Streamlit demo and the production workers, so the two paths never drift.

## 2. Two modes: local and production

The same `engine/` module is the single source of truth. Swap the frontier model for a local one and the rest of the pipeline is unchanged.

| Capability | Local (Streamlit) | Production (Kubernetes) |
|---|---|---|
| UI | Streamlit web app (Single + Batch tabs) | React/Next.js (out of scope) |
| API | Direct function call | FastAPI behind Kong API Gateway |
| Auth | None (local dev) | JWT/OIDC at the gateway |
| Rate limiting | None | Kong `rate-limiting-advanced` (per consumer + per IP) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` on CPU | Same model, batched on GPU |
| LLM | `ollama` + `llama3.1:8b` (optional) | `vLLM`-served OSS model OR managed frontier API |
| Queue | `ThreadPoolExecutor` (bounded, `DOC_INTEL_WORKERS` env override) | Redis + Celery |
| Storage | SQLite at `~/.cache/doc-intel/store.sqlite` | Postgres / Aurora (same interface) |
| Cache | `diskcache` | Redis |
| Circuit breaker | In-process `_Breaker` | `pybreaker` per dependency (LLM/OCR/S3) |
| Autoscale | n/a | HPA (API) + KEDA on queue depth (workers) + Karpenter (GPU nodes) |
| Load balance | n/a | ALB (API) + K8s Service round-robin (workers) |

## 3. Production architecture — walkthrough

### 3.1 API gateway (Kong)

Kong terminates TLS and handles every cross-cutting concern before the API pods see traffic. Configured declaratively in [`pipeline/deploy/api-gateway/kong.yaml`](pipeline/deploy/api-gateway/kong.yaml).

- **Auth** — JWT plugin validates RS256 tokens against the IdP's JWKS. Invalid tokens are rejected at the edge, so the API tier never burns CPU on bad requests.
- **Per-tenant rate limits** — `rate-limiting-advanced` backed by Redis — 100/min, 5 000/hour, 100 000/day per consumer. Tenant plan determines the bucket size.
- **Per-IP DDoS damper** — 20 req/s per source IP.
- **Request size cap** — 32 MB for direct uploads; larger files use signed S3 upload URLs so the gateway never buffers them.
- **Response cache** — 5-second `proxy-cache` on idempotent GETs (status polls), so thousands of clients polling `/v1/extractions/{job_id}` do not flood Redis or the API.
- **WAF + correlation-id + bot detection**.

### 3.2 API tier (stateless FastAPI)

Stateless = easy to scale. On `POST /v1/extractions`:

1. Auth middleware re-validates JWT (defense in depth), resolves `tenant_id` and scopes.
2. PDF body is streamed to S3 (SSE-AES256, VPC endpoint). Key = `sha256(body)`.
3. A Celery task is enqueued with `task_id = idempotency_key or uuid`. The caller gets `202 Accepted` and a `job_id` in < 100 ms.
4. HPA scales the API tier on CPU + measured `http_requests_per_second`. [`pipeline/deploy/kubernetes/api-deployment.yaml`](pipeline/deploy/kubernetes/api-deployment.yaml)

### 3.3 Message queue (Redis + Celery)

Heavy work **never** runs on the API thread.

- Broker: Redis (or RabbitMQ for tenants that require it).
- `task_acks_late=True` + `task_reject_on_worker_lost=True` — a worker that dies mid-task re-queues the job; at-least-once delivery.
- Idempotency keys make duplicate submissions safe — the task ID is the key, so a retry is a no-op.
- Priority queues per tenant tier (`extract.free`, `extract.premium`) with weighted fair-queueing at the worker level.

**Local demo equivalent.** [`engine/job_queue.py`](engine/job_queue.py) exposes the same `submit(file, bytes) → doc_id` interface backed by a bounded `ThreadPoolExecutor` (default 2 workers, `DOC_INTEL_WORKERS` env override). Excess uploads wait in the executor's internal queue — the "batch-by-batch" behavior the Streamlit dashboard demonstrates. Swapping this module for a Celery task in production is a one-file change; the calling code is unchanged.

### 3.3.1 Document + extraction store

Outputs are persisted, not thrown away in a JSON file.

- Schema: [`engine/store.py`](engine/store.py) — `documents` table (status, timestamps, pages, latency, error_message) and `extractions` table (one row per (document, field) with `question`, `value_verbatim`, `value`, `citation` as JSON).
- Status progression: `queued → processing → completed | failed` with per-row timestamps for each transition, so the dashboard can compute end-to-end latency and queue-wait separately.
- Failure capture: any exception inside the worker is written to `documents.error_message` (truncated to 2 KB) and the row stays discoverable in the dashboard with status `failed`. No upload silently disappears.
- Local: SQLite in WAL mode with a module-level lock (SQLite allows concurrent reads but serializes writes). Production: the same function signatures backed by a SQLAlchemy engine against Postgres / Aurora / CloudSQL — callers do not change.

### 3.3.2 Dashboard

The Streamlit **Batch & Dashboard** tab is the operator-facing view on the store. It surfaces:

- live status metrics (Queued / Processing / Completed / Failed / In flight);
- a documents table with filename, status, pages, latency, backend, queued-at, and the first 80 characters of any error message;
- a drill-in panel that shows extractions for a completed document or the full failure reason for a failed one.

In production the same data is exposed through FastAPI endpoints (`GET /v1/documents`, `GET /v1/documents/{id}`) and a React dashboard; the Streamlit tab is the demo-grade equivalent of that page.

### 3.4 Cache (don't re-do work)

Caching is **content-addressed** at three layers, all keyed by SHA-256 of input so identical work is never repeated:

| Layer | Key | What it saves |
|---|---|---|
| Parsed document | `parse:sha256(pdf)` | pdfplumber re-parse (~1-3 s per 100 pages) |
| Embedding vector | `emb:model_id:sha256(text)` | re-embedding the same page text |
| LLM extraction | `llm:model:sha256((question,page))` | re-invoking the LLM on identical (question, page) pairs |
| Full extraction | `extraction:backend:sha256(pdf)` | the entire pipeline on a re-uploaded document |

Implementation: [`engine/cache.py`](engine/cache.py) — a single `@cached(namespace)` decorator; Redis in prod, `diskcache` locally. This directly satisfies the "don't re-execute same embedded query again" requirement — an identical (model, text) pair is a microsecond Redis hit, never a GPU call.

### 3.5 Circuit breakers

Per-dependency breakers isolate failure domains so one bad upstream cannot take the whole job down. [`pipeline/app/infra/circuit_breaker.py`](pipeline/app/infra/circuit_breaker.py)

- `LLM_BREAKER` — 5 failures → open for 30 s. While open, the extractor falls back to the deterministic path for that field instead of burning wallclock on retries.
- `OCR_BREAKER` — 3 failures → open for 60 s. OCR fallback (Textract) protected separately because OCR calls are expensive.
- `S3_BREAKER` — 10 failures → open for 15 s. S3 rarely fails but when it does, fast failure beats cascading retries.

The `@with_breaker(LLM_BREAKER, fallback=None)` decorator returns `None` on open state — the orchestrator then falls back to the deterministic rule for that field. Net effect: a flaky LLM degrades the job from "5/5 fields via LLM" to "3/5 via LLM + 2/5 via deterministic" — never to "0/5 fields, job failed".

### 3.6 GPU worker pool + autoscaling + load balancing

[`pipeline/deploy/kubernetes/gpu-workers.yaml`](pipeline/deploy/kubernetes/gpu-workers.yaml)

- **Worker pod topology** — each pod runs the Celery worker AND a co-located Ollama (or vLLM) container sharing a GPU. The LLM call is `localhost:11434` — no network egress, no cross-AZ hop. This is also how the pod is the unit of load balancing for the GPU: one pod, one GPU, one model-serving process.
- **Scale to zero** — `minReplicaCount: 0` (KEDA). When the queue is empty, workers drain, pods terminate, Karpenter deprovisions the GPU nodes, spend goes to **$0** for idle GPUs.
- **Scale up on queue depth** — KEDA `ScaledObject` with Redis trigger: `listLength: 5` ⇒ one new pod per 5 queued jobs. Linear scaling up to 20 replicas.
- **Node-level autoscale** — Karpenter provisions G5/G6 instances (NVIDIA A10G / L4 GPUs) on demand, prefers spot when available, consolidates underutilized nodes, expires after 24 h for security hygiene.
- **Load balancing** — job dispatch across workers is round-robin via the Celery broker (Redis). Workers don't sticky-bind to documents, so queue depth is evenly consumed; a slow job only blocks its own worker slot, not the queue.
- **Fairness across GPUs** — `prefetch-multiplier=1` ensures a worker pulls one job at a time; faster GPUs simply drain the queue faster.

## 4. Models and parsing strategy

### 4.1 Layout parsing (tier-1)

`pdfplumber` for text + table extraction. Tables come back as structured `TableBlock` objects with separate `header`, `rows`, and cell iteration `(row_label, col_label, cell_value)`. Column alignment is preserved — critical for the "Fulfillment equipment" edge case (see `DEBRIEF.md`).

A stateful automaton walks the document and stamps each page with the `Item X.` and `Note N —` headings it falls under, so citations carry item/note context without the model needing to re-derive it. This is what lets *Foreign Exchange Risk* be cited against *Item 7A* rather than the identically-named section under *Item 1A*.

### 4.1.1 Generic structural primitives (schema as code, not constants)

Deterministic verbatim rules live in [`engine/patterns.py`](engine/patterns.py) as three reusable primitives:

- `extract_row_rightmost_money(text, row_label, must_follow=...)` — pulls the rightmost `$`-token from a row label (10-K segment tables are chronological L→R, so the rightmost column is always the most-recent fiscal year).
- `extract_phrase_with_integer(text, anchor, lead_words=...)` — matches `<lead> <integer> … <anchor>` by shape, not by literal count.
- `extract_row_value_by_label(text, row_label, value_pattern)` — reads the value cell on a "label … value" line where `value_pattern` is a regex on *shape* (e.g. `<Word> to <word> years?`), not on the literal value.

No dollar amount, employee count, or lifespan is hard-coded anywhere in [`engine/fields.py`](engine/fields.py). The same field list therefore works against Amazon's 2024, 2025, and future 10-Ks — and against any filing with a similar Item/Note structure (Microsoft, Alphabet, Meta, etc.) without code changes.

### 4.2 OCR fallback (tier-2)

When per-page text density is below 0.05 chars/kpx², Amazon Textract `AnalyzeDocument` runs with `FeatureTypes=['TABLES','FORMS']`. The geometry-aware output is re-stitched into the same `TableBlock` schema, so downstream code is OCR-agnostic.

### 4.3 Chunking — page-as-unit

10-Ks are the pathological case for naive 512-token chunking: table headers land in one chunk, table bodies in another, and the disambiguating sentence ("Other equipment consists primarily of fulfillment equipment") in a third. We retrieve at **page granularity** because every target field resolves to a span on a specific page. For cross-page narratives, the enclosing `Item`/`Note` section is the fallback unit.

### 4.4 Hybrid retrieval (BM25 ⊕ embeddings)

[`engine/local_embeddings.py`](engine/local_embeddings.py)

- **BM25** (rank-bm25) for exact tokens — "AWS", "Fulfillment equipment", "$ 128,725".
- **Dense** — sentence-transformers (local) or Voyage-3 / `text-embedding-3-large` (managed).
- **Fusion** — Reciprocal Rank Fusion (`1/(60+rank)`), robust to score-scale differences, no normalization headaches.

### 4.5 LLM selection

Two profiles, the pipeline doesn't care which:

| Profile | Local | Production |
|---|---|---|
| Purpose | Demo, air-gapped customers | High-throughput SaaS |
| Model | `llama3.1:8b` via Ollama (also Qwen2.5, Mistral work) | Claude Sonnet 4.6 **or** vLLM-served Llama-3.3-70B on A10G |
| Temperature | 0 | 0 |
| Output | `format="json"` | tool-use with `emit_extraction` schema |
| Grounded on | Single candidate page | Single candidate page |

The key design choice: the model **never sees the whole document** — only the retrieved candidate page and its structured table cells. It cannot "recall" a number from pre-training because the prompt contains the literal page; its job is copy-paste, not recall.

## 5. Zero-hallucination — four stacked defenses

1. **Scope restriction** — prompt contains only the candidate page + table cells.
2. **Tool schema / JSON format contract** — model MUST return a `verbatim` string + 40+ char `surrounding_text`.
3. **Post-call substring verification** — `verify_verbatim()` normalizes whitespace and asserts membership. A failure drops the result (no re-ask, no paraphrase).
4. **Cell-grounded numeric check** — for numeric fields, the parsed `value` must match the digits in `verbatim` (so `$ 128,725` cannot become `128.7 billion`).

## 6. Security

- **AuthN** — OAuth2/OIDC at Kong, short-lived JWTs (30 min) with `jti` for revocation.
- **AuthZ** — scopes `extract:write`, `extract:read`, `admin`. Tenant isolation on every read path; `job_id` of another tenant returns 404, not 403 (prevents enumeration).
- **Transport** — TLS 1.2+, HSTS, WAF with OWASP Top-10.
- **Input hardening** — MIME allow-list, 100 MB hard cap, SHA-256 as storage key (dedupes replays, prevents path injection).
- **Storage** — S3 SSE-AES256, VPC endpoint, lifecycle 30 d on raw uploads.
- **Secrets** — AWS Secrets Manager (rotated), never env vars in plaintext.
- **Audit** — append-only log to Postgres + S3 Object Lock: `{job_id, tenant, user, sha256, model, status, latency}`. Direct line to SOX-style traceability.
- **PII redaction** — request/response body logger strips SSN/credit-card/email patterns.

## 7. Observability

- Structured JSON logs with `job_id`, `tenant`, `page_no`, `field_key`.
- OpenTelemetry traces end-to-end (gateway → API → Celery → LLM → S3) with model + token-count span attributes.
- Prometheus metrics: `extraction_latency_seconds` (by field), `verbatim_verification_failures_total`, `llm_tokens_total`, `ocr_fallback_total`, `breaker_state` (per breaker, per state).
- Grafana dashboards per tenant; alert on `verbatim_verification_failures_total` rising — that's the canary for upstream model drift.

## 8. Running it

### Local (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Optional local LLM:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

### Production

```bash
kubectl apply -f pipeline/deploy/kubernetes/
deck gateway sync pipeline/deploy/api-gateway/kong.yaml
```

## 9. Folder map

```
RAG/
├── streamlit_app.py                 # Streamlit UI — Single + Batch/Dashboard tabs
├── engine/                          # Shared extraction engine (local + prod)
│   ├── extractor.py                 #   orchestrator (schema-versioned cache)
│   ├── layout_parser.py             #   pdfplumber + Item/Note state machine
│   ├── fields.py                    #   5 field specs (schema-as-code, value-free)
│   ├── patterns.py                  #   generic structural primitives
│   ├── local_embeddings.py          #   sentence-transformers + BM25 + RRF
│   ├── local_llm.py                 #   Ollama client + in-proc circuit breaker
│   ├── cache.py                     #   diskcache / Redis, content-addressed
│   ├── store.py                     #   SQLite doc + extraction store
│   ├── job_queue.py                 #   bounded worker pool (local Celery stand-in)
│   └── verify.py                    #   zero-hallucination gate
├── pipeline/                        # Production stack
│   ├── app/
│   │   ├── main.py                  #   FastAPI (auth, submit, poll)
│   │   ├── schemas.py
│   │   ├── workers/ …               #   Celery workers, LLM/retrieval glue
│   │   └── infra/circuit_breaker.py #   pybreaker, per dependency
│   └── deploy/
│       ├── api-gateway/kong.yaml    #   JWT, rate limit, cache, WAF
│       └── kubernetes/              #   Deployments, HPA, KEDA, Karpenter
├── extraction_output.json           # Final deliverable — 5 extractions + citations
├── ARCHITECTURE.md                  # This file
├── ARCHITECTURE_DIAGRAM.svg         # Visual of the prod architecture
├── DEBRIEF.md                       # Hardest technical challenge
└── case-study.pdf                   # Source document (Amazon 10-K, FY2025)
```

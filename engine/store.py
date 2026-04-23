"""
Document + extraction store.

Local demo: SQLite file under ~/.cache/doc-intel/store.sqlite. Production:
swap `_conn()` for a SQLAlchemy engine pointing at Postgres / Aurora /
CloudSQL and the rest of the code keeps working. The job queue and
Streamlit dashboard depend only on the functions exported here.

Schema
------
  documents     one row per uploaded PDF
    id            uuid
    filename      str
    sha256        str                     (content-address; also the cache key)
    size_bytes    int
    backend       str                     (auto | llm | deterministic)
    status        str                     (queued | processing | completed | failed)
    error_message str   nullable          (populated when status='failed')
    queued_at     iso-8601
    started_at    iso-8601 nullable
    completed_at  iso-8601 nullable
    pages         int      nullable
    latency_s     real     nullable

  extractions   one row per (document, field_key)
    id            autoinc
    document_id   uuid FK -> documents.id ON DELETE CASCADE
    field_key     str                     (e.g. aws_net_sales)
    question      str
    value_verbatim str
    value_json    json                    (int | float | str — kind-agnostic)
    citation_json json                    (full citation record)
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sqlite3
import threading
import uuid
from typing import Any

_DB = pathlib.Path.home() / ".cache" / "doc-intel" / "store.sqlite"
_DB.parent.mkdir(parents=True, exist_ok=True)

# SQLite allows concurrent reads but writes must be serialized. Streamlit
# rerenders plus the thread-pool worker pool both touch this DB, so a
# module-level lock is the simplest correct answer for the demo.
_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB), isolation_level=None, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    """Idempotent — safe to call on every request."""
    with _LOCK, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id             TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                sha256         TEXT NOT NULL,
                size_bytes     INTEGER NOT NULL,
                backend        TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'queued',
                error_message  TEXT,
                queued_at      TEXT NOT NULL,
                started_at     TEXT,
                completed_at   TEXT,
                pages          INTEGER,
                latency_s      REAL
            );
            CREATE INDEX IF NOT EXISTS idx_docs_status  ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_docs_queued  ON documents(queued_at DESC);

            CREATE TABLE IF NOT EXISTS extractions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id     TEXT NOT NULL
                                REFERENCES documents(id) ON DELETE CASCADE,
                field_key       TEXT NOT NULL,
                question        TEXT,
                value_verbatim  TEXT,
                value_json      TEXT,
                citation_json   TEXT NOT NULL,
                UNIQUE(document_id, field_key)
            );
            """
        )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def enqueue(filename: str, sha256: str, size: int, backend: str) -> str:
    init()
    doc_id = str(uuid.uuid4())
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO documents(id,filename,sha256,size_bytes,backend,status,queued_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (doc_id, filename, sha256, size, backend, "queued", _now()),
        )
    return doc_id


def mark_processing(doc_id: str) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE documents SET status='processing', started_at=? WHERE id=?",
            (_now(), doc_id),
        )


def mark_completed(doc_id: str, payload: dict[str, Any], latency: float) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE documents SET status='completed', completed_at=?, "
            "pages=?, latency_s=? WHERE id=?",
            (_now(), payload["document"]["total_pages"], latency, doc_id),
        )
        for key, ext in payload["extractions"].items():
            c.execute(
                "INSERT OR REPLACE INTO extractions"
                "(document_id, field_key, question, value_verbatim, value_json, citation_json) "
                "VALUES(?,?,?,?,?,?)",
                (
                    doc_id,
                    key,
                    ext.get("question"),
                    ext.get("value_verbatim"),
                    json.dumps(ext.get("value"), ensure_ascii=False),
                    json.dumps(ext.get("citation"), ensure_ascii=False),
                ),
            )


def mark_failed(doc_id: str, error: str) -> None:
    """Record a failure reason. `error` is truncated to 2KB to avoid
    accidentally persisting multi-MB tracebacks from a wild exception."""
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE documents SET status='failed', completed_at=?, "
            "error_message=? WHERE id=?",
            (_now(), (error or "")[:2000], doc_id),
        )


def list_documents(limit: int = 200) -> list[dict]:
    init()
    with _LOCK, _conn() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT * FROM documents ORDER BY queued_at DESC LIMIT ?", (limit,)
            )
        ]


def get_document(doc_id: str) -> dict | None:
    init()
    with _LOCK, _conn() as c:
        row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None


def get_extractions(doc_id: str) -> list[dict]:
    init()
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT * FROM extractions WHERE document_id=? ORDER BY id", (doc_id,)
        ).fetchall()
    return [
        {
            "field_key": r["field_key"],
            "question": r["question"],
            "value_verbatim": r["value_verbatim"],
            "value": json.loads(r["value_json"]) if r["value_json"] else None,
            "citation": json.loads(r["citation_json"]) if r["citation_json"] else None,
        }
        for r in rows
    ]


def status_counts() -> dict[str, int]:
    """Dashboard metric source — {status: count}."""
    init()
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def reset_all() -> None:
    """Demo-only: wipe documents + extractions. Not exposed in production."""
    init()
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM extractions")
        c.execute("DELETE FROM documents")

"""
Bounded worker-pool job queue for the local Streamlit demo.

Interface
---------
  submit(filename, pdf_bytes, backend) -> doc_id
  submit_many([(name, bytes), ...])    -> [doc_id, ...]
  pending_count()                      -> int

Both calls return immediately; extraction runs in a background thread.
Progress is observed through `engine.store.list_documents()` which the
Dashboard tab renders.

Why a bounded pool?
-------------------
pdfplumber + sentence-transformers pin a CPU core each; running 50
uploads at once just thrashes. MAX_CONCURRENCY workers run in parallel
and any excess submissions wait in the executor's internal queue —
matching the user's "batch by batch" requirement.

Production path
---------------
In production this module is replaced by `pipeline/workers/extract.py`
(Celery task) + Redis broker + GPU worker pool with KEDA autoscaling.
Application code only calls `submit(...)`/`status(...)` — swapping the
backend doesn't touch the UI.

Failure model
-------------
Any exception inside the worker is captured, logged, and written to the
`documents.error_message` column with status='failed'. The dashboard
shows the failure reason; no upload silently disappears.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from engine import store
from engine.extractor import extract_from_pdf

log = logging.getLogger("engine.queue")

# Override via env var if a host has spare cores.
MAX_CONCURRENCY = int(os.environ.get("DOC_INTEL_WORKERS", "2"))

_executor: ThreadPoolExecutor | None = None
_pending = 0
_pending_lock = threading.Lock()
_init_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    """Module-level singleton — shared across Streamlit sessions."""
    global _executor
    with _init_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=MAX_CONCURRENCY,
                thread_name_prefix="extract-",
            )
    return _executor


def pending_count() -> int:
    """Number of jobs accepted but not yet finished. Purely informational."""
    with _pending_lock:
        return _pending


def submit(filename: str, pdf_bytes: bytes, backend: str = "auto") -> str:
    """Register the file in the store and schedule it on the worker pool.

    Returns the document id. Status transitions (queued→processing→
    completed/failed) are observable via `engine.store.get_document()`.
    """
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    doc_id = store.enqueue(filename, sha, len(pdf_bytes), backend)

    with _pending_lock:
        global _pending
        _pending += 1

    _pool().submit(_run, doc_id, filename, pdf_bytes, backend)
    return doc_id


def submit_many(
    items: Iterable[tuple[str, bytes]], backend: str = "auto"
) -> list[str]:
    return [submit(name, data, backend) for name, data in items]


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def _run(doc_id: str, filename: str, pdf_bytes: bytes, backend: str) -> None:
    t0 = time.time()
    try:
        store.mark_processing(doc_id)
        result = extract_from_pdf(pdf_bytes, filename=filename, backend=backend)
        store.mark_completed(doc_id, result.to_dict(), time.time() - t0)
        log.info("extracted %s in %.1fs", filename, time.time() - t0)
    except Exception as e:
        # The circuit-breaker inside the extractor already isolates per-dep
        # failures; anything that reaches here is a genuine job failure
        # (corrupt PDF, disk full, etc.) and goes to the dashboard.
        log.exception("job failed for %s", filename)
        store.mark_failed(doc_id, f"{type(e).__name__}: {e}")
    finally:
        with _pending_lock:
            global _pending
            _pending = max(_pending - 1, 0)

"""
Streamlit UI for the Enterprise Document Intelligence extractor.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Optional — for local LLM-grounded extraction:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b
The app auto-detects Ollama and switches the badge to green if reachable.
Without Ollama the deterministic backend runs — the same zero-hallucination
guarantee applies either way because both paths go through the verbatim gate.

Two tabs:

  1. Single extraction — upload one PDF, get JSON immediately. Result is
     also persisted to the DB so it shows up in the dashboard.

  2. Batch & Dashboard — upload many PDFs at once, they're queued and
     processed concurrently (bounded pool) with live status, failure
     reasons, and per-file extractions stored in SQLite.

Storage lives in `engine.store` (SQLite locally, Postgres in production —
same interface). The job queue lives in `engine.job_queue`
(ThreadPoolExecutor locally, Celery + Redis in production — same API).
"""
from __future__ import annotations

import hashlib
import json
import time

import streamlit as st

from engine import extract_from_pdf
from engine import store, job_queue
from engine.local_llm import status as llm_status

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📄",
    layout="wide",
)

store.init()

# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("Document Intelligence")
    st.caption("Verbatim key-value extraction with citations.")
    st.divider()

    backend = st.radio(
        "Extraction backend",
        options=["auto", "llm", "deterministic"],
        index=0,
        help=(
            "auto — try local LLM, fall back to deterministic rules per-field.\n"
            "llm — force local Ollama LLM.\n"
            "deterministic — rule-based page predicates only (fastest, offline)."
        ),
    )

    st.subheader("Local LLM status")
    s = llm_status()
    dot = "🟢" if s["available"] else "🔴"
    st.write(f"{dot} Ollama at `{s['host']}`")
    st.write(f"Model: `{s['model']}`")
    st.write(f"Circuit breaker open: `{s['breaker_open']}`")

    st.divider()
    st.caption(f"Worker concurrency: `{job_queue.MAX_CONCURRENCY}`")
    st.caption(
        "Zero-hallucination guarantee: every extracted value is "
        "substring-verified against its cited page before emission."
    )

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
st.title("📄 Enterprise Document Intelligence")

tab_single, tab_batch = st.tabs(["Single extraction", "Batch & Dashboard"])


# =========================================================================== #
# TAB 1 — single synchronous extraction                                        #
# =========================================================================== #
with tab_single:
    st.markdown(
        "Upload a PDF (e.g. a 10-K annual report) and receive a strictly-typed "
        "JSON payload of verbatim extractions with page-level citations."
    )

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="single_uploader",
    )

    if uploaded is None:
        st.info("Upload a PDF to start.")
    else:
        pdf_bytes = uploaded.read()
        st.success(f"Received **{uploaded.name}** ({len(pdf_bytes):,} bytes).")

        col_a, _ = st.columns([1, 3])
        with col_a:
            run = st.button(
                "▶ Run extraction", type="primary", use_container_width=True
            )

        if run:
            progress = st.progress(0, text="Parsing PDF…")
            t0 = time.time()

            # Register the run in the store up-front so it's visible in the
            # dashboard even if the extractor throws mid-run. This is the
            # same codepath the batch queue takes — the only difference is
            # that this tab waits for the result synchronously.
            sha = hashlib.sha256(pdf_bytes).hexdigest()
            doc_id = store.enqueue(uploaded.name, sha, len(pdf_bytes), backend)
            store.mark_processing(doc_id)

            try:
                progress.progress(10, text="Parsing PDF layout + tables…")
                result = extract_from_pdf(
                    pdf_bytes, filename=uploaded.name, backend=backend
                )
                progress.progress(100, text="Done.")
                elapsed = time.time() - t0
                payload = result.to_dict()
                store.mark_completed(doc_id, payload, elapsed)
            except Exception as e:
                progress.empty()
                store.mark_failed(doc_id, f"{type(e).__name__}: {e}")
                st.error(f"Extraction failed: {e}")
                st.stop()

            # --------- header metrics --------------------------------------
            n_fields = len(payload["extractions"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pages parsed", payload["document"]["total_pages"])
            m2.metric("Fields extracted", n_fields)
            m3.metric("Latency", f"{elapsed:.1f}s")
            m4.metric("Backend", payload["provenance"]["backend"])
            st.divider()

            # --------- extractions -----------------------------------------
            st.subheader("Extracted values")
            for key, ext in payload["extractions"].items():
                with st.expander(
                    f"**{key}**  ·  page {ext['citation']['page']}",
                    expanded=True,
                ):
                    if ext.get("question"):
                        st.markdown("**Question**")
                        st.info(ext["question"])
                    left, right = st.columns([2, 3])
                    with left:
                        st.markdown("**Verbatim value**")
                        st.code(
                            ext.get("value_verbatim") or ext.get("verbatim"),
                            language=None,
                        )
                        if "value" in ext and ext["value"] is not None:
                            st.markdown("**Parsed value**")
                            st.write(ext["value"])
                    with right:
                        st.markdown("**Citation**")
                        st.json(ext["citation"], expanded=False)

            st.divider()
            st.subheader("Full JSON payload")
            st.json(payload, expanded=False)
            st.download_button(
                label="⬇ Download extraction_output.json",
                data=json.dumps(payload, indent=2, ensure_ascii=False).encode(),
                file_name="extraction_output.json",
                mime="application/json",
            )
            st.caption(
                f"Extraction backend: `{payload['provenance']['backend']}` · "
                f"stored as doc `{doc_id[:8]}…` in the dashboard."
            )


# =========================================================================== #
# TAB 2 — batch upload + live dashboard                                        #
# =========================================================================== #
with tab_batch:
    st.markdown(
        "Drop in multiple PDFs — they're queued and processed **concurrently** "
        f"({job_queue.MAX_CONCURRENCY} at a time). Any excess files wait in "
        "line. Failures are captured with their error reason rather than "
        "crashing the pipeline."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        batch_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="batch_uploader",
        )
    with c2:
        st.write("")  # vertical spacer
        st.write("")
        if st.button("🧹 Reset dashboard", use_container_width=True,
                     help="Demo helper — wipes the local store."):
            store.reset_all()
            st.success("Store cleared.")

    if batch_files:
        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submit = st.button(
                f"▶ Submit {len(batch_files)} file(s) to queue",
                type="primary",
                use_container_width=True,
            )
        if submit:
            job_queue.submit_many(
                [(f.name, f.read()) for f in batch_files], backend=backend
            )
            st.success(
                f"Queued {len(batch_files)} file(s). "
                "They'll move through queued → processing → completed/failed."
            )

    # -------------------- live status panel ---------------------------------
    st.divider()
    counts = store.status_counts()
    pending = job_queue.pending_count()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Queued",     counts.get("queued", 0))
    m2.metric("Processing", counts.get("processing", 0))
    m3.metric("Completed",  counts.get("completed", 0))
    m4.metric("Failed",     counts.get("failed", 0))
    m5.metric("In flight",  pending, help="Jobs submitted but not yet finished.")

    refresh_col, auto_col = st.columns([1, 4])
    with refresh_col:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with auto_col:
        if pending > 0:
            st.caption(
                "Jobs are still running — click Refresh to see the latest "
                "status. (Production UI uses WebSocket push; this is the "
                "pull-based demo equivalent.)"
            )

    # -------------------- documents table -----------------------------------
    docs = store.list_documents(limit=200)
    if not docs:
        st.info("No documents yet. Upload a PDF (single tab or batch above).")
    else:
        st.subheader(f"Documents ({len(docs)})")

        # Summary row so the dashboard answers "am I seeing failures?" at a
        # glance without expanding every entry.
        rows = []
        for d in docs:
            rows.append({
                "id": d["id"][:8],
                "filename": d["filename"],
                "status": d["status"],
                "pages": d["pages"] or "",
                "latency_s": f"{d['latency_s']:.1f}" if d["latency_s"] else "",
                "backend": d["backend"],
                "queued_at": d["queued_at"][:19].replace("T", " "),
                "error": (d["error_message"] or "")[:80],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Drill-in: pick a document and inspect its extractions or failure.
        labels = [
            f"{d['filename']}  ·  {d['status']}  ·  {d['id'][:8]}" for d in docs
        ]
        choice = st.selectbox("Inspect a document", options=range(len(docs)),
                              format_func=lambda i: labels[i])
        d = docs[choice]

        status_badge = {
            "queued":     "⏳ queued",
            "processing": "⚙️ processing",
            "completed":  "✅ completed",
            "failed":     "❌ failed",
        }.get(d["status"], d["status"])
        st.markdown(f"**{d['filename']}** — {status_badge}")

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.write(f"**SHA-256:** `{d['sha256'][:16]}…`")
        meta_col2.write(f"**Pages:** {d['pages'] or '—'}")
        meta_col3.write(f"**Latency:** {d['latency_s']:.1f}s"
                        if d["latency_s"] else "**Latency:** —")

        if d["status"] == "failed":
            st.error(
                "**Failure reason**\n\n"
                f"```\n{d['error_message'] or 'no error message recorded'}\n```"
            )
        elif d["status"] == "completed":
            exts = store.get_extractions(d["id"])
            if not exts:
                st.warning("No extractions recorded.")
            else:
                for ext in exts:
                    with st.expander(
                        f"**{ext['field_key']}**  ·  page "
                        f"{ext['citation'].get('page', '?')}",
                        expanded=False,
                    ):
                        if ext.get("question"):
                            st.info(ext["question"])
                        st.code(ext["value_verbatim"] or "", language=None)
                        st.json(ext["citation"], expanded=False)
        else:
            st.info(
                "Job still in flight. Hit **↻ Refresh** above to update."
            )

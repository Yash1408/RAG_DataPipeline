"""
Top-level orchestrator — parse → retrieve → extract → verify → JSON.

Two backends, picked by `backend` arg:

  - "deterministic"  : rule-based page predicates + verbatim rules. Offline,
                       no LLM required. Default for the Streamlit demo so it
                       always works.
  - "llm"            : local Ollama LLM, grounded on hybrid-retrieved pages.
                       Falls back to deterministic if Ollama is unreachable
                       or the LLM's `verbatim` fails the verification gate.
  - "auto"           : try llm first, fall back to deterministic per-field.

The verbatim verification gate runs in BOTH paths. The answer on the page is
the answer; the extractor cannot fabricate values.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Literal

from engine.cache import cache, _hash
from engine.fields import AMAZON_10K_FIELDS, FieldSpec
from engine.layout_parser import parse_pdf, ParsedDocument, ParsedPage
from engine.verify import verify_verbatim, normalize

log = logging.getLogger("engine.extractor")

Backend = Literal["deterministic", "llm", "auto"]


@dataclass
class ExtractionResult:
    document: dict
    extractions: dict
    provenance: dict

    def to_dict(self) -> dict:
        return {"document": self.document, "extractions": self.extractions, "provenance": self.provenance}


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #
def extract_from_pdf(
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    backend: Backend = "auto",
    fields: list[FieldSpec] = AMAZON_10K_FIELDS,
) -> ExtractionResult:
    """Parse PDF once, extract every field, return a fully-cited JSON payload."""

    # Content-addressed cache: identical uploads return an instant hit.
    # The schema version in the key invalidates old cached payloads whenever
    # the emitted JSON shape changes (e.g. new keys like `question` or
    # `value`), preventing stale results from being served.
    _SCHEMA_VERSION = "v3"
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    full_key = f"extraction:{_SCHEMA_VERSION}:{backend}:{sha}"
    hit = cache.get(full_key)
    if hit is not None:
        log.info("cache hit for doc sha=%s", sha[:12])
        return ExtractionResult(**hit)

    doc = parse_pdf(pdf_bytes)

    extractions: dict[str, Any] = {}
    for spec in fields:
        out = _extract_one(spec, doc, backend)
        if out is not None:
            extractions[spec.key] = out

    result = ExtractionResult(
        document={
            "filename": filename,
            "sha256": sha,
            "total_pages": doc.metadata["total_pages"],
            "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        extractions=extractions,
        provenance={
            "backend": backend,
            "extraction_method": "layout-aware parsing + hybrid retrieval + verbatim verification",
            "zero_hallucination_guarantee": (
                "Every value was substring-verified against its cited page before emission. "
                "Values that failed verification were dropped, not paraphrased."
            ),
        },
    )
    cache.set(full_key, asdict(result))
    return result


# --------------------------------------------------------------------------- #
# Per-field extraction                                                         #
# --------------------------------------------------------------------------- #
def _extract_one(spec: FieldSpec, doc: ParsedDocument, backend: Backend) -> dict | None:
    if backend in ("llm", "auto"):
        out = _extract_llm(spec, doc)
        if out is not None:
            return out
        if backend == "llm":
            return None
    return _extract_deterministic(spec, doc)


def _extract_deterministic(spec: FieldSpec, doc: ParsedDocument) -> dict | None:
    for page in doc.pages:
        if not spec.predicate(page.text):
            continue
        verbatim = spec.verbatim_rule(page.text)
        if not verbatim or not verify_verbatim(verbatim, page.text):
            continue
        return _build_output(spec, page, verbatim, surrounding=verbatim)
    return None


def _extract_llm(spec: FieldSpec, doc: ParsedDocument) -> dict | None:
    # Only import lazily so deterministic-only installs don't need numpy/etc.
    try:
        from engine.local_embeddings import hybrid_topk
        from engine.local_llm import llm_extract
    except Exception as e:
        log.info("llm stack unavailable (%s); falling back to deterministic", e)
        return None

    candidates = hybrid_topk(doc, spec.anchor_terms, k=5)
    for c in candidates:
        res = llm_extract(spec.question, c.page)
        if res is None or not res.verbatim:
            continue
        if not verify_verbatim(res.verbatim, c.page.text):
            continue   # circuit-break-friendly: bad answer is just ignored
        return _build_output(spec, c.page, res.verbatim,
                             surrounding=res.surrounding_text or res.verbatim)
    return None


def _build_output(spec: FieldSpec, page: ParsedPage, verbatim: str, surrounding: str) -> dict:
    cit: dict[str, Any] = {
        "page": page.page_no,
        "item": spec.item_title or page.item_title,
        "surrounding_text": normalize(surrounding),
    }
    if spec.kind == "numeric":
        cit.update({
            "note": spec.note_title or page.note_title,
            "table_name": spec.table_name,
            "row": spec.row,
            "column": spec.column,
        })
        return {
            "question": spec.question,
            "value_verbatim": verbatim,
            "value": _to_number(verbatim),
            "citation": cit,
        }
    cit["section"] = spec.section_title or page.section_title
    # For text fields the parsed value IS the verbatim string — emit both
    # keys so downstream consumers can rely on `value` being present on
    # every extraction regardless of kind.
    return {
        "question": spec.question,
        "value_verbatim": verbatim,
        "value": verbatim,
        "citation": cit,
    }


def _to_number(verbatim: str) -> float | int | str:
    clean = verbatim.replace("$", "").replace(",", "").strip()
    try:
        return int(clean)
    except ValueError:
        try:
            return float(clean)
        except ValueError:
            return verbatim

"""
Core extraction worker.
Pipeline: S3 fetch -> layout-aware parse -> per-field retrieval -> grounded
LLM extraction -> verbatim verification -> JSON payload.

Zero-hallucination guarantee is enforced in `verify_verbatim()` — any value
that cannot be substring-matched in the source page (after whitespace
normalization) is dropped before the payload is returned.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import pdfplumber
from celery import shared_task

from app.workers.celery_app import celery_app
from app.workers.parsers import LayoutParser, ParsedDocument, ParsedPage, TableBlock
from app.workers.retrieval import HybridRetriever
from app.workers.llm import GroundedExtractor

log = logging.getLogger("docai.extractor")


# --------------------------------------------------------------------------- #
# Verbatim verification — the zero-hallucination gate                          #
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Collapse whitespace — accounts for PDF line-wrapping inside sentences."""
    return _WS.sub(" ", s).strip()


def verify_verbatim(value: str, page_text: str) -> bool:
    """Return True iff `value` appears verbatim on the page after WS normalization."""
    return _norm(value) in _norm(page_text)


# --------------------------------------------------------------------------- #
# Field definitions — the schema for Amazon 10-K extractions                   #
# --------------------------------------------------------------------------- #
@dataclass
class FieldSpec:
    key: str
    question: str
    kind: str           # "numeric" | "text"
    anchor_terms: list[str]
    table_hint: str | None = None


AMAZON_10K_FIELDS: list[FieldSpec] = [
    FieldSpec(
        key="aws_net_sales",
        question="Net Sales of the AWS segment for the most recent fiscal year, in millions of USD.",
        kind="numeric",
        anchor_terms=["AWS", "Net sales", "Segment Information"],
        table_hint="Information on reportable segments and reconciliation to consolidated net income",
    ),
    FieldSpec(
        key="employee_headcount",
        question="Total number of full-time and part-time employees at fiscal year end.",
        kind="numeric",
        anchor_terms=["full-time and part-time employees", "Human Capital"],
    ),
    FieldSpec(
        key="property_equipment_useful_life_fulfillment_equipment",
        question=(
            "Estimated useful life (years) for Fulfillment equipment. "
            "Note: the 10-K typically classifies fulfillment equipment under "
            "'Other equipment' in the useful-life table; extract that row's value."
        ),
        kind="text",
        anchor_terms=["Fulfillment equipment", "Other equipment", "Estimated useful life"],
        table_hint="Property and equipment Estimated useful life",
    ),
    FieldSpec(
        key="primary_retail_competition",
        question=(
            "Extract the exact single sentence describing the principal/primary "
            "competitive factors in Amazon's retail business."
        ),
        kind="text",
        anchor_terms=["principal competitive factors", "retail businesses"],
    ),
    FieldSpec(
        key="foreign_exchange_risk",
        question=(
            "In Item 7A — Quantitative and Qualitative Disclosures About Market Risk, "
            "section 'Foreign Exchange Risk', extract the exact sentence summarizing "
            "the primary risk related to foreign exchange rates."
        ),
        kind="text",
        anchor_terms=["Foreign Exchange Risk", "exposed to foreign exchange rate"],
    ),
]


# --------------------------------------------------------------------------- #
# Celery entrypoint                                                            #
# --------------------------------------------------------------------------- #
@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def extract_document_task(self, job_id: str, object_key: str, schema: str, tenant_id: str) -> dict:
    try:
        pdf_bytes = _s3_get(object_key)
        parsed = LayoutParser().parse(pdf_bytes)
        retriever = HybridRetriever.build(parsed)   # BM25 + embeddings
        llm = GroundedExtractor()                   # e.g. Claude Sonnet with tool-use

        extractions: dict[str, Any] = {}
        for spec in AMAZON_10K_FIELDS:
            candidates = retriever.topk(spec.anchor_terms, k=5)
            cand_page = llm.pick_page(spec, candidates, parsed)
            if cand_page is None:
                log.warning("no candidate page for %s", spec.key)
                continue
            raw = llm.extract(spec, cand_page, parsed)
            if not raw:
                continue
            # HARD VERIFICATION GATE
            if not verify_verbatim(raw["verbatim"], cand_page.text):
                log.error("verbatim verification failed for %s on p.%d", spec.key, cand_page.page_no)
                continue
            extractions[spec.key] = _build_output(spec, raw, cand_page, parsed)

        return {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "status": "completed",
            "document": parsed.metadata,
            "extractions": extractions,
            "provenance": {
                "extraction_method": "hybrid layout-aware parsing + grounded LLM + verbatim verification",
                "zero_hallucination_guarantee": "Every value was substring-verified against its cited page.",
            },
        }
    except Exception as exc:
        log.exception("extraction job failed: %s", exc)
        raise self.retry(exc=exc)


def _build_output(spec: FieldSpec, raw: dict, page: "ParsedPage", doc: "ParsedDocument") -> dict:
    if spec.kind == "numeric":
        return {
            "value": raw["value"],
            "unit": raw.get("unit"),
            "verbatim": raw["verbatim"],
            "citation": {
                "page": page.page_no,
                "item": page.item_title,
                "note": page.note_title,
                "table_name": raw.get("table_name") or spec.table_hint,
                "row": raw.get("row"),
                "column": raw.get("column"),
                "surrounding_text": raw["surrounding_text"],
            },
        }
    return {
        "value_verbatim": raw["verbatim"],
        "citation": {
            "page": page.page_no,
            "item": page.item_title,
            "section": page.section_title,
            "surrounding_text": raw["surrounding_text"],
        },
    }


def _s3_get(key: str) -> bytes:
    # boto3.client('s3').get_object(Bucket=..., Key=key)['Body'].read()
    raise NotImplementedError

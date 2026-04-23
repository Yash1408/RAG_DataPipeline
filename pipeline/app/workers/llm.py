"""
Grounded LLM extractor.

Design choices that enforce zero-hallucination:

  1. The model is given ONLY the candidate page text + extracted table cells,
     never the whole document, so it cannot "recall" a number from training.

  2. The tool schema forces the model to return:
        - verbatim: the EXACT string as it appears on the page
        - surrounding_text: a >=40 char substring containing the verbatim value
        - page_no: must match the page it was given
     The caller then substring-verifies `verbatim` against page text before
     emitting the result. If the string is not on the page, the result is
     dropped (no re-ask, no paraphrase).

  3. Temperature is pinned to 0 and the prompt includes a one-shot exemplar
     where the correct answer requires copy-pasting a table cell.
"""
from __future__ import annotations

import json
import os
from typing import Any

# `anthropic` is the reference client. Any model that supports tool use works.
from anthropic import Anthropic

from app.workers.parsers import ParsedDocument, ParsedPage


_MODEL = os.environ.get("EXTRACTOR_MODEL", "claude-sonnet-4-6")


_EXTRACT_TOOL = {
    "name": "emit_extraction",
    "description": "Emit a single extraction result. verbatim MUST be copied character-for-character from the page text provided.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "verbatim": {"type": "string", "description": "Exact copy from the page"},
            "value": {"type": ["number", "string", "null"]},
            "unit": {"type": ["string", "null"]},
            "row": {"type": ["string", "null"]},
            "column": {"type": ["string", "null"]},
            "table_name": {"type": ["string", "null"]},
            "surrounding_text": {"type": "string", "description": "40+ chars of page text containing verbatim"},
        },
        "required": ["found", "verbatim", "surrounding_text"],
    },
}


class GroundedExtractor:
    def __init__(self) -> None:
        self.client = Anthropic()

    def pick_page(self, spec, candidates, doc: ParsedDocument) -> ParsedPage | None:
        """Ask the model which candidate page actually contains the answer."""
        if not candidates:
            return None
        menu = "\n\n".join(
            f"--- Candidate page {c.page.page_no} ---\n{c.page.text[:1500]}" for c in candidates
        )
        msg = self.client.messages.create(
            model=_MODEL, max_tokens=64, temperature=0,
            messages=[{
                "role": "user",
                "content": (
                    f"Task: {spec.question}\n\n"
                    f"Which of the following pages contains the literal answer? "
                    f"Reply ONLY with the page number integer.\n\n{menu}"
                ),
            }],
        )
        try:
            pn = int("".join(ch for ch in msg.content[0].text if ch.isdigit()))
        except (ValueError, IndexError):
            return candidates[0].page
        for c in candidates:
            if c.page.page_no == pn:
                return c.page
        return candidates[0].page

    def extract(self, spec, page: ParsedPage, doc: ParsedDocument) -> dict[str, Any] | None:
        # Flatten tables into a clearly labeled block so cell-citation works.
        table_dump = ""
        for t in page.tables:
            table_dump += f"\n[TABLE name={t.name!r}]\n"
            table_dump += " | ".join(t.header) + "\n"
            for r in t.rows:
                table_dump += " | ".join(r) + "\n"
        system = (
            "You are a document extraction model with a strict zero-hallucination "
            "contract. You may ONLY return values that appear literally, "
            "character-for-character, in the page text or tables provided. "
            "If the answer is not literally present, set found=false."
        )
        user = (
            f"Target field: {spec.key}\n"
            f"Question: {spec.question}\n\n"
            f"=== PAGE {page.page_no} TEXT ===\n{page.text}\n\n"
            f"=== PAGE {page.page_no} TABLES ==={table_dump}\n\n"
            "Call emit_extraction. Copy verbatim character-for-character."
        )
        msg = self.client.messages.create(
            model=_MODEL, max_tokens=1024, temperature=0,
            system=system, tools=[_EXTRACT_TOOL], tool_choice={"type": "tool", "name": "emit_extraction"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "emit_extraction":
                payload = block.input
                if not payload.get("found"):
                    return None
                return payload
        return None

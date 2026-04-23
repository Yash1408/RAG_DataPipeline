"""The zero-hallucination gate.

Every extraction — whether produced by deterministic rules or by a local
LLM — must pass through `verify_verbatim()` before it is emitted. This is
the ONE function that turns the system from 'fuzzy RAG' into 'audit-grade'.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Collapse all whitespace. PDF line-wrapping inserts \\n inside sentences;
    the sentence on the page is still a single sentence, so we compare in
    whitespace-normalized form."""
    return _WS.sub(" ", s).strip()


def verify_verbatim(value: str, page_text: str) -> bool:
    """Return True iff `value` appears on `page_text` after WS normalization."""
    if not value:
        return False
    return normalize(value) in normalize(page_text)


def extract_sentence(page_text: str, anchor: str) -> str | None:
    """Return the single sentence on `page_text` containing `anchor`, verbatim.
    Walks back to the previous terminator and forward to the next period."""
    t = normalize(page_text)
    i = t.find(anchor)
    if i == -1:
        return None
    start = max((t.rfind(term, 0, i) for term in (". ", "! ", "? ")), default=-1)
    start = 0 if start == -1 else start + 2
    end = len(t)
    for term in (". ", "! ", "? "):
        j = t.find(term, i)
        if j != -1:
            end = min(end, j + 1)
    return t[start:end].strip()

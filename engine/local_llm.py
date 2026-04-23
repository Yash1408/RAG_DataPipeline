"""
Local LLM backend — Ollama.

Install once on the host/GPU node:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b            # or mistral, qwen2.5, etc.

The extractor sends ONLY the candidate page + the field question. Temperature
is 0 and a JSON schema constrains the reply. The output then goes through
`verify_verbatim()` — no verbatim-match, no emission.

A simple in-process circuit breaker opens after 3 consecutive failures and
fails fast for the next 30 s, so a flaky LLM does not wedge the whole job.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from engine.cache import cache, _hash
from engine.layout_parser import ParsedPage

_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# --------------------------------------------------------------------------- #
# Minimal in-process circuit breaker                                           #
# --------------------------------------------------------------------------- #
class _Breaker:
    def __init__(self, fail_threshold: int = 3, reset_after_s: float = 30.0) -> None:
        self._fails = 0
        self._open_until = 0.0
        self._fail_threshold = fail_threshold
        self._reset_after = reset_after_s

    def allow(self) -> bool:
        return time.time() >= self._open_until

    def on_success(self) -> None:
        self._fails = 0

    def on_failure(self) -> None:
        self._fails += 1
        if self._fails >= self._fail_threshold:
            self._open_until = time.time() + self._reset_after
            self._fails = 0


_breaker = _Breaker()


# --------------------------------------------------------------------------- #
# Ollama client wrapper                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class LLMResult:
    verbatim: str | None
    surrounding_text: str | None
    raw: dict


def _ollama_available() -> bool:
    try:
        import ollama  # type: ignore
        ollama.Client(host=_HOST).list()
        return True
    except Exception:
        return False


def llm_extract(question: str, page: ParsedPage) -> LLMResult | None:
    """Grounded extraction — returns None if Ollama is unreachable, the
    breaker is open, or the call raises. The orchestrator falls back to the
    deterministic rule in that case. Result is cached by (model, question,
    page text hash) so identical retries are free."""
    if not _breaker.allow():
        return None

    key = f"llm:{_MODEL}:{_hash((question, page.text))}"
    hit = cache.get(key)
    if hit is not None:
        return LLMResult(**hit)

    try:
        import ollama  # type: ignore
    except ImportError:
        return None

    try:
        client = ollama.Client(host=_HOST)
        # Flatten tables for the model.
        table_dump = ""
        for t in page.tables:
            table_dump += f"\n[TABLE {t.name}]\n" + " | ".join(t.header) + "\n"
            for r in t.rows:
                table_dump += " | ".join(r) + "\n"
        system = (
            "You are a document-extraction model with a strict zero-hallucination "
            "contract. Only return values that appear LITERALLY on the page text "
            "given. If the answer is not literally present, return an empty "
            "verbatim string."
        )
        user = (
            f"Question: {question}\n\n"
            f"--- PAGE {page.page_no} TEXT ---\n{page.text}\n\n"
            f"--- PAGE {page.page_no} TABLES ---{table_dump}\n\n"
            "Return JSON: {\"verbatim\": \"...\", \"surrounding_text\": \"...\"} "
            "where verbatim is copied character-for-character from the page."
        )
        resp = client.chat(
            model=_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format="json",
            options={"temperature": 0, "num_predict": 512},
        )
        payload = json.loads(resp["message"]["content"])
        result = LLMResult(
            verbatim=payload.get("verbatim") or None,
            surrounding_text=payload.get("surrounding_text") or None,
            raw=payload,
        )
        _breaker.on_success()
        cache.set(key, {"verbatim": result.verbatim,
                        "surrounding_text": result.surrounding_text,
                        "raw": result.raw})
        return result
    except Exception:
        _breaker.on_failure()
        return None


def status() -> dict[str, Any]:
    return {
        "backend": "ollama",
        "model": _MODEL,
        "host": _HOST,
        "available": _ollama_available(),
        "breaker_open": not _breaker.allow(),
    }

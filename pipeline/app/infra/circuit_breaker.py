"""
Circuit breaker for external calls (LLM, OCR, S3).

Built on pybreaker. The key insight is per-dependency breakers: one flaky
LLM backend must not open the breaker that protects S3, and vice versa.
A breaker in the OPEN state fails fast (returns None within microseconds),
so a partially failing upstream cannot burn through the worker's wallclock
budget and turn a single-field failure into a whole-job failure.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

import pybreaker

log = logging.getLogger("docai.circuit_breaker")


class _BreakerListener(pybreaker.CircuitBreakerListener):
    def __init__(self, name: str) -> None:
        self.name = name

    def state_change(self, cb: pybreaker.CircuitBreaker, old_state, new_state) -> None:
        log.warning("breaker %s: %s -> %s", self.name, old_state.name, new_state.name)


def make_breaker(name: str, fail_max: int = 5, reset_timeout_s: int = 30) -> pybreaker.CircuitBreaker:
    return pybreaker.CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout_s,
        name=name,
        listeners=[_BreakerListener(name)],
    )


# Per-dependency breakers — isolate failure domains.
LLM_BREAKER = make_breaker("llm", fail_max=5, reset_timeout_s=30)
OCR_BREAKER = make_breaker("ocr", fail_max=3, reset_timeout_s=60)
S3_BREAKER = make_breaker("s3", fail_max=10, reset_timeout_s=15)


def with_breaker(breaker: pybreaker.CircuitBreaker, fallback: Any = None) -> Callable:
    """Decorator: calls through the breaker; on OPEN state returns `fallback`
    without invoking the wrapped function (and without raising) — so a single
    field failure doesn't break the whole extraction job."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(fn, *args, **kwargs)
            except pybreaker.CircuitBreakerError:
                log.info("breaker %s open — returning fallback", breaker.name)
                return fallback
            except Exception as e:
                log.warning("call via breaker %s failed: %s", breaker.name, e)
                return fallback
        return wrapper
    return deco

"""
Content-addressed cache for expensive work: parsing, embeddings, LLM output.

Local dev uses diskcache (no service dependency). Production swaps in Redis
by setting `CACHE_BACKEND=redis` and `REDIS_URL`. Keys are SHA-256 of the
input, so an identical PDF re-upload is a microsecond cache hit and NEVER
re-runs the embedding model or the LLM — satisfying the 'don't re-embed
the same query' requirement.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

_BACKEND = os.environ.get("CACHE_BACKEND", "disk")


def _hash(obj: Any) -> str:
    if isinstance(obj, bytes):
        return hashlib.sha256(obj).hexdigest()
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class _DiskCache:
    def __init__(self, path: str = ".cache") -> None:
        try:
            import diskcache  # type: ignore
            self._c = diskcache.Cache(path)
            self._mode = "diskcache"
        except Exception:
            # Zero-dep fallback: pickle files under .cache/
            Path(path).mkdir(exist_ok=True, parents=True)
            self._c = Path(path)
            self._mode = "files"

    @staticmethod
    def _safe(key: str) -> str:
        # Filesystem-safe key: hash any separators/colons/slashes away.
        return hashlib.sha1(key.encode()).hexdigest()

    def get(self, key: str) -> Any:
        if self._mode == "diskcache":
            return self._c.get(key)
        p = self._c / f"{self._safe(key)}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if self._mode == "diskcache":
            self._c.set(key, value, expire=ttl)
            return
        p = self._c / f"{self._safe(key)}.pkl"
        with open(p, "wb") as f:
            pickle.dump(value, f)


class _RedisCache:
    def __init__(self) -> None:
        import redis  # type: ignore
        self._c = redis.Redis.from_url(os.environ["REDIS_URL"])

    def get(self, key: str) -> Any:
        b = self._c.get(key)
        return pickle.loads(b) if b else None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._c.set(key, pickle.dumps(value), ex=ttl)


def _make() -> Any:
    return _RedisCache() if _BACKEND == "redis" else _DiskCache()


cache = _make()


def cached(namespace: str, ttl: int | None = 7 * 24 * 3600):
    """Decorator: caches function output keyed by (namespace, args hash)."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            key = f"{namespace}:{_hash((args, kwargs))}"
            hit = cache.get(key)
            if hit is not None:
                return hit
            val = fn(*args, **kwargs)
            cache.set(key, val, ttl=ttl)
            return val
        return wrapper
    return deco

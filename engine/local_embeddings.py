"""
Local embedding model + hybrid retriever.

Default model: sentence-transformers/all-MiniLM-L6-v2 (22M params, CPU-friendly).
Swap via `EMBEDDING_MODEL` env var (e.g. BAAI/bge-small-en-v1.5).

Embeddings are cached by (model_id, sha256(text)) so re-uploading the same
document never re-runs the encoder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from engine.cache import cache, _hash
from engine.layout_parser import ParsedDocument, ParsedPage

_MODEL_ID = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class _LocalEmbedder:
    """Lazy-loaded sentence-transformers encoder. Falls back to a hashed
    bag-of-words encoder if sentence-transformers is missing, broken, or
    depends on a half-installed stack (e.g. transformers pulling torchvision).
    The Streamlit demo therefore ALWAYS runs — pip surprises cannot brick it.
    """

    def __init__(self) -> None:
        self._model = None     # None = unloaded; "tfidf" = fallback mode; else real encoder
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Broadest possible catch — ModuleNotFoundError, ImportError, and
        # any runtime error raised during the import cascade (e.g.
        # transformers.models.zoedepth requiring torchvision).
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(_MODEL_ID)
            self._dim = self._model.get_sentence_embedding_dimension()
        except BaseException as e:   # BaseException: also catches import-time SystemExit
            import logging
            logging.getLogger("engine.embeddings").info(
                "sentence-transformers unavailable (%s: %s) — using hashed BoW fallback",
                type(e).__name__, e,
            )
            self._model = "tfidf"

    def encode(self, texts: List[str]) -> np.ndarray:
        self._load()
        if self._model == "tfidf":
            return self._tfidf(texts)
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        except BaseException:
            # Even if the model loaded, some runtime paths (quantization,
            # CUDA hiccups, tokenizer errors) can fail mid-encode. Degrade.
            self._model = "tfidf"
            return self._tfidf(texts)

    def _tfidf(self, texts: List[str]) -> np.ndarray:
        # Deterministic fallback: hash-based bag-of-words vectors, L2-normalized.
        dim = 384
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % dim] += 1.0
            n = np.linalg.norm(out[i]) or 1.0
            out[i] /= n
        return out


_embedder = _LocalEmbedder()


def embed(texts: List[str]) -> np.ndarray:
    """Cached page-level embedding. Key = (model, text-hash)."""
    keys = [f"emb:{_MODEL_ID}:{_hash(t.encode())}" for t in texts]
    hits = [cache.get(k) for k in keys]
    miss_idx = [i for i, h in enumerate(hits) if h is None]
    if miss_idx:
        fresh = _embedder.encode([texts[i] for i in miss_idx])
        for j, i in enumerate(miss_idx):
            hits[i] = fresh[j]
            cache.set(keys[i], fresh[j])
    return np.vstack(hits).astype(np.float32)


@dataclass
class Candidate:
    page: ParsedPage
    score: float


def hybrid_topk(doc: ParsedDocument, anchor_terms: List[str], k: int = 5) -> List[Candidate]:
    """BM25 lexical + embedding cosine, fused by Reciprocal Rank Fusion."""
    from rank_bm25 import BM25Okapi

    corpus = [p.text for p in doc.pages]
    tokens = [_tok(t) for t in corpus]
    bm25 = BM25Okapi(tokens)
    q = " ".join(anchor_terms)
    bm = np.asarray(bm25.get_scores(_tok(q)))
    dense_matrix = embed(corpus)
    q_vec = embed([q])[0]
    dense = dense_matrix @ q_vec

    def ranks(s):
        order = np.argsort(-s)
        r = np.empty_like(order)
        r[order] = np.arange(len(s))
        return r + 1

    rrf = 1.0 / (60.0 + ranks(bm)) + 1.0 / (60.0 + ranks(dense))
    top = np.argsort(-rrf)[:k]
    return [Candidate(doc.pages[i], float(rrf[i])) for i in top]


def _tok(s: str) -> list[str]:
    return [w.lower() for w in s.replace("$", " $ ").split() if w.strip()]

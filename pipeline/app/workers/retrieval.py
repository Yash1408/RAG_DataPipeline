"""
Hybrid retriever: BM25 lexical search fused with embedding cosine similarity.

We index pages (not fine-grained chunks) for 10-K extraction because every
target field corresponds to a definite span on a specific page. Page-level
retrieval keeps the subsequent LLM call focused and eliminates chunk-boundary
bugs that commonly cause table headers and body rows to land in different
chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from app.workers.parsers import ParsedDocument, ParsedPage


def _tok(s: str) -> list[str]:
    return [w.lower() for w in s.replace("$", " $ ").split() if w.strip()]


@dataclass
class Candidate:
    page: ParsedPage
    score: float


class HybridRetriever:
    def __init__(self, doc: ParsedDocument, bm25: BM25Okapi, embs: np.ndarray) -> None:
        self.doc = doc
        self.bm25 = bm25
        self.embs = embs  # shape (n_pages, d)

    @classmethod
    def build(cls, doc: ParsedDocument) -> "HybridRetriever":
        tokens = [_tok(p.text) for p in doc.pages]
        bm25 = BM25Okapi(tokens)
        embs = _embed([p.text for p in doc.pages])
        return cls(doc, bm25, embs)

    def topk(self, anchor_terms: List[str], k: int = 5) -> List[Candidate]:
        query = " ".join(anchor_terms)
        bm25_scores = self.bm25.get_scores(_tok(query))
        q_emb = _embed([query])[0]
        # cosine sim (embeddings are L2-normalized by the embedding service)
        dense_scores = self.embs @ q_emb
        # Reciprocal Rank Fusion (k0=60 is the standard constant)
        k0 = 60.0
        bm25_rank = _ranks(bm25_scores)
        dense_rank = _ranks(dense_scores)
        rrf = 1.0 / (k0 + bm25_rank) + 1.0 / (k0 + dense_rank)
        top = np.argsort(-rrf)[:k]
        return [Candidate(self.doc.pages[i], float(rrf[i])) for i in top]


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks + 1  # 1-based


def _embed(texts: list[str]) -> np.ndarray:
    # Plug your embedding model here (Voyage-3, text-embedding-3-large, etc.).
    # Return L2-normalized vectors so dot product == cosine similarity.
    raise NotImplementedError

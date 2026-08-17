"""
Sparse lexical retrieval via BM25 (rank_bm25 library — real, local,
no network dependency, works identically here and in production).

Tokenization note: uses simple whitespace splitting. This is adequate for
the observed MSMARCO-XI text (space-delimited Kannada + English), but does
NOT do any stemming/lemmatization or Kannada-specific morphological
normalization. Documented as a known limitation — see
docs/retrieval-architecture.md.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be the same length")
        self.doc_ids = doc_ids
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_idx]

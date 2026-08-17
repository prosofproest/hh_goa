"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists (e.g. dense
vector results + BM25 sparse results) into a single fused ranking.

score(doc) = sum over each ranking list where doc appears of 1 / (k + rank)

Standard formulation (Cormack et al. 2009). `k` is configurable, not
hardcoded — default 60 is the commonly cited constant in IR literature,
but callers can override it.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """`ranked_lists` is a list of ranked id lists (best first), e.g.
    [dense_ranked_ids, sparse_ranked_ids]. Returns a single fused ranking
    as (id, fused_score) tuples, sorted best-first."""
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

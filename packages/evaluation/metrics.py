"""
Retrieval evaluation metrics — pure functions, no external dependencies.

All functions take:
    ranked_ids: list[str]   — retrieved item IDs, in rank order (best first)
    relevant_ids: set[str]  — the ground-truth relevant IDs for this query

These are standard IR metric definitions; unit-testable in isolation from
any embedding model, Qdrant instance, or real dataset (see
packages/evaluation/test_metrics.py-equivalent inline doctest-style checks
run during Phase 4 development).
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hit = len(top_k & relevant_ids)
    return hit / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if ANY relevant id appears in top-k, else 0.0."""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str], k: int | None = None) -> float:
    """MRR contribution for one query: 1/rank of the first relevant hit,
    0.0 if no relevant item appears within the first k (or anywhere, if
    k is None)."""
    if not relevant_ids:
        return 0.0
    ids = ranked_ids[:k] if k is not None else ranked_ids
    for i, item_id in enumerate(ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if item_id in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_single_query(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    mrr_k: int = 10,
) -> dict:
    result = {}
    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        result[f"hit@{k}"] = hit_at_k(ranked_ids, relevant_ids, k)
    result[f"mrr@{mrr_k}"] = reciprocal_rank(ranked_ids, relevant_ids, mrr_k)
    result[f"ndcg@{max(k_values)}"] = ndcg_at_k(ranked_ids, relevant_ids, max(k_values))
    return result


def aggregate_metrics(per_query_results: list[dict]) -> dict:
    """Mean of each metric across all queries. Assumes every dict in
    per_query_results has the same keys."""
    if not per_query_results:
        return {}
    keys = per_query_results[0].keys()
    agg = {}
    for key in keys:
        values = [r[key] for r in per_query_results]
        agg[key] = round(sum(values) / len(values), 4)
    agg["_num_queries"] = len(per_query_results)
    return agg

"""
Phase 3 (review) — Chunk Strategy Comparison Report

Reads the already-written, dedup-flagged chunk output
(data/processed/chunks/strategy=*/language=*/part.parquet — produced by
scripts/chunk_dataset.py, which already computes chunk_text_hash,
is_duplicate_chunk, duplicate_of_chunk_id, below_min_length per chunk) and
produces a structural comparison of the four strategies.

This script does NOT re-derive dedup logic; it reports on what
chunk_dataset.py already flagged, plus additional cross-cutting statistics
(length distribution, per-strategy duplicate rate, compactness) needed to
decide, in Phase 4, whether all four strategies are worth embedding.

IMPORTANT: this compares STRUCTURAL properties only (chunk counts,
duplication, length). It intentionally does NOT rank strategies by
retrieval quality — that requires embeddings + Recall@K/MRR/nDCG
evaluation, which is out of scope until Phase 4.

Usage:
    python scripts/compare_chunk_strategies.py --chunks-dir data/processed/chunks

Output:
    data/processed/chunk_strategy_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LENGTH_BUCKETS = [
    (0, 10, "0-9"),
    (10, 50, "10-49"),
    (50, 100, "50-99"),
    (100, 200, "100-199"),
    (200, 500, "200-499"),
    (500, 1000, "500-999"),
    (1000, float("inf"), "1000+"),
]


def bucket_for(length: int) -> str:
    for lo, hi, label in LENGTH_BUCKETS:
        if lo <= length < hi:
            return label
    return "1000+"


def main():
    parser = argparse.ArgumentParser(description="Compare chunking strategies structurally")
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    args = parser.parse_args()

    import pandas as pd

    chunks_dir = REPO_ROOT / args.chunks_dir
    parquet_files = sorted(chunks_dir.glob("strategy=*/language=*/part.parquet"))

    if not parquet_files:
        print(f"FATAL: no chunk files found under {chunks_dir}")
        sys.exit(1)

    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} chunks from {len(parquet_files)} partition file(s)")

    required_cols = {"chunk_strategy", "chunk_text_hash", "is_duplicate_chunk", "duplicate_of_chunk_id",
                      "below_min_length", "parent_passage_id", "query_id", "is_selected", "char_count"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"FATAL: chunk output is missing required columns for comparison: {missing}. "
              f"Re-run scripts/chunk_dataset.py with the updated code first.")
        sys.exit(1)

    total_chunks = len(df)

    # 1. chunks per strategy
    chunks_per_strategy = df["chunk_strategy"].value_counts().to_dict()

    # 2. unique vs duplicate chunk TEXT — two views:
    #    a) provenance-scoped (what chunk_dataset.py already flagged: dedup only within same parent+language)
    #    b) global text uniqueness (informational only — NOT used for any dedup action,
    #       since collapsing across different parent passages is explicitly disallowed)
    provenance_scoped_duplicates = int(df["is_duplicate_chunk"].sum())
    provenance_scoped_canonical = total_chunks - provenance_scoped_duplicates

    global_text_hash_counts = df["chunk_text_hash"].value_counts()
    global_unique_texts = int((global_text_hash_counts == 1).sum()) + int(len(global_text_hash_counts))
    # (the above double counts groups of size 1 — fix properly below)
    global_unique_texts = int(len(global_text_hash_counts))
    global_texts_appearing_more_than_once = int((global_text_hash_counts > 1).sum())
    global_duplicate_chunk_instances = int(total_chunks - global_unique_texts)

    # 3. average chunks per parent passage / per source record
    n_parent_passages = df["parent_passage_id"].nunique()
    n_source_records = df["query_id"].nunique()
    avg_chunks_per_passage = round(total_chunks / n_parent_passages, 3) if n_parent_passages else None
    avg_chunks_per_record = round(total_chunks / n_source_records, 3) if n_source_records else None

    # 4. selected vs non-selected
    selected = int(df["is_selected"].sum())
    non_selected = total_chunks - selected

    # 5. chunk length distribution (overall, and per strategy)
    df["_length_bucket"] = df["char_count"].apply(bucket_for)
    length_distribution_overall = df["_length_bucket"].value_counts().to_dict()
    length_distribution_by_strategy = {
        strat: group["_length_bucket"].value_counts().to_dict()
        for strat, group in df.groupby("chunk_strategy")
    }

    # 6. below-min-length percentage
    below_min = int(df["below_min_length"].sum())
    below_min_pct = round(below_min / total_chunks * 100, 2) if total_chunks else 0.0
    below_min_by_strategy = {
        strat: int(group["below_min_length"].sum()) for strat, group in df.groupby("chunk_strategy")
    }

    # 7. per-strategy duplicate rate (of THIS strategy's chunks, how many are
    #    flagged as duplicates of a chunk from a DIFFERENT/preferred strategy
    #    for the same passage — this directly answers "how much overlap
    #    exists between strategies")
    dup_rate_by_strategy = {}
    for strat, group in df.groupby("chunk_strategy"):
        n = len(group)
        n_dup = int(group["is_duplicate_chunk"].sum())
        dup_rate_by_strategy[strat] = {
            "total_chunks": n,
            "duplicate_of_another_chunk": n_dup,
            "duplicate_rate_pct": round(n_dup / n * 100, 2) if n else 0.0,
        }

    # 8. compactness (average length) per strategy
    avg_length_by_strategy = {
        strat: round(group["char_count"].mean(), 2) for strat, group in df.groupby("chunk_strategy")
    }
    most_compact_strategy = min(avg_length_by_strategy, key=avg_length_by_strategy.get)
    least_compact_strategy = max(avg_length_by_strategy, key=avg_length_by_strategy.get)

    fewest_chunks_strategy = min(chunks_per_strategy, key=chunks_per_strategy.get)
    most_chunks_strategy = max(chunks_per_strategy, key=chunks_per_strategy.get)

    # 9. selected passages represented (unique parent_passage_id with is_selected=1
    #    that have at least one chunk in the output, by strategy)
    selected_passages_represented_by_strategy = {
        strat: int(group.loc[group["is_selected"] == 1, "parent_passage_id"].nunique())
        for strat, group in df.groupby("chunk_strategy")
    }

    report = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_dir": str(chunks_dir),
        "total_chunks_analyzed": total_chunks,
        "chunks_per_strategy": chunks_per_strategy,
        "duplication": {
            "provenance_scoped": {
                "description": "Duplicates ONLY within the same (parent_passage_id, language) — "
                                "this is what scripts/chunk_dataset.py already flags and is safe to act on.",
                "canonical_chunks": provenance_scoped_canonical,
                "duplicate_chunks": provenance_scoped_duplicates,
                "duplicate_ratio": round(provenance_scoped_duplicates / total_chunks, 4) if total_chunks else None,
            },
            "global_text_uniqueness_informational_only": {
                "description": "Text-identical chunks ACROSS THE ENTIRE CORPUS, including across "
                                "different source passages. INFORMATIONAL ONLY — never used to drop "
                                "chunks, since collapsing across different parent passages would "
                                "conflate distinct source documents.",
                "unique_texts": global_unique_texts,
                "texts_appearing_more_than_once": global_texts_appearing_more_than_once,
                "total_instances_beyond_first_occurrence": global_duplicate_chunk_instances,
            },
        },
        "average_chunks_per_parent_passage": avg_chunks_per_passage,
        "average_chunks_per_source_record": avg_chunks_per_record,
        "unique_parent_passages": n_parent_passages,
        "unique_source_records": n_source_records,
        "selected_vs_non_selected": {"selected": selected, "non_selected": non_selected},
        "selected_passages_represented_by_strategy": selected_passages_represented_by_strategy,
        "chunk_length_distribution_overall": length_distribution_overall,
        "chunk_length_distribution_by_strategy": length_distribution_by_strategy,
        "below_min_length": {
            "total": below_min,
            "percent_of_all_chunks": below_min_pct,
            "by_strategy": below_min_by_strategy,
        },
        "duplicate_rate_by_strategy": dup_rate_by_strategy,
        "average_chunk_length_chars_by_strategy": avg_length_by_strategy,
        "summary_answers": {
            "strategy_with_fewest_chunks": fewest_chunks_strategy,
            "strategy_with_most_chunks": most_chunks_strategy,
            "most_compact_strategy_by_avg_length": most_compact_strategy,
            "least_compact_strategy_by_avg_length": least_compact_strategy,
            "overlap_between_strategies_pct_of_total": round(
                provenance_scoped_duplicates / total_chunks * 100, 2
            ) if total_chunks else None,
            "genuinely_unique_chunks_provenance_scoped": provenance_scoped_canonical,
            "note_on_retrieval_quality": (
                "This report does NOT determine which strategy is best for retrieval. "
                "That requires embeddings and Recall@K/MRR/nDCG evaluation, planned for Phase 4."
            ),
        },
    }

    out_path = REPO_ROOT / "data" / "processed" / "chunk_strategy_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nComparison report written to: {out_path}")
    print(f"Chunks per strategy: {chunks_per_strategy}")
    print(f"Provenance-scoped duplicate ratio: {report['duplication']['provenance_scoped']['duplicate_ratio']}")
    print(f"Most compact strategy: {most_compact_strategy} | Least compact: {least_compact_strategy}")


if __name__ == "__main__":
    main()

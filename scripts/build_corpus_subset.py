"""
Phase 4 — Step 2: Build Retrieval Corpus Subset

For the first embedding/retrieval benchmark we do NOT embed the entire
canonical chunk corpus. This script builds a representative SUBSET:

    1. All chunks belonging to the benchmark queries' own source records
       (this guarantees every ground-truth-relevant chunk is present in
       the corpus — otherwise Recall@K would be meaningless).
    2. Hard negatives: non-selected passages from the SAME source records
       as benchmark queries (is_selected=0, same query_id) — these are
       "hard" because they are topically adjacent (same source document)
       but not the correct answer passage.
    3. Random negatives: a deterministic random sample of chunks from
       OTHER records entirely, to ensure the corpus isn't trivially small
       and retrieval has to actually discriminate across documents.

Sampling methodology (documented, not hidden):
    - Positive set: every chunk whose parent_passage_id is in ANY benchmark
      query's ground_truth_parent_passage_ids.
    - Hard negative set: every OTHER chunk sharing query_id with a
      benchmark query (i.e. sibling passages from the same record).
    - Random negative set: deterministically seeded random sample of
      remaining chunks, sized to reach --target-corpus-size in total.

Usage:
    python scripts/build_corpus_subset.py \
        --benchmark-queries data/processed/retrieval_benchmark_queries.jsonl \
        --chunks-dir data/processed/chunks_full \
        --output data/processed/retrieval_corpus_subset.parquet \
        --target-corpus-size 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Build a representative retrieval corpus subset")
    parser.add_argument("--benchmark-queries", required=True)
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--output", default="data/processed/retrieval_corpus_subset.parquet")
    parser.add_argument("--target-corpus-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies", default=None,
        help="Comma-separated strategies to include in the corpus (default: all present in chunks-dir)",
    )
    args = parser.parse_args()

    import pandas as pd

    chunks_dir = REPO_ROOT / args.chunks_dir
    parquet_files = sorted(chunks_dir.glob("strategy=*/language=*/part.parquet"))
    if not parquet_files:
        print(f"FATAL: no chunk files found under {chunks_dir}")
        sys.exit(1)

    if args.strategies:
        wanted = set(s.strip() for s in args.strategies.split(","))
        parquet_files = [pf for pf in parquet_files if pf.parent.parent.name.split("=")[1] in wanted]

    print(f"Loading {len(parquet_files)} chunk partition(s)...")
    chunk_df = pd.concat([pd.read_parquet(pf) for pf in parquet_files], ignore_index=True)
    print(f"Total chunks available: {len(chunk_df)}")

    with open(REPO_ROOT / args.benchmark_queries, "r", encoding="utf-8") as f:
        benchmark = [json.loads(line) for line in f]
    print(f"Loaded {len(benchmark)} benchmark queries")

    benchmark_query_ids = set(b["query_id"] for b in benchmark)
    positive_passage_ids = set()
    for b in benchmark:
        positive_passage_ids.update(b["ground_truth_parent_passage_ids"])

    # 1. Positive chunks: parent_passage_id is a ground-truth-relevant passage
    is_positive = chunk_df["parent_passage_id"].isin(positive_passage_ids)
    positive_chunks = chunk_df[is_positive]

    # 2. Hard negatives: same query_id as a benchmark query, but NOT a positive passage
    is_same_record = chunk_df["query_id"].isin(benchmark_query_ids)
    hard_negative_chunks = chunk_df[is_same_record & ~is_positive]

    # 3. Random negatives: everything else, deterministically sampled to fill the target size
    remaining_pool = chunk_df[~is_same_record]
    already_selected = len(positive_chunks) + len(hard_negative_chunks)
    n_random_needed = max(0, args.target_corpus_size - already_selected)
    n_random = min(n_random_needed, len(remaining_pool))

    rng = random.Random(args.seed)
    if n_random > 0 and len(remaining_pool) > 0:
        sampled_idx = rng.sample(range(len(remaining_pool)), n_random)
        random_negative_chunks = remaining_pool.iloc[sampled_idx]
    else:
        random_negative_chunks = remaining_pool.iloc[0:0]

    corpus = pd.concat([positive_chunks, hard_negative_chunks, random_negative_chunks], ignore_index=True)
    corpus = corpus.drop_duplicates(subset=["chunk_id"])

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output_path, index=False)

    # Verify: every ground-truth passage_id must have at least one chunk in the corpus,
    # for at least one strategy, or Recall@K will be structurally impossible to satisfy.
    missing_ground_truth = positive_passage_ids - set(corpus["parent_passage_id"])

    manifest = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_dir": str(chunks_dir),
        "benchmark_queries_file": args.benchmark_queries,
        "seed": args.seed,
        "target_corpus_size": args.target_corpus_size,
        "actual_corpus_size": len(corpus),
        "positive_chunks": len(positive_chunks),
        "hard_negative_chunks": len(hard_negative_chunks),
        "random_negative_chunks": len(random_negative_chunks),
        "positive_parent_passage_ids": len(positive_passage_ids),
        "missing_ground_truth_passage_ids": list(missing_ground_truth),
        "missing_ground_truth_count": len(missing_ground_truth),
        "chunks_by_strategy": corpus["chunk_strategy"].value_counts().to_dict(),
        "output_file": str(output_path),
        "sampling_methodology": (
            "positive = all chunks whose parent_passage_id is a benchmark query's ground truth; "
            "hard_negative = other chunks sharing query_id with a benchmark query (sibling passages); "
            "random_negative = seeded random sample of chunks from unrelated records, filling up to target_corpus_size"
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus subset written to: {output_path} ({len(corpus)} chunks)")
    print(f"  positive: {len(positive_chunks)}, hard_negative: {len(hard_negative_chunks)}, "
          f"random_negative: {len(random_negative_chunks)}")
    if missing_ground_truth:
        print(f"WARNING: {len(missing_ground_truth)} ground-truth passage_id(s) have NO chunk in this corpus "
              f"(likely because --strategies excluded the strategy that contains them, or they were dropped "
              f"as duplicates upstream). Recall@K cannot reach 100% for these queries. See manifest for the list.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

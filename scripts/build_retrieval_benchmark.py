"""
Phase 4 — Step 1: Build Retrieval Benchmark (evaluation queries)

Builds a deterministic, stratified sample of evaluation queries from the
normalized MSMARCO-XI records. Each benchmark entry carries the query text
(target-language and English), query_type, and GROUND-TRUTH relevance
(which parent_passage_id(s) / chunk_id(s) are relevant), derived from
`is_selected` — used ONLY as evaluation ground truth here, never fed to the
retrieval system as an input signal.

CRITICAL ANTI-LEAKAGE: Answer / Eng_Answer are read only to decide
ANSWER_PRESENT vs NO_ANSWER bookkeeping; their text is never stored as
retrieval input in the benchmark file.

Stratification: approximately preserves the source query_type distribution
across DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION, via deterministic
seeded sampling (--seed, default 42) — same seed always produces the same
benchmark set for reproducibility.

Usage:
    python scripts/build_retrieval_benchmark.py \
        --input data/processed/normalized_full/normalized.parquet \
        --output data/processed/retrieval_benchmark_queries.jsonl \
        --size 1000 --seed 42

Output: one JSON object per line (JSONL) with:
    query_id, query, Eng_Query, query_type, source_lang, target_lang,
    has_answer, answer_status, ground_truth_parent_passage_ids,
    ground_truth_chunk_ids_by_strategy (populated only if a chunk output
    dir is supplied via --chunks-dir, since chunk_ids depend on chunking)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.ingestion.reconstruct import reconstruct_record  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build stratified retrieval evaluation benchmark")
    parser.add_argument("--input", required=True, help="Path to normalized.parquet")
    parser.add_argument("--output", default="data/processed/retrieval_benchmark_queries.jsonl")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunks-dir", default=None,
        help="Optional: path to a chunk output dir (strategy=*/language=*/part.parquet). "
             "If given, resolves ground_truth_chunk_ids per strategy for each selected passage. "
             "If omitted, only ground_truth_parent_passage_ids are recorded.",
    )
    parser.add_argument(
        "--require-answer", action="store_true",
        help="If set, only sample from ANSWER_PRESENT records (excludes NO_ANSWER records from "
             "the retrieval benchmark; NO_ANSWER records are still useful for guardrail testing "
             "in a later phase, so they are INCLUDED by default).",
    )
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} normalized records from {args.input}")

    if args.require_answer:
        df = df[df["has_answer"] == True]  # noqa: E712
        print(f"Filtered to ANSWER_PRESENT only: {len(df)} records")

    # Only sample from records that actually have at least one selected
    # passage (is_selected=1 somewhere) — otherwise there is no retrieval
    # ground truth to evaluate against for that query. NO_ANSWER records
    # with zero selected passages are legitimately excluded from THIS
    # benchmark (they belong to guardrail/no-answer evaluation instead).
    def has_selected(row):
        passages = json.loads(row["passages"]) if isinstance(row["passages"], str) else row["passages"]
        return any(p.get("is_selected") == 1 for p in passages)

    df["_has_selected"] = df.apply(has_selected, axis=1)
    eligible = df[df["_has_selected"]]
    print(f"Eligible records (>=1 selected passage): {len(eligible)}")

    # Stratify by query_type, proportional to eligible distribution
    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for _, row in eligible.iterrows():
        by_type[row["query_type"]].append(row.to_dict())

    type_counts = {k: len(v) for k, v in by_type.items()}
    total_eligible = sum(type_counts.values())
    print(f"Eligible distribution by query_type: {type_counts}")

    target_size = min(args.size, total_eligible)
    sampled_rows = []
    remaining = target_size
    types_sorted = sorted(by_type.keys())
    for i, qtype in enumerate(types_sorted):
        pool = by_type[qtype]
        rng.shuffle(pool)
        # proportional allocation, with the last type absorbing rounding remainder
        if i == len(types_sorted) - 1:
            n = remaining
        else:
            n = round(target_size * len(pool) / total_eligible)
            n = min(n, len(pool), remaining)
        sampled_rows.extend(pool[:n])
        remaining -= n

    rng.shuffle(sampled_rows)
    print(f"Sampled {len(sampled_rows)} benchmark queries (target was {target_size})")

    # Optionally resolve chunk-level ground truth
    chunk_lookup = None
    if args.chunks_dir:
        chunks_dir = REPO_ROOT / args.chunks_dir
        parquet_files = sorted(chunks_dir.glob("strategy=*/language=*/part.parquet"))
        if not parquet_files:
            print(f"WARNING: --chunks-dir given but no chunk files found at {chunks_dir}; "
                  f"ground_truth_chunk_ids_by_strategy will be empty.")
        else:
            print(f"Loading {len(parquet_files)} chunk partition(s) to resolve ground-truth chunk IDs...")
            chunk_dfs = [pd.read_parquet(pf) for pf in parquet_files]
            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            selected_chunks = chunk_df[chunk_df["is_selected"] == 1]
            chunk_lookup = defaultdict(lambda: defaultdict(list))
            for _, crow in selected_chunks.iterrows():
                chunk_lookup[crow["parent_passage_id"]][crow["chunk_strategy"]].append(crow["chunk_id"])

    output_records = []
    query_type_counts_final = defaultdict(int)
    for row in sampled_rows:
        record = reconstruct_record(row)
        selected_passage_ids = [
            f"{record.query_id}_p{p.passage_index}" for p in record.passages if p.is_selected == 1
        ]

        chunk_ids_by_strategy = {}
        if chunk_lookup is not None:
            for ppid in selected_passage_ids:
                for strat, cids in chunk_lookup.get(ppid, {}).items():
                    chunk_ids_by_strategy.setdefault(strat, []).extend(cids)

        entry = {
            "query_id": record.query_id,
            "query": record.query_target,
            "Eng_Query": record.query_english,
            "query_type": record.query_type,
            "source_lang": record.source_lang,
            "target_lang": record.target_lang,
            "has_answer": record.has_answer,
            "answer_status": record.answer_status,
            "ground_truth_parent_passage_ids": selected_passage_ids,
            "ground_truth_chunk_ids_by_strategy": chunk_ids_by_strategy,
        }
        output_records.append(entry)
        query_type_counts_final[record.query_type] += 1

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    manifest = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": args.input,
        "seed": args.seed,
        "requested_size": args.size,
        "actual_size": len(output_records),
        "eligible_pool_size": total_eligible,
        "eligible_distribution_by_query_type": type_counts,
        "final_benchmark_distribution_by_query_type": dict(query_type_counts_final),
        "require_answer_only": args.require_answer,
        "chunks_dir_used_for_ground_truth": args.chunks_dir,
        "output_file": str(output_path),
        "anti_leakage_note": (
            "Answer/Eng_Answer text is never stored in benchmark entries. "
            "is_selected is used only to compute ground_truth_parent_passage_ids "
            "(evaluation ground truth), never as a retrieval input."
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark queries written to: {output_path}")
    print(f"Manifest written to: {manifest_path}")
    print(f"Final distribution: {dict(query_type_counts_final)}")


if __name__ == "__main__":
    main()

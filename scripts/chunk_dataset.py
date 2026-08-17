"""
Phase 3 — Step 2: Multi-Strategy Chunking

Reads the normalized parquet produced by scripts/ingest_dataset.py, runs
all configured chunking strategies (passage, paragraph, sentence, adaptive)
over every record, and writes the resulting chunks as partitioned parquet:

    data/processed/chunks/strategy=<name>/language=<lang>/part.parquet

Also writes the final combined ingestion_report.json (merging
normalization-time stats with chunk-time stats), per Phase 3 spec section 10.

Usage:
    python scripts/chunk_dataset.py \
        --input data/processed/normalized/normalized.parquet \
        --output-dir data/processed/chunks \
        --limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.chunking.selector import run_strategies, ALL_STRATEGIES  # noqa: E402
from packages.chunking.chunk_dedup import run_chunk_dedup  # noqa: E402
from packages.ingestion.reconstruct import reconstruct_record as _reconstruct_record  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run multi-strategy chunking over normalized records")
    parser.add_argument("--input", required=True, help="Path to normalized.parquet")
    parser.add_argument("--output-dir", default="data/processed/chunks", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N normalized records")
    parser.add_argument(
        "--strategies", default=",".join(ALL_STRATEGIES),
        help="Comma-separated list of strategies to run",
    )
    parser.add_argument(
        "--normalization-report", default="data/processed/normalization_report.json",
        help="Path to the normalization_report.json produced by ingest_dataset.py, merged into the final report",
    )
    parser.add_argument(
        "--min-chunk-chars", type=int, default=10,
        help="Chunks shorter than this (in characters) are flagged (below_min_length=True), never silently deleted",
    )
    parser.add_argument(
        "--drop-duplicate-chunks", action="store_true",
        help="If set, physically exclude chunks flagged is_duplicate_chunk=True from the written output "
             "(they are always flagged in the report either way; this only controls whether they are WRITTEN)",
    )
    parser.add_argument(
        "--drop-below-min-length", action="store_true",
        help="If set, physically exclude chunks flagged below_min_length=True from the written output",
    )
    args = parser.parse_args()

    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"FATAL: input not found: {input_path}. Run scripts/ingest_dataset.py first.")
        sys.exit(1)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    output_dir = REPO_ROOT / args.output_dir

    t_start = time.time()

    df = pd.read_parquet(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} normalized records from {input_path}")
    print(f"Running strategies: {strategies}")

    all_chunk_dicts = []
    chunks_by_strategy = Counter()
    chunks_by_language = Counter()
    chunks_by_query_type = Counter()
    selected_chunks = 0
    non_selected_chunks = 0
    chunk_lengths = []

    total_canonical = 0
    total_duplicate_chunks = 0
    total_below_min_length = 0
    total_passage_groups = 0
    dropped_duplicates = 0
    dropped_below_min = 0

    for i, (_, row) in enumerate(df.iterrows()):
        record = _reconstruct_record(row.to_dict())
        chunks = run_strategies(record, strategies)

        # Provenance-scoped dedup + min-length flagging, per record (parent_passage_id
        # is unique per record+passage, so per-record scoping == global scoping here,
        # while keeping memory bounded for large datasets).
        dedup_stats = run_chunk_dedup(chunks, min_chars=args.min_chunk_chars)
        total_canonical += dedup_stats["canonical_chunks"]
        total_duplicate_chunks += dedup_stats["duplicate_chunks"]
        total_below_min_length += dedup_stats["below_min_length_chunks"]
        total_passage_groups += dedup_stats["passage_groups"]

        for c in chunks:
            if args.drop_duplicate_chunks and c.is_duplicate_chunk:
                dropped_duplicates += 1
                continue
            if args.drop_below_min_length and c.below_min_length:
                dropped_below_min += 1
                continue

            d = c.to_flat_dict()
            all_chunk_dicts.append(d)
            chunks_by_strategy[c.chunk_strategy] += 1
            chunks_by_language[c.language] += 1
            chunks_by_query_type[c.query_type] += 1
            if c.is_selected:
                selected_chunks += 1
            else:
                non_selected_chunks += 1
            chunk_lengths.append(c.char_count)

        if (i + 1) % 5000 == 0:
            print(f"  ... chunked {i + 1}/{len(df)} records, {len(all_chunk_dicts)} chunks written so far")

    print(f"Finished chunking. Total chunks generated (pre-drop): "
          f"{total_canonical + total_duplicate_chunks}")
    print(f"  Canonical (non-duplicate): {total_canonical}")
    print(f"  Duplicate (same text, same parent passage, different strategy): {total_duplicate_chunks}")
    print(f"  Below min-length ({args.min_chunk_chars} chars): {total_below_min_length}")
    if args.drop_duplicate_chunks:
        print(f"  Dropped from output (duplicate): {dropped_duplicates}")
    if args.drop_below_min_length:
        print(f"  Dropped from output (below min length): {dropped_below_min}")
    print(f"Total chunks WRITTEN to output: {len(all_chunk_dicts)}")

    chunks_df = pd.DataFrame(all_chunk_dicts)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    if len(chunks_df) > 0:
        for strategy, group in chunks_df.groupby("chunk_strategy"):
            strat_dir = output_dir / f"strategy={strategy}"
            strat_dir.mkdir(parents=True, exist_ok=True)
            for lang, lang_group in group.groupby("language"):
                lang_dir = strat_dir / f"language={lang}"
                lang_dir.mkdir(parents=True, exist_ok=True)
                out_file = lang_dir / "part.parquet"
                lang_group.to_parquet(out_file, index=False)
                written_files.append(str(out_file))
    else:
        print("WARNING: no chunks produced.")

    elapsed = time.time() - t_start

    # Merge with normalization report if available
    norm_report_path = REPO_ROOT / args.normalization_report
    norm_report = {}
    if norm_report_path.exists():
        with open(norm_report_path, "r", encoding="utf-8") as f:
            norm_report = json.load(f)
    else:
        print(f"WARNING: normalization report not found at {norm_report_path}; "
              f"final report will omit ingestion-time stats.")

    combined_report = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": norm_report.get("input_file"),
        "input_rows": norm_report.get("input_rows_examined"),
        "malformed_records": norm_report.get("malformed_records"),
        "empty_passages": norm_report.get("empty_passages_records"),
        "duplicate_query_ids": (
            norm_report.get("dedup_stats", {}).get("content_distinct_duplicate_query_ids")
        ),
        "exact_duplicate_records": (
            norm_report.get("dedup_stats", {}).get("exact_duplicates_dropped")
        ),
        "answer_present_records": norm_report.get("answer_present_records"),
        "no_answer_records": norm_report.get("no_answer_records"),
        "normalized_records_chunked": len(df),
        "output_chunks": len(all_chunk_dicts),
        "chunks_by_strategy": dict(chunks_by_strategy),
        "chunks_by_language": dict(chunks_by_language),
        "chunks_by_query_type": dict(chunks_by_query_type),
        "selected_chunks": selected_chunks,
        "non_selected_chunks": non_selected_chunks,
        "average_chunk_length_chars": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else None,
        "min_chunk_length_chars": min(chunk_lengths) if chunk_lengths else None,
        "max_chunk_length_chars": max(chunk_lengths) if chunk_lengths else None,
        "chunk_dedup": {
            "total_chunks_generated_pre_drop": total_canonical + total_duplicate_chunks,
            "canonical_chunks": total_canonical,
            "duplicate_chunks_same_parent_passage": total_duplicate_chunks,
            "duplicate_ratio": round(
                total_duplicate_chunks / (total_canonical + total_duplicate_chunks), 4
            ) if (total_canonical + total_duplicate_chunks) else None,
            "passage_groups_considered": total_passage_groups,
            "below_min_length_chunks": total_below_min_length,
            "min_chunk_chars_threshold": args.min_chunk_chars,
            "drop_duplicate_chunks_enabled": args.drop_duplicate_chunks,
            "drop_below_min_length_enabled": args.drop_below_min_length,
            "dropped_duplicate_chunks_from_output": dropped_duplicates,
            "dropped_below_min_length_from_output": dropped_below_min,
            "dedup_scope": "STRICTLY within (parent_passage_id, language) — never collapses chunks from different source passages",
        },
        "strategies_run": strategies,
        "output_directory": str(output_dir),
        "output_files_written": written_files,
        "normalization_processing_time_seconds": norm_report.get("processing_time_seconds"),
        "chunking_processing_time_seconds": round(elapsed, 3),
        "total_processing_time_seconds": round(
            (norm_report.get("processing_time_seconds") or 0) + elapsed, 3
        ),
    }

    report_path = REPO_ROOT / "data" / "processed" / "ingestion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)

    print(f"\nCombined ingestion_report.json written to: {report_path}")
    print(f"Chunk output written under: {output_dir}")
    print(f"Elapsed (chunking only): {elapsed:.2f}s")


if __name__ == "__main__":
    main()

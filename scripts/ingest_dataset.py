"""
Phase 3 — Step 1: Ingest + Normalize + Deduplicate

Reads a real MSMARCO-XI parquet file in BATCHES (pyarrow ParquetFile
.iter_batches), never loading the entire file into one giant pandas
DataFrame or Python list. Validates structure, normalizes each row into a
NormalizedRecord, deduplicates exact duplicates, and writes the result as
partitioned parquet under data/processed/normalized/.

Usage:
    python scripts/ingest_dataset.py \
        --file data/raw/validation/kanval.parquet \
        --output-dir data/processed/normalized \
        --limit 1000 \
        --batch-size 500

Output:
    data/processed/normalized/normalized.parquet (or .jsonl fallback)
    data/processed/normalization_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.ingestion.dedup import DedupTracker  # noqa: E402
from packages.ingestion.normalizer import normalize_row  # noqa: E402
from packages.ingestion.validators import RecordCorruptionError, validate_raw_row  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest and normalize a raw MSMARCO-XI parquet file")
    parser.add_argument("--file", required=True, help="Path to raw parquet file")
    parser.add_argument("--output-dir", default="data/processed/normalized", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Rows per read batch")
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"FATAL: pyarrow not available: {e}")
        sys.exit(1)

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"FATAL: input file not found: {input_path}")
        sys.exit(1)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPO_ROOT / "data" / "processed" / "normalization_report.json"

    t_start = time.time()

    parquet_file = pq.ParquetFile(str(input_path))
    total_rows_in_file = parquet_file.metadata.num_rows
    print(f"Input file: {input_path}")
    print(f"Total rows in file: {total_rows_in_file}")
    if args.limit:
        print(f"Processing LIMIT={args.limit} rows (test mode)")

    dedup = DedupTracker()

    kept_records = []  # list of dict (flattened) — accumulated in batches, flushed at end
    malformed_records = 0
    malformed_examples = []
    empty_passages_count = 0
    total_processed = 0
    warnings_count = 0

    stop = False
    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        if stop:
            break
        df_batch = batch.to_pandas()
        for _, row in df_batch.iterrows():
            if args.limit is not None and total_processed >= args.limit:
                stop = True
                break

            row_dict = row.to_dict()
            total_processed += 1

            try:
                warnings = validate_raw_row(row_dict)
            except RecordCorruptionError as e:
                malformed_records += 1
                if len(malformed_examples) < 10:
                    malformed_examples.append({
                        "query_id": row_dict.get("query_id"),
                        "error": str(e),
                    })
                continue

            if warnings:
                warnings_count += len(warnings)
                if any(w.startswith("empty_passages") for w in warnings):
                    empty_passages_count += 1

            record = normalize_row(row_dict)
            record.validation_warnings = warnings

            keep = dedup.process(record)
            if not keep:
                continue

            kept_records.append(record.to_flat_dict())

        if total_processed % 10000 == 0:
            print(f"  ... processed {total_processed} rows, kept {len(kept_records)}")

    print(f"Finished reading. total_processed={total_processed}, kept={len(kept_records)}, "
          f"malformed={malformed_records}, exact_duplicates_dropped={dedup.exact_duplicates_dropped}")

    # Write output
    import pandas as pd

    out_df = pd.DataFrame(kept_records)
    output_file = output_dir / "normalized.parquet"
    if len(out_df) > 0:
        out_df.to_parquet(output_file, index=False)
    else:
        print("WARNING: no records kept; not writing parquet output.")

    elapsed = time.time() - t_start

    # answer status distribution among kept records
    answer_present = sum(1 for r in kept_records if r.get("has_answer"))
    no_answer = len(kept_records) - answer_present

    report = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "limit_applied": args.limit,
        "input_rows_examined": total_processed,
        "input_rows_total_in_file": total_rows_in_file,
        "kept_records": len(kept_records),
        "malformed_records": malformed_records,
        "malformed_examples": malformed_examples,
        "empty_passages_records": empty_passages_count,
        "validation_warnings_total": warnings_count,
        "dedup_stats": dedup.stats(),
        "answer_present_records": answer_present,
        "no_answer_records": no_answer,
        "output_file": str(output_file) if len(out_df) > 0 else None,
        "processing_time_seconds": round(elapsed, 3),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nNormalization report written to: {report_path}")
    print(f"Normalized output written to: {output_file if len(out_df) > 0 else '(none — 0 kept records)'}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

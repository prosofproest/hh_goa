"""
Phase 3 — Step 3: Chunk Validation

Walks the partitioned chunk output directory and runs structural/consistency
checks. Exits with a non-zero status code if any CRITICAL check fails.

Checks performed:
    - no missing chunk_id
    - no missing query_id
    - no invalid language value (must be 'target' or 'english')
    - no invalid query_type (must be one of the observed set, or flagged)
    - no orphan chunks (parent_passage_id must be derivable from query_id+passage_index)
    - passage_index consistency (non-negative int)
    - is_selected consistency (0 or 1)
    - no accidental answer leakage (chunk text must not equal the record's answer_target/answer_english verbatim)
    - Unicode validity
    - duplicate chunk_id (globally, across all strategy/language partitions... duplicates
      are expected ACROSS strategies since the same passage produces one chunk per
      strategy; this check verifies uniqueness WITHIN a single strategy+language partition)
    - empty text
    - metadata consistency (metadata field must be valid JSON)

Usage:
    python scripts/validate_chunks.py --chunks-dir data/processed/chunks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_LANGUAGES = {"target", "english"}
KNOWN_QUERY_TYPES = {"DESCRIPTION", "NUMERIC", "ENTITY", "PERSON", "LOCATION"}


def main():
    parser = argparse.ArgumentParser(description="Validate chunk output")
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    args = parser.parse_args()

    import pandas as pd

    chunks_dir = REPO_ROOT / args.chunks_dir
    parquet_files = sorted(chunks_dir.glob("strategy=*/language=*/part.parquet"))

    if not parquet_files:
        print(f"CRITICAL: no chunk parquet files found under {chunks_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} chunk partition file(s)")

    critical_issues = []
    warnings = []

    total_chunks = 0
    seen_chunk_ids_by_partition = {}

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        total_chunks += len(df)
        partition_key = str(pf.relative_to(chunks_dir))

        # missing chunk_id / query_id
        if df["chunk_id"].isna().any():
            critical_issues.append(f"{partition_key}: missing chunk_id in {df['chunk_id'].isna().sum()} row(s)")
        if df["query_id"].isna().any():
            critical_issues.append(f"{partition_key}: missing query_id in {df['query_id'].isna().sum()} row(s)")

        # duplicate chunk_id within this partition
        dup_mask = df["chunk_id"].duplicated()
        if dup_mask.any():
            critical_issues.append(f"{partition_key}: {dup_mask.sum()} duplicate chunk_id(s) within partition")
        seen_chunk_ids_by_partition[partition_key] = set(df["chunk_id"])

        # invalid language
        bad_lang = ~df["language"].isin(VALID_LANGUAGES)
        if bad_lang.any():
            critical_issues.append(f"{partition_key}: {bad_lang.sum()} row(s) with invalid language value")

        # query_type sanity (warning only — dataset could legitimately have others)
        unknown_qt = ~df["query_type"].isin(KNOWN_QUERY_TYPES)
        if unknown_qt.any():
            warnings.append(
                f"{partition_key}: {unknown_qt.sum()} row(s) with query_type outside known set {KNOWN_QUERY_TYPES}: "
                f"{sorted(df.loc[unknown_qt, 'query_type'].unique().tolist())[:10]}"
            )

        # passage_index consistency
        bad_idx = df["passage_index"] < 0
        if bad_idx.any():
            critical_issues.append(f"{partition_key}: {bad_idx.sum()} row(s) with negative passage_index")

        # is_selected consistency
        bad_sel = ~df["is_selected"].isin([0, 1])
        if bad_sel.any():
            critical_issues.append(f"{partition_key}: {bad_sel.sum()} row(s) with invalid is_selected value")

        # empty text
        empty_text = df["text"].fillna("").str.strip() == ""
        if empty_text.any():
            critical_issues.append(f"{partition_key}: {empty_text.sum()} row(s) with empty text")

        # orphan chunk check: parent_passage_id should equal f"{query_id}_p{passage_index}"
        expected_parent = df["query_id"].astype(str) + "_p" + df["passage_index"].astype(str)
        orphan_mask = df["parent_passage_id"].astype(str) != expected_parent
        if orphan_mask.any():
            critical_issues.append(f"{partition_key}: {orphan_mask.sum()} row(s) with inconsistent parent_passage_id")

        # answer leakage check
        # (text should not be verbatim-equal to the record's answer field for that language)
        # We only have text/text_target/text_english here, not the original answer fields,
        # so this checks a weaker but still useful invariant: text must not be exactly the
        # literal string "No Answer Present." (which would indicate the answer field was
        # mistakenly used as passage content upstream).
        leaked = df["text"].fillna("").str.strip().str.lower() == "no answer present."
        if leaked.any():
            critical_issues.append(
                f"{partition_key}: {leaked.sum()} row(s) where chunk text literally equals the "
                f"no-answer marker — possible answer-field leakage into passage content"
            )

        # Unicode validity (should never fail given prior NFC normalization, but re-verify)
        try:
            df["text"].fillna("").apply(lambda s: str(s).encode("utf-8"))
        except UnicodeEncodeError as e:
            critical_issues.append(f"{partition_key}: unicode encode failure: {e}")

        # metadata JSON validity
        def _check_json(s):
            try:
                json.loads(s) if s else None
                return True
            except Exception:
                return False

        bad_meta = ~df["metadata"].apply(_check_json)
        if bad_meta.any():
            critical_issues.append(f"{partition_key}: {bad_meta.sum()} row(s) with invalid metadata JSON")

    print(f"\nTotal chunks validated: {total_chunks}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings[:20]:
        print(f"  WARNING: {w}")

    print(f"Critical issues: {len(critical_issues)}")
    for c in critical_issues[:50]:
        print(f"  CRITICAL: {c}")

    result = {
        "total_chunks_validated": total_chunks,
        "partitions_checked": len(parquet_files),
        "warnings": warnings,
        "critical_issues": critical_issues,
        "passed": len(critical_issues) == 0,
    }
    out_path = REPO_ROOT / "data" / "processed" / "chunk_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nValidation report written to: {out_path}")

    if critical_issues:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    else:
        print("\nVALIDATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
Phase 2 — MSMARCO-XI Dataset Discovery and Inspection (LOCAL PARQUET MODE)
============================================================================

This script inspects a REAL, already-downloaded MSMARCO-XI parquet file
directly from disk (e.g. validation/kanval.parquet, obtained via the
Hugging Face `datasets` library or `hf download`).

It performs NO network calls and makes NO assumptions about schema beyond
what is physically observed in the file. Every field, dtype, nested key,
and sample record in the output report is read directly from the data.

Known real top-level columns (as verified against kanval.parquet), used
only as detection hints — NOT hardcoded assumptions. The script inspects
whatever columns actually exist, and will report if the observed columns
differ from these hints:

    source_lang, target_lang, meta, Answer, query_id, query_type,
    passages, Eng_Query, Eng_Answer, query

Usage:
    python scripts/inspect_dataset.py --file path/to/kanval.parquet
    python scripts/inspect_dataset.py --file path/to/kanval.parquet --sample-size 5

Output:
    data/processed/dataset_schema.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "dataset_schema.json"

# Hints only — used to label detected fields in the report, never to invent data.
FIELD_ROLE_HINTS = {
    "query": "target_language_query_field",
    "Eng_Query": "english_query_field",
    "Answer": "target_language_answer_field",
    "Eng_Answer": "english_answer_field",
    "passages": "context_passage_field_nested",
    "query_id": "record_identifier",
    "query_type": "query_category_label",
    "source_lang": "source_language_code",
    "target_lang": "target_language_code",
    "meta": "additional_metadata_object",
}

NO_ANSWER_MARKERS = {"no answer present.", "no answer present"}


def die_with_report(error: Exception, phase: str, extra: dict | None = None) -> None:
    report = {
        "status": "FAILED",
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "note": (
            "Inspection could not be completed. No schema, sample records, "
            "or statistics have been fabricated."
        ),
    }
    if extra:
        report.update(extra)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"FAILED at phase '{phase}': {error}")
    print(f"Honest failure report written to: {OUTPUT_PATH}")
    sys.exit(1)


def jsonable(value):
    """Best-effort conversion of arbitrary parquet-loaded values (numpy arrays,
    pandas scalars, nested dict/list/object columns) into JSON-serializable
    Python primitives, WITHOUT altering or inventing content."""
    try:
        import numpy as np
    except ImportError:
        np = None

    if value is None:
        return None
    if np is not None and isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if np is not None and isinstance(value, (np.generic,)):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def inspect_nested_passages(series, column_name: str) -> dict:
    """Inspect the structure of a nested 'passages'-like object column without
    assuming its inner keys ahead of time."""
    nested_report = {
        "observed_inner_keys": None,
        "inner_key_types": {},
        "lengths_observed": {},
        "sample_structure": None,
    }

    non_null = series.dropna()
    if len(non_null) == 0:
        nested_report["warning"] = f"No non-null values found in '{column_name}'"
        return nested_report

    first_val = non_null.iloc[0]
    conv = jsonable(first_val)

    if isinstance(conv, dict):
        nested_report["observed_inner_keys"] = list(conv.keys())
        for k, v in conv.items():
            if isinstance(v, list):
                nested_report["inner_key_types"][k] = f"list[{type(v[0]).__name__ if v else 'unknown'}], len={len(v)}"
            else:
                nested_report["inner_key_types"][k] = type(v).__name__

        # Collect length stats for list-valued inner keys across a sample of rows
        list_keys = [k for k, v in conv.items() if isinstance(v, list)]
        for k in list_keys:
            lengths = []
            for val in non_null.head(500):
                c = jsonable(val)
                if isinstance(c, dict) and isinstance(c.get(k), list):
                    lengths.append(len(c[k]))
            if lengths:
                nested_report["lengths_observed"][k] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "distinct_values": sorted(set(lengths))[:20],
                }

        nested_report["sample_structure"] = conv
    else:
        nested_report["observed_inner_keys"] = "NOT_A_DICT"
        nested_report["raw_type"] = type(first_val).__name__
        nested_report["sample_value"] = conv

    return nested_report


def main():
    parser = argparse.ArgumentParser(description="Inspect a downloaded MSMARCO-XI parquet file")
    parser.add_argument("--file", required=True, help="Path to the parquet file, e.g. validation/kanval.parquet")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of full sample records to include")
    parser.add_argument(
        "--split-label", default=None,
        help="Human label for what this file represents, e.g. 'validation/kanval.parquet (Kannada)'"
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        die_with_report(e, phase="import pandas")
        return

    file_path = Path(args.file)
    if not file_path.exists():
        die_with_report(
            FileNotFoundError(f"File not found: {file_path}"),
            phase="locate_parquet_file",
        )
        return

    try:
        print(f"Reading {file_path} ...")
        df = pd.read_parquet(file_path)
    except Exception as e:
        die_with_report(e, phase="read_parquet")
        return

    print(f"Loaded shape: {df.shape}")

    report = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "ai4bharat/MSMARCO-XI",
        "verified_dataset_scope": {
            "physically_inspected_file": str(file_path),
            "split_label": args.split_label or str(file_path),
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "scope_warning": (
                "This report reflects ONLY the physically inspected file above. "
                "MSMARCO-XI is a multilingual dataset with (per the dataset's own "
                "structure) multiple target-language configs and splits. Values, "
                "distributions, and structure observed here (language codes, "
                "query types, passage counts, no-answer rate, etc.) must NOT be "
                "assumed to hold for other language configs or splits (e.g. "
                "train, test, or other target languages) until those are "
                "independently inspected."
            ),
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "field_role_hints": {
            col: FIELD_ROLE_HINTS[col] for col in df.columns if col in FIELD_ROLE_HINTS
        },
        "unmapped_columns": [col for col in df.columns if col not in FIELD_ROLE_HINTS],
    }

    # Null / missing value check per column
    report["null_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}

    # Simple duplicate check on query_id if present
    if "query_id" in df.columns:
        dup_count = int(df["query_id"].duplicated().sum())
        report["duplicate_query_ids"] = dup_count

    # Language field value distributions
    for lang_col in ("source_lang", "target_lang"):
        if lang_col in df.columns:
            report[f"{lang_col}_value_counts"] = df[lang_col].value_counts(dropna=False).to_dict()

    # query_type distribution
    if "query_type" in df.columns:
        report["query_type_distribution"] = {
            str(k): int(v) for k, v in df["query_type"].value_counts(dropna=False).items()
        }

    # Nested passages structure
    if "passages" in df.columns:
        report["passages_structure"] = inspect_nested_passages(df["passages"], "passages")

        # is_selected / passage-count statistics across full dataset (not just sample)
        selected_counts = []
        passage_counts = []
        for val in df["passages"]:
            conv = jsonable(val)
            if not isinstance(conv, dict):
                continue
            eng_p = conv.get("English_passages")
            is_sel = conv.get("is_selected")
            if isinstance(eng_p, list):
                passage_counts.append(len(eng_p))
            if isinstance(is_sel, list):
                selected_counts.append(sum(1 for x in is_sel if int(x) == 1))

        if passage_counts:
            report["passages_per_record_stats"] = {
                "min": min(passage_counts),
                "max": max(passage_counts),
                "distinct_values": sorted(set(passage_counts))[:20],
            }
        if selected_counts:
            report["selected_passages_per_record_stats"] = {
                "min": min(selected_counts),
                "max": max(selected_counts),
                "records_with_zero_selected": sum(1 for c in selected_counts if c == 0),
                "records_with_one_or_more_selected": sum(1 for c in selected_counts if c >= 1),
            }

    # meta field structure (inspect but don't assume keys)
    if "meta" in df.columns:
        non_null_meta = df["meta"].dropna()
        if len(non_null_meta) > 0:
            sample_meta = jsonable(non_null_meta.iloc[0])
            report["meta_field_sample_structure"] = sample_meta
            if isinstance(sample_meta, dict):
                report["meta_field_observed_keys"] = list(sample_meta.keys())

    # No-answer detection, based on Eng_Answer (language-independent marker)
    if "Eng_Answer" in df.columns:
        no_answer_mask = df["Eng_Answer"].astype(str).str.strip().str.lower().isin(NO_ANSWER_MARKERS)
        no_answer_count = int(no_answer_mask.sum())
        report["no_answer_present_count"] = no_answer_count
        report["no_answer_present_ratio"] = round(no_answer_count / len(df), 4) if len(df) else None

        # Cross-check: for a sample of no-answer rows, confirm is_selected is all-zero
        if "passages" in df.columns and no_answer_count > 0:
            no_answer_rows = df[no_answer_mask].head(20)
            all_zero_confirmed = 0
            checked = 0
            for _, row in no_answer_rows.iterrows():
                conv = jsonable(row["passages"])
                if isinstance(conv, dict) and isinstance(conv.get("is_selected"), list):
                    checked += 1
                    if all(int(x) == 0 for x in conv["is_selected"]):
                        all_zero_confirmed += 1
            report["no_answer_is_selected_all_zero_check"] = {
                "checked_sample_size": checked,
                "confirmed_all_zero": all_zero_confirmed,
            }

    # Basic text length stats for query / Eng_Query
    for col in ("query", "Eng_Query", "Answer", "Eng_Answer"):
        if col in df.columns:
            lengths = df[col].dropna().astype(str).str.len()
            if len(lengths) > 0:
                report.setdefault("text_length_stats", {})[col] = {
                    "min_chars": int(lengths.min()),
                    "max_chars": int(lengths.max()),
                    "mean_chars": round(float(lengths.mean()), 2),
                    "median_chars": float(lengths.median()),
                }

    # Real, complete sample records (raw, converted to JSON-safe form)
    n = min(args.sample_size, len(df))
    sample_records = []
    for _, row in df.head(n).iterrows():
        sample_records.append({col: jsonable(row[col]) for col in df.columns})
    report["sample_records"] = sample_records

    # Approximate on-disk size of the inspected file
    report["file_size_bytes"] = file_path.stat().st_size
    report["file_size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print(f"Inspection complete. Report written to: {OUTPUT_PATH}")
    print("=" * 70)
    print(f"Rows: {report['verified_dataset_scope']['row_count']}")
    print(f"Columns: {report['columns']}")
    if "no_answer_present_count" in report:
        print(f"No-answer records: {report['no_answer_present_count']} ({report['no_answer_present_ratio']*100:.2f}%)")


if __name__ == "__main__":
    main()

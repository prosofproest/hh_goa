"""
Reconstructs a NormalizedRecord from a flattened row of normalized.parquet
(the inverse of NormalizedRecord.to_flat_dict()). Shared by
scripts/chunk_dataset.py and Phase 4 scripts so this logic exists in
exactly one place.
"""

from __future__ import annotations

import json

from .schema import NormalizedRecord, PassageEntry


def reconstruct_record(row: dict) -> NormalizedRecord:
    passages_raw = json.loads(row["passages"]) if isinstance(row["passages"], str) else row["passages"]
    passages = [
        PassageEntry(
            passage_index=p["passage_index"],
            text_english=p["text_english"],
            text_target=p["text_target"],
            is_selected=p["is_selected"],
        )
        for p in passages_raw
    ]
    meta_raw = row.get("meta")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else None

    warnings_raw = row.get("validation_warnings")
    warnings = json.loads(warnings_raw) if isinstance(warnings_raw, str) and warnings_raw else []

    return NormalizedRecord(
        query_id=str(row["query_id"]),
        raw_query_id=int(row["raw_query_id"]),
        source_lang=row["source_lang"],
        target_lang=row["target_lang"],
        query_type=row["query_type"],
        query_target=row["query_target"],
        query_english=row["query_english"],
        answer_target=row["answer_target"],
        answer_english=row["answer_english"],
        has_answer=bool(row["has_answer"]),
        answer_status=row["answer_status"],
        passages=passages,
        meta=meta,
        source_record_hash=row["source_record_hash"],
        is_duplicate_query_id=bool(row.get("is_duplicate_query_id", False)),
        is_exact_duplicate=bool(row.get("is_exact_duplicate", False)),
        validation_warnings=warnings,
    )

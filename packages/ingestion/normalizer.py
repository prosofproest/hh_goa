"""
Raw row -> NormalizedRecord conversion.

This is intentionally a thin, explicit mapping (no guessing): every target
field is sourced from exactly one verified raw column.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from .schema import NormalizedRecord, PassageEntry, canonicalize_for_hash, compute_answer_status


def _nfc(text: Any) -> str:
    """Unicode NFC normalization; safe no-op for None/non-str."""
    if text is None:
        return ""
    s = str(text)
    return unicodedata.normalize("NFC", s)


def _as_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def normalize_row(row: dict[str, Any]) -> NormalizedRecord:
    """Convert one validated raw row into a NormalizedRecord.

    Assumes the row has already passed `validators.validate_raw_row`
    (structural corruption raises before this is called).
    """
    raw_query_id = int(row["query_id"])

    passages_raw = row["passages"]
    eng_p = _as_list(passages_raw.get("English_passages"))
    trans_p = _as_list(passages_raw.get("Translated_passages"))
    is_sel = _as_list(passages_raw.get("is_selected"))

    passages: list[PassageEntry] = []
    for i in range(len(trans_p)):
        passages.append(
            PassageEntry(
                passage_index=i,
                text_english=_nfc(eng_p[i]) if i < len(eng_p) else "",
                text_target=_nfc(trans_p[i]),
                is_selected=int(is_sel[i]) if i < len(is_sel) and str(is_sel[i]).strip() != "" else 0,
            )
        )

    eng_answer = _nfc(row.get("Eng_Answer"))
    has_answer, answer_status = compute_answer_status(eng_answer)

    meta_val = row.get("meta")
    if meta_val is not None and hasattr(meta_val, "tolist"):
        meta_val = meta_val.tolist()

    record = NormalizedRecord(
        query_id=str(raw_query_id),
        raw_query_id=raw_query_id,
        source_lang=_nfc(row.get("source_lang")),
        target_lang=_nfc(row.get("target_lang")),
        query_type=_nfc(row.get("query_type")),
        query_target=_nfc(row.get("query")),
        query_english=_nfc(row.get("Eng_Query")),
        answer_target=_nfc(row.get("Answer")),
        answer_english=eng_answer,
        has_answer=has_answer,
        answer_status=answer_status,
        passages=passages,
        meta=meta_val if isinstance(meta_val, dict) else ({"raw": meta_val} if meta_val not in (None, "") else None),
        source_record_hash=canonicalize_for_hash(row),
    )
    return record

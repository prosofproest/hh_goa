"""
Structural validation for raw MSMARCO-XI rows, applied BEFORE normalization.

Detects (without silently discarding):
    - null/missing required fields
    - malformed nested `passages` object
    - mismatched list lengths inside `passages`
    - empty passage lists
    - non-Unicode-safe text

Returns a list of warning strings per record; the caller decides whether a
record is unusable (fails loudly) or usable-with-warnings.
"""

from __future__ import annotations

from typing import Any

REQUIRED_TOP_LEVEL_FIELDS = [
    "source_lang",
    "target_lang",
    "query_id",
    "query_type",
    "passages",
    "Eng_Query",
    "Eng_Answer",
    "query",
    "Answer",
]

REQUIRED_PASSAGE_KEYS = ["English_passages", "Translated_passages", "is_selected"]


class RecordCorruptionError(Exception):
    """Raised when a record is structurally unusable (not just imperfect)."""


def validate_raw_row(row: dict[str, Any]) -> list[str]:
    """Validate one raw row. Returns a list of non-fatal warnings.
    Raises RecordCorruptionError for unrecoverable structural corruption.
    """
    warnings: list[str] = []

    for f in REQUIRED_TOP_LEVEL_FIELDS:
        if f not in row:
            raise RecordCorruptionError(f"Missing required top-level field: '{f}'")

    if row.get("query_id") is None:
        raise RecordCorruptionError("query_id is null")

    passages = row.get("passages")
    if passages is None:
        raise RecordCorruptionError("passages is null")

    if not isinstance(passages, dict):
        raise RecordCorruptionError(f"passages is not a dict/object (got {type(passages).__name__})")

    for key in REQUIRED_PASSAGE_KEYS:
        if key not in passages:
            raise RecordCorruptionError(f"passages missing required key: '{key}'")

    eng_p = passages.get("English_passages")
    trans_p = passages.get("Translated_passages")
    is_sel = passages.get("is_selected")

    # Coerce numpy arrays to lists for length checks
    def _as_list(x):
        if x is None:
            return None
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    eng_p_l = _as_list(eng_p)
    trans_p_l = _as_list(trans_p)
    is_sel_l = _as_list(is_sel)

    if eng_p_l is None or trans_p_l is None or is_sel_l is None:
        raise RecordCorruptionError("One or more passage sub-fields is null")

    if len(eng_p_l) == 0 and len(trans_p_l) == 0:
        warnings.append("empty_passages: both English_passages and Translated_passages are empty")

    if not (len(eng_p_l) == len(trans_p_l) == len(is_sel_l)):
        raise RecordCorruptionError(
            f"Mismatched passage list lengths: English_passages={len(eng_p_l)}, "
            f"Translated_passages={len(trans_p_l)}, is_selected={len(is_sel_l)}"
        )

    # Unicode sanity check (should never fail in Python 3 str, but guards against
    # surrogate-escaped/broken bytes that sometimes leak through parquet readers)
    for field_name in ("query", "Eng_Query", "Answer", "Eng_Answer"):
        val = row.get(field_name)
        if val is not None:
            try:
                str(val).encode("utf-8")
            except UnicodeEncodeError:
                warnings.append(f"unicode_encode_issue:{field_name}")

    for i, txt in enumerate(trans_p_l):
        if txt is not None:
            try:
                str(txt).encode("utf-8")
            except UnicodeEncodeError:
                warnings.append(f"unicode_encode_issue:Translated_passages[{i}]")

    # is_selected values should be 0/1 ints
    for i, v in enumerate(is_sel_l):
        try:
            iv = int(v)
            if iv not in (0, 1):
                warnings.append(f"unexpected_is_selected_value:index={i}:value={v}")
        except (TypeError, ValueError):
            warnings.append(f"non_integer_is_selected:index={i}:value={v}")

    return warnings

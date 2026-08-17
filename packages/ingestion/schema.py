"""
Normalized record schema for MSMARCO-XI ingestion.

This module defines the internal representation used AFTER raw parquet rows
are read and validated, but BEFORE chunking. It is deliberately close to the
real, verified schema (see data/processed/dataset_schema.json) rather than a
speculative "generic RAG document" schema — every field here maps to a field
that was physically observed in validation/kanval.parquet.

Verified source columns (from Phase 2 inspection):
    source_lang, target_lang, meta, Answer, query_id, query_type,
    passages (English_passages, Translated_passages, is_selected),
    Eng_Query, Eng_Answer, query
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

NO_ANSWER_MARKERS = {"no answer present.", "no answer present"}


@dataclass
class PassageEntry:
    """A single (English, Translated, is_selected) triple from the `passages`
    nested structure of one source record."""

    passage_index: int
    text_english: str
    text_target: str
    is_selected: int


@dataclass
class NormalizedRecord:
    """One fully normalized MSMARCO-XI record, ready for chunking.

    Nothing from the original record is discarded: passages, relevance
    labels, both language variants of query/answer, and query_type are all
    preserved explicitly.
    """

    query_id: str  # stringified; may carry a __dupN suffix for content-distinct duplicates
    raw_query_id: int
    source_lang: str
    target_lang: str
    query_type: str

    query_target: str  # `query` column (target-language query)
    query_english: str  # `Eng_Query` column

    answer_target: str  # `Answer` column
    answer_english: str  # `Eng_Answer` column

    has_answer: bool
    answer_status: str  # "ANSWER_PRESENT" | "NO_ANSWER"

    passages: list[PassageEntry] = field(default_factory=list)

    meta: dict[str, Any] | None = None

    source_record_hash: str = ""  # computed from canonicalized content
    is_duplicate_query_id: bool = False  # True if this query_id appeared more than once
    is_exact_duplicate: bool = False  # True if this exact record (by hash) is a repeat

    validation_warnings: list[str] = field(default_factory=list)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten for parquet/JSONL storage (passages serialized as JSON string
        of lists, matching common columnar-storage practice for nested data)."""
        d = asdict(self)
        d["passages"] = json.dumps(
            [asdict(p) for p in self.passages], ensure_ascii=False
        )
        d["meta"] = json.dumps(self.meta, ensure_ascii=False) if self.meta is not None else None
        d["validation_warnings"] = json.dumps(self.validation_warnings, ensure_ascii=False)
        return d


def compute_answer_status(eng_answer: str | None) -> tuple[bool, str]:
    """Determine has_answer / answer_status from the English answer field,
    since it is language-independent (avoids depending on the exact
    no-answer marker string in every target language)."""
    if eng_answer is None:
        return False, "NO_ANSWER"
    normalized = str(eng_answer).strip().lower()
    if normalized in NO_ANSWER_MARKERS or normalized == "":
        return False, "NO_ANSWER"
    return True, "ANSWER_PRESENT"


def canonicalize_for_hash(row: dict[str, Any]) -> str:
    """Produce a deterministic canonical string representation of a raw
    record's meaningful content, used to compute source_record_hash.

    Excludes nothing semantically important; only normalizes key order and
    whitespace so that identical content hashes identically regardless of
    dict ordering or incidental whitespace differences.
    """
    canonical = {
        "source_lang": (row.get("source_lang") or "").strip(),
        "target_lang": (row.get("target_lang") or "").strip(),
        "query_type": (row.get("query_type") or "").strip(),
        "query": (row.get("query") or "").strip(),
        "Eng_Query": (row.get("Eng_Query") or "").strip(),
        "Answer": (row.get("Answer") or "").strip(),
        "Eng_Answer": (row.get("Eng_Answer") or "").strip(),
        "passages": row.get("passages"),
    }
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

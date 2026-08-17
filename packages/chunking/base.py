"""
Shared chunk representation and text utilities for all chunking strategies.

Token counting note: this project does not yet have an embedding-model
tokenizer wired in (embeddings are Phase 5). Token counts here are an
approximate whitespace/punctuation-based proxy, clearly labeled as such.
This is an intentional, honest choice rather than fabricating a "real"
token count from an unselected tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

# Sentence boundary heuristic: splits on '.', '!', '?', and the Devanagari/
# Indic danda '।' followed by whitespace or end-of-string. This is a
# structural heuristic (no NLP model), documented as approximate — MSMARCO-XI
# translated (Kannada) text generally uses Latin-style punctuation per the
# observed samples, but the danda is included defensively for broader
# Indic-language reuse.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")

_WORD_RE = re.compile(r"\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count via whitespace-delimited word count.
    Documented as an approximation, not a real model tokenizer count."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter. Falls back to the whole text as one
    'sentence' if no boundaries are found."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split on blank-line paragraph boundaries. Falls back to treating the
    whole text as a single paragraph if no blank lines are present (common
    for MSMARCO-style single-block passages)."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


def normalize_text_for_dedup(text: str) -> str:
    """Canonicalize text for chunk-level duplicate detection: collapse all
    whitespace runs to a single space, strip, and lowercase. This is
    deliberately more aggressive than display normalization — it is used
    ONLY to decide whether two chunks are duplicate content, never to alter
    stored/displayed text."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def compute_text_hash(text: str) -> str:
    import hashlib
    normalized = normalize_text_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    chunk_id: str
    query_id: str
    source_lang: str
    target_lang: str
    language: str  # language of `text` (the primary field) — "target" or "english"
    query_type: str

    passage_index: int
    is_selected: int

    chunk_strategy: str
    chunk_index: int  # index of this chunk within its parent passage's chunk sequence
    parent_passage_id: str

    text: str  # primary text (== text_target by default; see docs)
    text_english: str
    text_target: str

    has_answer: bool
    answer_status: str

    source_record_hash: str

    token_count_estimate: int = 0
    char_count: int = 0
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- chunk-level dedup / quality gating (populated by a post-pass, not
    # by individual strategy modules — see packages/chunking/chunk_dedup.py) ---
    chunk_text_hash: str = ""
    is_duplicate_chunk: bool = False
    duplicate_of_chunk_id: str = ""  # empty string if not a duplicate (parquet-friendly; avoids mixed None/str columns)
    below_min_length: bool = False

    def finalize(self) -> "Chunk":
        self.char_count = len(self.text or "")
        self.token_count_estimate = estimate_tokens(self.text or "")
        self.chunk_text_hash = compute_text_hash(self.text or "")
        return self

    def to_flat_dict(self) -> dict[str, Any]:
        import json
        d = asdict(self)
        d["metadata"] = json.dumps(self.metadata, ensure_ascii=False)
        return d

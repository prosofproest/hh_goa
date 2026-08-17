"""
Strategy C — SENTENCE-AWARE CHUNKING

Segments a passage into sentences, then greedily combines consecutive
sentences until a configurable [MIN_TOKENS, TARGET_TOKENS, MAX_TOKENS]
budget is met, with a small configurable overlap of sentences between
neighboring chunks.

MIN_TOKENS / TARGET_TOKENS / MAX_TOKENS and overlap are all configurable
(not hardcoded), per the competition requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.ingestion.schema import NormalizedRecord
from .base import Chunk, split_sentences, estimate_tokens
from .quality import compute_quality_score

STRATEGY_NAME = "sentence"


@dataclass
class SentenceChunkConfig:
    min_tokens: int = 30
    target_tokens: int = 100
    max_tokens: int = 160
    overlap_sentences: int = 1


def _build_sentence_windows(sentences: list[str], cfg: SentenceChunkConfig) -> list[str]:
    if not sentences:
        return []

    windows: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        current: list[str] = []
        current_tokens = 0
        j = i
        while j < n:
            t = estimate_tokens(sentences[j])
            if current and current_tokens + t > cfg.max_tokens:
                break
            current.append(sentences[j])
            current_tokens += t
            j += 1
            if current_tokens >= cfg.target_tokens:
                break

        # If this window is below min_tokens and there are more sentences
        # available, try to pull in one more to avoid tiny fragments.
        while current_tokens < cfg.min_tokens and j < n:
            t = estimate_tokens(sentences[j])
            current.append(sentences[j])
            current_tokens += t
            j += 1

        windows.append(" ".join(current))

        if j >= n:
            break

        # advance start index with sentence-level overlap
        i = max(i + 1, j - cfg.overlap_sentences)

    return windows


def chunk_record(record: NormalizedRecord, cfg: SentenceChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or SentenceChunkConfig()
    chunks: list[Chunk] = []

    for p in record.passages:
        primary_text = p.text_target.strip() or p.text_english.strip()
        if not primary_text:
            continue
        language = "target" if p.text_target.strip() else "english"
        parent_passage_id = f"{record.query_id}_p{p.passage_index}"

        sentences = split_sentences(primary_text)
        windows = _build_sentence_windows(sentences, cfg)
        if not windows:
            continue

        for idx, window_text in enumerate(windows):
            if not window_text.strip():
                continue
            chunk_id = f"{parent_passage_id}_{STRATEGY_NAME}_{idx}"
            q = compute_quality_score(
                window_text, target_tokens=cfg.target_tokens,
                sibling_texts=[w for j, w in enumerate(windows) if j != idx],
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                query_id=record.query_id,
                source_lang=record.source_lang,
                target_lang=record.target_lang,
                language=language,
                query_type=record.query_type,
                passage_index=p.passage_index,
                is_selected=p.is_selected,
                chunk_strategy=STRATEGY_NAME,
                chunk_index=idx,
                parent_passage_id=parent_passage_id,
                text=window_text,
                text_english=p.text_english if language == "target" else window_text,
                text_target=p.text_target if language == "target" else "",
                has_answer=record.has_answer,
                answer_status=record.answer_status,
                source_record_hash=record.source_record_hash,
                quality_score=q["quality_score"],
                metadata={
                    "quality_breakdown": q,
                    "min_tokens": cfg.min_tokens,
                    "target_tokens": cfg.target_tokens,
                    "max_tokens": cfg.max_tokens,
                    "overlap_sentences": cfg.overlap_sentences,
                    "num_windows_in_passage": len(windows),
                },
            ).finalize()
            chunks.append(chunk)

    return chunks

"""
Strategy A — PASSAGE-AS-CHUNK

Each original translated passage becomes exactly one retrieval chunk.
No splitting, regardless of length. This is the baseline strategy and the
safest fallback (it can never break a passage mid-sentence).
"""

from __future__ import annotations

from packages.ingestion.schema import NormalizedRecord
from .base import Chunk
from .quality import compute_quality_score

STRATEGY_NAME = "passage"


def chunk_record(record: NormalizedRecord) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in record.passages:
        if not p.text_target.strip() and not p.text_english.strip():
            continue  # skip genuinely empty passage slots

        parent_passage_id = f"{record.query_id}_p{p.passage_index}"
        chunk_id = f"{parent_passage_id}_{STRATEGY_NAME}_0"

        q = compute_quality_score(p.text_target or p.text_english)

        chunk = Chunk(
            chunk_id=chunk_id,
            query_id=record.query_id,
            source_lang=record.source_lang,
            target_lang=record.target_lang,
            language="target" if p.text_target.strip() else "english",
            query_type=record.query_type,
            passage_index=p.passage_index,
            is_selected=p.is_selected,
            chunk_strategy=STRATEGY_NAME,
            chunk_index=0,
            parent_passage_id=parent_passage_id,
            text=p.text_target if p.text_target.strip() else p.text_english,
            text_english=p.text_english,
            text_target=p.text_target,
            has_answer=record.has_answer,
            answer_status=record.answer_status,
            source_record_hash=record.source_record_hash,
            quality_score=q["quality_score"],
            metadata={"quality_breakdown": q, "split_reason": "whole_passage"},
        ).finalize()
        chunks.append(chunk)
    return chunks

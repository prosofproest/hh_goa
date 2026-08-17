"""
Strategy D — ADAPTIVE CHUNKING

Automatically selects how to chunk each passage based on its own length and
sentence density (and, at the metadata level, query_type — see below):

    SHORT   (<= short_max_tokens)         -> keep whole (passage-as-chunk)
    MEDIUM  (<= medium_max_tokens)        -> paragraph/sentence grouping
    LONG    (<= long_max_tokens)          -> sentence-window chunking
    VERY_LONG (> long_max_tokens)         -> recursive structural splitting
                                             (halve at nearest sentence
                                             boundary until each piece is
                                             under long_max_tokens)

HONESTY NOTE: "recursive structural splitting" is a boundary-aware halving
of text at sentence boundaries — NOT an embedding-based semantic split.
True semantic chunking (embedding similarity drop-off) requires the
embedding model chosen in Phase 5 and is out of scope for Phase 3, which
explicitly excludes building embeddings. This module is named "adaptive"
per the competition's required strategy list; where the spec's broader
architecture document additionally requests embedding-based "semantic
chunking" as a distinct strategy, that will be added in a later phase once
an embedding model has been benchmarked and selected.

query_type is not used to change the splitting *algorithm* here — per the
Phase 3 spec, query_type should be used as a retrieval/filtering/ranking
*signal*, not to fork the dataset into five separate pipelines. It is
already carried through as chunk metadata on every strategy (see base.Chunk
.query_type), which is sufficient for that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.ingestion.schema import NormalizedRecord
from .base import Chunk, split_sentences, estimate_tokens
from .quality import compute_quality_score
from .sentence import SentenceChunkConfig, _build_sentence_windows

STRATEGY_NAME = "adaptive"


@dataclass
class AdaptiveConfig:
    short_max_tokens: int = 40
    medium_max_tokens: int = 120
    long_max_tokens: int = 250
    overlap_sentences: int = 1


def _recursive_split(sentences: list[str], max_tokens: int) -> list[str]:
    """Recursively halve a sentence list at its midpoint until every
    resulting group is under max_tokens (structural, boundary-aware)."""
    total = sum(estimate_tokens(s) for s in sentences)
    if total <= max_tokens or len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []

    mid = len(sentences) // 2
    left = _recursive_split(sentences[:mid], max_tokens)
    right = _recursive_split(sentences[mid:], max_tokens)
    return left + right


def _classify(total_tokens: int, cfg: AdaptiveConfig) -> str:
    if total_tokens <= cfg.short_max_tokens:
        return "SHORT"
    if total_tokens <= cfg.medium_max_tokens:
        return "MEDIUM"
    if total_tokens <= cfg.long_max_tokens:
        return "LONG"
    return "VERY_LONG"


def chunk_record(record: NormalizedRecord, cfg: AdaptiveConfig | None = None) -> list[Chunk]:
    cfg = cfg or AdaptiveConfig()
    chunks: list[Chunk] = []

    for p in record.passages:
        primary_text = p.text_target.strip() or p.text_english.strip()
        if not primary_text:
            continue
        language = "target" if p.text_target.strip() else "english"
        parent_passage_id = f"{record.query_id}_p{p.passage_index}"

        total_tokens = estimate_tokens(primary_text)
        length_class = _classify(total_tokens, cfg)
        sentences = split_sentences(primary_text)

        if length_class == "SHORT":
            pieces = [primary_text]
            method = "kept_whole"
        elif length_class == "MEDIUM":
            sent_cfg = SentenceChunkConfig(
                min_tokens=max(10, cfg.short_max_tokens // 2),
                target_tokens=cfg.medium_max_tokens,
                max_tokens=cfg.medium_max_tokens + 30,
                overlap_sentences=cfg.overlap_sentences,
            )
            pieces = _build_sentence_windows(sentences, sent_cfg)
            method = "sentence_window_medium"
        elif length_class == "LONG":
            sent_cfg = SentenceChunkConfig(
                min_tokens=max(20, cfg.medium_max_tokens // 2),
                target_tokens=cfg.long_max_tokens - 40,
                max_tokens=cfg.long_max_tokens,
                overlap_sentences=cfg.overlap_sentences,
            )
            pieces = _build_sentence_windows(sentences, sent_cfg)
            method = "sentence_window_long"
        else:  # VERY_LONG
            pieces = _recursive_split(sentences, cfg.long_max_tokens)
            method = "recursive_structural_split"

        pieces = [pc for pc in pieces if pc and pc.strip()]
        if not pieces:
            continue

        for idx, piece in enumerate(pieces):
            chunk_id = f"{parent_passage_id}_{STRATEGY_NAME}_{idx}"
            q = compute_quality_score(
                piece, target_tokens=cfg.medium_max_tokens,
                sibling_texts=[pc for j, pc in enumerate(pieces) if j != idx],
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
                text=piece,
                text_english=p.text_english if language == "target" else piece,
                text_target=p.text_target if language == "target" else "",
                has_answer=record.has_answer,
                answer_status=record.answer_status,
                source_record_hash=record.source_record_hash,
                quality_score=q["quality_score"],
                metadata={
                    "quality_breakdown": q,
                    "length_class": length_class,
                    "adaptive_method": method,
                    "passage_total_tokens_estimate": total_tokens,
                    "num_pieces_in_passage": len(pieces),
                },
            ).finalize()
            chunks.append(chunk)

    return chunks

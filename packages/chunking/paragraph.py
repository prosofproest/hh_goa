"""
Strategy B — PARAGRAPH-AWARE CHUNKING

For longer passages: split on paragraph boundaries (blank lines). If a
passage contains no explicit paragraph breaks (common for MSMARCO-style
single-block passages, confirmed structurally short in the current
dataset), it falls back to grouping sentences into paragraph-sized chunks
using a token budget, so the strategy remains meaningful even without
literal newline-separated paragraphs.

Configurable overlap is applied between neighboring paragraph-level chunks
when a passage is split into more than one chunk.
"""

from __future__ import annotations

from packages.ingestion.schema import NormalizedRecord
from .base import Chunk, split_paragraphs, split_sentences, estimate_tokens
from .quality import compute_quality_score

STRATEGY_NAME = "paragraph"

DEFAULT_TARGET_TOKENS = 150
DEFAULT_OVERLAP_TOKENS = 20
# Below this token count, a passage is left whole rather than "chunked" into
# a single trivial paragraph (avoids pointless 1:1 duplication of Strategy A).
MIN_TOKENS_TO_SPLIT = 40


def _group_sentences_by_budget(sentences: list[str], target_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback grouping when no literal paragraph breaks exist: greedily
    pack sentences into chunks up to target_tokens, with a small sentence-
    level overlap between consecutive chunks."""
    if not sentences:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        t = estimate_tokens(sent)
        if current and current_tokens + t > target_tokens:
            groups.append(current)
            # overlap: carry trailing sentences whose token sum <= overlap_tokens
            overlap_sents = []
            overlap_sum = 0
            for s in reversed(current):
                st = estimate_tokens(s)
                if overlap_sum + st > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_sum += st
            current = list(overlap_sents)
            current_tokens = overlap_sum
        current.append(sent)
        current_tokens += t

    if current:
        groups.append(current)

    return [" ".join(g) for g in groups]


def chunk_record(
    record: NormalizedRecord,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for p in record.passages:
        primary_text = p.text_target.strip() or p.text_english.strip()
        if not primary_text:
            continue

        language = "target" if p.text_target.strip() else "english"
        parent_passage_id = f"{record.query_id}_p{p.passage_index}"

        total_tokens = estimate_tokens(primary_text)
        if total_tokens <= MIN_TOKENS_TO_SPLIT:
            # Too short to meaningfully paragraph-split; keep whole (still
            # tagged under this strategy so retrieval-strategy comparisons
            # remain apples-to-apples across all passages).
            pieces = [primary_text]
            split_reason = "below_min_tokens_kept_whole"
        else:
            paragraphs = split_paragraphs(primary_text)
            if len(paragraphs) > 1:
                pieces = paragraphs
                split_reason = "literal_paragraph_boundaries"
            else:
                sentences = split_sentences(primary_text)
                pieces = _group_sentences_by_budget(sentences, target_tokens, overlap_tokens)
                split_reason = "sentence_grouped_fallback_no_literal_paragraphs"

        sibling_texts = pieces  # for duplication-penalty comparison within the same passage

        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunk_id = f"{parent_passage_id}_{STRATEGY_NAME}_{idx}"
            q = compute_quality_score(
                piece, target_tokens=target_tokens,
                sibling_texts=[s for j, s in enumerate(sibling_texts) if j != idx],
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
                    "split_reason": split_reason,
                    "target_tokens": target_tokens,
                    "overlap_tokens": overlap_tokens,
                    "num_pieces_in_passage": len(pieces),
                },
            ).finalize()
            chunks.append(chunk)

    return chunks

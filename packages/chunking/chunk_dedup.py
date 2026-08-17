"""
Chunk-level deduplication — PROVENANCE-SCOPED.

Motivation (found via real 1,000-record test on kanval.parquet): for short
passages, all four chunking strategies legitimately produce identical text
(a short passage kept whole is the same string whether you call it
"passage", "paragraph", "sentence", or "adaptive" chunking). This means a
large fraction of the 4x chunk multiplier is not four genuinely different
retrieval views — it's the same text stored four times.

RULE (per explicit requirement): deduplication is scoped to
    (parent_passage_id, language)
i.e. we only ever collapse chunks that come from the SAME source passage.
Two different passages that happen to contain identical text (e.g. two
records both containing a generic filler sentence) are NEVER collapsed —
that would conflate genuinely different documents/provenance, which is
explicitly disallowed.

This is a non-destructive pass by default: duplicate chunks are FLAGGED
(is_duplicate_chunk=True, duplicate_of_chunk_id=<canonical id>), not
deleted, and the canonical chunk's metadata records which strategies
contributed identical text. A separate, explicit --drop-duplicate-chunks
flag in scripts/chunk_dataset.py controls whether duplicates are physically
excluded from the written parquet output (for embedding-cost efficiency in
Phase 4/5) — dropping is opt-in, never silent.
"""

from __future__ import annotations

from collections import defaultdict

from .base import Chunk

# Canonical-selection preference order: prefer the structurally "safest"
# strategy as the canonical representative when multiple strategies tie.
STRATEGY_PREFERENCE = ("passage", "paragraph", "sentence", "adaptive")


def dedupe_chunks_within_passage(chunks: list[Chunk]) -> dict:
    """Mutates `chunks` in place, setting dedup fields. Operates on a list
    of chunks that has ALREADY been grouped by (parent_passage_id, language)
    by the caller — this function does not itself filter by provenance, so
    callers must ensure that invariant.

    Returns summary stats for this passage group.
    """
    if not chunks:
        return {"canonical_count": 0, "duplicate_count": 0}

    # Group by text hash
    by_hash: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_hash[c.chunk_text_hash].append(c)

    canonical_count = 0
    duplicate_count = 0

    for text_hash, group in by_hash.items():
        if len(group) == 1:
            group[0].is_duplicate_chunk = False
            group[0].duplicate_of_chunk_id = ""
            canonical_count += 1
            continue

        # Pick canonical by strategy preference order, then by chunk_id for determinism
        def sort_key(c: Chunk):
            try:
                pref_idx = STRATEGY_PREFERENCE.index(c.chunk_strategy)
            except ValueError:
                pref_idx = len(STRATEGY_PREFERENCE)
            return (pref_idx, c.chunk_id)

        group_sorted = sorted(group, key=sort_key)
        canonical = group_sorted[0]
        canonical.is_duplicate_chunk = False
        canonical.duplicate_of_chunk_id = ""
        canonical.metadata["duplicate_chunk_ids"] = [c.chunk_id for c in group_sorted[1:]]
        canonical.metadata["contributing_strategies"] = sorted(set(c.chunk_strategy for c in group_sorted))
        canonical_count += 1

        for dup in group_sorted[1:]:
            dup.is_duplicate_chunk = True
            dup.duplicate_of_chunk_id = canonical.chunk_id
            duplicate_count += 1

    return {"canonical_count": canonical_count, "duplicate_count": duplicate_count}


def apply_min_length_flag(chunks: list[Chunk], min_chars: int) -> int:
    """Flags (does not delete) chunks below a configurable minimum character
    length. Returns the count flagged."""
    flagged = 0
    for c in chunks:
        if c.char_count < min_chars:
            c.below_min_length = True
            flagged += 1
        else:
            c.below_min_length = False
    return flagged


def run_chunk_dedup(all_chunks: list[Chunk], min_chars: int = 10) -> dict:
    """Full dedup + min-length-flag pass over a full chunk set (typically
    all chunks for one record, or a whole dataset if memory allows).

    Groups strictly by (parent_passage_id, language) before deduping, so
    cross-passage collapsing can never happen.
    """
    groups: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for c in all_chunks:
        groups[(c.parent_passage_id, c.language)].append(c)

    total_canonical = 0
    total_duplicate = 0
    for key, group in groups.items():
        stats = dedupe_chunks_within_passage(group)
        total_canonical += stats["canonical_count"]
        total_duplicate += stats["duplicate_count"]

    below_min = apply_min_length_flag(all_chunks, min_chars)

    return {
        "total_chunks": len(all_chunks),
        "passage_groups": len(groups),
        "canonical_chunks": total_canonical,
        "duplicate_chunks": total_duplicate,
        "below_min_length_chunks": below_min,
        "min_chars_threshold": min_chars,
    }

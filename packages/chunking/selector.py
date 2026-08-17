"""
Runs the configured set of chunking strategies over a NormalizedRecord and
returns all resulting chunks tagged with their strategy name.

This is the "multi-view chunking" entry point: the same source passage
produces multiple chunk representations (passage / paragraph / sentence /
adaptive), each stored with strategy metadata so retrieval-quality
experiments (Phase 4+) can compare them directly.
"""

from __future__ import annotations

from packages.ingestion.schema import NormalizedRecord
from . import adaptive, paragraph, passage, sentence
from .base import Chunk

ALL_STRATEGIES = ("passage", "paragraph", "sentence", "adaptive")


def run_strategies(record: NormalizedRecord, strategies: list[str] | None = None) -> list[Chunk]:
    strategies = strategies or list(ALL_STRATEGIES)
    chunks: list[Chunk] = []

    if "passage" in strategies:
        chunks.extend(passage.chunk_record(record))
    if "paragraph" in strategies:
        chunks.extend(paragraph.chunk_record(record))
    if "sentence" in strategies:
        chunks.extend(sentence.chunk_record(record))
    if "adaptive" in strategies:
        chunks.extend(adaptive.chunk_record(record))

    return chunks

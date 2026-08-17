"""
Chunk quality scoring — STRUCTURAL v1.

quality_score = semantic_coherence + information_density + boundary_quality
                - duplication_penalty

IMPORTANT HONESTY NOTE: embeddings do not exist yet at this phase (Phase 5).
`semantic_coherence` here is therefore a STRUCTURAL proxy (sentence-count /
length consistency), not an embedding-similarity measure. Once embeddings
are available (Phase 5+), this module should be revisited and the
semantic_coherence term replaced/augmented with actual embedding-based
cosine coherence between sentences in the chunk. This is documented in
docs/ingestion-and-chunking.md.
"""

from __future__ import annotations

from .base import split_sentences


def _semantic_coherence_proxy(text: str, target_tokens: int) -> float:
    """Structural proxy: chunks close to the target token length, with a
    reasonable sentence count (not a single fragment, not dozens of
    unrelated fragments), score higher. Range approx [0, 1]."""
    sentences = split_sentences(text)
    n_sent = len(sentences)
    if n_sent == 0:
        return 0.0
    # Prefer 1-6 sentences per chunk as a coherence heuristic
    if 1 <= n_sent <= 6:
        sentence_score = 1.0
    else:
        sentence_score = max(0.0, 1.0 - (n_sent - 6) * 0.08)
    return round(sentence_score, 3)


def _information_density(text: str) -> float:
    """Ratio of unique whitespace-delimited tokens to total tokens. Very
    repetitive text scores low; varied text scores high. Range [0, 1]."""
    words = text.split()
    if not words:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    return round(unique_ratio, 3)


def _boundary_quality(text: str) -> float:
    """Rewards chunks that start with a capital/Unicode-letter and end with
    sentence-ending punctuation (i.e. don't look like a mid-sentence cut).
    Range [0, 1]."""
    text = text.strip()
    if not text:
        return 0.0
    score = 0.0
    if text[-1] in ".!?।\"'”":
        score += 0.5
    else:
        score += 0.1
    if text[0].isupper() or not text[0].isascii():
        score += 0.5
    else:
        score += 0.2
    return round(min(score, 1.0), 3)


def _duplication_penalty(text: str, sibling_texts: list[str]) -> float:
    """Penalizes chunks that substantially overlap (as sets of words) with
    sibling chunks from the same passage/strategy. Range [0, 0.5]."""
    if not sibling_texts:
        return 0.0
    words = set(text.lower().split())
    if not words:
        return 0.0
    max_overlap = 0.0
    for sib in sibling_texts:
        sib_words = set(sib.lower().split())
        if not sib_words:
            continue
        overlap = len(words & sib_words) / len(words)
        max_overlap = max(max_overlap, overlap)
    return round(min(max_overlap * 0.5, 0.5), 3)


def compute_quality_score(
    text: str,
    target_tokens: int = 120,
    sibling_texts: list[str] | None = None,
) -> dict:
    coherence = _semantic_coherence_proxy(text, target_tokens)
    density = _information_density(text)
    boundary = _boundary_quality(text)
    dup_penalty = _duplication_penalty(text, sibling_texts or [])

    total = round(coherence + density + boundary - dup_penalty, 3)
    return {
        "quality_score": total,
        "semantic_coherence_proxy": coherence,
        "information_density": density,
        "boundary_quality": boundary,
        "duplication_penalty": dup_penalty,
    }

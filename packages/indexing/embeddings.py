"""
Embedding model abstraction for Phase 4 retrieval benchmarking.

IMPORTANT ENVIRONMENT CONSTRAINT (documented, not hidden):
This development sandbox's outbound network is restricted to a fixed
domain allowlist (pypi.org, npmjs.org, github.com, etc.) and does NOT
include huggingface.co (confirmed in Phase 2 — 403 host_not_allowed).
Real multilingual embedding models (candidates below) are hosted on the
Hugging Face Hub, so they CANNOT be downloaded or benchmarked for real
inside this sandbox. This module still implements the REAL production
interface (SentenceTransformerEmbedder) so it is ready to run wherever
Hugging Face is reachable (e.g. the user's local machine), and additionally
provides a TfidfHashingEmbedder used ONLY to exercise indexing/retrieval/
evaluation code paths locally — it is NOT a semantic embedding model and
must never be used to report real retrieval-quality numbers.

Candidate multilingual models for the real Phase 4 benchmark (to be run on
a machine with Hugging Face access), chosen for stated reasons, not
assumed as "the" answer:

    1. intfloat/multilingual-e5-base
       - 278M params, 768-dim, supports 100+ languages incl. Kannada
       - Strong benchmark performance on MIRACL/MTEB multilingual retrieval
       - Requires "query: " / "passage: " prefixing convention (E5 family)

    2. BAAI/bge-m3
       - 568M params, 1024-dim, explicit multi-lingual + multi-granularity
         (dense + sparse + ColBERT-style multi-vector in one model)
       - Larger; higher expected quality but higher latency/memory cost
       - Native multi-vector output could feed the "late-interaction
         reranker" requirement directly

    3. ai4bharat/indic-sentence-bert-nli (or similar AI4Bharat IndicBERT-
       family sentence encoder)
       - Trained specifically on Indic languages including Kannada
       - Likely better Kannada-specific quality, smaller than bge-m3
       - Narrower language coverage than the two above (Indic-focused only)

These three are NOT benchmarked with real numbers in this sandbox run —
see docs/retrieval-architecture.md for the honest status and exact
commands to run the real benchmark locally.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Returns an (N, dimension) float32 array, L2-normalized (so dot
        product == cosine similarity)."""
        ...

    def embed_with_timing(self, texts: list[str], is_query: bool = False) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        vecs = self.embed(texts, is_query=is_query)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return vecs, elapsed_ms

    def warm_up(self, n_calls: int = 5, sample_text: str = "warm up query for device initialization") -> float:
        """Default no-op-cost warm-up for embedders that don't need it
        (e.g. the local proxy). Real backends override this."""
        t0 = time.perf_counter()
        for _ in range(n_calls):
            self.embed([sample_text], is_query=True)
        return (time.perf_counter() - t0) * 1000


class SentenceTransformerEmbedder(EmbeddingModel):
    """REAL production embedder. Requires `sentence-transformers` and
    network access to Hugging Face to download model weights on first use.
    Not runnable inside this development sandbox (see module docstring) —
    implemented against the real library API so it works unmodified
    wherever HF is reachable.
    """

    # E5-family models expect this prefixing convention; other model
    # families (e.g. bge-m3) do not require it. Configurable per model.
    E5_STYLE_MODELS = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large",
                        "intfloat/multilingual-e5-small"}

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Install with: "
                "pip install sentence-transformers torch --break-system-packages"
            ) from e

        # Explicit device verification — NEVER silently fall back. If the
        # caller asked for "mps" (or "cuda") and it isn't actually
        # available, fail loudly rather than quietly running on CPU, which
        # would invalidate any latency numbers claimed for that device.
        if device == "mps":
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "device='mps' was requested but torch.backends.mps.is_available() "
                    "is False on this machine. Refusing to silently fall back to CPU. "
                    "Either fix the MPS setup or explicitly pass device='cpu'."
                )
        elif device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' was requested but torch.cuda.is_available() is False. "
                    "Refusing to silently fall back to CPU."
                )

        self.name = model_name
        self.requested_device = device
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        self._is_e5_style = model_name in self.E5_STYLE_MODELS

        # Verify the model's parameters actually ended up on the requested
        # device (belt-and-braces on top of the availability check above —
        # some library versions have been known to silently downgrade).
        actual_device = str(next(self._model.parameters()).device)
        self.actual_device = actual_device
        self.device_mismatch = not actual_device.startswith(device.split(":")[0])
        if self.device_mismatch:
            raise RuntimeError(
                f"Requested device='{device}' but model parameters are actually on "
                f"'{actual_device}'. Refusing to report latency numbers under a false "
                f"device label. Investigate the sentence-transformers/torch device "
                f"resolution before proceeding."
            )

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._is_e5_style:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


class TfidfHashingEmbedder(EmbeddingModel):
    """LOCAL TEST-ONLY PROXY. Uses scikit-learn's HashingVectorizer +
    TF-IDF-style weighting to produce fixed-dimension vectors purely to
    exercise indexing/retrieval/evaluation code paths without any network
    access or GPU/large-model dependency.

    THIS IS NOT A SEMANTIC EMBEDDING MODEL. It has no notion of meaning,
    synonyms, or cross-lingual similarity — it is a bag-of-character-
    n-grams hash, chosen specifically so it still produces *some* non-zero
    signal for Kannada text (unlike a whitespace-tokenizer proxy, which
    would be uninformative for space-delimited-but-non-Latin scripts).
    Any retrieval-quality numbers produced with this embedder MUST be
    labeled as code-path verification only, never as real benchmark
    results.
    """

    def __init__(self, dimension: int = 256, ngram_range: tuple[int, int] = (2, 4)):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"TEST_ONLY_tfidf_hashing_dim{dimension}"
        self.dimension = dimension
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            analyzer="char_wb",
            ngram_range=ngram_range,
            norm="l2",
            alternate_sign=False,
        )

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        vecs = matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def get_embedder(model_name: str, device: str = "cpu", **kwargs) -> EmbeddingModel:
    """Factory. `model_name` starting with 'TEST_ONLY:' routes to the local
    proxy embedder (for sandbox/CI use, `device` is ignored); anything else
    is treated as a real sentence-transformers model name and `device` is
    passed through (e.g. 'cpu', 'mps', 'cuda')."""
    if model_name.startswith("TEST_ONLY:"):
        dim = int(model_name.split("dim")[-1]) if "dim" in model_name else 256
        return TfidfHashingEmbedder(dimension=dim)
    return SentenceTransformerEmbedder(model_name, device=device)

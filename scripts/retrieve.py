"""
Phase 4 — scripts/retrieve.py

Single-query retrieval against a previously built Qdrant index (see
build_qdrant_index.py). Outputs structured JSON exactly per the Phase 4
spec, with a per-stage latency breakdown.

Guardrail-preparation fields (retrieved scores, top-k evidence, source
chunk IDs, metadata) are included precisely so a later grounding-check
phase can consume this output without re-querying retrieval.

Usage:
    python scripts/retrieve.py \
        --query "ಬೆಂಗಳೂರು ಎಲ್ಲಿದೆ" \
        --model "TEST_ONLY:dim256" \
        --collection voice_rag_hhgoa \
        --qdrant-path data/processed/qdrant_local \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.indexing.embeddings import get_embedder  # noqa: E402
from packages.indexing.qdrant_index import QdrantIndex  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a single retrieval query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collection", default="voice_rag_hhgoa")
    parser.add_argument("--qdrant-path", default="data/processed/qdrant_local")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--filter-language", default=None, help="Optional legitimate filter: 'target' or 'english'"
    )
    parser.add_argument(
        "--filter-query-type", default=None,
        help="Optional legitimate filter: DESCRIPTION/NUMERIC/ENTITY/PERSON/LOCATION",
    )
    args = parser.parse_args()

    t_client0 = time.perf_counter()
    embedder = get_embedder(args.model)
    qdrant_path = None if args.qdrant_url else str(REPO_ROOT / args.qdrant_path)
    qdrant = QdrantIndex(
        collection_name=args.collection, dimension=embedder.dimension,
        path=qdrant_path, url=args.qdrant_url, create=False,
    )
    client_init_ms = (time.perf_counter() - t_client0) * 1000

    # RAG-core latency starts here (model + client init is a one-time process
    # startup cost in a real server, not a per-request cost — reported
    # separately, never folded into the per-query latency_ms below).
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    qvec = embedder.embed([args.query], is_query=True)[0]
    embedding_ms = (time.perf_counter() - t0) * 1000

    query_filter = {}
    if args.filter_language:
        query_filter["language"] = args.filter_language
    if args.filter_query_type:
        query_filter["query_type"] = args.filter_query_type

    results, retrieval_ms = qdrant.search(qvec, top_k=args.top_k, query_filter=query_filter or None)

    total_ms = (time.perf_counter() - t_total0) * 1000

    # NOTE: corpus text is not stored redundantly here — the Qdrant payload
    # in this schema does not carry chunk text (kept lean per spec section
    # 6's metadata list). A production retrieve.py would join chunk_id back
    # against the chunk parquet/a text store to populate "text"; that join
    # is deliberately out of scope for this benchmark-focused script — see
    # docs/retrieval-architecture.md "Known limitations".
    retrieved_chunks = [
        {
            "chunk_id": r["chunk_id"],
            "score": round(r["score"], 6),
            "text": None,  # see note above
            "metadata": r["payload"],
        }
        for r in results
    ]

    output = {
        "query": args.query,
        "retrieved_chunks": retrieved_chunks,
        "latency_ms": round(total_ms, 3),
        "latency_breakdown_ms": {
            "embedding_ms": round(embedding_ms, 3),
            "qdrant_search_ms": round(retrieval_ms, 3),
            "total_ms": round(total_ms, 3),
        },
        "process_startup_ms_excluded_from_latency_ms": round(client_init_ms, 3),
        "guardrail_preparation": {
            "top_k_requested": args.top_k,
            "results_returned": len(retrieved_chunks),
            "filters_applied": query_filter or None,
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

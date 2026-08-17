"""
Phase 4 — Step 3: Run Retrieval Benchmark

Orchestrates the full benchmark:
    1. Load corpus subset + benchmark queries
    2. For each configured embedding model:
        a. Embed the corpus (batched), build a Qdrant index
        b. Build a BM25 sparse index over the same corpus
        c. For each benchmark query: run dense retrieval, hybrid (RRF)
           retrieval, recording per-stage latency
        d. Compute Recall@K / Hit@K / MRR@10 / nDCG@10 per query, and
           aggregate overall + broken down by query_type
    3. Compute latency percentiles (P50/P70/P90/P95/P99/P100) per stage
       across ALL queries (not a single best-case number)
    4. Write data/processed/retrieval_benchmark_report.json

CRITICAL ANTI-LEAKAGE: is_selected / Answer / Eng_Answer are never passed
to the embedder or the retrieval index as input text or as a filter. They
are read ONLY from the benchmark queries' pre-computed
`ground_truth_parent_passage_ids` (see build_retrieval_benchmark.py) for
scoring after the fact.

Usage:
    python scripts/run_retrieval_benchmark.py \
        --corpus data/processed/retrieval_corpus_subset.parquet \
        --benchmark-queries data/processed/retrieval_benchmark_queries.jsonl \
        --config configs/retrieval_benchmark.yaml \
        --models "TEST_ONLY:dim256"     # override config's model list, comma-separated
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.indexing.embeddings import get_embedder  # noqa: E402
from packages.indexing.qdrant_index import QdrantIndex  # noqa: E402
from packages.indexing.sparse import BM25Index  # noqa: E402
from packages.indexing.fusion import reciprocal_rank_fusion  # noqa: E402
from packages.evaluation.metrics import evaluate_single_query, aggregate_metrics  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((p / 100) * (len(s) - 1))))
    return round(s[idx], 3)


def latency_percentiles(values: list[float]) -> dict:
    return {
        "p50_ms": percentile(values, 50),
        "p70_ms": percentile(values, 70),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "p100_ms": percentile(values, 100),
        "mean_ms": round(sum(values) / len(values), 3) if values else 0.0,
        "n": len(values),
    }


def resolve_relevant_chunk_ids(benchmark_entry: dict, strategy_filter: str | None = None) -> set[str]:
    """Ground truth chunk IDs for a benchmark query, across whichever
    strategies are present in ground_truth_chunk_ids_by_strategy (or a
    single strategy, if strategy_filter is given)."""
    by_strategy = benchmark_entry.get("ground_truth_chunk_ids_by_strategy", {})
    if not by_strategy:
        return set()
    if strategy_filter:
        return set(by_strategy.get(strategy_filter, []))
    all_ids = set()
    for ids in by_strategy.values():
        all_ids.update(ids)
    return all_ids


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 4 retrieval benchmark")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--benchmark-queries", required=True)
    parser.add_argument("--config", default="configs/retrieval_benchmark.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model names, overrides config")
    parser.add_argument("--top-k", type=int, default=None, help="Overrides config top_k")
    parser.add_argument("--output", default="data/processed/retrieval_benchmark_report.json")
    parser.add_argument("--device", default="cpu", help="Device for real embedding models: cpu, mps, cuda")
    parser.add_argument("--warmup-calls", type=int, default=5,
                         help="Number of throwaway embed() calls to run before timed measurement begins")
    args = parser.parse_args()

    import pandas as pd
    import yaml

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    top_k = args.top_k or config.get("top_k", 30)
    k_values = tuple(config.get("eval_k_values", [1, 3, 5, 10]))
    mrr_k = config.get("mrr_k", 10)
    rrf_k = config.get("rrf_k", 60)
    modes = config.get("retrieval_modes", ["dense"])

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = [m["name"] for m in config.get("embedding_models", [])]
        if not model_names:
            model_names = [config.get("local_test_only_embedder", "TEST_ONLY:dim256")]

    corpus_df = pd.read_parquet(REPO_ROOT / args.corpus)
    print(f"Corpus: {len(corpus_df)} chunks")

    with open(REPO_ROOT / args.benchmark_queries, "r", encoding="utf-8") as f:
        benchmark = [json.loads(line) for line in f]
    print(f"Benchmark queries: {len(benchmark)}")

    corpus_ids = corpus_df["chunk_id"].tolist()
    corpus_texts = corpus_df["text"].tolist()

    print("Building BM25 sparse index over corpus...")
    bm25 = BM25Index(corpus_ids, corpus_texts)

    all_model_results = {}

    for model_name in model_names:
        print(f"\n=== Embedding model: {model_name} ===")
        try:
            embedder = get_embedder(model_name, device=args.device)
        except ImportError as e:
            print(f"SKIPPING {model_name}: {e}")
            all_model_results[model_name] = {"status": "SKIPPED", "reason": str(e)}
            continue
        except RuntimeError as e:
            # Explicit device-mismatch/unavailability failure — surface it,
            # never silently retry on CPU (that would misreport the device
            # for any latency numbers produced).
            print(f"FAILED {model_name}: {e}")
            all_model_results[model_name] = {"status": "FAILED_DEVICE_CHECK", "reason": str(e)}
            continue
        except Exception as e:
            print(f"SKIPPING {model_name}: failed to load ({type(e).__name__}: {e})")
            all_model_results[model_name] = {"status": "SKIPPED", "reason": f"{type(e).__name__}: {e}"}
            continue

        device_info = {
            "requested_device": getattr(embedder, "requested_device", args.device),
            "actual_device": getattr(embedder, "actual_device", "n/a (proxy embedder)"),
            "device_mismatch_detected": getattr(embedder, "device_mismatch", False),
        }
        print(f"Device verification: requested={device_info['requested_device']}, "
              f"actual={device_info['actual_device']}")

        print(f"Warming up ({args.warmup_calls} calls, excluded from all reported latency)...")
        warmup_ms = embedder.warm_up(n_calls=args.warmup_calls)
        print(f"Warm-up complete in {warmup_ms:.1f}ms (not counted toward any percentile below)")

        # --- Build Qdrant index ---
        collection_name = f"{config.get("qdrant_collection_name", "bench")}_{abs(hash(model_name)) % 100000}"
        qdrant = QdrantIndex(
            collection_name=collection_name,
            dimension=embedder.dimension,
            path=":memory:",
            distance=config.get("qdrant_distance", "Cosine"),
        )

        # --- Memory-safe corpus embedding + immediate Qdrant upsert ---
        # Apple M1/MPS: never hold all corpus embeddings in RAM.
        import gc
        import numpy as np

        batch_size = 32
        embed_corpus_t0 = time.perf_counter()
        total_upsert_ms = 0.0

        print(
            f"Embedding corpus in memory-safe batches of {batch_size}..."
        )

        for i in range(0, len(corpus_texts), batch_size):
            end_batch = min(i + batch_size, len(corpus_texts))

            batch_texts = corpus_texts[i:end_batch]
            batch_ids = corpus_ids[i:end_batch]

            batch_vecs = embedder.embed(
                batch_texts,
                is_query=False,
            )

            batch_vecs = np.asarray(
                batch_vecs,
                dtype=np.float32,
            )

            batch_payloads = [
                {
                    "parent_passage_id": row["parent_passage_id"],
                    "query_id": row["query_id"],
                    "language": row["language"],
                    "source_lang": row["source_lang"],
                    "target_lang": row["target_lang"],
                    "query_type": row["query_type"],
                    "chunk_strategy": row["chunk_strategy"],
                    "passage_index": int(row["passage_index"]),
                    "is_selected": int(row["is_selected"]),
                }
                for _, row in corpus_df.iloc[i:end_batch].iterrows()
            ]

            batch_upsert_ms = qdrant.upsert(
                batch_ids,
                batch_vecs,
                batch_payloads,
            )

            total_upsert_ms += batch_upsert_ms

            del batch_vecs
            del batch_payloads
            del batch_texts
            del batch_ids

            gc.collect()

            try:
                import torch

                if (
                    hasattr(torch, "mps")
                    and torch.backends.mps.is_available()
                ):
                    torch.mps.empty_cache()
            except Exception:
                pass

            if (
                i == 0
                or end_batch == len(corpus_texts)
                or end_batch % 1000 == 0
            ):
                elapsed = time.perf_counter() - embed_corpus_t0

                print(
                    f"  {end_batch}/{len(corpus_texts)} chunks "
                    f"({end_batch / len(corpus_texts) * 100:.1f}%) "
                    f"| elapsed={elapsed:.1f}s"
                )

        embed_corpus_ms = (
            time.perf_counter() - embed_corpus_t0
        ) * 1000

        upsert_ms = total_upsert_ms

        print(
            f"Corpus embedding + indexing complete: "
            f"{len(corpus_texts)} chunks in {embed_corpus_ms:.1f}ms"
        )

        print(
            f"Qdrant upsert total: {upsert_ms:.1f}ms"
        )

        # --- Per-query retrieval + evaluation ---
        per_mode_results = {mode: [] for mode in modes}
        latency_stages = {
            "embedding_ms": [], "dense_ms": [], "sparse_ms": [], "fusion_ms": [], "total_dense_ms": [], "total_hybrid_ms": [],
        }

        for entry in benchmark:
            query_text = entry["query"] or entry["Eng_Query"]
            relevant_ids = resolve_relevant_chunk_ids(entry)
            if not relevant_ids:
                continue  # no ground truth chunks resolvable for this query (e.g. chunks-dir not given at benchmark-build time)

            t0 = time.perf_counter()
            qvec = embedder.embed([query_text], is_query=True)[0]
            embed_ms = (time.perf_counter() - t0) * 1000
            latency_stages["embedding_ms"].append(embed_ms)

            dense_results, dense_ms = qdrant.search(qvec, top_k=top_k)
            latency_stages["dense_ms"].append(dense_ms)
            dense_ranked = [r["chunk_id"] for r in dense_results]

            if "dense" in modes:
                metrics = evaluate_single_query(dense_ranked, relevant_ids, k_values, mrr_k)
                metrics["query_type"] = entry["query_type"]
                per_mode_results["dense"].append(metrics)
                latency_stages["total_dense_ms"].append(embed_ms + dense_ms)

            if "hybrid_rrf" in modes:
                t_sparse0 = time.perf_counter()
                sparse_results = bm25.search(query_text, top_k=top_k)
                sparse_ms = (time.perf_counter() - t_sparse0) * 1000
                latency_stages["sparse_ms"].append(sparse_ms)
                sparse_ranked = [cid for cid, _ in sparse_results]

                t_fuse0 = time.perf_counter()
                fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=rrf_k)
                fused_ranked = [cid for cid, _ in fused]
                fusion_ms = (time.perf_counter() - t_fuse0) * 1000
                latency_stages["fusion_ms"].append(fusion_ms)

                metrics = evaluate_single_query(fused_ranked, relevant_ids, k_values, mrr_k)
                metrics["query_type"] = entry["query_type"]
                per_mode_results["hybrid_rrf"].append(metrics)
                latency_stages["total_hybrid_ms"].append(embed_ms + dense_ms + sparse_ms + fusion_ms)

        n_evaluated = len(per_mode_results.get(modes[0], []))
        print(f"Evaluated {n_evaluated} / {len(benchmark)} benchmark queries "
              f"(remainder had no resolvable ground-truth chunk IDs)")

        mode_summaries = {}
        for mode, results in per_mode_results.items():
            if not results:
                continue
            overall = aggregate_metrics([{k: v for k, v in r.items() if k != "query_type"} for r in results])
            by_qtype = defaultdict(list)
            for r in results:
                by_qtype[r["query_type"]].append({k: v for k, v in r.items() if k != "query_type"})
            qtype_summary = {qt: aggregate_metrics(rs) for qt, rs in by_qtype.items()}
            mode_summaries[mode] = {"overall": overall, "by_query_type": qtype_summary}

        all_model_results[model_name] = {
            "status": "COMPLETE",
            "dimension": embedder.dimension,
            "is_test_only_proxy": model_name.startswith("TEST_ONLY:"),
            "device": device_info,
            "warmup_ms_excluded_from_latency": round(warmup_ms, 2),
            "warmup_calls": args.warmup_calls,
            "corpus_size": len(corpus_ids),
            "queries_evaluated": n_evaluated,
            "embed_corpus_total_ms": round(embed_corpus_ms, 2),
            "embed_corpus_ms_per_chunk": round(embed_corpus_ms / len(corpus_texts), 4),
            "qdrant_upsert_ms": round(upsert_ms, 2),
            "latency_percentiles": {
                stage: latency_percentiles(vals) for stage, vals in latency_stages.items() if vals
            },
            "retrieval_modes": mode_summaries,
        }

    report = {
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_used": config,
        "corpus_file": args.corpus,
        "corpus_size": len(corpus_df),
        "benchmark_queries_file": args.benchmark_queries,
        "benchmark_query_count": len(benchmark),
        "top_k": top_k,
        "models": all_model_results,
        "anti_leakage_note": (
            "is_selected/Answer/Eng_Answer were never used as embedder input or as a "
            "Qdrant query filter. Ground truth was applied only in post-hoc scoring."
        ),
    }

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nBenchmark report written to: {output_path}")


if __name__ == "__main__":
    main()

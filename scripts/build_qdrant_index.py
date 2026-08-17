"""
Phase 4 — Build a persistent, resumable Qdrant index.

M1/MPS-safe:
- Embeds only one batch at a time.
- Immediately upserts each batch.
- Never concatenates the entire corpus into RAM.
- Progress is checkpointed to disk.
- Re-running the same command resumes from completed batches.
- Existing Qdrant points are safe to overwrite because point IDs are
  deterministic from chunk_id.

Example:

python3 scripts/build_qdrant_index.py \
  --corpus data/processed/retrieval_corpus_subset.parquet \
  --model "intfloat/multilingual-e5-base" \
  --collection voice_rag_hhgoa_e5_50k \
  --qdrant-path data/processed/qdrant_e5_50k \
  --device mps \
  --batch-size 16
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from packages.indexing.embeddings import get_embedder
from packages.indexing.qdrant_index import QdrantIndex


def main():
    parser = argparse.ArgumentParser(
        description="Build a resumable persistent Qdrant index"
    )

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collection", default="voice_rag_hhgoa")
    parser.add_argument(
        "--qdrant-path",
        default="data/processed/qdrant_local",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Real Qdrant server URL. If supplied, --qdrant-path is ignored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding/upsert batch size. 16 is conservative for M1/MPS.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device: cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint JSON path.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    corpus_path = REPO_ROOT / args.corpus
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    corpus_df = pd.read_parquet(corpus_path)

    total = len(corpus_df)

    print(f"Corpus: {total} chunks")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")

    # ------------------------------------------------------------
    # Load embedder with explicit device.
    # ------------------------------------------------------------

    embedder = get_embedder(
        args.model,
        device=args.device,
    )

    print(
        f"Embedder: {embedder.name}, "
        f"dimension={embedder.dimension}"
    )

    print(
        f"Device verification: "
        f"requested={getattr(embedder, 'requested_device', args.device)}, "
        f"actual={getattr(embedder, 'actual_device', 'unknown')}"
    )

    # Warm-up happens once and is NOT part of corpus timing.

    if hasattr(embedder, "warm_up"):
        print("Warming up...")
        warmup_ms = embedder.warm_up(n_calls=3)
        print(f"Warm-up: {warmup_ms:.1f}ms")

    # ------------------------------------------------------------
    # Persistent Qdrant location.
    # ------------------------------------------------------------

    qdrant_path = (
        None
        if args.qdrant_url
        else str(REPO_ROOT / args.qdrant_path)
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else REPO_ROOT
        / "data"
        / "processed"
        / f"qdrant_index_checkpoint_{args.collection}.json"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Resume detection.
    # ------------------------------------------------------------

    start_batch = 0

    if checkpoint_path.exists():
        try:
            with checkpoint_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                checkpoint = json.load(f)

            compatible = (
                checkpoint.get("corpus") == args.corpus
                and checkpoint.get("model") == args.model
                and checkpoint.get("collection") == args.collection
                and checkpoint.get("device") in {"mps", "cpu"}
                and checkpoint.get("total_chunks") == total
                and checkpoint.get("batch_size") == args.batch_size
            )

            if compatible:
                start_batch = int(
                    checkpoint.get(
                        "next_batch",
                        0,
                    )
                )

                print(
                    f"Checkpoint found: resuming at "
                    f"batch {start_batch}"
                )
            else:
                print(
                    "Existing checkpoint is incompatible "
                    "with this run. Starting from batch 0."
                )

        except Exception as exc:
            print(
                f"Could not read checkpoint ({exc}); "
                "starting from batch 0."
            )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # If no checkpoint exists, create=True recreates the collection.
    #
    # If a compatible checkpoint exists, create=False preserves it.
    # ------------------------------------------------------------

    create_collection = start_batch == 0

    qdrant = QdrantIndex(
        collection_name=args.collection,
        dimension=embedder.dimension,
        path=qdrant_path,
        url=args.qdrant_url,
        create=create_collection,
    )

    print(
        f"Qdrant collection: {args.collection}"
    )

    print(
        f"Existing points: {qdrant.count()}"
    )

    # ------------------------------------------------------------
    # Streaming embedding + upsert.
    # ------------------------------------------------------------

    total_embed_ms = 0.0
    total_upsert_ms = 0.0

    total_batches = (
        total + args.batch_size - 1
    ) // args.batch_size

    overall_t0 = time.perf_counter()

    for batch_no in range(
        start_batch,
        total_batches,
    ):
        start = batch_no * args.batch_size
        end = min(
            start + args.batch_size,
            total,
        )

        batch_df = corpus_df.iloc[start:end]

        texts = batch_df["text"].tolist()
        ids = batch_df["chunk_id"].tolist()

        payloads = [
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
            for _, row in batch_df.iterrows()
        ]

        # -------------------------
        # Embed ONLY this batch.
        # -------------------------

        embed_t0 = time.perf_counter()

        vectors = embedder.embed(
            texts,
            is_query=False,
        )

        batch_embed_ms = (
            time.perf_counter() - embed_t0
        ) * 1000

        total_embed_ms += batch_embed_ms

        # -------------------------
        # Immediately upsert.
        # -------------------------

        batch_upsert_ms = qdrant.upsert(
            ids,
            vectors,
            payloads,
        )

        total_upsert_ms += batch_upsert_ms

        # -------------------------
        # Release batch memory.
        # -------------------------

        del vectors
        del texts
        del ids
        del payloads
        del batch_df

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

        # -------------------------
        # Checkpoint AFTER successful upsert.
        # -------------------------

        next_batch = batch_no + 1

        checkpoint = {
            "status": (
                "COMPLETE"
                if next_batch >= total_batches
                else "IN_PROGRESS"
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "corpus": args.corpus,
            "model": args.model,
            "device": args.device,
            "collection": args.collection,
            "qdrant_path": qdrant_path,
            "qdrant_url": args.qdrant_url,
            "dimension": embedder.dimension,
            "total_chunks": total,
            "batch_size": args.batch_size,
            "completed_batches": next_batch,
            "total_batches": total_batches,
            "next_batch": next_batch,
            "points_written": end,
        }

        with checkpoint_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                checkpoint,
                f,
                indent=2,
            )

        # -------------------------
        # Progress output.
        # -------------------------

        elapsed = (
            time.perf_counter()
            - overall_t0
        )

        percent = (
            end / total * 100
            if total
            else 100
        )

        print(
            f"{end}/{total} "
            f"({percent:.1f}%) | "
            f"batch {batch_no + 1}/{total_batches} | "
            f"embed={batch_embed_ms:.1f}ms | "
            f"upsert={batch_upsert_ms:.1f}ms | "
            f"elapsed={elapsed:.1f}s"
        )

    # ------------------------------------------------------------
    # Final manifest.
    # ------------------------------------------------------------

    point_count = qdrant.count()

    manifest = {
        "status": "COMPLETE",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": args.model,
        "device": args.device,
        "actual_device": getattr(
            embedder,
            "actual_device",
            "unknown",
        ),
        "dimension": embedder.dimension,
        "collection": args.collection,
        "corpus_file": args.corpus,
        "corpus_size": total,
        "batch_size": args.batch_size,
        "embed_total_ms": round(
            total_embed_ms,
            2,
        ),
        "upsert_total_ms": round(
            total_upsert_ms,
            2,
        ),
        "qdrant_path": qdrant_path,
        "qdrant_url": args.qdrant_url,
        "qdrant_point_count": point_count,
        "checkpoint": str(
            checkpoint_path
        ),
    }

    manifest_path = (
        REPO_ROOT
        / "data"
        / "processed"
        / "qdrant_index_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Index complete: {point_count} points"
    )
    print(
        f"Collection: {args.collection}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Checkpoint: {checkpoint_path}"
    )


if __name__ == "__main__":
    main()

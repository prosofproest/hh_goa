# Retrieval Architecture (Phase 4)

## Honest status of this phase

**Everything in this document's code was built and tested in a sandbox
that cannot reach Hugging Face** (confirmed in Phase 2: `huggingface.co`
returns `403 host_not_allowed` from this environment's egress proxy). This
has direct consequences for what could and could not be verified here:

| Component | Tested here with | Status |
|---|---|---|
| Retrieval metrics (Recall@K, MRR, Hit@K, nDCG) | Hand-verified unit tests with known expected values | ✅ Correct, real |
| RRF fusion | Hand-verified unit test | ✅ Correct, real |
| BM25 sparse retrieval | `rank_bm25`, real library, real local execution | ✅ Real |
| Qdrant indexing/search/filtering | `qdrant-client`, real library, local `:memory:`/on-disk mode | ✅ Real |
| Chunk-level dedup / anti-leakage filter guard | Real code, tested | ✅ Real |
| Benchmark stratification & reproducibility | Real code, tested against synthetic data, seed-reproducibility verified | ✅ Real |
| **Real multilingual embedding models** (multilingual-e5, bge-m3, indic-sentence-bert) | **Not run** — requires downloading weights from Hugging Face | ❌ **Not benchmarked in this sandbox** |
| **Real retrieval-quality numbers on real MSMARCO-XI data** | **Not run** — no real corpus/benchmark queries were embedded with a real model | ❌ **Not measured yet** |

Every numeric result described below as "synthetic" was produced with
`TfidfHashingEmbedder` (`packages/indexing/embeddings.py`) — a
character-n-gram hashing vectorizer used **only** to exercise code paths
(indexing, search, fusion, metric computation, latency measurement). It has
no semantic understanding and must never be cited as a retrieval-quality
result. See "How to run the real benchmark" below for exact commands to
run on a machine with Hugging Face access.

## Pipeline

```
corpus subset (parquet)              benchmark queries (jsonl)
        |                                       |
        v                                       v
 [embed corpus, batched]              [ground truth: is_selected
        |                              -> parent_passage_id, NEVER
        v                              fed to embedder/index]
 [Qdrant: upsert dense vectors
  + legitimate-only payload:
  language, query_type,
  chunk_strategy, source_lang,
  target_lang, passage_index;
  is_selected stored for OFFLINE
  EVAL ONLY, never filtered on]
        |
        v
 for each benchmark query:
   embed query -> dense search (top_k)
   [+ BM25 sparse search (top_k)
    -> Reciprocal Rank Fusion]
        |
        v
 [score against ground truth:
  Recall@1/3/5/10, Hit@1/3/5/10,
  MRR@10, nDCG@10 — overall AND
  broken down by query_type]
        |
        v
 [latency percentiles P50-P100
  per stage, across ALL queries]
        |
        v
 data/processed/retrieval_benchmark_report.json
```

## Anti-leakage guarantees (verified, not just claimed)

1. **`QdrantIndex.search()` raises `ValueError`** if a caller passes
   `is_selected` as a filter key — tested directly (see Phase 3→4
   transition test log). Only `language`, `query_type`, `chunk_strategy`
   are accepted as legitimate routing/filter signals.
2. **Benchmark queries never embed `Answer`/`Eng_Answer` text.**
   `build_retrieval_benchmark.py` reads `is_selected` only to compute
   `ground_truth_parent_passage_ids` — a field consumed exclusively by the
   post-hoc scoring step in `run_retrieval_benchmark.py`, never passed to
   `embedder.embed()` or `qdrant.search()`.
3. **Qdrant payload stores `is_selected`** (per the required metadata
   schema) but it is documented, in-code, as "OFFLINE EVAL ONLY" at every
   call site that writes it.

## Corpus subset sampling methodology

`scripts/build_corpus_subset.py` builds three groups, deterministically
seeded:

1. **Positive chunks** — every chunk whose `parent_passage_id` is a
   ground-truth-relevant passage for some benchmark query. Required so
   Recall@K is even structurally achievable.
2. **Hard negatives** — other chunks from the *same* source record
   (`query_id`) as a benchmark query, but not the selected passage. These
   are the hardest negatives because they share topic/document context.
3. **Random negatives** — a seeded random sample from unrelated records,
   filling up to `--target-corpus-size`, so retrieval must discriminate
   across genuinely different documents, not just within one record's
   sibling passages.

The manifest written alongside the corpus (`*.manifest.json`) reports
exact counts for each group and flags any ground-truth passage that ended
up with zero corpus chunks (which would make perfect recall structurally
impossible for that query — this can happen if `--strategies` excludes the
strategy containing the ground-truth chunk).

## Embedding model candidates (not yet benchmarked with real numbers)

See `configs/retrieval_benchmark.yaml` for the full list and per-model
rationale (`intfloat/multilingual-e5-base`, `BAAI/bge-m3`,
`ai4bharat/indic-sentence-bert-nli`). Selection criteria to be applied once
real numbers exist: Kannada support, embedding dimensionality, inference
speed, memory footprint, measured Recall@K/MRR/nDCG, and licensing — **no
model is pre-selected**; `run_retrieval_benchmark.py` accepts a
comma-separated `--models` list and reports all of them side-by-side.

## Query routing (implemented, deterministic, no LLM)

`scripts/retrieve.py` accepts `--filter-language` and `--filter-query-type`
as the only legitimate routing/filter signals, matching the spec's
"deterministic or very lightweight" router requirement. A full router
module tying query-length/script-detection heuristics to these filters
automatically is not yet built — Phase 4 validates the underlying
filter mechanism works (see Qdrant anti-leakage tests above); wiring a
router module on top is a small, low-risk follow-up once real embedding
results indicate which signals actually help.

## Reranking — deliberately not added yet

Per spec section 12, Phase 4 benchmarks dense and hybrid (RRF) retrieval
**without** a reranker first (`configs/retrieval_benchmark.yaml` has
`reranker.enabled: false`). Whether a lightweight reranker is worth its
latency cost is an empirical question that requires real Recall/MRR
numbers to answer — premature to add before those exist.

## Latency budget — what's real, what's projected

Measured (synthetic-embedder, local `:memory:` Qdrant, 172-chunk corpus,
20 queries) — **illustrative of the code path's overhead, not of real
retrieval latency at production corpus scale**:

- Query embedding: ~0.3-0.5ms (with the lightweight hashing proxy; a real
  transformer-based encoder will be substantially slower — likely
  single-digit to double-digit ms on CPU, lower on GPU — this must be
  re-measured with a real model before any 200ms claim is made)
- Qdrant dense search: ~0.7-1.7ms at this small corpus size
- BM25 sparse search: ~0.07-0.12ms
- RRF fusion: ~0.02-0.06ms
- **Qdrant client + collection init: ~1000ms** — this is a one-time
  process-startup cost (confirmed via a real bug found and fixed during
  this phase: `scripts/retrieve.py` originally recreated/wiped the
  collection on every invocation, which both broke correctness and
  inflated the apparent per-query latency; fixed by adding a
  `create=False` connect-only mode to `QdrantIndex`). In a real server
  process this happens once at startup, not per request — but it is a
  real cost that must be accounted for in deployment design (keep the
  Qdrant client alive across requests, never re-instantiate per query).

**What Phase 4 has NOT yet established:** real per-stage latency at the
full ~998,513-chunk (or even the 50,000-chunk target) corpus size with a
real embedding model. Corpus size materially affects Qdrant search latency
(HNSW graph traversal is sub-linear but not free), and CPU-based
transformer embedding is orders of magnitude slower than the hashing
proxy used here. **No 200ms compliance claim can be made from this data.**
The real benchmark run (see below) is required before that budget can be
filled in honestly.

## How to run the real benchmark (on a machine with Hugging Face access)

```bash
# 1. Full ingestion + chunking (if not already done from Phase 3)
python3 scripts/ingest_dataset.py --file data/raw/validation/kanval.parquet --output-dir data/processed/normalized_full
python3 scripts/chunk_dataset.py --input data/processed/normalized_full/normalized.parquet --output-dir data/processed/chunks_full --drop-duplicate-chunks

# 2. Build benchmark queries (stratified, 1000, seeded)
python3 scripts/build_retrieval_benchmark.py \
  --input data/processed/normalized_full/normalized.parquet \
  --output data/processed/retrieval_benchmark_queries.jsonl \
  --size 1000 --seed 42 \
  --chunks-dir data/processed/chunks_full

# 3. Build a representative corpus subset (not the full 998K chunks)
python3 scripts/build_corpus_subset.py \
  --benchmark-queries data/processed/retrieval_benchmark_queries.jsonl \
  --chunks-dir data/processed/chunks_full \
  --output data/processed/retrieval_corpus_subset.parquet \
  --target-corpus-size 50000 --seed 42

# 4. Install real embedding dependencies
pip install sentence-transformers torch --break-system-packages

# 5. Run the real benchmark against 2-3 real multilingual models
python3 scripts/run_retrieval_benchmark.py \
  --corpus data/processed/retrieval_corpus_subset.parquet \
  --benchmark-queries data/processed/retrieval_benchmark_queries.jsonl \
  --config configs/retrieval_benchmark.yaml \
  --models "intfloat/multilingual-e5-base,BAAI/bge-m3"

# 6. Inspect data/processed/retrieval_benchmark_report.json for real
#    Recall@K/MRR/nDCG/latency numbers, broken down by model and query_type.
```

## Known limitations (honest, as of Phase 4)

1. **No real embedding-quality numbers exist yet** — see status table above.
2. **`scripts/retrieve.py`'s output does not populate `text`** for
   retrieved chunks — the Qdrant payload schema intentionally excludes
   chunk text (kept lean, per the spec's metadata field list). A
   production version would join `chunk_id` back against the chunk
   parquet corpus (or store text in a separate fast key-value store) to
   populate it; this join is out of scope for the benchmark-focused
   script built in this phase.
3. **BM25 tokenization is whitespace-only** — no Kannada-specific
   morphological analysis or stemming. This is a known limitation that
   should be revisited if hybrid retrieval's real-model numbers show BM25
   underperforming due to tokenization mismatches.
4. **Qdrant payload indexes have no effect in local (`:memory:`/on-disk)
   mode** — confirmed via a runtime warning from the qdrant-client
   library itself. Filtering still works correctly (verified), but
   filter *performance* at scale requires a real Qdrant server, which is
   explicitly out of scope until a later deployment phase.
5. **Corpus subset sampling has not been run against the real 998,513-chunk
   corpus** — `build_corpus_subset.py` was only tested against synthetic
   data in this sandbox; its logic (positive/hard-negative/random-negative
   split) should be spot-checked against the real corpus's actual size
   distribution when run for real.

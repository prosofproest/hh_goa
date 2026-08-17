# Ingestion & Chunking Architecture (Phase 3)

## Scope and verified data

This document describes the ingestion + chunking pipeline built against the
**verified** schema of `data/raw/validation/kanval.parquet` (see
`data/processed/dataset_schema.json`, produced in Phase 2):

- 97,941 rows, 10 columns
- `source_lang = eng_Latn`, `target_lang = kan_Knda` (Kannada validation split only)
- Columns: `source_lang, target_lang, meta, Answer, query_id, query_type, passages, Eng_Query, Eng_Answer, query`
- `passages` is a nested object with `English_passages`, `Translated_passages`, `is_selected` (parallel lists, 1–27 entries per record)
- `query_type` ∈ {DESCRIPTION, NUMERIC, ENTITY, PERSON, LOCATION}
- 44.92% of records are no-answer records (`Eng_Answer == "No Answer Present."`)

**Nothing in this document generalizes those statistics to other MSMARCO-XI
languages/splits** — they are true only for the physically inspected Kannada
validation file until other files are independently inspected.

## Pipeline overview

```
raw parquet (streamed in batches)
    -> validators.validate_raw_row()      [structural checks; raises on corruption]
    -> normalizer.normalize_row()         [-> NormalizedRecord]
    -> dedup.DedupTracker.process()       [exact-duplicate drop; content-distinct query_id preserved]
    -> normalized.parquet + normalization_report.json
    -> selector.run_strategies()          [passage / paragraph / sentence / adaptive]
    -> chunks/strategy=*/language=*/part.parquet + ingestion_report.json
    -> validate_chunks.py                 [structural + consistency checks, exit code]
```

Two scripts implement this: `scripts/ingest_dataset.py` (normalize + dedup)
and `scripts/chunk_dataset.py` (multi-strategy chunking). They are separate
so the (expensive, I/O-bound) normalization pass and the (CPU-bound)
chunking pass can be run, tested, and re-run independently — e.g. you can
re-chunk with different strategy configs without re-reading the 460 MB
source file.

## Why each chunking strategy exists

| Strategy | When it's the best choice | What it protects against |
|---|---|---|
| **passage** | Baseline / control group for retrieval evaluation. Also the right choice when passages are already short (most MSMARCO passages). | Never breaks a passage mid-sentence; simplest possible retrieval unit. |
| **paragraph** | Passages with genuine internal paragraph structure, or passages long enough that a single embedding vector would blur multiple sub-topics. | Splitting mid-thought; groups sentences into topic-sized units. |
| **sentence** | Passages where recall of a *specific fact* matters more than topical grouping — smaller windows increase the chance a short, precise span is retrieved. | Chunks that are too large to score well against short factual queries. |
| **adaptive** | Production default candidate: applies passage/paragraph/sentence logic *conditionally* based on the passage's own length, so short filler passages aren't artificially split and long passages aren't left as one diffuse vector. | Both over-splitting short text and under-splitting long text. |

All four strategies run over **every** passage (multi-view chunking): the
same source passage produces up to 4 differently-shaped chunk
representations, each tagged with `chunk_strategy` in its metadata. This
lets Phase 4's retrieval evaluation compare strategies head-to-head
(Recall@K, MRR, nDCG per strategy) rather than committing to one strategy
by assumption.

**Not yet implemented, and why:** the broader project brief additionally
describes "semantic chunking" (splitting where *embedding* similarity drops
below a threshold) and "hierarchical chunking" (document → section →
paragraph → child chunk parent/child tree). Phase 3 explicitly excludes
building embeddings, and MSMARCO-XI passages are single, flat text blocks
with no section markup — so true embedding-based semantic chunking and
multi-level document hierarchy don't yet have the inputs they need. The
`adaptive` strategy's `VERY_LONG` branch uses a **structural** recursive
split (halving at sentence boundaries) as an honest stand-in, clearly
labeled as such in `packages/chunking/adaptive.py`. True semantic chunking
should be revisited in Phase 5 once an embedding model is benchmarked and
selected.

## How overlap is determined

- **paragraph strategy**: when a passage has no literal paragraph breaks
  (the common case for MSMARCO-style single-block passages), it falls back
  to grouping sentences into a token budget (`target_tokens`, default 150)
  with sentence-level overlap capped at `overlap_tokens` (default 20) —
  i.e. trailing sentences from one chunk whose combined length is under the
  overlap budget are repeated at the start of the next chunk.
- **sentence strategy**: explicit `SentenceChunkConfig(min_tokens,
  target_tokens, max_tokens, overlap_sentences)` — all four are
  constructor arguments, not hardcoded constants, per the competition
  requirement. `overlap_sentences` controls how many trailing sentences
  from the previous window are repeated at the start of the next.
- All overlap amounts are configurable per-call; the CLI scripts currently
  expose the strategy defaults, and the same functions can be called with
  custom `SentenceChunkConfig` / `AdaptiveConfig` instances for the Phase 4
  chunking experiments (`ChunkingExperiment`).

## How multilingual text is handled

Every chunk stores **three** text fields, never merged by default:

- `text_target` — the Kannada (`Translated_passages[i]`) text
- `text_english` — the English (`English_passages[i]`) text
- `text` — the "primary" field used for chunk splitting and quality scoring;
  currently `text_target` when non-empty, else `text_english`

This lets Phase 4 retrieval experiments compare Kannada-only, English-only,
and bilingual (concatenated at query time, not storage time) retrieval
without needing to re-chunk.

## How relevance labels are preserved

Each `PassageEntry` inside a `NormalizedRecord` keeps `passage_index` and
`is_selected` exactly as they appeared in the source `is_selected` array.
Every chunk derived from that passage — regardless of chunking strategy —
inherits the same `is_selected` value and a `parent_passage_id` of the form
`{query_id}_p{passage_index}`, so `is_selected` can be traced back
unambiguously for Recall@K / MRR / nDCG evaluation in Phase 4.

The `Answer` / `Eng_Answer` fields are **never** used as chunk text — they
are preserved only as record-level `answer_target` / `answer_english` /
`has_answer` / `answer_status`, for use as evaluation ground truth, not as
retrievable context. `scripts/validate_chunks.py` includes a check that
flags any chunk whose text is literally the no-answer marker string, as a
guard against this specific leakage mode.

## How no-answer examples are preserved

No-answer records (`Eng_Answer == "No Answer Present."`, case/whitespace
normalized) are **kept**, not dropped. `has_answer=False` and
`answer_status="NO_ANSWER"` are stored on the `NormalizedRecord` and
propagated to every chunk derived from that record's passages. These
records are earmarked for **guardrail evaluation** in later phases (does
the system correctly refuse to fabricate an answer when none exists in the
retrieved evidence?).

## How deduplication works

`source_record_hash` is a SHA-256 hash of a canonicalized subset of the raw
record's meaningful fields (`source_lang, target_lang, query_type, query,
Eng_Query, Answer, Eng_Answer, passages`), with dict keys sorted and
whitespace-trimmed so incidental formatting differences don't cause false
non-duplicates.

- **Exact duplicates** (identical hash) are dropped after the first
  occurrence — silently in terms of row count, but explicitly counted in
  `normalization_report.json` (`dedup_stats.exact_duplicates_dropped`).
- **Same `query_id`, different content** is never dropped. The record is
  kept and its `query_id` is suffixed (`__dup1`, `__dup2`, ...) so it
  remains addressable without colliding with the original. This was tested
  directly (see Phase 3 test run below) against a synthetic case built to
  exercise exactly this path.

## Why this representation suits low-latency retrieval later

- Chunk output is columnar (Parquet), partitioned by `strategy=` and
  `language=`, so Phase 4/5 indexing can load exactly one
  strategy/language combination at a time without scanning irrelevant data.
- Every chunk carries `is_selected`, `query_type`, `has_answer`, and
  `quality_score` as first-class columns (not buried in free text), so
  Qdrant payload filtering and quality-based re-ranking (Phase 4+) don't
  require re-parsing text.
- No embeddings or index structures are built in this phase — chunk output
  is embedding-model-agnostic, so the Phase 5 embedding-model benchmark can
  run against the same chunk files regardless of which model is ultimately
  selected.

## Known limitations (honest, as of Phase 3)

1. **Token counts are approximate.** `estimate_tokens()` is a whitespace
   word-count proxy, not a real tokenizer count from the model that will
   eventually be selected in Phase 5. This is documented in
   `packages/chunking/base.py`.
2. **Quality scoring is structural, not semantic.** `quality_score`
   currently combines a sentence-count heuristic, lexical-diversity ratio,
   and boundary-punctuation check — not embedding-based coherence, since no
   embedding model has been selected yet. See `packages/chunking/quality.py`
   for the explicit breakdown and the plan to revisit this in Phase 5+.
2. **Sentence splitting is rule-based** (regex on `. ! ? ।`), not a
   language model or NLP-library sentence tokenizer. This was a deliberate
   choice to avoid adding a heavy/unverified dependency at this stage; it
   should be revisited if evaluation in Phase 4 shows sentence-boundary
   errors materially hurting chunk quality for Kannada text specifically.
3. **Paragraph splitting depends on literal blank lines**, which MSMARCO-XI
   passages generally do not contain (single-block passages), so in
   practice the paragraph strategy's fallback (sentence-grouped-by-budget)
   is the dominant code path for this dataset. This is intentional and
   documented in `packages/chunking/paragraph.py`, not a bug.
4. **Only the Kannada validation file has been run through this pipeline
   end-to-end against real data** at the time of writing. The full 97,941
   record run has not yet been executed or reported on (that is the next
   step, pending user execution — see the Phase 3 report for how to run it).

## Chunk-level deduplication (added after real 1,000-record test)

The first real 1,000-record run against `kanval.parquet` produced ~40,000
chunks (≈40/record) with paragraph/sentence/adaptive counts all close to
the passage count. Investigation (`packages/chunking/chunk_dedup.py`)
confirmed this is expected: with default token thresholds (short≤40,
medium≤120 words) and typical MSMARCO-XI passage lengths (~40-70 words),
most passages don't reach the length where paragraph/sentence/adaptive
splitting actually diverges from "keep whole" — so all four strategies
legitimately produce identical text for the majority of passages.

**Dedup rule, strictly scoped:** two chunks are only ever compared for
duplication if they share the same `(parent_passage_id, language)` — i.e.
they must come from the *same source passage*. Chunks from different
passages are never collapsed, even if their text happens to be identical,
per explicit requirement (different passages are different documents/
provenance, even when their content coincides).

**Non-destructive by default:** `scripts/chunk_dataset.py` always computes
`chunk_text_hash`, `is_duplicate_chunk`, and `duplicate_of_chunk_id` for
every chunk and writes them to the output — duplicates are flagged, not
silently removed. The canonical chunk (preferred in order: passage >
paragraph > sentence > adaptive, tie-broken deterministically) records
`contributing_strategies` and `duplicate_chunk_ids` in its metadata.
Physically excluding duplicates from the written parquet is opt-in via
`--drop-duplicate-chunks`, so the decision to shrink the corpus before
embedding is explicit and auditable, not implicit.

Similarly, `--min-chunk-chars` (default 10) flags chunks below that length
via `below_min_length` rather than deleting them; `--drop-below-min-length`
makes exclusion explicit and opt-in.

See `scripts/compare_chunk_strategies.py` for the structural comparison
report this makes possible (chunks-per-strategy, duplicate rate per
strategy, chunk length distribution, compactness) — deliberately scoped to
structural properties only; it does not rank strategies by retrieval
quality, which requires embeddings (Phase 4+).

"""
Deduplication logic.

Rules (per Phase 3 spec):
    - Only exact duplicate records (identical source_record_hash) are
      deduplicated (i.e. dropped after the first occurrence).
    - Records that share a query_id but differ in content are NEVER
      dropped; they are preserved and given a distinguishing query_id
      suffix (__dup1, __dup2, ...) so no valid data is silently lost.

This is a streaming-friendly tracker: call `process(record)` once per
record in file order; it mutates the record's dedup flags in place and
returns False if the record should be skipped (exact duplicate).
"""

from __future__ import annotations

from collections import defaultdict

from .schema import NormalizedRecord


class DedupTracker:
    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()
        self._query_id_occurrences: dict[str, int] = defaultdict(int)
        self.exact_duplicates_dropped = 0
        self.content_distinct_duplicate_query_ids = 0

    def process(self, record: NormalizedRecord) -> bool:
        """Returns True if the record should be kept, False if it is an
        exact duplicate and should be dropped."""
        is_exact_dup = record.source_record_hash in self._seen_hashes
        record.is_exact_duplicate = is_exact_dup

        if is_exact_dup:
            self.exact_duplicates_dropped += 1
            return False

        self._seen_hashes.add(record.source_record_hash)

        occurrence = self._query_id_occurrences[record.query_id]
        self._query_id_occurrences[record.query_id] += 1

        if occurrence > 0:
            # Same query_id seen before, but different content (since we
            # already returned False above for exact hash duplicates).
            record.is_duplicate_query_id = True
            self.content_distinct_duplicate_query_ids += 1
            record.query_id = f"{record.query_id}__dup{occurrence}"

        return True

    def stats(self) -> dict:
        repeated_query_ids = sum(1 for v in self._query_id_occurrences.values() if v > 1)
        return {
            "unique_source_record_hashes": len(self._seen_hashes),
            "exact_duplicates_dropped": self.exact_duplicates_dropped,
            "content_distinct_duplicate_query_ids": self.content_distinct_duplicate_query_ids,
            "query_ids_with_multiple_occurrences": repeated_query_ids,
        }

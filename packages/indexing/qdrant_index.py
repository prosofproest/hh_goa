"""
Qdrant index build + query wrapper.

Uses the real qdrant-client library. Supports:
    - path=":memory:"  -> ephemeral in-process Qdrant (used for local dev/
      testing in this sandbox — genuinely functional, not a mock)
    - path="/some/dir" -> persistent local on-disk Qdrant (no server needed)
    - url="http://host:6333" -> real Qdrant server (production/Docker mode)

Payload schema stores exactly the metadata fields specified in the Phase 4
spec. `is_selected` is stored for OFFLINE EVALUATION purposes only — no
query path in this module ever filters or boosts using it.
"""

from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

PAYLOAD_FIELDS = [
    "chunk_id", "parent_passage_id", "query_id", "language", "source_lang",
    "target_lang", "query_type", "chunk_strategy", "passage_index", "is_selected",
]


class QdrantIndex:
    def __init__(self, collection_name: str, dimension: int, path: str | None = ":memory:",
                 url: str | None = None, distance: str = "Cosine", create: bool = True):
        """`create=True` (default) creates/recreates the collection — use for
        index-building. `create=False` connects to an EXISTING collection
        without touching its contents — use for querying (scripts/retrieve.py),
        otherwise every query process would wipe the index on startup."""
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path)

        self.collection_name = collection_name
        self.dimension = dimension

        distance_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Dot": qmodels.Distance.DOT,
            "Euclid": qmodels.Distance.EUCLID,
        }

        if create:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=dimension, distance=distance_map[distance]),
            )

            # Create payload indexes for frequently-filtered metadata fields
            # (language, query_type — legitimate routing signals; explicitly
            # NOT is_selected, which must never be used as a retrieval filter).
            for field_name, schema_type in [
                ("language", qmodels.PayloadSchemaType.KEYWORD),
                ("query_type", qmodels.PayloadSchemaType.KEYWORD),
                ("chunk_strategy", qmodels.PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=collection_name, field_name=field_name, field_schema=schema_type
                )
        else:
            if not self.client.collection_exists(collection_name):
                raise ValueError(
                    f"Collection '{collection_name}' does not exist at this Qdrant path/url. "
                    f"Build it first with scripts/build_qdrant_index.py, or pass create=True."
                )

    def upsert(self, chunk_ids: list[str], vectors, payloads: list[dict], batch_size: int = 256) -> float:
        """Returns upsert wall-clock time in ms."""
        t0 = time.perf_counter()
        points = []
        for cid, vec, payload in zip(chunk_ids, vectors, payloads):
            # Qdrant point IDs must be int or UUID; we keep the original
            # chunk_id in the payload for identity, and derive a
            # deterministic UUID from it for the point ID.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, cid))
            points.append(
                qmodels.PointStruct(id=point_id, vector=vec.tolist(), payload={**payload, "chunk_id": cid})
            )
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i + batch_size])
        return (time.perf_counter() - t0) * 1000

    def search(self, query_vector, top_k: int = 30, query_filter: dict | None = None) -> tuple[list[dict], float]:
        """Returns (results, latency_ms). `query_filter` may contain
        legitimate routing keys only: 'language', 'query_type',
        'chunk_strategy'. is_selected is intentionally not accepted here."""
        qfilter = None
        if query_filter:
            allowed = {"language", "query_type", "chunk_strategy"}
            disallowed = set(query_filter.keys()) - allowed
            if disallowed:
                raise ValueError(
                    f"Refusing to filter on disallowed field(s) {disallowed}. "
                    f"Only {allowed} are permitted as retrieval filters (is_selected "
                    f"must never be used as a production filter)."
                )
            conditions = [
                qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                for k, v in query_filter.items()
            ]
            qfilter = qmodels.Filter(must=conditions)

        t0 = time.perf_counter()
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        ).points
        latency_ms = (time.perf_counter() - t0) * 1000

        results = [{"chunk_id": h.payload["chunk_id"], "score": h.score, "payload": h.payload} for h in hits]
        return results, latency_ms

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

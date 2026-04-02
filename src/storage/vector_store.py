"""Qdrant embedded vector store for clinical trial embeddings."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM


class VectorStore:
    """Qdrant embedded vector store — no server needed."""

    def __init__(self, path: str | None = None, embedding_dim: int = EMBEDDING_DIM):
        self.path = path or QDRANT_PATH
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(path=self.path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

    def upsert(self, trial_id: str, vector: list[float], metadata: dict):
        """Insert or update a trial embedding with metadata."""
        point = PointStruct(
            id=hash(trial_id) % (2**63),
            vector=vector,
            payload={"trial_id": trial_id, **metadata},
        )
        self.client.upsert(collection_name=QDRANT_COLLECTION, points=[point])

    def upsert_batch(self, items: list[dict]):
        """Batch insert: each item has 'id', 'vector', 'metadata'."""
        points = [
            PointStruct(
                id=hash(item["id"]) % (2**63),
                vector=item["vector"],
                payload={"trial_id": item["id"], **item["metadata"]},
            )
            for item in items
        ]
        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        language: str | None = None,
        phase: str | None = None,
        drug: str | None = None,
        country: str | None = None,
    ) -> list[dict]:
        """Search for similar trials with optional metadata filters."""
        conditions = []
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        if phase:
            conditions.append(FieldCondition(key="phase", match=MatchValue(value=phase)))
        if drug:
            conditions.append(FieldCondition(key="drugs", match=MatchValue(value=drug)))
        if country:
            conditions.append(FieldCondition(key="country", match=MatchValue(value=country)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return [
            {
                "score": hit.score,
                "trial_id": hit.payload.get("trial_id"),
                **hit.payload,
            }
            for hit in results.points
        ]

    def count(self) -> int:
        """Get total number of vectors in the collection."""
        info = self.client.get_collection(QDRANT_COLLECTION)
        return info.points_count

    def delete_collection(self):
        """Delete the collection (for reset)."""
        self.client.delete_collection(QDRANT_COLLECTION)

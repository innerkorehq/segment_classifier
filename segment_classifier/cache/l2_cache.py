import asyncio
import json
import uuid
from pathlib import Path
import aiofiles
import numpy as np
from pydantic import ValidationError
from segment_classifier.models import ClusterRecord, ComponentType


class L2FuzzyCache:
    def __init__(
        self,
        cache_path: str,
        embeddings_path: str,
        similarity_threshold: float = 0.85,
        max_cluster_size: int = 50,
        persist_on_update: bool = True
    ):
        self._path = Path(cache_path)
        self._embeddings_path = Path(embeddings_path)
        self._similarity_threshold = similarity_threshold
        self._max_cluster_size = max_cluster_size
        self._persist_on_update = persist_on_update

        self._store: list[ClusterRecord] = []
        # Matrix storing centroids, parallel to self._store
        self._centroids: np.ndarray | None = None

        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if not self._path.exists() or not self._embeddings_path.exists():
            return

        async with aiofiles.open(self._path, "r", encoding="utf-8") as f:
            content = await f.read()
            if content.strip():
                try:
                    data = json.loads(content)
                    self._store = [ClusterRecord.model_validate(val) for val in data]
                except (json.JSONDecodeError, ValidationError):
                    self._store = []

        if self._store:
            try:
                self._centroids = np.load(str(self._embeddings_path))
                if len(self._store) != self._centroids.shape[0]:
                    # Mismatch between JSON and npy, reset
                    self._store = []
                    self._centroids = None
            except Exception:
                self._store = []
                self._centroids = None

    async def find_nearest(self, vector: list[float], threshold: float | None = None) -> ClusterRecord | None:
        async with self._lock:
            if not self._store or self._centroids is None:
                return None

            query = np.array(vector)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return None

            # Cosine similarity
            dot_products = np.dot(self._centroids, query)
            centroid_norms = np.linalg.norm(self._centroids, axis=1)
            # Avoid division by zero
            centroid_norms[centroid_norms == 0] = 1

            similarities = dot_products / (centroid_norms * query_norm)

            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            check_threshold = threshold if threshold is not None else self._similarity_threshold

            if best_sim >= check_threshold:
                return self._store[best_idx]
            return None

    async def add_to_cluster(self, cluster_id: str, fingerprint_hash: str, vector: list[float]) -> None:
        async with self._lock:
            idx = next((i for i, c in enumerate(self._store) if c.cluster_id == cluster_id), None)
            if idx is not None and self._centroids is not None:
                cluster = self._store[idx]

                # Check size
                if len(cluster.member_fingerprints) < self._max_cluster_size:
                    if fingerprint_hash not in cluster.member_fingerprints:
                        cluster.member_fingerprints.append(fingerprint_hash)

                        # Update centroid (running mean)
                        n = len(cluster.member_fingerprints)
                        old_centroid = self._centroids[idx]
                        new_vec = np.array(vector)
                        # (old_centroid * (n-1) + new_vec) / n
                        new_centroid = (old_centroid * (n - 1) + new_vec) / n

                        self._centroids[idx] = new_centroid
                        cluster.centroid_vector = new_centroid.tolist()

                        if self._persist_on_update:
                            await self._persist_unsafe()

    async def create_cluster(
        self,
        fingerprint_hash: str,
        vector: list[float],
        component_type: ComponentType,
        confidence: float,
    ) -> ClusterRecord:
        async with self._lock:
            cluster_id = str(uuid.uuid4())
            record = ClusterRecord(
                cluster_id=cluster_id,
                centroid_vector=vector,
                component_type=component_type,
                confidence=confidence,
                member_fingerprints=[fingerprint_hash]
            )

            self._store.append(record)

            new_vec = np.array([vector])
            if self._centroids is None:
                self._centroids = new_vec
            else:
                self._centroids = np.vstack([self._centroids, new_vec])

            if self._persist_on_update:
                await self._persist_unsafe()

            return record

    async def _persist_unsafe(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        data = [c.model_dump(mode="json") for c in self._store]
        async with aiofiles.open(self._path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))

        # Write npy
        if self._centroids is not None:
            # We can't do aiofiles easily for numpy, doing it sync for now
            # as it's a small matrix and we use it as memory store
            np.save(str(self._embeddings_path), self._centroids)

    async def persist(self) -> None:
        async with self._lock:
            await self._persist_unsafe()

    @property
    def size(self) -> int:
        return len(self._store)

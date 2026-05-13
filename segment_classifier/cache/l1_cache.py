import asyncio
import json
from pathlib import Path
import aiofiles
from pydantic import ValidationError
from segment_classifier.models import FingerprintRecord


class L1FingerprintCache:
    def __init__(self, cache_path: str, auto_persist_every: int = 50):
        self._path = Path(cache_path)
        self._store: dict[str, FingerprintRecord] = {}
        self._lock = asyncio.Lock()
        self._write_count = 0
        self._auto_persist_every = auto_persist_every

    async def load(self) -> None:
        if not self._path.exists():
            return

        async with aiofiles.open(self._path, "r", encoding="utf-8") as f:
            content = await f.read()
            if not content.strip():
                return

            try:
                data = json.loads(content)
                for key, val in data.items():
                    self._store[key] = FingerprintRecord.model_validate(val)
            except (json.JSONDecodeError, ValidationError) as e:
                # Log error in real app, we'll just ignore and start fresh here
                pass

    async def get(self, fingerprint_hash: str) -> FingerprintRecord | None:
        async with self._lock:
            return self._store.get(fingerprint_hash)

    async def set(self, fingerprint_hash: str, record: FingerprintRecord) -> None:
        async with self._lock:
            self._store[fingerprint_hash] = record
            self._write_count += 1
            if self._write_count >= self._auto_persist_every:
                self._write_count = 0
                await self._persist_unsafe()

    async def increment_hit(self, fingerprint_hash: str) -> None:
        async with self._lock:
            record = self._store.get(fingerprint_hash)
            if record:
                record.hit_count += 1
                self._store[fingerprint_hash] = record
                self._write_count += 1
                if self._write_count >= self._auto_persist_every:
                    self._write_count = 0
                    await self._persist_unsafe()

    async def _persist_unsafe(self) -> None:
        """Called internally when lock is already acquired."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in self._store.items()}
        async with aiofiles.open(self._path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))

    async def persist(self) -> None:
        async with self._lock:
            self._write_count = 0
            await self._persist_unsafe()

    @property
    def size(self) -> int:
        return len(self._store)

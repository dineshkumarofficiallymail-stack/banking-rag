"""Embedding providers behind one tiny interface.
- FastEmbedEmbedder: local ONNX model (BGE-small), no API key, no data egress.
- FakeEmbedder: deterministic hashed bag-of-words for tests/CI (OFFLINE_MODE=true).
The sync model runs in a worker thread so the event loop is never blocked."""

import asyncio
import hashlib
import math
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingDimensionError(RuntimeError):
    pass


class Embedder(Protocol):
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # noqa: S324 (not security)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str, dim: int) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = dim
        logger.info("embeddings.loaded", extra={"extra_fields": {"model": model_name, "dim": dim}})

    def _check(self, vec: list[float]) -> list[float]:
        if len(vec) != self.dim:
            raise EmbeddingDimensionError(
                f"Model returned {len(vec)} dims but EMBEDDING_DIM={self.dim}; fix .env and re-run init_db"
            )
        return vec

    def _embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [self._check([float(x) for x in v]) for v in self._model.embed(texts, batch_size=32)]

    def _embed_query(self, text: str) -> list[float]:
        return self._check([float(x) for x in next(iter(self._model.query_embed(text)))])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_docs, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_query, text)


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.offline_mode or s.embedding_provider == "fake":
        return FakeEmbedder(dim=s.embedding_dim)
    return FastEmbedEmbedder(model_name=s.embedding_model, cache_dir=s.fastembed_cache_path, dim=s.embedding_dim)

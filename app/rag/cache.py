"""Redis answer cache. Key = sha256(normalised question + filters + prompt version).
Disabled transparently when REDIS_URL is unset or Redis is unreachable."""

import hashlib
import json
from typing import Any

from app.core.logging import get_logger, log_fields
from app.rag.prompts import PROMPT_VERSION

logger = get_logger(__name__)


class ResponseCache:
    def __init__(self, redis: Any | None, ttl_seconds: int = 3600, prefix: str = "rag:answer:") -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        self.prefix = prefix

    @staticmethod
    def make_key(question: str, regulator: str | None, top_k: int) -> str:
        norm = " ".join(question.lower().split())
        raw = f"{PROMPT_VERSION}|{regulator or '*'}|{top_k}|{norm}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> dict | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(self.prefix + key)
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache.get_failed", **log_fields(error=type(exc).__name__))
            return None

    async def set(self, key: str, value: dict) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(self.prefix + key, json.dumps(value, default=str), ex=self.ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache.set_failed", **log_fields(error=type(exc).__name__))

"""Download every entry in data/sources.yaml into data/raw/. Safe to re-run (skips existing files).
Run: python -m scripts.download_sources"""

import asyncio
from pathlib import Path

import httpx
import yaml

from app.core.logging import get_logger, log_fields, setup_logging

logger = get_logger(__name__)
RAW_DIR = Path("data/raw")
HEADERS = {"User-Agent": "banking-rag-ingestor/0.1 (+https://github.com)"}


async def download_all(sources_path: Path = Path("data/sources.yaml")) -> None:
    cfg = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers=HEADERS) as client:
        for src in cfg.get("sources", []):
            target = RAW_DIR / src["filename"]
            if target.exists():
                logger.info("download.skip_existing", **log_fields(file=target.name))
                continue
            try:
                resp = await client.get(src["url"])
                resp.raise_for_status()
                target.write_bytes(resp.content)
                logger.info("download.ok", **log_fields(file=target.name, bytes=len(resp.content)))
            except httpx.HTTPError as exc:
                logger.error("download.failed", **log_fields(url=src["url"], error=str(exc)))


if __name__ == "__main__":
    setup_logging()
    asyncio.run(download_all())

"""Ingest documents into Postgres/pgvector.

python -m scripts.ingest --sources data/sources.yaml          # everything listed in the yaml
python -m scripts.ingest --path data/raw/x.pdf --regulator RBI --title "..." --url "..."
python -m scripts.ingest --path data/raw --regulator NPCI      # whole directory
"""

import argparse
import asyncio
from pathlib import Path

import yaml

from app.core.config import get_settings
from app.core.logging import get_logger, log_fields, setup_logging
from app.db.session import SessionLocal
from app.ingestion.loaders import SUPPORTED
from app.ingestion.pipeline import ingest_path

logger = get_logger(__name__)


async def run(args: argparse.Namespace) -> None:
    jobs: list[dict] = []
    if args.sources:
        cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8")) or {}
        for s in cfg.get("sources", []):
            jobs.append(
                {
                    "path": Path("data/raw") / s["filename"],
                    "regulator": s["regulator"],
                    "title": s.get("title"),
                    "source_url": s.get("url"),
                    "doc_type": s.get("doc_type", "circular"),
                }
            )
    else:
        p = Path(args.path)
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
        jobs = [
            {
                "path": f,
                "regulator": args.regulator,
                "title": args.title if not p.is_dir() else None,
                "source_url": args.url,
                "doc_type": args.doc_type,
            }
            for f in files
        ]

    total_chunks = 0
    async with SessionLocal() as db:
        for job in jobs:
            if not Path(job["path"]).exists():
                logger.warning("ingest.missing_file", **log_fields(path=str(job["path"])))
                continue
            result = await ingest_path(db, **job)
            total_chunks += result.chunks
            logger.info(
                "ingest.result",
                **log_fields(title=result.title, chunks=result.chunks, skipped=result.skipped, reason=result.reason),
            )
    logger.info("ingest.summary", **log_fields(documents=len(jobs), chunks=total_chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest regulatory documents")
    parser.add_argument("--sources", help="data/sources.yaml manifest")
    parser.add_argument("--path", help="file or directory")
    parser.add_argument("--regulator", default="RBI", choices=["RBI", "NPCI", "NACH", "OTHER"])
    parser.add_argument("--title")
    parser.add_argument("--url")
    parser.add_argument("--doc-type", default="circular")
    args = parser.parse_args()
    if not args.sources and not args.path:
        parser.error("provide --sources or --path")
    setup_logging(get_settings().log_level)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

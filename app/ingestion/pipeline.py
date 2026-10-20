"""load -> chunk -> embed -> persist. Idempotent on sha256 checksum."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, log_fields
from app.db.models import Chunk, Document
from app.ingestion.chunking import chunk_document
from app.ingestion.embeddings import get_embedder
from app.ingestion.loaders import LoadedDocument, load_file

logger = get_logger(__name__)
EMBED_BATCH = 64


@dataclass
class IngestResult:
    document_id: str | None
    title: str
    chunks: int
    skipped: bool
    reason: str | None = None


async def ingest_loaded(db: AsyncSession, loaded: LoadedDocument) -> IngestResult:
    existing = await db.scalar(select(Document).where(Document.checksum == loaded.checksum))
    if existing:
        return IngestResult(str(existing.id), loaded.title, 0, skipped=True, reason="duplicate checksum")

    text_chunks = chunk_document(loaded)
    if not text_chunks:
        return IngestResult(None, loaded.title, 0, skipped=True, reason="no chunks produced")

    embedder = get_embedder()
    vectors: list[list[float]] = []
    for i in range(0, len(text_chunks), EMBED_BATCH):
        batch = [f"{c.section}\n{c.content}" if c.section else c.content for c in text_chunks[i : i + EMBED_BATCH]]
        vectors.extend(await embedder.embed_documents(batch))

    document = Document(
        title=loaded.title,
        source_url=loaded.source_url,
        regulator=loaded.regulator,
        doc_type=loaded.doc_type,
        checksum=loaded.checksum,
        page_count=loaded.page_count,
    )
    db.add(document)
    await db.flush()

    db.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=c.index,
            page=c.page,
            section=c.section,
            content=c.content,
            token_count=c.token_count,
            embedding=vec,
            meta={"regulator": loaded.regulator, "doc_type": loaded.doc_type},
        )
        for c, vec in zip(text_chunks, vectors, strict=True)
    )
    await db.commit()
    logger.info("ingest.done", **log_fields(title=loaded.title, chunks=len(text_chunks), pages=loaded.page_count))
    return IngestResult(str(document.id), loaded.title, len(text_chunks), skipped=False)


async def ingest_path(
    db: AsyncSession,
    path: str | Path,
    *,
    regulator: str,
    title: str | None = None,
    source_url: str | None = None,
    doc_type: str = "circular",
) -> IngestResult:
    try:
        loaded = load_file(path, title=title, source_url=source_url, regulator=regulator, doc_type=doc_type)
    except ValueError as exc:
        logger.warning("ingest.skip", **log_fields(path=str(path), reason=str(exc)))
        return IngestResult(None, str(path), 0, skipped=True, reason=str(exc))
    return await ingest_loaded(db, loaded)

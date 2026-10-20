"""Corpus management: list (any user), upload + delete (admin only)."""

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import DB, CurrentUser, require_role
from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.ingestion.loaders import SUPPORTED
from app.ingestion.pipeline import ingest_path
from app.schemas.documents import DocumentOut, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])
AdminOnly = Depends(require_role("admin"))


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: DB, _: CurrentUser) -> list[DocumentOut]:
    stmt = (
        select(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [DocumentOut(**{**doc.__dict__, "chunk_count": n}) for doc, n in rows]


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, dependencies=[AdminOnly])
async def upload_document(
    db: DB,
    file: Annotated[UploadFile, File()],
    regulator: Annotated[str, Form(pattern="^(RBI|NPCI|NACH|OTHER)$")],
    title: Annotated[str | None, Form()] = None,
    source_url: Annotated[str | None, Form()] = None,
    doc_type: Annotated[str, Form()] = "circular",
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported type {suffix}; allowed: {sorted(SUPPORTED)}")
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {get_settings().max_upload_mb} MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = await ingest_path(
            db,
            tmp_path,
            regulator=regulator,
            title=title or Path(file.filename or "upload").stem,
            source_url=source_url,
            doc_type=doc_type,
        )
    finally:
        await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
    return UploadResponse(
        document_id=uuid.UUID(result.document_id) if result.document_id else None,
        title=result.title,
        chunks=result.chunks,
        skipped=result.skipped,
        reason=result.reason,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[AdminOnly])
async def delete_document(document_id: uuid.UUID, db: DB) -> None:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await db.delete(doc)  # chunks cascade

"""Hybrid retrieval: pgvector cosine (HNSW) + Postgres full-text (GIN) + exact circular-number match,
fused with Reciprocal Rank Fusion. Returns RetrievedChunk objects ready for prompting and citing."""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger, log_fields
from app.db.models import Chunk, Document
from app.ingestion.embeddings import get_embedder

logger = get_logger(__name__)

CIRCULAR_NO = re.compile(
    r"\b[A-Z]{2,6}/\d{4}-\d{2,4}/\d{1,4}\b"  # RBI/2023-24/117
    r"|\b[A-Z]{2,6}(?:\.[A-Za-z]{1,6})+\.?\s?\d+/[\d.]+/\d{4}-\d{2}\b"  # DPSS.CO.PD.No.629/02.01.014/2019-20
    r"|\bNPCI/\d{4}-\d{2}/[A-Z]+/\d{1,4}\b"  # NPCI/2023-24/NACH/001
)
RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    source_url: str | None
    regulator: str
    page: int | None
    section: str | None
    content: str
    score: float  # fused score in [0,1]
    vector_score: float | None  # cosine similarity if from vector leg
    keyword_hit: bool


class HybridRetriever:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.settings = get_settings()

    async def _vector_leg(self, qvec: list[float], k: int, regulator: str | None):
        dist = Chunk.embedding.cosine_distance(qvec)
        stmt = select(Chunk, Document, (1 - dist).label("sim")).join(Document, Chunk.document_id == Document.id)
        if regulator:
            stmt = stmt.where(Document.regulator == regulator)
        stmt = stmt.order_by(dist).limit(k)
        await self.db.execute(text("SET LOCAL hnsw.ef_search = 80"))
        return (await self.db.execute(stmt)).all()

    async def _keyword_leg(self, question: str, k: int, regulator: str | None):
        tsv = func.to_tsvector("english", Chunk.content)
        tsq = func.plainto_tsquery("english", question)
        rank = func.ts_rank_cd(tsv, tsq)
        stmt = (
            select(Chunk, Document, rank.label("rank")).join(Document, Chunk.document_id == Document.id).where(tsv.op("@@")(tsq))
        )
        if regulator:
            stmt = stmt.where(Document.regulator == regulator)
        stmt = stmt.order_by(rank.desc()).limit(k)
        return (await self.db.execute(stmt)).all()

    async def _exact_leg(self, question: str, k: int):
        hits = CIRCULAR_NO.findall(question)
        if not hits:
            return []
        stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
        conditions = [Chunk.content.ilike(f"%{h.strip()}%") for h in hits]
        return (await self.db.execute(stmt.where(or_(*conditions)).limit(k))).all()

    async def retrieve(self, question: str, *, k: int | None = None, regulator: str | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.top_k
        cand = k * 3
        qvec = await get_embedder().embed_query(question)

        vec_rows = await self._vector_leg(qvec, cand, regulator)
        kw_rows = await self._keyword_leg(question, cand, regulator)
        exact_rows = await self._exact_leg(question, k)

        fused: dict[uuid.UUID, float] = {}
        meta: dict[uuid.UUID, tuple[Chunk, Document, float | None, bool]] = {}

        for rank, (chunk, doc, sim) in enumerate(vec_rows):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            meta[chunk.id] = (chunk, doc, float(sim), False)
        for rank, (chunk, doc, _rank) in enumerate(kw_rows):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            prev = meta.get(chunk.id)
            meta[chunk.id] = (chunk, doc, prev[2] if prev else None, True)
        for rank, (chunk, doc) in enumerate(exact_rows):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 2.0 / (RRF_K + rank)  # exact id match: strong signal
            prev = meta.get(chunk.id)
            meta[chunk.id] = (chunk, doc, prev[2] if prev else None, True)

        if not fused:
            return []
        max_score = max(fused.values())
        results: list[RetrievedChunk] = []
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            chunk, doc, sim, kw = meta[cid]
            # Similarity floor only applies to chunks found purely by vector similarity
            if not kw and sim is not None and sim < self.settings.min_similarity:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    title=doc.title,
                    source_url=doc.source_url,
                    regulator=doc.regulator,
                    page=chunk.page,
                    section=chunk.section,
                    content=chunk.content,
                    score=round(score / max_score, 4),
                    vector_score=sim,
                    keyword_hit=kw,
                )
            )
            if len(results) >= k:
                break

        logger.info(
            "retrieve.done",
            **log_fields(k=k, vector=len(vec_rows), keyword=len(kw_rows), exact=len(exact_rows), returned=len(results)),
        )
        return results

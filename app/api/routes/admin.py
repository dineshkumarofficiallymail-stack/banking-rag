"""Operational visibility for admins: corpus size, grounded rate, cost, recent queries."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select

from app.api.deps import DB, require_role
from app.db.models import Chunk, Document, QueryLog
from app.schemas.documents import QueryLogOut, StatsOut

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_role("admin"))])


@router.get("/stats", response_model=StatsOut)
async def stats(db: DB) -> StatsOut:
    docs = await db.scalar(select(func.count(Document.id))) or 0
    chunks = await db.scalar(select(func.count(Chunk.id))) or 0
    row = (
        await db.execute(
            select(
                func.count(QueryLog.id),
                func.avg(case((QueryLog.grounded.is_(True), 1.0), else_=0.0)),
                func.sum(case((QueryLog.flagged_injection.is_(True), 1), else_=0)),
                func.avg(QueryLog.latency_ms),
                func.sum(QueryLog.input_tokens),
                func.sum(QueryLog.output_tokens),
            )
        )
    ).one()
    queries, grounded_rate, flags, avg_latency, in_tok, out_tok = row
    return StatsOut(
        documents=docs,
        chunks=chunks,
        queries=queries or 0,
        grounded_rate=round(float(grounded_rate or 0.0), 3),
        injection_flags=int(flags or 0),
        avg_latency_ms=round(float(avg_latency or 0.0), 1),
        total_input_tokens=int(in_tok or 0),
        total_output_tokens=int(out_tok or 0),
    )


@router.get("/queries", response_model=list[QueryLogOut])
async def recent_queries(db: DB, limit: int = Query(50, ge=1, le=500), flagged_only: bool = False) -> list[QueryLog]:
    stmt = select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
    if flagged_only:
        stmt = stmt.where(QueryLog.flagged_injection.is_(True))
    return list((await db.execute(stmt)).scalars().all())

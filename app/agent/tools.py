"""Tools the agent can call. All retrieval goes through AgentContext so every source the model saw
is numbered consistently and can be cited/verified afterwards. Date arithmetic is a deterministic
tool because LLMs are unreliable at it (and ops staff care about exact TAT deadlines)."""

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

from langchain_core.tools import tool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.rag.guardrails import sanitize_chunks
from app.rag.retriever import HybridRetriever, RetrievedChunk


@dataclass
class AgentContext:
    db: AsyncSession
    regulator: str | None = None
    top_k: int | None = None
    sources: list[RetrievedChunk] = field(default_factory=list)
    _index: dict[uuid.UUID, int] = field(default_factory=dict)

    def add(self, chunks: list[RetrievedChunk]) -> list[tuple[int, RetrievedChunk]]:
        numbered = []
        for c in chunks:
            if c.chunk_id not in self._index:
                self.sources.append(c)
                self._index[c.chunk_id] = len(self.sources)
            numbered.append((self._index[c.chunk_id], c))
        return numbered


def _format(numbered: list[tuple[int, RetrievedChunk]]) -> str:
    blocks = []
    for i, c in numbered:
        loc = ", ".join(x for x in (f"p.{c.page}" if c.page else "", c.section or "") if x)
        blocks.append(f"<source id={i}>\n[{i}] {c.title} ({c.regulator}{', ' + loc if loc else ''})\n{c.content}\n</source>")
    return "\n\n".join(blocks)


def build_tools(ctx: AgentContext) -> list:
    @tool
    async def search_regulations(query: str, regulator: str | None = None) -> str:
        """Search the ingested RBI / NPCI / NACH documents (semantic + keyword). Use this for any factual
        question about rules, timelines, limits or procedures. Returns numbered sources to cite as [n].
        `regulator` may be RBI, NPCI or NACH to narrow the search."""
        chunks = await HybridRetriever(ctx.db).retrieve(query, k=ctx.top_k, regulator=regulator or ctx.regulator)
        if not chunks:
            return "No relevant passages found."
        return _format(ctx.add(sanitize_chunks(chunks)))

    @tool
    async def lookup_circular(circular_number: str) -> str:
        """Find passages that mention an exact circular / notification number, e.g. 'RBI/2023-24/117'
        or 'DPSS.CO.PD.No.629/02.01.014/2019-20'. Use when the user quotes a reference number."""
        stmt = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.content.ilike(f"%{circular_number.strip()}%"))
            .limit(5)
        )
        rows = (await ctx.db.execute(stmt)).all()
        if not rows:
            return f"No passages mention {circular_number}."
        chunks = [
            RetrievedChunk(
                chunk_id=ch.id,
                document_id=doc.id,
                title=doc.title,
                source_url=doc.source_url,
                regulator=doc.regulator,
                page=ch.page,
                section=ch.section,
                content=ch.content,
                score=1.0,
                vector_score=None,
                keyword_hit=True,
            )
            for ch, doc in rows
        ]
        return _format(ctx.add(sanitize_chunks(chunks)))

    @tool
    async def list_documents() -> str:
        """List the titles and regulators of every ingested document, so you know what can be answered."""
        stmt = (
            select(Document.title, Document.regulator, func.count(Chunk.id))
            .join(Chunk)
            .group_by(Document.id)
            .order_by(Document.regulator)
        )
        rows = (await ctx.db.execute(stmt)).all()
        return "\n".join(f"- [{reg}] {title} ({n} chunks)" for title, reg, n in rows) or "No documents ingested."

    @tool
    def working_day_deadline(start_date: str, working_days: int, holidays: list[str] | None = None) -> str:
        """Deterministically compute the calendar date N working days (Mon-Fri, minus the given holidays as
        YYYY-MM-DD) after start_date (YYYY-MM-DD). Always use this for T+N / TAT deadline arithmetic instead
        of calculating dates yourself."""
        try:
            d = date.fromisoformat(start_date)
            skip = {date.fromisoformat(h) for h in (holidays or [])}
        except ValueError as exc:
            return f"Invalid date: {exc}"
        if not 0 <= working_days <= 60:
            return "working_days must be between 0 and 60"
        remaining = working_days
        while remaining > 0:
            d += timedelta(days=1)
            if d.weekday() < 5 and d not in skip:
                remaining -= 1
        return f"{working_days} working day(s) after {start_date} is {d.isoformat()} ({d.strftime('%A')})."

    return [search_regulations, lookup_circular, list_documents, working_day_deadline]

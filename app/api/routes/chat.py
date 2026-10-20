"""/chat (plain RAG, cached) and /chat/agent (LangGraph tool-calling). Both: JWT, rate limit,
injection guard, groundedness guard, audit log in background."""

import time

from fastapi import APIRouter, BackgroundTasks, Request

from app.agent.graph import run_agent
from app.api.deps import DB, CurrentUser
from app.core.config import get_settings
from app.core.logging import request_id_ctx
from app.rag.cache import ResponseCache
from app.rag.generator import answer_question
from app.rag.guardrails import detect_injection
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.query_log import record_query
from app.services.rate_limit import RateLimited

router = APIRouter(prefix="/chat", tags=["chat"])

INJECTION_REPLY = (
    "This request could not be processed because it appears to contain instructions directed at the assistant "
    "rather than a question about banking regulations. Please rephrase your question."
)


async def _guard(payload: ChatRequest, user, background: BackgroundTasks, start: float) -> ChatResponse | None:
    inj = await detect_injection(payload.question)
    if not inj.flagged:
        return None
    latency = int((time.perf_counter() - start) * 1000)
    background.add_task(
        record_query,
        user_id=user.id,
        request_id=request_id_ctx.get(),
        question=payload.question,
        answer=None,
        chunk_ids=[],
        tool_calls=[],
        grounded=None,
        flagged_injection=True,
        input_tokens=0,
        output_tokens=0,
        latency_ms=latency,
    )
    return ChatResponse(
        answer=INJECTION_REPLY,
        citations=[],
        grounded=True,
        confidence=0.0,
        flagged_injection=True,
        latency_ms=latency,
        request_id=request_id_ctx.get(),
    )


@router.post("", response_model=ChatResponse, summary="Grounded Q&A with citations (RAG)")
async def chat(
    payload: ChatRequest, request: Request, background: BackgroundTasks, db: DB, user: CurrentUser, _: RateLimited
) -> ChatResponse:
    start = time.perf_counter()
    if blocked := await _guard(payload, user, background, start):
        return blocked

    s = get_settings()
    cache = ResponseCache(request.app.state.redis, ttl_seconds=s.cache_ttl_seconds)
    key = cache.make_key(payload.question, payload.regulator, payload.top_k or s.top_k)
    if cached := await cache.get(key):
        cached.update(request_id=request_id_ctx.get(), latency_ms=int((time.perf_counter() - start) * 1000))
        return ChatResponse(**cached)

    result = await answer_question(db, payload.question, top_k=payload.top_k, regulator=payload.regulator)
    resp = ChatResponse(
        answer=result.answer,
        citations=result.citations,
        grounded=result.grounded,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        request_id=request_id_ctx.get(),
    )
    if result.grounded and result.citations:
        await cache.set(key, resp.model_dump(mode="json"))
    background.add_task(
        record_query,
        user_id=user.id,
        request_id=resp.request_id,
        question=payload.question,
        answer=result.answer,
        chunk_ids=[str(c.chunk_id) for c in result.retrieved],
        tool_calls=[],
        grounded=result.grounded,
        flagged_injection=False,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
    return resp


@router.post(
    "/agent", response_model=ChatResponse, summary="Tool-calling agent (multi-step search, circular lookup, deadline math)"
)
async def chat_agent(
    payload: ChatRequest, background: BackgroundTasks, db: DB, user: CurrentUser, _: RateLimited
) -> ChatResponse:
    start = time.perf_counter()
    if blocked := await _guard(payload, user, background, start):
        return blocked

    result = await run_agent(db, payload.question, regulator=payload.regulator, top_k=payload.top_k)
    resp = ChatResponse(
        answer=result.answer,
        citations=result.citations,
        grounded=result.grounded,
        confidence=result.confidence,
        tool_calls=result.tool_calls,
        latency_ms=result.latency_ms,
        request_id=request_id_ctx.get(),
    )
    background.add_task(
        record_query,
        user_id=user.id,
        request_id=resp.request_id,
        question=payload.question,
        answer=result.answer,
        chunk_ids=[str(c.chunk_id) for c in result.retrieved],
        tool_calls=result.tool_calls,
        grounded=result.grounded,
        flagged_injection=False,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
    return resp

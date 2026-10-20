"""The RAG pipeline: retrieve -> sanitize -> prompt -> generate -> cite -> verify groundedness.
`answer_question` is used directly by /chat (fast path) and wrapped as a tool by the agent."""

import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger, log_fields
from app.rag.citations import build_citations, renumber_answer
from app.rag.guardrails import check_groundedness, sanitize_chunks
from app.rag.llm import get_chat_model, usage_of
from app.rag.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt, format_sources
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.schemas.chat import Citation

logger = get_logger(__name__)


class TransientLLMError(RuntimeError):
    pass


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(TransientLLMError),
    reraise=True,
)
async def _generate(question: str, sources_block: str):
    try:
        return await get_chat_model().ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=build_user_prompt(question, sources_block))]
        )
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if any(t in name for t in ("RateLimit", "Overloaded", "Timeout", "APIConnection", "InternalServer")):
            raise TransientLLMError(name) from exc
        raise


async def answer_question(
    db: AsyncSession,
    question: str,
    *,
    top_k: int | None = None,
    regulator: str | None = None,
) -> RAGResult:
    s = get_settings()
    start = time.perf_counter()

    retrieved = await HybridRetriever(db).retrieve(question, k=top_k, regulator=regulator)
    if not retrieved:
        return RAGResult(
            answer=f"{NOT_FOUND_ANSWER} No relevant passages were retrieved"
            + (f" from {regulator} documents." if regulator else " from the ingested documents."),
            citations=[],
            grounded=True,
            confidence=1.0,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    retrieved = sanitize_chunks(retrieved)
    sources_block = format_sources(retrieved, s.max_context_chars)

    msg = await _generate(question, sources_block)
    raw_answer = str(msg.content).strip()
    inp, out = usage_of(msg)

    ground = await check_groundedness(raw_answer, sources_block)
    inp += ground.input_tokens
    out += ground.output_tokens

    if not ground.grounded:
        logger.warning("rag.ungrounded", **log_fields(score=ground.score, unsupported=ground.unsupported_claims))
        answer = (
            f"{NOT_FOUND_ANSWER} A draft answer was generated but failed verification against the sources "
            f"(faithfulness {ground.score:.2f}); please review the cited passages directly."
        )
        citations = build_citations(raw_answer, retrieved, include_uncited=True)
    else:
        citations = build_citations(raw_answer, retrieved)
        answer = renumber_answer(raw_answer, retrieved)

    # Confidence = faithfulness blended with retrieval strength of the top hit
    top_sim = retrieved[0].vector_score if retrieved[0].vector_score is not None else retrieved[0].score
    confidence = round(min(1.0, 0.7 * ground.score + 0.3 * max(0.0, float(top_sim))), 3)

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "rag.answer",
        **log_fields(
            latency_ms=latency_ms, grounded=ground.grounded, citations=len(citations), input_tokens=inp, output_tokens=out
        ),
    )
    return RAGResult(
        answer=answer,
        citations=citations,
        grounded=ground.grounded,
        confidence=confidence,
        retrieved=retrieved,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
    )

"""LangGraph tool-calling agent: agent -> (tools -> agent)* -> END, with a hard step budget.
After the loop we run the same groundedness check as the plain RAG path, over every source the agent saw."""

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState
from app.agent.tools import AgentContext, build_tools
from app.core.config import get_settings
from app.core.logging import get_logger, log_fields
from app.rag.citations import build_citations, renumber_answer
from app.rag.guardrails import check_groundedness
from app.rag.llm import get_chat_model, usage_of
from app.rag.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, format_sources
from app.rag.retriever import RetrievedChunk
from app.schemas.chat import Citation

logger = get_logger(__name__)

AGENT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Tool use:
- Always call `search_regulations` before answering a factual question; refine the query and search again if the first results are weak.
- Use `lookup_circular` when the user quotes a circular / notification number.
- Use `working_day_deadline` for any T+N or turnaround-time date calculation.
- Stop calling tools once you have enough sources; then answer with [n] citations referring to the source ids returned by the tools."""
)

FORCE_FINAL = "Tool budget exhausted. Answer now from the sources already retrieved, with [n] citations, or state that the information could not be found."


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float
    tool_calls: list[str] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


def content_text(msg: BaseMessage) -> str:
    """Anthropic returns a list of content blocks; normalise to plain text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text").strip()


def build_graph(llm_plain, llm_with_tools, tools: list, max_steps: int):
    async def agent_node(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            resp = await llm_plain.ainvoke(state["messages"] + [HumanMessage(content=FORCE_FINAL)])
            return {"messages": [resp], "steps": steps}
        resp = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [resp], "steps": steps + 1}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


async def run_agent(db: AsyncSession, question: str, *, regulator: str | None = None, top_k: int | None = None) -> AgentResult:
    s = get_settings()
    start = time.perf_counter()

    ctx = AgentContext(db=db, regulator=regulator, top_k=top_k)
    tools = build_tools(ctx)
    llm = get_chat_model()
    graph = build_graph(llm, llm.bind_tools(tools), tools, s.agent_max_steps)

    state = await graph.ainvoke(
        {"messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=question)], "steps": 0},
        config={"recursion_limit": 2 * s.agent_max_steps + 4},
    )

    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
    tool_calls = [tc["name"] for m in ai_msgs for tc in (m.tool_calls or [])]
    inp = sum(usage_of(m)[0] for m in ai_msgs)
    out = sum(usage_of(m)[1] for m in ai_msgs)
    raw_answer = content_text(state["messages"][-1]).strip() or NOT_FOUND_ANSWER

    sources_block = format_sources(ctx.sources, s.max_context_chars)
    ground = await check_groundedness(raw_answer, sources_block) if ctx.sources else None
    inp += ground.input_tokens if ground else 0
    out += ground.output_tokens if ground else 0

    if ground is None:
        answer, citations, grounded, confidence = (
            raw_answer if raw_answer.startswith(NOT_FOUND_ANSWER) else f"{NOT_FOUND_ANSWER} No sources were retrieved.",
            [],
            True,
            0.0,
        )
    elif not ground.grounded:
        logger.warning("agent.ungrounded", **log_fields(score=ground.score, unsupported=ground.unsupported_claims))
        answer = (
            f"{NOT_FOUND_ANSWER} A draft answer failed verification against the sources "
            f"(faithfulness {ground.score:.2f}); please review the cited passages directly."
        )
        citations, grounded, confidence = (
            build_citations(raw_answer, ctx.sources, include_uncited=True),
            False,
            round(0.5 * ground.score, 3),
        )
    else:
        citations = build_citations(raw_answer, ctx.sources)
        answer = renumber_answer(raw_answer, ctx.sources)
        top = ctx.sources[0]
        top_sim = top.vector_score if top.vector_score is not None else top.score
        grounded, confidence = True, round(min(1.0, 0.7 * ground.score + 0.3 * max(0.0, float(top_sim))), 3)

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "agent.done",
        **log_fields(
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            grounded=grounded,
            sources=len(ctx.sources),
            input_tokens=inp,
            output_tokens=out,
        ),
    )
    return AgentResult(
        answer=answer,
        citations=citations,
        grounded=grounded,
        confidence=confidence,
        tool_calls=tool_calls,
        retrieved=ctx.sources,
        input_tokens=inp,
        output_tokens=out,
        latency_ms=latency_ms,
    )

"""Chat-model factory. Real: Claude via langchain-anthropic. Offline: deterministic FakeChatModel so
tests and CI exercise the full pipeline without network or API keys."""

import json
import re
from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.config import get_settings


class FakeChatModel:
    """Mimics the subset of the LangChain chat interface we use (ainvoke, bind_tools)."""

    def __init__(self) -> None:
        self._tools: list = []

    def bind_tools(self, tools: list) -> "FakeChatModel":
        clone = FakeChatModel()
        clone._tools = list(tools)
        return clone

    async def ainvoke(self, messages: list[BaseMessage], **_: object) -> AIMessage:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        usage = {"input_tokens": len(str(system) + str(human)) // 4, "output_tokens": 40, "total_tokens": 0}

        if "faithfulness judge mode" in system:
            return AIMessage(content=json.dumps({"grounded": True, "score": 1.0, "unsupported_claims": []}), usage_metadata=usage)
        if "injection judge mode" in system:
            flagged = bool(re.search(r"ignore (all |previous |prior )?instructions|system prompt", str(human), re.I))
            return AIMessage(
                content=json.dumps({"injection": flagged, "reason": "pattern" if flagged else "none"}), usage_metadata=usage
            )

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if self._tools and not tool_msgs:
            # Agent mode, first turn: always search first (mirrors the real system prompt's instruction).
            question = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_regulations", "args": {"query": str(question)}, "id": "call_1", "type": "tool_call"}
                ],
                usage_metadata=usage,
            )
        haystack = "\n".join(str(m.content) for m in tool_msgs) if tool_msgs else str(human)
        m = re.search(r"<source id=(\d+)>\n.*?\n(.*?)\n</source>", haystack, re.S)
        if not m:
            return AIMessage(content="I could not find this in the available documents.", usage_metadata=usage)
        first_sentence = re.split(r"(?<=[.;])\s", m.group(2).strip())[0][:300]
        return AIMessage(content=f"According to the sources, {first_sentence} [{m.group(1)}]", usage_metadata=usage)


@lru_cache
def get_chat_model():
    s = get_settings()
    if s.offline_mode:
        return FakeChatModel()
    if not s.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (or set OFFLINE_MODE=true)")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=s.llm_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        api_key=s.anthropic_api_key,
        timeout=60,
        max_retries=2,
    )


def usage_of(msg: AIMessage) -> tuple[int, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    return int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))

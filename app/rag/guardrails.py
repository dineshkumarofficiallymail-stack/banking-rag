"""Input and output guardrails.
- detect_injection: fast regex pre-filter + optional LLM classifier for ambiguous inputs.
- sanitize_source: strips instruction-like lines from retrieved text (indirect injection via documents).
- check_groundedness: LLM-as-judge faithfulness check; falls back to lexical overlap when offline."""

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger, log_fields
from app.rag.llm import get_chat_model, usage_of
from app.rag.prompts import FAITHFULNESS_JUDGE_PROMPT, INJECTION_JUDGE_PROMPT, NOT_FOUND_ANSWER, build_judge_prompt
from app.rag.retriever import RetrievedChunk

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore (all|any|the|your)? ?(previous|prior|above|earlier)? ?(instructions|rules|prompts?)",
    r"disregard (the|your|all) (system|previous|above)",
    r"(reveal|print|show|output|repeat) (the|your) (system|hidden|initial) (prompt|instructions)",
    r"you are now (a|an|the) ",
    r"\bact as (a|an) (unfiltered|unrestricted|jailbroken|DAN)\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"forget (everything|all|your) (you|rules|instructions)",
    r"<\/?(system|instruction|prompt)>",
    r"\bBEGIN SYSTEM\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
_INSTRUCTION_LINE = re.compile(r"^\s*(ignore|disregard|you must|assistant:|system:|###\s*instruction)", re.I)


@dataclass
class InjectionResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class GroundednessResult:
    grounded: bool
    score: float
    unsupported_claims: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def quick_injection_scan(text: str) -> InjectionResult:
    reasons = []
    if _INJECTION_RE.search(text):
        reasons.append("instruction-override pattern")
    if _BASE64_BLOB.search(text):
        reasons.append("encoded payload")
    if text.count("\n") > 40:
        reasons.append("excessive structure")
    return InjectionResult(flagged=bool(reasons), reasons=reasons)


async def detect_injection(text: str, *, use_llm: bool = True) -> InjectionResult:
    quick = quick_injection_scan(text)
    if quick.flagged or not use_llm:
        return quick
    # Only ask the model for borderline inputs (keeps cost + latency low)
    if not re.search(r"\b(instruction|prompt|rule|role|pretend|system)\b", text, re.I):
        return quick
    try:
        msg = await get_chat_model().ainvoke(
            [SystemMessage(content=INJECTION_JUDGE_PROMPT), HumanMessage(content=f"USER TEXT:\n{text}")]
        )
        data = _parse_json(str(msg.content))
        if data.get("injection"):
            return InjectionResult(flagged=True, reasons=[f"llm: {data.get('reason', 'classified')}"])
    except Exception as exc:  # noqa: BLE001 — guardrail must never take the API down
        logger.warning("guardrail.injection_judge_failed", **log_fields(error=type(exc).__name__))
    return quick


def sanitize_source(content: str) -> str:
    kept = [ln for ln in content.splitlines() if not _INSTRUCTION_LINE.match(ln) and not _INJECTION_RE.search(ln)]
    return "\n".join(kept).replace("<source", "&lt;source").replace("</source", "&lt;/source")


def sanitize_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    for c in chunks:
        c.content = sanitize_source(c.content)
    return chunks


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def lexical_overlap(answer: str, sources_text: str) -> float:
    """Cheap fallback: share of answer content words that appear in the sources."""
    words = [w for w in re.findall(r"[a-z0-9]{4,}", answer.lower())]
    if not words:
        return 1.0
    src = set(re.findall(r"[a-z0-9]{4,}", sources_text.lower()))
    return sum(1 for w in words if w in src) / len(words)


async def check_groundedness(answer: str, sources_block: str, *, threshold: float = 0.75) -> GroundednessResult:
    if answer.strip().startswith(NOT_FOUND_ANSWER):
        return GroundednessResult(grounded=True, score=1.0)
    try:
        msg = await get_chat_model().ainvoke(
            [SystemMessage(content=FAITHFULNESS_JUDGE_PROMPT), HumanMessage(content=build_judge_prompt(answer, sources_block))]
        )
        data = _parse_json(str(msg.content))
        score = float(data.get("score", 0.0))
        inp, out = usage_of(msg)
        return GroundednessResult(
            grounded=bool(data.get("grounded", False)) and score >= threshold,
            score=round(score, 3),
            unsupported_claims=[str(x) for x in data.get("unsupported_claims", [])][:5],
            input_tokens=inp,
            output_tokens=out,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("guardrail.judge_failed_fallback_lexical", **log_fields(error=type(exc).__name__))
        score = lexical_overlap(answer, sources_block)
        return GroundednessResult(grounded=score >= threshold, score=round(score, 3))

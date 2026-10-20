"""Map [n] markers in the answer back to retrieved chunks and build API Citation objects."""

import re

from app.rag.retriever import RetrievedChunk
from app.schemas.chat import Citation

_MARKER = re.compile(r"\[(\d{1,2})\]")
SNIPPET_CHARS = 280


def cited_indices(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _MARKER.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen


def make_snippet(content: str) -> str:
    text = " ".join(content.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rsplit(" ", 1)[0] + "…"


def build_citations(answer: str, chunks: list[RetrievedChunk], *, include_uncited: bool = False) -> list[Citation]:
    idx = cited_indices(answer, len(chunks))
    if include_uncited or not idx:
        idx = list(range(1, len(chunks) + 1))
    return [
        Citation(
            chunk_id=c.chunk_id,
            document_title=c.title,
            source_url=c.source_url,
            page=c.page,
            section=c.section,
            score=c.score,
            snippet=make_snippet(c.content),
        )
        for c in (chunks[i - 1] for i in idx)
    ]


def renumber_answer(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Make citation numbers contiguous (1..n) in the order they appear, matching build_citations order."""
    idx = cited_indices(answer, len(chunks))
    mapping = {old: new for new, old in enumerate(idx, start=1)}
    return _MARKER.sub(lambda m: f"[{mapping.get(int(m.group(1)), m.group(1))}]", answer)

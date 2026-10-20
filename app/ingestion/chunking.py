"""Section-aware, page-preserving chunker.
Why not a fixed window: circulars are organised as numbered clauses ("3.2 Mandate validity").
Keeping a clause together and tagging each chunk with its section heading improves both
retrieval precision and the quality of citations shown to the user."""

import re
from dataclasses import dataclass

from app.ingestion.loaders import LoadedDocument

NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|[A-Z]\.|[IVXLC]+\.|\([a-z0-9]{1,3}\)|[a-z]\))\s+[A-Z]")
SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class TextChunk:
    index: int
    page: int | None
    section: str | None
    content: str

    @property
    def token_count(self) -> int:
        return max(1, len(self.content) // 4)  # cheap estimate; good enough for budgeting


def is_heading(line: str) -> bool:
    line = line.strip()
    if not 3 <= len(line) <= 110 or line.endswith((".", ",", ";")):
        return False
    if NUMBERED_HEADING.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out, buf = [], ""
    for sent in SENTENCE_SPLIT.split(paragraph):
        if len(buf) + len(sent) + 1 > max_chars and buf:
            out.append(buf.strip())
            buf = ""
        buf += sent + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_document(doc: LoadedDocument, *, max_chars: int = 1200, overlap_chars: int = 150) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    buf: list[str] = []
    buf_len = 0
    buf_page: int | None = None
    section: str | None = None
    carry = ""

    def flush() -> None:
        nonlocal buf, buf_len, buf_page, carry
        if not buf:
            return
        body = "\n".join(buf).strip()
        content = (carry + "\n" + body).strip() if carry else body
        chunks.append(TextChunk(index=len(chunks), page=buf_page, section=section, content=content))
        carry = body[-overlap_chars:] if overlap_chars and len(body) > overlap_chars else ""
        buf, buf_len, buf_page = [], 0, None

    for page in doc.pages:
        for para in re.split(r"\n\s*\n", page.text):
            para = para.strip()
            if not para:
                continue
            if is_heading(para.splitlines()[0]) and len(para.splitlines()) == 1:
                flush()
                section = para[:512]
                continue
            for piece in _split_long(para, max_chars):
                if buf_len + len(piece) > max_chars and buf:
                    flush()
                if buf_page is None:
                    buf_page = page.number
                buf.append(piece)
                buf_len += len(piece) + 1
    flush()
    return chunks

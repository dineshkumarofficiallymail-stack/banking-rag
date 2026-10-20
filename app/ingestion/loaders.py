"""Turn a file (PDF / HTML / TXT / MD) into a LoadedDocument: per-page cleaned text + sha256 checksum.
Regulatory PDFs are messy (hyphenated line breaks, repeated headers), so we normalise aggressively."""

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass
class Page:
    number: int
    text: str


@dataclass
class LoadedDocument:
    title: str
    regulator: str
    doc_type: str
    checksum: str
    pages: list[Page] = field(default_factory=list)
    source_url: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_file(
    path: str | Path,
    *,
    title: str | None = None,
    source_url: str | None = None,
    regulator: str = "RBI",
    doc_type: str = "circular",
) -> LoadedDocument:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    raw = p.read_bytes()
    doc = LoadedDocument(
        title=title or p.stem.replace("_", " ").replace("-", " ").strip(),
        regulator=regulator.upper(),
        doc_type=doc_type,
        checksum=_checksum(raw),
        source_url=source_url,
    )

    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                doc.pages.append(Page(number=i, text=text))
    elif p.suffix.lower() in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(raw.decode("utf-8", errors="ignore"))
        doc.pages.append(Page(number=1, text=clean_text("".join(parser.parts))))
    else:
        doc.pages.append(Page(number=1, text=clean_text(raw.decode("utf-8", errors="ignore"))))

    if not any(pg.text for pg in doc.pages):
        raise ValueError(f"No extractable text in {p.name} (scanned PDF? run OCR first)")
    return doc

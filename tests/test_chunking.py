from app.ingestion.chunking import chunk_document, is_heading
from app.ingestion.loaders import LoadedDocument, Page, clean_text


def _doc(*pages: str) -> LoadedDocument:
    return LoadedDocument(
        title="t", regulator="RBI", doc_type="c", checksum="x", pages=[Page(i + 1, p) for i, p in enumerate(pages)]
    )


def test_heading_detection():
    assert is_heading("2.1 Mandate Validity")
    assert is_heading("3. Presentation and Returns")
    assert is_heading("ANNEXURE A")
    assert not is_heading("The mandate shall be valid for thirty days.")
    assert not is_heading("2.1 this ends with a period.")


def test_sections_and_pages_preserved():
    chunks = chunk_document(_doc("1. Scope\n\nFirst para.", "2.1 Validity\n\nSecond para on page two."))
    assert [c.section for c in chunks] == ["1. Scope", "2.1 Validity"]
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[1].index == 1


def test_long_paragraph_split_with_overlap():
    para = "Sponsor banks must respond within one working day. " * 60
    chunks = chunk_document(_doc(para), max_chars=500, overlap_chars=100)
    assert len(chunks) > 3
    assert all(len(c.content) <= 500 + 100 + 1 for c in chunks)
    # overlap: tail of chunk n appears at head of chunk n+1
    assert chunks[0].content[-50:] in chunks[1].content


def test_clean_text_rejoins_hyphenation():
    assert clean_text("compen-\nsation  payable") == "compensation payable"

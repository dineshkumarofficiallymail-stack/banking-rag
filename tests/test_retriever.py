from app.db.session import SessionLocal
from app.rag.retriever import HybridRetriever
from tests.conftest import integration

pytestmark = integration


async def test_retrieves_relevant_chunk(seeded_corpus):
    async with SessionLocal() as db:
        res = await HybridRetriever(db).retrieve("destination bank accept or reject mandate registration working days", k=3)
    assert res and "nach" in res[0].title.lower()
    assert "three working days" in res[0].content


async def test_regulator_filter(seeded_corpus):
    async with SessionLocal() as db:
        res = await HybridRetriever(db).retrieve("mandate registration", k=5, regulator="RBI")
    assert all(c.regulator == "RBI" for c in res)


async def test_exact_circular_number_boosted(seeded_corpus):
    async with SessionLocal() as db:
        res = await HybridRetriever(db).retrieve("Which circular is SAMPLE/2024-25/001?", k=3)
    assert res and "SAMPLE/2024-25/001" in res[0].content and res[0].keyword_hit


async def test_duplicate_ingest_is_skipped(seeded_corpus):
    from app.ingestion.pipeline import ingest_path
    from tests.conftest import SAMPLE_DOCS

    async with SessionLocal() as db:
        res = await ingest_path(db, SAMPLE_DOCS / "sample_nach_mandates.txt", regulator="NACH")
    assert res.skipped and res.reason == "duplicate checksum"

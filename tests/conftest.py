"""Test configuration.
- Unit tests run anywhere (OFFLINE_MODE: fake embeddings + fake LLM, no network).
- Integration tests (marked `integration`) need a pgvector Postgres: set TEST_DATABASE_URL, e.g.
    TEST_DATABASE_URL=postgresql+asyncpg://rag:pass@localhost:5432/banking_rag_test
  GitHub Actions provides one via a service container. Without it, they are skipped."""

import os
from pathlib import Path

import pytest

# Must be set before any `app.*` import because Settings is cached at import time.
os.environ.setdefault("OFFLINE_MODE", "true")
os.environ.setdefault("JWT_SECRET", "test-secret-test-secret-test-secret-test-secret")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("ALLOW_REGISTRATION", "true")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
TEST_DB = os.getenv("TEST_DATABASE_URL")
os.environ["DATABASE_URL"] = TEST_DB or "postgresql+asyncpg://unused:unused@localhost:5432/unused"
os.environ.pop("REDIS_URL", None)

integration = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")
SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "evals" / "sample_docs"


@pytest.fixture(scope="session")
async def schema():
    """Create tables once per session, truncate between tests via `clean_db`."""
    if not TEST_DB:
        pytest.skip("integration")
    from scripts.init_db import init_db

    await init_db()
    yield


@pytest.fixture
async def clean_db(schema):
    from sqlalchemy import text

    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        await db.execute(text("TRUNCATE query_logs, chunks, documents, users CASCADE"))
        await db.commit()
    yield


@pytest.fixture
async def client(clean_db):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    app.state.redis = None  # lifespan isn't run under ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _register_and_login(client, email: str, password: str = "Passw0rd!xyz") -> str:
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    r = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture
async def user_token(client) -> str:
    return await _register_and_login(client, "analyst@example.com")


@pytest.fixture
async def admin_token(client) -> str:
    from sqlalchemy import update

    from app.db.models import User
    from app.db.session import SessionLocal

    await _register_and_login(client, "admin@example.com")
    async with SessionLocal() as db:
        await db.execute(update(User).where(User.email == "admin@example.com").values(role="admin"))
        await db.commit()
    return await _register_and_login_existing(client, "admin@example.com")


async def _register_and_login_existing(client, email: str, password: str = "Passw0rd!xyz") -> str:
    r = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    return r.json()["access_token"]


@pytest.fixture
async def seeded_corpus(clean_db):
    from app.db.session import SessionLocal
    from app.ingestion.pipeline import ingest_path

    regs = {
        "sample_nach_mandates.txt": "NACH",
        "sample_rbi_failed_transactions.txt": "RBI",
        "sample_kyc_master_direction.txt": "RBI",
    }
    async with SessionLocal() as db:
        for name, reg in regs.items():
            res = await ingest_path(db, SAMPLE_DOCS / name, regulator=reg, title=name.replace("_", " ").removesuffix(".txt"))
            assert not res.skipped, res.reason
    yield


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

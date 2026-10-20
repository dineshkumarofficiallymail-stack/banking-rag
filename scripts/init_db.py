"""Create the pgvector extension, all tables, the full-text index, and (optionally) seed an admin.
Run: python -m scripts.init_db"""

import asyncio

from sqlalchemy import select, text

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.security import hash_password
from app.db.models import Base, User
from app.db.session import SessionLocal, engine

logger = get_logger(__name__)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # GIN index for the keyword half of hybrid search
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_chunks_content_fts ON chunks USING gin (to_tsvector('english', content))")
        )
    logger.info("db.schema_ready")

    s = get_settings()
    if s.admin_email and s.admin_password:
        async with SessionLocal() as db:
            exists = await db.scalar(select(User).where(User.email == s.admin_email))
            if not exists:
                db.add(User(email=s.admin_email, hashed_password=hash_password(s.admin_password), role="admin"))
                await db.commit()
                logger.info("db.admin_seeded")


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    asyncio.run(init_db())

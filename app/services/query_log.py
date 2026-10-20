"""Persist an audit record for every answer. Runs as a FastAPI BackgroundTask with its own session,
so logging never adds latency to the response and a logging failure never fails the request."""

import uuid

from app.core.logging import get_logger, log_fields
from app.db.models import QueryLog
from app.db.session import SessionLocal

logger = get_logger(__name__)


async def record_query(
    *,
    user_id: uuid.UUID | None,
    request_id: str,
    question: str,
    answer: str | None,
    chunk_ids: list[str],
    tool_calls: list[str],
    grounded: bool | None,
    flagged_injection: bool,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
) -> None:
    try:
        async with SessionLocal() as db:
            db.add(
                QueryLog(
                    user_id=user_id,
                    request_id=request_id,
                    question=question[:4000],
                    answer=answer,
                    retrieved_chunk_ids=chunk_ids,
                    tool_calls=tool_calls,
                    grounded=grounded,
                    flagged_injection=flagged_injection,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )
            )
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("querylog.write_failed", **log_fields(error=type(exc).__name__, request_id=request_id))

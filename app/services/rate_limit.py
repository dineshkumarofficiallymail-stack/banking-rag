"""Per-user fixed-window rate limiter on Redis (INCR + EXPIRE). Fails open if Redis is unavailable:
losing rate limiting is preferable to taking the API down."""

import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status

from app.api.deps import CurrentUser
from app.core.config import get_settings
from app.core.logging import get_logger, log_fields

logger = get_logger(__name__)
WINDOW_SECONDS = 60


async def check_rate_limit(redis, key: str, limit: int) -> tuple[bool, int, int]:
    """Returns (allowed, remaining, reset_epoch)."""
    now = int(time.time())
    window_key = f"ratelimit:{key}:{now // WINDOW_SECONDS}"
    reset = (now // WINDOW_SECONDS + 1) * WINDOW_SECONDS
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(window_key)
            pipe.expire(window_key, WINDOW_SECONDS)
            count, _ = await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ratelimit.redis_unavailable_fail_open", **log_fields(error=type(exc).__name__))
        return True, limit, reset
    return count <= limit, max(0, limit - int(count)), reset


async def rate_limit(request: Request, response: Response, user: CurrentUser) -> None:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return
    limit = get_settings().rate_limit_per_minute
    allowed, remaining, reset = await check_rate_limit(redis, str(user.id), limit)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(max(1, reset - int(time.time())))},
        )


RateLimited = Annotated[None, Depends(rate_limit)]

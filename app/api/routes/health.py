from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.api.deps import DB

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response, db: DB) -> dict:
    """Readiness: dependencies reachable. Returns 503 so load balancers stop routing."""
    checks: dict[str, str] = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {type(exc).__name__}"
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        checks["redis"] = "disabled"
    else:
        try:
            await redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["redis"] = f"error: {type(exc).__name__}"
    healthy = all(v in ("ok", "disabled") for v in checks.values())
    response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if healthy else "degraded", "checks": checks}

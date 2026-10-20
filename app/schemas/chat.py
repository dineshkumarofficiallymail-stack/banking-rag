from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    regulator: str | None = Field(default=None, pattern="^(RBI|NPCI|NACH|OTHER)$")


class Citation(BaseModel):
    chunk_id: UUID
    document_title: str
    source_url: str | None = None
    page: int | None = None
    section: str | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    flagged_injection: bool = False
    tool_calls: list[str] = []
    latency_ms: int
    request_id: str

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    regulator: str
    doc_type: str
    source_url: str | None
    page_count: int
    chunk_count: int
    created_at: datetime


class UploadResponse(BaseModel):
    document_id: UUID | None
    title: str
    chunks: int
    skipped: bool
    reason: str | None = None


class StatsOut(BaseModel):
    documents: int
    chunks: int
    queries: int
    grounded_rate: float
    injection_flags: int
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int


class QueryLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    request_id: str | None
    question: str
    grounded: bool | None
    flagged_injection: bool
    tool_calls: list
    latency_ms: int
    created_at: datetime

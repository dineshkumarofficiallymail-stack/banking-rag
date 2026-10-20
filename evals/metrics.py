"""Retrieval and answer-quality metrics. Pure functions -> easy to unit test and reason about."""

from dataclasses import dataclass, field


def hit_at_k(ranked_titles: list[str], expected_source: str | None, k: int) -> float:
    if expected_source is None:
        return 1.0  # out-of-scope questions have no retrieval target
    needle = expected_source.lower()
    return 1.0 if any(needle in t.lower() for t in ranked_titles[:k]) else 0.0


def reciprocal_rank(ranked_titles: list[str], expected_source: str | None) -> float:
    if expected_source is None:
        return 1.0
    needle = expected_source.lower()
    for i, t in enumerate(ranked_titles, start=1):
        if needle in t.lower():
            return 1.0 / i
    return 0.0


def keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def context_precision(chunks_text: list[str], keywords: list[str]) -> float:
    """Share of retrieved chunks that contain at least one expected keyword (noise measure)."""
    if not chunks_text or not keywords:
        return 1.0
    return sum(1 for c in chunks_text if any(k.lower() in c.lower() for k in keywords)) / len(chunks_text)


@dataclass
class EvalRow:
    id: str
    hit: float
    rr: float
    context_precision: float
    answer_keyword_coverage: float
    faithfulness: float
    grounded: bool
    not_found_correct: bool | None
    latency_ms: int
    input_tokens: int
    output_tokens: int


@dataclass
class EvalReport:
    rows: list[EvalRow] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.rows) or 1
        oos = [r for r in self.rows if r.not_found_correct is not None]
        return {
            "n": len(self.rows),
            "hit_rate@k": round(sum(r.hit for r in self.rows) / n, 3),
            "mrr": round(sum(r.rr for r in self.rows) / n, 3),
            "context_precision": round(sum(r.context_precision for r in self.rows) / n, 3),
            "answer_keyword_coverage": round(sum(r.answer_keyword_coverage for r in self.rows) / n, 3),
            "faithfulness": round(sum(r.faithfulness for r in self.rows) / n, 3),
            "grounded_rate": round(sum(1 for r in self.rows if r.grounded) / n, 3),
            "oos_refusal_accuracy": round(sum(1 for r in oos if r.not_found_correct) / len(oos), 3) if oos else None,
            "p50_latency_ms": sorted(r.latency_ms for r in self.rows)[len(self.rows) // 2] if self.rows else 0,
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in self.rows),
        }

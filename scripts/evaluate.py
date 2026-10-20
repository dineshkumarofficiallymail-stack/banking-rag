"""RAG evaluation: retrieval quality (hit@k, MRR, context precision) + answer quality
(faithfulness via judge, keyword coverage, out-of-scope refusal). Exits 1 if below thresholds -> CI gate.

  python -m scripts.evaluate --golden evals/golden_set.jsonl --k 6 --out evals/report.json --min-hit-rate 0.8 --min-faithfulness 0.8
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, log_fields, setup_logging
from app.db.session import SessionLocal
from app.rag.generator import answer_question
from app.rag.guardrails import check_groundedness
from app.rag.prompts import NOT_FOUND_ANSWER, format_sources
from app.rag.retriever import HybridRetriever
from evals.metrics import EvalReport, EvalRow, context_precision, hit_at_k, keyword_coverage, reciprocal_rank

logger = get_logger(__name__)


async def evaluate(golden: Path, k: int) -> EvalReport:
    items = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = EvalReport()
    async with SessionLocal() as db:
        for item in items:
            chunks = await HybridRetriever(db).retrieve(item["question"], k=k, regulator=item.get("regulator"))
            titles = [c.title for c in chunks]
            result = await answer_question(db, item["question"], top_k=k, regulator=item.get("regulator"))

            if chunks and not result.answer.startswith(NOT_FOUND_ANSWER):
                faith = (await check_groundedness(result.answer, format_sources(chunks, get_settings().max_context_chars))).score
            else:
                faith = 1.0 if result.answer.startswith(NOT_FOUND_ANSWER) else 0.0

            expect_nf = item.get("expect_not_found", False)
            row = EvalRow(
                id=item["id"],
                hit=hit_at_k(titles, item.get("expected_source"), k),
                rr=reciprocal_rank(titles, item.get("expected_source")),
                context_precision=context_precision([c.content for c in chunks], item.get("expected_keywords", [])),
                answer_keyword_coverage=0.0 if expect_nf else keyword_coverage(result.answer, item.get("expected_keywords", [])),
                faithfulness=faith,
                grounded=result.grounded,
                not_found_correct=result.answer.startswith(NOT_FOUND_ANSWER) if expect_nf else None,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            if expect_nf:
                row.answer_keyword_coverage = 1.0 if row.not_found_correct else 0.0
            report.rows.append(row)
            logger.info("eval.item", **log_fields(id=row.id, hit=row.hit, faith=row.faithfulness, ms=row.latency_ms))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="evals/golden_set.jsonl")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--out", default="evals/report.json")
    p.add_argument("--min-hit-rate", type=float, default=0.0)
    p.add_argument("--min-faithfulness", type=float, default=0.0)
    args = p.parse_args()

    setup_logging(get_settings().log_level)
    report = asyncio.run(evaluate(Path(args.golden), args.k))
    summary = report.summary()
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": [asdict(r) for r in report.rows]}, indent=2))

    print(f"\n{'id':<10}{'hit':>5}{'rr':>6}{'ctxP':>6}{'kw':>6}{'faith':>7}{'ms':>7}")
    for r in report.rows:
        print(
            f"{r.id:<10}{r.hit:>5.0f}{r.rr:>6.2f}{r.context_precision:>6.2f}{r.answer_keyword_coverage:>6.2f}{r.faithfulness:>7.2f}{r.latency_ms:>7}"
        )
    print("\nSUMMARY:", json.dumps(summary, indent=2))

    failed = []
    if summary["hit_rate@k"] < args.min_hit_rate:
        failed.append(f"hit_rate@k {summary['hit_rate@k']} < {args.min_hit_rate}")
    if summary["faithfulness"] < args.min_faithfulness:
        failed.append(f"faithfulness {summary['faithfulness']} < {args.min_faithfulness}")
    if failed:
        raise SystemExit("EVAL GATE FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()

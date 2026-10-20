# Banking Regulatory RAG + Agent

Grounded question answering over public **RBI / NPCI / NACH** documents for bank operations staff, with
source citations, hallucination and prompt-injection guardrails, a LangGraph tool-calling agent, JWT auth,
Redis caching and rate limiting, a full audit trail, automated RAG evaluation and CI.

> Built by a core-banking (Finacle) engineer moving into applied LLM engineering: the design decisions below
> are the ones that matter in a bank — auditability, deterministic behaviour where possible, and refusing
> rather than guessing.

## Features

| Area | What's implemented |
|---|---|
| Ingestion | PDF/HTML/TXT loaders, section-aware chunking (keeps clause numbers + page), local BGE embeddings, idempotent on sha256 |
| Retrieval | **Hybrid**: pgvector HNSW cosine + Postgres full-text (GIN) + exact circular-number match, fused with Reciprocal Rank Fusion |
| Generation | Claude via `langchain-anthropic`; numbered `<source>` blocks; mandatory `[n]` citations |
| Agent | LangGraph loop with tools `search_regulations`, `lookup_circular`, `list_documents`, `working_day_deadline`; hard step budget |
| Guardrails | Regex + LLM prompt-injection classifier; sanitisation of retrieved text (indirect injection); LLM-as-judge faithfulness check; refuse-when-ungrounded |
| API | FastAPI async, Pydantic v2, JWT (PyJWT + bcrypt), role guard (analyst/admin), per-user Redis rate limit, request-id + JSON logs |
| Data | PostgreSQL + pgvector via async SQLAlchemy 2.0 / asyncpg; `query_logs` audit table (chunks used, tokens, latency, grounded, flagged) |
| Evals | hit@k, MRR, context precision, answer keyword coverage, faithfulness, out-of-scope refusal accuracy; CI gate |
| Ops | Docker (non-root, healthcheck), Compose, GitHub Actions (lint → tests on pgvector service → eval gate → image build), ECS Fargate / App Runner / Azure Container Apps configs |

## Architecture

```
Client ──JWT──▶ FastAPI ──▶ injection guard ──▶ Redis cache ──▶ LangGraph agent ──▶ tools ──▶ Hybrid retriever (pgvector + FTS + RRF)
                                                                      │
                                                                      ▼
                                      answer + [n] citations ──▶ faithfulness judge ──▶ confidence ──▶ QueryLog (audit)
```

Request flow for `POST /api/v1/chat/agent`:
1. Authenticate (JWT) and rate-limit (Redis, per user).
2. Prompt-injection guard: regex pre-filter, LLM classifier only for borderline inputs.
3. Agent loop (max `AGENT_MAX_STEPS`): model decides which tools to call; every retrieved chunk is numbered once in `AgentContext`.
4. Final answer must cite `[n]`; the faithfulness judge scores it against *all* sources the agent saw.
5. If ungrounded → the API returns a refusal plus the retrieved passages so staff can read the primary source.
6. Audit record written in a background task (no added latency).

## Quick start

```bash
git clone <your-repo-url> && cd banking-rag
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                     # set POSTGRES_PASSWORD, JWT_SECRET, ANTHROPIC_API_KEY
docker compose up -d db redis
python -m scripts.init_db
python -m scripts.download_sources && python -m scripts.ingest --sources data/sources.yaml
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs → `POST /auth/register` → **Authorize** → `POST /chat/agent`.

Generate a JWT secret: `python -c "import secrets; print(secrets.token_urlsafe(64))"`.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/v1/auth/register` | – | create analyst (disable in prod with `ALLOW_REGISTRATION=false`) |
| POST | `/api/v1/auth/login` | – | OAuth2 password form → JWT |
| GET | `/api/v1/auth/me` | user | current user |
| POST | `/api/v1/chat` | user | RAG answer with citations (cached) |
| POST | `/api/v1/chat/agent` | user | tool-calling agent |
| GET | `/api/v1/documents` | user | corpus listing |
| POST | `/api/v1/documents/upload` | admin | multipart upload + ingest |
| DELETE | `/api/v1/documents/{id}` | admin | remove document + chunks |
| GET | `/api/v1/admin/stats` | admin | grounded rate, injection flags, latency, token totals |
| GET | `/api/v1/admin/queries` | admin | recent audit records |
| GET | `/api/v1/health`, `/ready` | – | liveness / readiness (DB + Redis) |

Example:
```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login -d 'username=admin@example.com&password=...' | jq -r .access_token)
curl -s localhost:8000/api/v1/chat/agent -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"What is the TAT for a failed IMPS transaction and what compensation applies?","regulator":"RBI"}' | jq
```

## Tests and evaluation

```bash
pytest -q                                  # unit tests (offline, fake LLM + embeddings)
TEST_DATABASE_URL=postgresql+asyncpg://rag:<pw>@localhost:5432/banking_rag_test pytest -q   # + integration tests
make eval                                  # retrieval + faithfulness report, fails below thresholds
```
`OFFLINE_MODE=true` swaps the LLM and embedder for deterministic fakes so the whole pipeline (agent included)
runs in CI without network or API keys. The eval gate in CI runs against a small synthetic corpus in
`evals/sample_docs/`; point `--golden` at your own set for the real corpus.

## Configuration

All settings are environment variables (see `.env.example`); nothing is hard-coded. Key ones:
`DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `ANTHROPIC_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`/`EMBEDDING_DIM`,
`TOP_K`, `MIN_SIMILARITY`, `AGENT_MAX_STEPS`, `RATE_LIMIT_PER_MINUTE`, `OFFLINE_MODE`.

## Deployment

- **Docker**: `docker compose up --build` (runs `init_db` then uvicorn).
- **AWS ECS Fargate**: `deploy/aws/ecs-task-def.json` — secrets from Secrets Manager, logs to CloudWatch; use RDS PostgreSQL (`CREATE EXTENSION vector`) + ElastiCache.
- **AWS App Runner**: `deploy/aws/apprunner-service.json` with `aws apprunner create-service --cli-input-json`.
- **Azure Container Apps**: `deploy/azure/containerapp.bicep` — Postgres Flexible Server (allow-list `vector`) + Azure Cache for Redis.
- Substitute `${AWS_ACCOUNT_ID}`, `${AWS_REGION}`, `${IMAGE_TAG}` with `envsubst` before applying.

## Design decisions (and the alternatives considered)

- **pgvector over a dedicated vector DB** — one transactional store for documents, chunks, users and audit logs; joins and filters (regulator) in SQL; banks already run Postgres. Switch when the corpus is tens of millions of vectors.
- **Hybrid retrieval** — circular numbers and clause ids are exact tokens that embeddings blur; FTS catches them. RRF fuses rankings without calibrating scores.
- **Local embeddings (BGE-small, 384-d)** — no per-token embedding bill, no regulatory text leaving the VPC, deterministic in CI. Trade-off: slightly lower recall than large hosted models; mitigated by hybrid search.
- **LLM-as-judge faithfulness gate** — doubles LLM cost per answer but turns hallucinations into explicit refusals. Fallback to lexical overlap keeps the API up if the judge call fails.
- **Deterministic tools for arithmetic** — TAT / T+N deadlines are computed in Python, never by the model.
- **Agent with a step budget** — multi-hop questions get more than one search; cost and latency stay bounded.
- **Fail-open rate limiting, fail-closed auth** — losing throttling is acceptable; losing authentication is not.
- **Migrations** — `scripts/init_db.py` creates the schema; move to Alembic once the schema stabilises.

## Roadmap

Streaming responses (SSE), reranking (cross-encoder) stage, conversation memory with LangGraph checkpointer,
Alembic migrations, OpenTelemetry tracing, per-regulator freshness tracking of circulars.

## Licence

MIT. Regulatory documents belong to their respective publishers; this project only indexes publicly available material.

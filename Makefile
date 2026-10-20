.PHONY: install dev-up dev-down db ingest run test lint eval docker-build docker-up docker-down

install:
	pip install -r requirements-dev.txt

dev-up:
	docker compose up -d db redis

dev-down:
	docker compose down

db:
	python -m scripts.init_db

ingest:
	python -m scripts.download_sources && python -m scripts.ingest --sources data/sources.yaml

run:
	uvicorn app.main:app --reload --port 8000

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

eval:
	python -m scripts.evaluate --golden evals/golden_set.jsonl --k 6 --out evals/report.json --min-hit-rate 0.8 --min-faithfulness 0.8

docker-build:
	docker build -t banking-rag:local .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

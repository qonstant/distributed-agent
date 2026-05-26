SHELL := /bin/bash

.PHONY: help \
	rag-build rag-up rag-down rag-clean rag-chunks-md rag-chunks-manual \
	class guard clarity suff ret eval \
	landing-build landing-up landing-down \
	rabbitmq-up rabbitmq-down rabbitmq-logs \
	go-build go-up go-down go-clean \
	db-up db-down db-smoke db-smoke-test \
	migrateup migrateup1 migratedown migratedown1 \
	go-test go-test-verbose python-test python-test-verbose \
	test test-verbose coverage coverage-html \
	test-docker coverage-docker

RAG_IMAGE := rag
RAG_DOCKERFILE := python/rag_api/Dockerfile
RAG_BUILD_CTX := python/rag_api
RAG_COMPOSE_FILE := python/rag_api/docker-compose.yml
RAG_LEGACY_DIR := python/RAG
RAG_EVAL_DIR := $(RAG_LEGACY_DIR)/evaluation
RAG_MARKDOWN_TOOL := $(RAG_LEGACY_DIR)/markdown/markdown.py
RAG_DOCX2PDF_OUTPUT_DIR := $(RAG_LEGACY_DIR)/docx2pdf/output
RAG_MARKDOWN_DST_DIR := $(RAG_LEGACY_DIR)/markdown/docs_md
RAG_NOTEBOOK := rag.ipynb
RAG_NOTEBOOK_OUTPUT := rag.executed.ipynb
RAG_OUT_DIR := out
RAG_DOC_PREFIX ?= italy
CLASS_EVAL_SCRIPT := $(RAG_EVAL_DIR)/classification_eval.py
CLASS_EVAL_CSV := $(RAG_EVAL_DIR)/query_mappings.csv
GUARD_EVAL_SCRIPT := $(RAG_EVAL_DIR)/guardrail_eval.py
GUARD_EVAL_CSV := $(RAG_EVAL_DIR)/guardrail_mappings.csv
CLARITY_EVAL_SCRIPT := $(RAG_EVAL_DIR)/clarity_eval.py
CLARITY_EVAL_CSV := $(RAG_EVAL_DIR)/clarity_mappings.csv
SUFF_EVAL_SCRIPT := $(RAG_EVAL_DIR)/sufficiency_eval.py
SUFF_EVAL_CSV := $(RAG_EVAL_DIR)/sufficiency_mappings.csv
RET_EVAL_SCRIPT := $(RAG_EVAL_DIR)/retrieval_eval.py
RET_EVAL_CSV := $(RAG_EVAL_DIR)/query_mappings.csv
EVAL_LIMIT ?=
EVAL_REFRESH ?= 1
EVAL_EXTRA ?=

LANDING_DIR := frontend/landing-page
LANDING_COMPOSE_FILE := $(LANDING_DIR)/docker-compose.yml
LANDING_IMAGE_NAME := landing-page:latest

GO_IMAGE := distributed-agent-go
GO_DOCKERFILE := golang/Dockerfile
GO_BUILD_CTX := golang
GO_COMPOSE_FILE := golang/docker-compose.yml
GO_RABBITMQ_COMPOSE_FILE := golang/docker-compose.rabbitmq.yml
GO_DIR := golang
GO_GOCACHE := $(GO_DIR)/.gocache
GO_COVERAGE_FILE := $(GO_DIR)/coverage.out
GO_TOOLCHAIN_IMAGE := golang:1.24
PYTHON ?= python
PYTHON_RAG_DIR := python/rag_api
DB_COMPOSE_FILE := golang/docker-compose.db.yml
MIGRATIONS_PATH := golang/db/migrations
ENV_FILE := ./.env

help:
	@echo "Available commands:"
	@echo ""
	@echo "RAG:"
	@echo "  make rag-build"
	@echo "  make rag-up"
	@echo "  make rag-down"
	@echo "  make rag-clean"
	@echo "  make rag-chunks-md"
	@echo "  make rag-chunks-manual"
	@echo "  make class     # classification metrics, fresh LLM calls by default"
	@echo "  make guard     # guardrail metrics, fresh LLM calls by default"
	@echo "  make clarity   # clarity/rewrite metrics, fresh LLM calls by default"
	@echo "  make suff      # retrieval sufficiency metrics, fresh LLM calls by default"
	@echo "  make ret       # retrieval metrics over local python/RAG/out artifacts"
	@echo "  make eval      # run guard, class, clarity, suff, and ret metrics"
	@echo "  make class EVAL_REFRESH=     # reuse cached classification predictions"
	@echo "  make guard EVAL_REFRESH=     # reuse cached guardrail predictions"
	@echo "  make clarity EVAL_REFRESH=   # reuse cached clarity predictions"
	@echo "  make suff EVAL_REFRESH=      # reuse cached sufficiency predictions"
	@echo "  make ret EVAL_REFRESH=       # reuse cached retrieval predictions"
	@echo ""
	@echo "Landing page:"
	@echo "  make landing-build"
	@echo "  make landing-up"
	@echo "  make landing-down"
	@echo ""
	@echo "RabbitMQ:"
	@echo "  make rabbitmq-up"
	@echo "  make rabbitmq-down"
	@echo "  make rabbitmq-logs"
	@echo ""
	@echo "Go backend:"
	@echo "  make go-build"
	@echo "  make go-up"
	@echo "  make go-down"
	@echo "  make go-clean"
	@echo "  make go-test"
	@echo "  make go-test-verbose"
	@echo "  make coverage"
	@echo "  make coverage-html"
	@echo "  make test-docker"
	@echo "  make coverage-docker"
	@echo ""
	@echo "Python RAG:"
	@echo "  make python-test"
	@echo "  make python-test-verbose"
	@echo ""
	@echo "All tests:"
	@echo "  make test"
	@echo "  make test-verbose"
	@echo ""
	@echo "Database:"
	@echo "  make db-up"
	@echo "  make db-down"
	@echo "  make db-smoke"
	@echo "  make db-smoke-test"
	@echo ""
	@echo "Migrations:"
	@echo "  make migrateup"
	@echo "  make migrateup1"
	@echo "  make migratedown"
	@echo "  make migratedown1"
	@echo ""
	@echo "Note: DB_URL must be provided through environment variables."

# ----------------------------
# RAG
# ----------------------------

rag-build:
	docker build --no-cache -t "$(RAG_IMAGE)" -f "$(RAG_DOCKERFILE)" "$(RAG_BUILD_CTX)"

rag-up:
	docker compose -f "$(RAG_COMPOSE_FILE)" up -d --build

rag-down:
	docker compose -f "$(RAG_COMPOSE_FILE)" down

rag-clean: 
	-docker rmi -f "$(RAG_IMAGE)" || true

rag-chunks-md:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(RAG_MARKDOWN_TOOL)" "$(RAG_DOCX2PDF_OUTPUT_DIR)" -o "$(RAG_MARKDOWN_DST_DIR)" -d "$(RAG_DOC_PREFIX)"

rag-chunks-manual: rag-chunks-md
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$OPENAI_API_KEY" || (echo "OPENAI_API_KEY is not set" && exit 1); \
	mkdir -p "$(RAG_LEGACY_DIR)/$(RAG_OUT_DIR)"; \
	cd "$(RAG_LEGACY_DIR)" && \
	"$(PYTHON)" -m jupyter nbconvert --to notebook --execute "$(RAG_NOTEBOOK)" \
		--ExecutePreprocessor.timeout=-1 \
		--output "$(RAG_NOTEBOOK_OUTPUT)" \
		--output-dir "$(RAG_OUT_DIR)"

class:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(CLASS_EVAL_SCRIPT)" --csv "$(CLASS_EVAL_CSV)" \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_REFRESH),--refresh-cache,) \
		$(EVAL_EXTRA)

guard:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(GUARD_EVAL_SCRIPT)" --csv "$(GUARD_EVAL_CSV)" \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_REFRESH),--refresh-cache,) \
		$(EVAL_EXTRA)

clarity:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(CLARITY_EVAL_SCRIPT)" --csv "$(CLARITY_EVAL_CSV)" \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_REFRESH),--refresh-cache,) \
		$(EVAL_EXTRA)

suff:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(SUFF_EVAL_SCRIPT)" --csv "$(SUFF_EVAL_CSV)" \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_REFRESH),--refresh-cache,) \
		$(EVAL_EXTRA)

ret:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	"$(PYTHON)" "$(RET_EVAL_SCRIPT)" --csv "$(RET_EVAL_CSV)" --out-dir "$(RAG_LEGACY_DIR)/$(RAG_OUT_DIR)" \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_REFRESH),--refresh-cache,) \
		$(EVAL_EXTRA)

eval: guard class clarity suff ret

# ----------------------------
# Landing page
# ----------------------------

landing-build:
	@set -a; [ -f "$(LANDING_DIR)/.env" ] && source "$(LANDING_DIR)/.env" || true; set +a; \
	cd "$(LANDING_DIR)" && docker build --build-arg NEXT_PUBLIC_SCRIPT_URL="$$NEXT_PUBLIC_SCRIPT_URL" -t "$(LANDING_IMAGE_NAME)" .

landing-up:
	@set -a; [ -f "$(LANDING_DIR)/.env" ] && source "$(LANDING_DIR)/.env" || true; set +a; \
	docker compose -f "$(LANDING_COMPOSE_FILE)" up -d --build

landing-down:
	docker compose -f "$(LANDING_COMPOSE_FILE)" down

# ----------------------------
# RabbitMQ
# ----------------------------

rabbitmq-up:
	@set -a; [ -f "$(GO_DIR)/.env.rabbitmq" ] && source "$(GO_DIR)/.env.rabbitmq" || true; set +a; \
	docker compose -f "$(GO_RABBITMQ_COMPOSE_FILE)" up -d

rabbitmq-down:
	docker compose -f "$(GO_RABBITMQ_COMPOSE_FILE)" down

rabbitmq-logs:
	docker compose -f "$(GO_RABBITMQ_COMPOSE_FILE)" logs -f

# ----------------------------
# Go backend
# ----------------------------

go-build:
	docker build -t "$(GO_IMAGE)" -f "$(GO_DOCKERFILE)" "$(GO_BUILD_CTX)"

go-up:
	docker compose -f "$(GO_COMPOSE_FILE)" up -d --build

go-down:
	docker compose -f "$(GO_COMPOSE_FILE)" down

go-clean:
	-docker rmi -f "$(GO_IMAGE)" || true

go-test:
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go test ./...

go-test-verbose:
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go test -v ./...

python-test:
	cd "$(PYTHON_RAG_DIR)" && "$(PYTHON)" -m unittest discover -s tests

python-test-verbose:
	cd "$(PYTHON_RAG_DIR)" && "$(PYTHON)" -m unittest discover -s tests -v

test: go-test python-test

test-verbose: go-test-verbose python-test-verbose

coverage:
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go test -coverpkg=./... -coverprofile=coverage.out ./...
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go tool cover -func=coverage.out

coverage-html:
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go test -coverpkg=./... -coverprofile=coverage.out ./...
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go tool cover -func=coverage.out
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" go tool cover -html=coverage.out -o coverage.html

test-docker:
	docker run --rm \
		-v "$(CURDIR):/workspace" \
		-w /workspace/"$(GO_DIR)" \
		--user "$$(id -u):$$(id -g)" \
		-e GOCACHE=/workspace/"$(GO_DIR)"/.gocache \
		"$(GO_TOOLCHAIN_IMAGE)" \
		sh -c 'go test ./...'

coverage-docker:
	docker run --rm \
		-v "$(CURDIR):/workspace" \
		-w /workspace/"$(GO_DIR)" \
		--user "$$(id -u):$$(id -g)" \
		-e GOCACHE=/workspace/"$(GO_DIR)"/.gocache \
		"$(GO_TOOLCHAIN_IMAGE)" \
		sh -c 'go test -coverpkg=./... -coverprofile=coverage.out ./... && go tool cover -func=coverage.out'

# ----------------------------
# Database
# ----------------------------

db-up:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	docker compose -f "$(DB_COMPOSE_FILE)" up -d

db-down:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	docker compose -f "$(DB_COMPOSE_FILE)" down -v

db-smoke:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL" || (echo "DB_URL is not set" && exit 1); \
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" DB_URL="$$DB_URL" go run ./cmd/db_access_smoke

db-smoke-test:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL_TEST" || (echo "DB_URL_TEST is not set" && exit 1); \
	cd "$(GO_DIR)" && env GOCACHE="$$(pwd)/.gocache" DB_URL="$$DB_URL_TEST" go run ./cmd/db_access_smoke

# ----------------------------
# Database migrations
# ----------------------------

migrateup:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL" || (echo "DB_URL is not set" && exit 1); \
	migrate -path "$(MIGRATIONS_PATH)" -database "$$DB_URL" -verbose up

migrateup1:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL" || (echo "DB_URL is not set" && exit 1); \
	migrate -path "$(MIGRATIONS_PATH)" -database "$$DB_URL" -verbose up 1

migratedown:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL" || (echo "DB_URL is not set" && exit 1); \
	migrate -path "$(MIGRATIONS_PATH)" -database "$$DB_URL" -verbose down

migratedown1:
	@set -a; [ -f "$(ENV_FILE)" ] && source "$(ENV_FILE)" || true; set +a; \
	test -n "$$DB_URL" || (echo "DB_URL is not set" && exit 1); \
	migrate -path "$(MIGRATIONS_PATH)" -database "$$DB_URL" -verbose down 1

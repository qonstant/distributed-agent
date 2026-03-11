SHELL := /bin/bash

.PHONY: help \
	rag-build rag-up rag-down rag-clean \
	go-build go-up go-down go-clean \
	migrateup migrateup1 migratedown migratedown1

RAG_IMAGE := rag
RAG_DOCKERFILE := python/rag_api/Dockerfile
RAG_BUILD_CTX := python/rag_api
RAG_COMPOSE_FILE := python/rag_api/docker-compose.yml

GO_IMAGE := distributed-agent-go
GO_DOCKERFILE := golang/Dockerfile
GO_BUILD_CTX := golang
GO_COMPOSE_FILE := golang/docker-compose.yml
MIGRATIONS_PATH := golang/db/migrations

help:
	@echo "Available commands:"
	@echo ""
	@echo "RAG:"
	@echo "  make rag-build"
	@echo "  make rag-up"
	@echo "  make rag-down"
	@echo "  make rag-clean"
	@echo ""
	@echo "Go backend:"
	@echo "  make go-build"
	@echo "  make go-up"
	@echo "  make go-down"
	@echo "  make go-clean"
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

# ----------------------------
# Database migrations
# ----------------------------

migrateup:
	@test -n "$(DB_URL)" || (echo "DB_URL is not set" && exit 1)
	migrate -path "$(MIGRATIONS_PATH)" -database "$(DB_URL)" -verbose up

migrateup1:
	@test -n "$(DB_URL)" || (echo "DB_URL is not set" && exit 1)
	migrate -path "$(MIGRATIONS_PATH)" -database "$(DB_URL)" -verbose up 1

migratedown:
	@test -n "$(DB_URL)" || (echo "DB_URL is not set" && exit 1)
	migrate -path "$(MIGRATIONS_PATH)" -database "$(DB_URL)" -verbose down

migratedown1:
	@test -n "$(DB_URL)" || (echo "DB_URL is not set" && exit 1)
	migrate -path "$(MIGRATIONS_PATH)" -database "$(DB_URL)" -verbose down 1
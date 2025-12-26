# Makefile (repo root)
SHELL := /bin/bash
.PHONY: help rag build up down rag-up rag-down clean

IMAGE := rag
RAW_BUILD_ARGS ?= --no-cache

# --- Detect Dockerfile (priority order) ---
DOCKERFILE := $(strip $(shell \
  if [ -f Dockerfile ]; then echo Dockerfile; \
  elif [ -f python/rag_api/Dockerfile ]; then echo python/rag_api/Dockerfile; \
  elif [ -f golang/Dockerfile ]; then echo golang/Dockerfile; \
  else echo ""; fi))

# --- Build context: use the directory containing the Dockerfile, or '.' if Dockerfile is root-name ---
ifeq ($(DOCKERFILE),)
BUILD_CTX := .
else
# if DOCKERFILE contains a slash, use its directory; otherwise use current dir
ifneq (,$(findstring /,$(DOCKERFILE)))
BUILD_CTX := $(dir $(DOCKERFILE))
# strip trailing slash from BUILD_CTX (makes printed output nicer)
BUILD_CTX := $(patsubst %/,%,$(BUILD_CTX))
else
BUILD_CTX := .
endif
endif

# --- Collect docker-compose files that actually exist ---
_COMPOSE_CANDIDATES := docker-compose.yml golang/docker-compose.yml python/rag_api/docker-compose.yml
COMPOSE_FILES := $(foreach f,$(_COMPOSE_CANDIDATES),$(if $(wildcard $(f)),$(f)))
COMPOSE_ARGS := $(patsubst %,-f %,$(COMPOSE_FILES))

help:
	@echo "Usage:"
	@echo "  make rag up       # build image '$(IMAGE)' then 'docker compose up -d' (if compose files exist)"
	@echo "  make rag down     # 'docker compose down' (if compose files exist) and remove image '$(IMAGE)'"
	@echo "  make build        # only build image"
	@echo "  make clean        # remove image '$(IMAGE)' (force)"
	@echo ""
	@echo "Detected Dockerfile: '$(DOCKERFILE)'"
	@echo "Build context: '$(BUILD_CTX)'"
	@echo "Detected compose files: $(COMPOSE_FILES)"
	@echo ""
	@echo "Override on command line: e.g."
	@echo "  make DOCKERFILE=golang/Dockerfile build"
	@echo "  make DOCKERFILE=python/rag_api/Dockerfile BUILD_CTX=python/rag_api rag up"

# noop so `make rag up` runs `rag` then `up`
rag:
	@true

# build image using Dockerfile and proper build context
build:
ifneq ($(DOCKERFILE),)
	@echo "[make] Building image '$(IMAGE)' using Dockerfile: $(DOCKERFILE) with context: $(BUILD_CTX)"
	docker build $(RAW_BUILD_ARGS) -t "$(IMAGE)" -f "$(DOCKERFILE)" "$(BUILD_CTX)"
else
	@echo "[make] ERROR: No Dockerfile found (checked root, python/rag_api/, golang/)." >&2
	@exit 1
endif

# up: build then docker compose up -d (if any compose files exist)
up: build
ifneq ($(COMPOSE_ARGS),)
	@echo "[make] Starting docker compose: docker compose $(COMPOSE_ARGS) up -d"
	docker compose $(COMPOSE_ARGS) up -d
else
	@echo "[make] WARNING: No docker-compose.yml found (checked root, golang/, python/rag_api/)." >&2
	@echo "[make] Built image '$(IMAGE)'. Add a docker-compose.yml if you want to start containers." >&2
endif

# down: docker compose down (if exists) then remove image
down:
ifneq ($(COMPOSE_ARGS),)
	@echo "[make] Running: docker compose $(COMPOSE_ARGS) down"
	docker compose $(COMPOSE_ARGS) down
else
	@echo "[make] No docker-compose.yml found; skipping 'docker compose down'." >&2
endif
	@echo "[make] Removing image: $(IMAGE) (if exists)"
	-@docker rmi -f "$(IMAGE)" || true

# convenience aliases so user can call `make rag up` and `make rag down` (rag target above is a noop)
rag-up: up
rag-down: down

clean:
	@echo "[make] Removing image: $(IMAGE) (if exists)"
	-@docker rmi -f "$(IMAGE)" || true

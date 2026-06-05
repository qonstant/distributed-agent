#!/usr/bin/env python3
"""Run one query through the production RAG pipeline with trace logs enabled."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from rag_service.application.query_service import QueryService
from rag_service.infrastructure.config import load_settings
from rag_service.infrastructure.faiss_store import FaissMetadataStore
from rag_service.infrastructure.lightrag_store import LightRAGMetadataStore
from rag_service.infrastructure.openai_gateway import OpenAIGateway


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug one full RAG answer pipeline from guardrail to final answer."
    )
    parser.add_argument("--query", required=True, help="User question to run.")
    parser.add_argument("--conversation-id", default="", help="Optional conversation id.")
    parser.add_argument("--preferred-name", default="", help="Optional preferred user name.")
    parser.add_argument("--raw-k", type=int, default=128, help="Candidate chunks to retrieve.")
    parser.add_argument("--top-for-llm", type=int, default=5, help="Filtered chunks sent to the answer LLM.")
    parser.add_argument(
        "--trace-format",
        choices=("pretty", "json"),
        default=os.getenv("RAG_TRACE_LOG_FORMAT", "pretty"),
        help="Trace log format.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def load_store(settings: Any):
    if settings.retrieval_backend == "lightrag":
        try:
            return LightRAGMetadataStore.load(settings)
        except Exception:
            if settings.retrieval_fallback == "faiss":
                print("[debug] LightRAG unavailable; falling back to FAISS")
                return FaissMetadataStore.load(settings)
            raise
    return FaissMetadataStore.load(settings)


def result_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [result_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: result_to_jsonable(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    configure_logging()

    os.environ["RAG_TRACE_LOGS"] = "true"
    os.environ["RAG_TRACE_LOG_FORMAT"] = args.trace_format

    settings = load_settings()
    gateway = OpenAIGateway(settings)
    store = load_store(settings)
    service = QueryService(
        gateway,
        store,
        conversation_memory=None,
        trace_enabled=True,
        trace_max_chars=settings.trace_log_max_chars,
        trace_log_format=args.trace_format,
    )
    result = service.handle_query(
        args.query,
        conversation_id=args.conversation_id or None,
        preferred_name=args.preferred_name or None,
        raw_k=args.raw_k,
        top_for_llm=args.top_for_llm,
    )

    print("\n=== Final Result ===")
    print(json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    redis_url: Optional[str]
    s3_endpoint: Optional[str]
    s3_access_key: Optional[str]
    s3_secret: Optional[str]
    s3_bucket_vectors: Optional[str]
    s3_use_ssl: bool
    s3_verify: bool
    release_prefix: Optional[str]
    out_dir: Path
    meta_json_path: Path
    faiss_index_path: Path
    retrieval_backend: str = "faiss"
    retrieval_fallback: str = "faiss"
    lightrag_dir: Path = Path("out/lightrag")
    lightrag_s3_prefix: str = "lightrag"
    lightrag_query_mode: str = "naive"
    lightrag_llm_model: str = "gpt-4o-mini"
    lightrag_embed_model: str = "text-embedding-3-small"
    lightrag_embed_dim: int = 1536
    optional_artifacts: List[str] = field(default_factory=list)
    embed_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    class_model: str = "gpt-4o-mini"
    conversation_key_prefix: str = "chat:conv:"
    conversation_max_items: int = 8
    trace_logs_enabled: bool = True
    trace_log_max_chars: int = 240
    trace_log_format: str = "pretty"


def load_settings() -> Settings:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

    s3_use_ssl = os.getenv("S3_USE_SSL", "false").lower() in ("1", "true", "yes")
    s3_verify_raw = os.getenv("S3_VERIFY", "").lower()
    s3_verify = False if s3_verify_raw in ("0", "false", "no") else s3_use_ssl

    out_dir = Path(os.getenv("OUT_DIR", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lightrag_dir = Path(os.getenv("LIGHTRAG_DIR", str(out_dir / "lightrag")))

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    class_model = os.getenv("CLASS_MODEL", llm_model)
    lightrag_embed_dim_raw = os.getenv("LIGHTRAG_EMBED_DIM", "1536")
    try:
        lightrag_embed_dim = max(1, int(lightrag_embed_dim_raw))
    except ValueError:
        lightrag_embed_dim = 1536
    conversation_max_items_raw = os.getenv("CONVERSATION_MEMORY_MAX_ITEMS", "8")
    try:
        conversation_max_items = max(1, int(conversation_max_items_raw))
    except ValueError:
        conversation_max_items = 8

    trace_log_max_chars_raw = os.getenv("RAG_TRACE_LOG_MAX_CHARS", "240")
    try:
        trace_log_max_chars = max(40, int(trace_log_max_chars_raw))
    except ValueError:
        trace_log_max_chars = 240

    trace_logs_enabled = os.getenv("RAG_TRACE_LOGS", "true").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    trace_log_format = os.getenv("RAG_TRACE_LOG_FORMAT", "pretty").strip().lower()
    if trace_log_format not in {"pretty", "json"}:
        trace_log_format = "pretty"

    return Settings(
        openai_api_key=openai_api_key,
        redis_url=os.getenv("REDIS_URL"),
        s3_endpoint=os.getenv("S3_ENDPOINT"),
        s3_access_key=os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY"),
        s3_secret=os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET"),
        s3_bucket_vectors=os.getenv("S3_BUCKET_VECTORS"),
        s3_use_ssl=s3_use_ssl,
        s3_verify=s3_verify,
        release_prefix=os.getenv("RELEASE_PREFIX"),
        out_dir=out_dir,
        meta_json_path=out_dir / "meta.json",
        faiss_index_path=out_dir / "index.faiss",
        retrieval_backend=os.getenv("RAG_RETRIEVAL_BACKEND", "faiss").strip().lower(),
        retrieval_fallback=os.getenv("RAG_RETRIEVAL_FALLBACK", "faiss").strip().lower(),
        lightrag_dir=lightrag_dir,
        lightrag_s3_prefix=os.getenv("LIGHTRAG_S3_PREFIX", "lightrag").strip().strip("/") or "lightrag",
        lightrag_query_mode=os.getenv("LIGHTRAG_QUERY_MODE", "naive").strip().lower() or "naive",
        lightrag_llm_model=os.getenv("LIGHTRAG_LLM_MODEL", llm_model),
        lightrag_embed_model=os.getenv("LIGHTRAG_EMBED_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-small")),
        lightrag_embed_dim=lightrag_embed_dim,
        optional_artifacts=["embeddings.npy", "ids.npy", "chunks.jsonl", "manifest.json", "faq.jsonl"],
        llm_model=llm_model,
        class_model=class_model,
        conversation_key_prefix=os.getenv("CONVERSATION_MEMORY_KEY_PREFIX", "chat:conv:"),
        conversation_max_items=conversation_max_items,
        trace_logs_enabled=trace_logs_enabled,
        trace_log_max_chars=trace_log_max_chars,
        trace_log_format=trace_log_format,
    )

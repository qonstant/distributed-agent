from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
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
    optional_artifacts: List[str] = field(default_factory=list)
    embed_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    class_model: str = "gpt-4o-mini"


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

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    class_model = os.getenv("CLASS_MODEL", llm_model)

    return Settings(
        openai_api_key=openai_api_key,
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
        optional_artifacts=["embeddings.npy", "ids.npy", "chunks.jsonl", "manifest.json"],
        llm_model=llm_model,
        class_model=class_model,
    )

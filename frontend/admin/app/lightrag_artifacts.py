from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config as BotoConfig


LIGHTRAG_REQUIRED_FILES = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_s3_client() -> tuple[Any, str]:
    endpoint = os.getenv("S3_ENDPOINT", "").strip()
    bucket = os.getenv("S3_BUCKET_VECTORS", "").strip()
    access_key = (os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET") or "").strip()
    use_ssl = _env_bool("S3_USE_SSL", default=False)
    verify_raw = os.getenv("S3_VERIFY", "").strip().lower()
    verify = False if verify_raw in {"0", "false", "no"} else use_ssl

    if not endpoint or not bucket or not access_key or not secret_key:
        raise RuntimeError("S3_ENDPOINT, S3_BUCKET_VECTORS, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY are required")

    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"{'https' if use_ssl else 'http'}://{endpoint}"

    print(f"[lightrag-upload] S3 endpoint: {endpoint}", flush=True)
    print(f"[lightrag-upload] S3 bucket: {bucket}", flush=True)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        verify=verify,
    )
    return client, bucket


def _iter_lightrag_files(working_dir: Path) -> list[Path]:
    return sorted(path for path in working_dir.iterdir() if path.is_file() and not path.name.startswith("."))


def upload_lightrag_release(working_dir: Path, *, prefix: str | None = None) -> dict[str, Any]:
    if not working_dir.exists() or not working_dir.is_dir():
        raise RuntimeError(f"LightRAG working directory not found: {working_dir}")

    missing = [filename for filename in LIGHTRAG_REQUIRED_FILES if not (working_dir / filename).exists()]
    if missing:
        raise RuntimeError(f"LightRAG artifacts are incomplete; missing: {', '.join(missing)}")

    files = _iter_lightrag_files(working_dir)
    if not files:
        raise RuntimeError(f"LightRAG working directory has no files: {working_dir}")

    s3, bucket = _build_s3_client()
    root_prefix = (prefix or os.getenv("LIGHTRAG_S3_PREFIX") or "lightrag").strip().strip("/")
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_prefix = f"{root_prefix}/releases/{build_id}"

    uploaded: list[str] = []
    for path in files:
        key = f"{release_prefix}/{path.name}"
        s3.upload_file(str(path), bucket, key)
        uploaded.append(key)

    manifest = {
        "build_id": build_id,
        "release_prefix": release_prefix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": [path.name for path in files],
    }
    manifest_key = f"{release_prefix}/manifest.json"
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    uploaded.append(manifest_key)

    pointer_key = f"{root_prefix}/releases/current"
    s3.put_object(
        Bucket=bucket,
        Key=pointer_key,
        Body=release_prefix.encode("utf-8"),
        ContentType="text/plain",
    )

    return {
        "bucket": bucket,
        "build_id": build_id,
        "release_prefix": release_prefix,
        "pointer_key": pointer_key,
        "uploaded": uploaded,
    }

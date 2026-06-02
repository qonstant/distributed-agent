#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.client import Config as BotoConfig


RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parents[1]
MARKDOWN_TOOL = RAG_DIR / "markdown" / "markdown.py"
LIGHTRAG_EVAL = RAG_DIR / "evaluation" / "lightrag_eval.py"
DEFAULT_SOURCE_DIR = RAG_DIR / "docx2pdf" / "output"
DEFAULT_MARKDOWN_DIR = RAG_DIR / "markdown" / "docs_md"
DEFAULT_WORKING_DIR = RAG_DIR / "out" / "lightrag"
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_s3_client() -> tuple[Any, str]:
    endpoint = os.getenv("S3_ENDPOINT", "").strip()
    bucket = os.getenv("S3_BUCKET_VECTORS", "").strip()
    access_key = (os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET") or "").strip()
    use_ssl = env_bool("S3_USE_SSL", default=False)
    verify_raw = os.getenv("S3_VERIFY", "").strip().lower()
    verify = False if verify_raw in {"0", "false", "no"} else use_ssl

    if not endpoint or not bucket or not access_key or not secret_key:
        raise RuntimeError("S3_ENDPOINT, S3_BUCKET_VECTORS, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY are required")

    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"{'https' if use_ssl else 'http'}://{endpoint}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        verify=verify,
    )
    return client, bucket


def list_source_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    normalized_prefix = prefix.strip("/")
    prefix_arg = f"{normalized_prefix}/" if normalized_prefix else ""
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_arg):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            if Path(key).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS:
                keys.append(key)
    return sorted(keys)


def local_path_for_key(key: str, source_prefix: str, source_dir: Path, strip_prefix: bool) -> Path:
    rel = key.strip("/")
    normalized_prefix = source_prefix.strip("/")
    if strip_prefix and normalized_prefix and rel.startswith(f"{normalized_prefix}/"):
        rel = rel[len(normalized_prefix) + 1 :]
    return source_dir / rel


def download_sources(
    s3: Any,
    bucket: str,
    keys: Iterable[str],
    source_prefix: str,
    source_dir: Path,
    strip_prefix: bool,
) -> list[Path]:
    downloaded: list[Path] = []
    for key in keys:
        target = local_path_for_key(key, source_prefix, source_dir, strip_prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[lightrag-s3] download s3://{bucket}/{key} -> {target}", flush=True)
        s3.download_file(bucket, key, str(target))
        downloaded.append(target)
    return downloaded


def upload_markdowns(s3: Any, bucket: str, markdown_dir: Path, prefix: str) -> dict[str, Any]:
    if not markdown_dir.exists():
        raise RuntimeError(f"markdown directory not found: {markdown_dir}")

    root_prefix = prefix.strip().strip("/")
    uploaded: list[str] = []
    for path in sorted(markdown_dir.rglob("*.md")):
        rel = path.relative_to(markdown_dir).as_posix()
        key = f"{root_prefix}/{rel}" if root_prefix else rel
        print(f"[lightrag-s3] upload markdown {path} -> s3://{bucket}/{key}", flush=True)
        s3.upload_file(str(path), bucket, key)
        uploaded.append(key)

    manifest = {
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "markdown_prefix": root_prefix,
        "file_count": len(uploaded),
        "files": uploaded,
    }
    manifest_key = f"{root_prefix}/manifest.json" if root_prefix else "manifest.json"
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    uploaded.append(manifest_key)
    return manifest


def run_command(args: list[str], cwd: Path) -> None:
    print(f"[lightrag-s3] run: {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def clean_for_full(source_dir: Path, markdown_dir: Path, working_dir: Path) -> None:
    for path in (source_dir, markdown_dir, working_dir):
        if path.exists():
            print(f"[lightrag-s3] remove {path}", flush=True)
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightRAG artifacts from source documents stored in S3.")
    parser.add_argument("--mode", choices=("full", "continue"), default=os.getenv("LIGHTRAG_PIPELINE_MODE", "continue"))
    parser.add_argument("--source-prefix", default=os.getenv("LIGHTRAG_SOURCE_PREFIX", "italy"))
    parser.add_argument("--markdown-prefix", default=os.getenv("LIGHTRAG_MARKDOWN_S3_PREFIX", "markdowns"))
    parser.add_argument("--source-dir", default=os.getenv("LIGHTRAG_SOURCE_DIR", str(DEFAULT_SOURCE_DIR)))
    parser.add_argument("--markdown-dir", default=os.getenv("LIGHTRAG_MARKDOWN_DIR", str(DEFAULT_MARKDOWN_DIR)))
    parser.add_argument("--working-dir", default=os.getenv("LIGHTRAG_WORK_DIR", str(DEFAULT_WORKING_DIR)))
    parser.add_argument("--doc-prefix", default=os.getenv("RAG_DOC_PREFIX", os.getenv("LIGHTRAG_DOC_PREFIX", "italy")))
    parser.add_argument("--strip-source-prefix", action="store_true", default=env_bool("LIGHTRAG_STRIP_SOURCE_PREFIX", True))
    parser.add_argument("--no-strip-source-prefix", action="store_false", dest="strip_source_prefix")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    markdown_dir = Path(args.markdown_dir)
    working_dir = Path(args.working_dir)

    if args.mode == "full":
        clean_for_full(source_dir, markdown_dir, working_dir)

    source_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    s3, bucket = build_s3_client()
    source_keys = list_source_keys(s3, bucket, args.source_prefix)
    if not source_keys:
        raise RuntimeError(f"no source documents found in s3://{bucket}/{args.source_prefix.strip('/')}/")
    print(f"[lightrag-s3] found {len(source_keys)} source document(s)", flush=True)

    download_sources(
        s3,
        bucket,
        source_keys,
        source_prefix=args.source_prefix,
        source_dir=source_dir,
        strip_prefix=args.strip_source_prefix,
    )

    run_command(
        [
            sys.executable,
            str(MARKDOWN_TOOL),
            str(source_dir),
            "-o",
            str(markdown_dir),
            "-d",
            args.doc_prefix,
        ],
        cwd=REPO_ROOT,
    )
    upload_markdowns(s3, bucket, markdown_dir, args.markdown_prefix)

    lightrag_args = [
        sys.executable,
        str(LIGHTRAG_EVAL),
        "--docs-dir",
        str(markdown_dir),
        "--working-dir",
        str(working_dir),
        "--index-only",
    ]
    if args.mode == "full":
        lightrag_args.append("--reset")
    else:
        lightrag_args.append("--rebuild")
    run_command(lightrag_args, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()

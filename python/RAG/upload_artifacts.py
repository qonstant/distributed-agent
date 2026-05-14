#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig


REQUIRED_FILES = ("meta.json", "index.faiss")
OPTIONAL_FILES = (
    "chunks.jsonl",
    "embeddings.npy",
    "ids.npy",
    "manifest.json",
    "eval_report.json",
    "classification_eval_report.json",
    "guardrail_eval_report.json",
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")


def build_s3_client():
    endpoint = os.getenv("S3_ENDPOINT", "").strip()
    bucket = os.getenv("S3_BUCKET_VECTORS", "").strip()
    access_key = (os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET") or "").strip()
    use_ssl = env_bool("S3_USE_SSL", default=False)
    verify = not (os.getenv("S3_VERIFY", "").strip().lower() in ("0", "false", "no"))

    if not endpoint or not bucket or not access_key or not secret_key:
        raise RuntimeError("S3_ENDPOINT, S3_BUCKET_VECTORS, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY are required")

    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"{'https' if use_ssl else 'http'}://{endpoint}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        verify=verify if use_ssl else False,
    )
    return client, bucket


def upload_file(s3, bucket: str, local_path: Path, key: str) -> None:
    print(f"[upload] {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload RAG artifacts to the vectors bucket.")
    parser.add_argument("--source-dir", default="out", help="Directory containing meta.json and index.faiss")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f"source directory not found: {source_dir}")

    release_prefix = (os.getenv("RELEASE_PREFIX") or "").strip()
    prefix = release_prefix.rstrip("/") if release_prefix else ""

    s3, bucket = build_s3_client()

    for filename in REQUIRED_FILES:
        file_path = source_dir / filename
        if not file_path.exists():
            raise RuntimeError(f"required artifact missing: {file_path}")
        key = f"{prefix}/{filename}" if prefix else filename
        upload_file(s3, bucket, file_path, key)

    for filename in OPTIONAL_FILES:
        file_path = source_dir / filename
        if not file_path.exists():
            continue
        key = f"{prefix}/{filename}" if prefix else filename
        upload_file(s3, bucket, file_path, key)

    if prefix:
        current_key = "releases/current"
        print(f"[upload] pointer -> s3://{bucket}/{current_key} = {prefix}")
        s3.put_object(Bucket=bucket, Key=current_key, Body=prefix.encode("utf-8"), ContentType="text/plain")

    print("[upload] done")


if __name__ == "__main__":
    main()

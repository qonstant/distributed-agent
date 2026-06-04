#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.client import Config as BotoConfig


RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parents[1]
MARKDOWN_TOOL = RAG_DIR / "markdown" / "markdown.py"
LIGHTRAG_EVAL = RAG_DIR / "evaluation" / "lightrag_eval.py"
DEFAULT_WORKING_DIR = Path(tempfile.gettempdir()) / "nomadmit-lightrag-work"
SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
LIGHTRAG_REQUIRED_FILES = {
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
}
BUILD_CONTEXT_FILE = "s3_build_context.json"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_s3_client() -> tuple[Any, str, str]:
    endpoint = os.getenv("S3_ENDPOINT", "").strip()
    vectors_bucket = os.getenv("S3_BUCKET_VECTORS", "").strip()
    docs_bucket = (os.getenv("S3_BUCKET") or vectors_bucket).strip()
    access_key = (os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET") or "").strip()
    use_ssl = env_bool("S3_USE_SSL", default=False)
    verify_raw = os.getenv("S3_VERIFY", "").strip().lower()
    verify = False if verify_raw in {"0", "false", "no"} else use_ssl

    if not endpoint or not vectors_bucket or not docs_bucket or not access_key or not secret_key:
        raise RuntimeError(
            "S3_ENDPOINT, S3_BUCKET, S3_BUCKET_VECTORS, "
            "S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY are required"
        )

    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"{'https' if use_ssl else 'http'}://{endpoint}"

    print(f"[lightrag-s3] S3 endpoint: {endpoint}", flush=True)
    print(f"[lightrag-s3] S3 docs bucket: {docs_bucket}", flush=True)
    print(f"[lightrag-s3] S3 vectors bucket: {vectors_bucket}", flush=True)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
        verify=verify,
    )
    return client, docs_bucket, vectors_bucket


def _prefix(value: str) -> str:
    return value.strip().strip("/")


def _join_key(prefix: str, name: str) -> str:
    root = _prefix(prefix)
    return f"{root}/{name}" if root else name


def list_source_objects(s3: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    normalized_prefix = prefix.strip("/")
    prefix_arg = f"{normalized_prefix}/" if normalized_prefix else ""
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_arg):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            if Path(key).suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS:
                objects.append(dict(item))
    return sorted(objects, key=lambda item: str(item.get("Key") or ""))


def source_signature(source_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for item in source_objects:
        signature.append(
            {
                "key": str(item.get("Key") or ""),
                "etag": str(item.get("ETag") or "").strip('"'),
                "size": int(item.get("Size") or 0),
            }
        )
    return signature


def s3_json_or_none(s3: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception:
        return None


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


def download_markdowns_from_manifest(
    s3: Any,
    bucket: str,
    markdown_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    files = [str(key) for key in manifest.get("files") or [] if str(key).endswith(".md")]
    if not files:
        return False

    markdown_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lightrag-s3] reusing {len(files)} markdown file(s) from S3 manifest", flush=True)
    root_prefix = _prefix(str(manifest.get("markdown_prefix") or ""))
    for key in files:
        rel = key[len(root_prefix) + 1 :] if root_prefix and key.startswith(f"{root_prefix}/") else Path(key).name
        target = markdown_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[lightrag-s3] download markdown s3://{bucket}/{key} -> {target}", flush=True)
        s3.download_file(bucket, key, str(target))
    return True


def upload_markdowns(
    s3: Any,
    bucket: str,
    markdown_dir: Path,
    prefix: str,
    source_sig: list[dict[str, Any]],
) -> dict[str, Any]:
    if not markdown_dir.exists():
        raise RuntimeError(f"markdown directory not found: {markdown_dir}")

    root_prefix = prefix.strip().strip("/")
    uploaded: list[str] = []
    markdown_files = sorted(markdown_dir.rglob("*.md"))
    print(f"[lightrag-s3] uploading {len(markdown_files)} markdown file(s) to prefix {root_prefix}", flush=True)
    for path in markdown_files:
        rel = path.relative_to(markdown_dir).as_posix()
        key = f"{root_prefix}/{rel}" if root_prefix else rel
        print(f"[lightrag-s3] upload markdown {path} -> s3://{bucket}/{key}", flush=True)
        s3.upload_file(str(path), bucket, key)
        uploaded.append(key)

    manifest = {
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "markdown_prefix": root_prefix,
        "source_signature": source_sig,
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


def working_dir_has_required_files(working_dir: Path) -> bool:
    return all((working_dir / filename).exists() for filename in LIGHTRAG_REQUIRED_FILES)


def build_context_path(working_dir: Path) -> Path:
    return working_dir / BUILD_CONTEXT_FILE


def load_local_build_context(working_dir: Path) -> dict[str, Any] | None:
    path = build_context_path(working_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_build_context(
    working_dir: Path,
    *,
    source_sig: list[dict[str, Any]],
    markdown_prefix: str,
    docs_bucket: str,
    vectors_bucket: str,
) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docs_bucket": docs_bucket,
        "vectors_bucket": vectors_bucket,
        "markdown_prefix": _prefix(markdown_prefix),
        "source_signature": source_sig,
    }
    build_context_path(working_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_context_matches(working_dir: Path, source_sig: list[dict[str, Any]]) -> bool:
    context = load_local_build_context(working_dir)
    return bool(context and context.get("source_signature") == source_sig)


def download_current_lightrag_release(
    s3: Any,
    bucket: str,
    working_dir: Path,
    *,
    root_prefix: str,
    source_sig: list[dict[str, Any]],
) -> bool:
    pointer_key = _join_key(root_prefix, "releases/current")
    try:
        response = s3.get_object(Bucket=bucket, Key=pointer_key)
        release_prefix = response["Body"].read().decode("utf-8").strip().strip("/")
    except Exception:
        return False
    if not release_prefix:
        return False

    manifest = s3_json_or_none(s3, bucket, _join_key(release_prefix, "manifest.json"))
    if not manifest:
        return False

    files = [str(name) for name in manifest.get("files") or []]
    if BUILD_CONTEXT_FILE not in files:
        print("[lightrag-s3] current LightRAG release has no build context; cannot prove it matches sources", flush=True)
        return False

    tmp_dir = working_dir.with_name(f".{working_dir.name}.download")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        key = _join_key(release_prefix, name)
        target = tmp_dir / name
        print(f"[lightrag-s3] hydrate LightRAG artifact s3://{bucket}/{key} -> {target}", flush=True)
        s3.download_file(bucket, key, str(target))

    if not build_context_matches(tmp_dir, source_sig):
        print("[lightrag-s3] current LightRAG release does not match source documents; rebuilding graph", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    if not working_dir_has_required_files(tmp_dir):
        print("[lightrag-s3] current LightRAG release is missing required files; rebuilding graph", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    if working_dir.exists():
        shutil.rmtree(working_dir)
    tmp_dir.replace(working_dir)
    print(f"[lightrag-s3] reused current LightRAG release {release_prefix}", flush=True)
    return True


def run_command(args: list[str], cwd: Path) -> None:
    print(f"[lightrag-s3] run: {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def count_markdowns(markdown_dir: Path) -> int:
    return sum(1 for _path in markdown_dir.rglob("*.md")) if markdown_dir.exists() else 0


def clean_for_full(working_dir: Path) -> None:
    for path in (working_dir,):
        if path.exists():
            print(f"[lightrag-s3] remove {path}", flush=True)
            shutil.rmtree(path)


def remove_temp_dir(path: Path) -> None:
    if path.exists():
        print(f"[lightrag-s3] cleanup temp {path}", flush=True)
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightRAG artifacts from source documents stored in S3.")
    parser.add_argument("--mode", choices=("full", "continue"), default=os.getenv("LIGHTRAG_PIPELINE_MODE", "continue"))
    parser.add_argument("--source-prefix", default=os.getenv("LIGHTRAG_SOURCE_PREFIX", "italy"))
    parser.add_argument("--markdown-prefix", default=os.getenv("LIGHTRAG_MARKDOWN_S3_PREFIX", "markdowns"))
    parser.add_argument("--source-dir", default=os.getenv("LIGHTRAG_SOURCE_DIR", ""))
    parser.add_argument("--markdown-dir", default=os.getenv("LIGHTRAG_MARKDOWN_DIR", ""))
    parser.add_argument("--working-dir", default=os.getenv("LIGHTRAG_WORK_DIR", str(DEFAULT_WORKING_DIR)))
    parser.add_argument("--doc-prefix", default=os.getenv("RAG_DOC_PREFIX", os.getenv("LIGHTRAG_DOC_PREFIX", "italy")))
    parser.add_argument("--strip-source-prefix", action="store_true", default=env_bool("LIGHTRAG_STRIP_SOURCE_PREFIX", True))
    parser.add_argument("--no-strip-source-prefix", action="store_false", dest="strip_source_prefix")
    args = parser.parse_args()

    temp_root = Path(tempfile.mkdtemp(prefix="nomadmit-lightrag-"))
    source_dir = Path(args.source_dir) if args.source_dir else temp_root / "source_docs"
    markdown_dir = Path(args.markdown_dir) if args.markdown_dir else temp_root / "markdowns"
    working_dir = Path(args.working_dir)

    try:
        print(f"[lightrag-s3] temp root: {temp_root}", flush=True)
        print(f"[lightrag-s3] source prefix: {args.source_prefix}", flush=True)
        print(f"[lightrag-s3] markdown S3 prefix: {args.markdown_prefix}", flush=True)
        print(f"[lightrag-s3] LightRAG scratch dir: {working_dir}", flush=True)
        print("[lightrag-s3] persistent inputs/outputs stay in S3; local paths are temporary build scratch", flush=True)
        if args.mode == "full":
            clean_for_full(working_dir)

        source_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        s3, docs_bucket, vectors_bucket = build_s3_client()
        source_objects = list_source_objects(s3, docs_bucket, args.source_prefix)
        if not source_objects:
            raise RuntimeError(f"no source documents found in s3://{docs_bucket}/{args.source_prefix.strip('/')}/")
        source_keys = [str(item.get("Key") or "") for item in source_objects]
        current_source_signature = source_signature(source_objects)
        print(f"[lightrag-s3] found {len(source_keys)} source document(s)", flush=True)

        markdown_manifest_key = _join_key(args.markdown_prefix, "manifest.json")
        markdown_manifest = s3_json_or_none(s3, vectors_bucket, markdown_manifest_key)
        markdowns_reused = False
        if args.mode == "continue" and markdown_manifest and markdown_manifest.get("source_signature") == current_source_signature:
            markdowns_reused = download_markdowns_from_manifest(s3, vectors_bucket, markdown_dir, markdown_manifest)

        if markdowns_reused:
            print("[lightrag-s3] source documents unchanged; skipped markdown regeneration", flush=True)
        else:
            if args.mode == "continue":
                print("[lightrag-s3] markdown manifest missing or source documents changed; regenerating markdowns", flush=True)
            download_sources(
                s3,
                docs_bucket,
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
            print(f"[lightrag-s3] created {count_markdowns(markdown_dir)} markdown file(s)", flush=True)
            upload_markdowns(s3, vectors_bucket, markdown_dir, args.markdown_prefix, current_source_signature)

        if args.mode == "continue":
            if build_context_matches(working_dir, current_source_signature):
                print("[lightrag-s3] local LightRAG scratch matches source documents; continuing existing graph state", flush=True)
            elif download_current_lightrag_release(
                s3,
                vectors_bucket,
                working_dir,
                root_prefix=os.getenv("LIGHTRAG_S3_PREFIX", "lightrag"),
                source_sig=current_source_signature,
            ):
                print("[lightrag-s3] source documents unchanged; reused LightRAG graph artifacts", flush=True)
            elif working_dir.exists():
                print("[lightrag-s3] local LightRAG scratch does not match source documents; resetting graph scratch", flush=True)
                shutil.rmtree(working_dir)
        write_build_context(
            working_dir,
            source_sig=current_source_signature,
            markdown_prefix=args.markdown_prefix,
            docs_bucket=docs_bucket,
            vectors_bucket=vectors_bucket,
        )

        print("[lightrag-s3] starting LightRAG indexing from generated markdowns", flush=True)
        lightrag_args = [
            sys.executable,
            str(LIGHTRAG_EVAL),
            "--docs-dir",
            str(markdown_dir),
            "--working-dir",
            str(working_dir),
            "--index-only",
        ]
        # This wrapper owns scratch cleanup. Passing --reset here would delete
        # the S3 source-context marker and make interrupted builds harder to resume.
        lightrag_args.append("--rebuild")
        run_command(lightrag_args, cwd=REPO_ROOT)
        write_build_context(
            working_dir,
            source_sig=current_source_signature,
            markdown_prefix=args.markdown_prefix,
            docs_bucket=docs_bucket,
            vectors_bucket=vectors_bucket,
        )
    finally:
        if not args.source_dir and not args.markdown_dir:
            remove_temp_dir(temp_root)


if __name__ == "__main__":
    main()

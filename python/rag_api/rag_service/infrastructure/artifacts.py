from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from rag_service.infrastructure.config import Settings

REQUIRED_ARTIFACTS = ("meta.json", "index.faiss")
LIGHTRAG_REQUIRED_ARTIFACTS = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
)


def _create_s3_client(settings: Settings) -> Optional[Any]:
    try:
        import boto3
        from botocore.client import Config as BotoConfig
    except Exception:
        print("[s3] boto3 not installed; skipping S3 support")
        return None

    if not settings.s3_endpoint or not settings.s3_bucket_vectors:
        print("[s3] S3_ENDPOINT or S3_BUCKET_VECTORS not set; skipping S3")
        return None

    endpoint = settings.s3_endpoint.strip()
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        scheme = "https" if settings.s3_use_ssl else "http"
        endpoint_url = f"{scheme}://{endpoint}"
    else:
        endpoint_url = endpoint

    cfg = BotoConfig(signature_version="s3v4")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret,
            config=cfg,
            verify=settings.s3_verify,
        )
        print(f"[s3] boto3 client created endpoint={endpoint_url} verify={settings.s3_verify}")
        return s3
    except Exception as exc:
        print("[s3] boto3 client creation failed:", exc)
        return None


def _is_not_found_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = str(error.get("Code") or "").strip()
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return True

    message = str(exc).lower()
    return any(token in message for token in ("nosuchkey", "not found", "key not found", "(404)"))


def _get_release_prefix_from_s3(s3: Any, settings: Settings) -> Optional[str]:
    if settings.release_prefix:
        print(f"[s3] using configured release prefix: {settings.release_prefix.rstrip('/')}")
        return settings.release_prefix.rstrip("/")
    key = "releases/current"
    try:
        response = s3.get_object(Bucket=settings.s3_bucket_vectors, Key=key)
        body = response["Body"].read().decode("utf-8")
        prefix = body.strip()
        if prefix:
            print(f"[s3] resolved release prefix from {key}: {prefix.rstrip('/')}")
        return prefix.rstrip("/") if prefix else None
    except Exception as exc:
        if _is_not_found_error(exc):
            print(f"[s3] release pointer {key} not found; using bucket root artifacts")
        else:
            print(f"[s3] failed to resolve release prefix from {key}: {exc}")
        return None


def _download_to_temp(s3: Any, bucket: str, key: str, target_path: Path) -> Path:
    tmp_path = target_path.with_name(f".{target_path.name}.download")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    s3.download_file(bucket, key, str(tmp_path))
    return tmp_path


def _download_required_artifacts(s3: Any, settings: Settings, prefix: Optional[str]) -> bool:
    key_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
    label = f"prefix {prefix}" if prefix else "bucket root"
    downloaded: dict[str, Path] = {}

    try:
        for filename, target_path in (
            ("meta.json", settings.meta_json_path),
            ("index.faiss", settings.faiss_index_path),
        ):
            key = f"{key_prefix}{filename}"
            downloaded[filename] = _download_to_temp(s3, settings.s3_bucket_vectors, key, target_path)
    except Exception as exc:
        for tmp_path in downloaded.values():
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"[s3] failed to download required artifacts from {label}: {exc}")
        return False

    for filename, tmp_path in downloaded.items():
        target_path = settings.out_dir / filename
        tmp_path.replace(target_path)

    for filename in settings.optional_artifacts:
        key = f"{key_prefix}{filename}"
        target_path = settings.out_dir / filename
        try:
            s3.download_file(settings.s3_bucket_vectors, key, str(target_path))
        except Exception as exc:
            if _is_not_found_error(exc):
                print(f"[s3] optional artifact missing at {key}; continuing")
            else:
                print(f"[s3] failed to download optional artifact from {key}: {exc}")

    print(f"[s3] downloaded artifacts from {label}")
    return True


def _get_lightrag_release_prefix_from_s3(s3: Any, settings: Settings) -> Optional[str]:
    root_prefix = settings.lightrag_s3_prefix.strip().strip("/")
    key = f"{root_prefix}/releases/current" if root_prefix else "releases/current"
    try:
        response = s3.get_object(Bucket=settings.s3_bucket_vectors, Key=key)
        body = response["Body"].read().decode("utf-8")
        prefix = body.strip().strip("/")
        if prefix:
            print(f"[s3] resolved LightRAG release prefix from {key}: {prefix}")
        return prefix or None
    except Exception as exc:
        if _is_not_found_error(exc):
            print(f"[s3] LightRAG release pointer {key} not found")
        else:
            print(f"[s3] failed to resolve LightRAG release prefix from {key}: {exc}")
        return None


def _download_lightrag_artifacts(s3: Any, settings: Settings, prefix: str) -> bool:
    manifest_key = f"{prefix.rstrip('/')}/manifest.json"
    try:
        response = s3.get_object(Bucket=settings.s3_bucket_vectors, Key=manifest_key)
        manifest = response["Body"].read().decode("utf-8")
        import json

        parsed = json.loads(manifest)
        files = [str(name) for name in parsed.get("files") or [] if str(name).strip()]
    except Exception as exc:
        print(f"[s3] failed to load LightRAG manifest from {manifest_key}: {exc}")
        return False

    required_missing = [name for name in LIGHTRAG_REQUIRED_ARTIFACTS if name not in files]
    if required_missing:
        print(f"[s3] LightRAG manifest is missing required files: {', '.join(required_missing)}")
        return False

    tmp_dir = settings.lightrag_dir.with_name(f".{settings.lightrag_dir.name}.download")
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename in files:
            key = f"{prefix.rstrip('/')}/{filename}"
            target_path = tmp_dir / filename
            s3.download_file(settings.s3_bucket_vectors, key, str(target_path))
    except Exception as exc:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[s3] failed to download LightRAG artifacts from prefix {prefix}: {exc}")
        return False

    import shutil

    if settings.lightrag_dir.exists():
        shutil.rmtree(settings.lightrag_dir)
    tmp_dir.replace(settings.lightrag_dir)
    print(f"[s3] downloaded LightRAG artifacts from prefix {prefix}")
    return True


def _have_local_required_artifacts(settings: Settings) -> bool:
    return all((settings.out_dir / filename).exists() for filename in REQUIRED_ARTIFACTS)


def _have_local_lightrag_artifacts(settings: Settings) -> bool:
    return all((settings.lightrag_dir / filename).exists() for filename in LIGHTRAG_REQUIRED_ARTIFACTS)


def ensure_local_artifacts(settings: Settings) -> None:
    s3 = _create_s3_client(settings)
    if s3 is None:
        if _have_local_required_artifacts(settings):
            print("[startup] S3 client not available; using existing local artifacts")
            return
        print("[s3] S3 client not available; expecting local out/ to contain artifacts.")
        return

    prefix = _get_release_prefix_from_s3(s3, settings)
    if prefix and _download_required_artifacts(s3, settings, prefix):
        return

    if _download_required_artifacts(s3, settings, prefix=None):
        return

    if _have_local_required_artifacts(settings):
        print("[startup] S3 refresh failed; using existing local artifacts")
        return

    print("[startup] required artifacts are missing locally and could not be downloaded from S3")


def ensure_local_lightrag_artifacts(settings: Settings) -> None:
    s3 = _create_s3_client(settings)
    if s3 is None:
        if _have_local_lightrag_artifacts(settings):
            print("[startup] S3 client not available; using existing local LightRAG artifacts")
            return
        print("[s3] S3 client not available; expecting local LightRAG artifacts.")
        return

    prefix = _get_lightrag_release_prefix_from_s3(s3, settings)
    if prefix and _download_lightrag_artifacts(s3, settings, prefix):
        return

    if _have_local_lightrag_artifacts(settings):
        print("[startup] LightRAG refresh failed; using existing local artifacts")
        return

    print("[startup] LightRAG artifacts are missing locally and could not be downloaded from S3")

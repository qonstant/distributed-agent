from __future__ import annotations

from typing import Any, Optional

from rag_service.infrastructure.config import Settings


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
        print(f"[s3] no release prefix found at {key}: {exc}")
        return None


def ensure_local_artifacts(settings: Settings) -> None:
    need_meta = not settings.meta_json_path.exists()
    need_index = not settings.faiss_index_path.exists()
    if not (need_meta or need_index):
        print("[startup] local artifacts present, skipping S3 download")
        return

    s3 = _create_s3_client(settings)
    if s3 is None:
        print("[s3] S3 client not available; expecting local out/ to contain artifacts.")
        return

    prefix = _get_release_prefix_from_s3(s3, settings)
    if prefix:
        try:
            s3.download_file(
                settings.s3_bucket_vectors,
                f"{prefix}/meta.json",
                str(settings.meta_json_path),
            )
            s3.download_file(
                settings.s3_bucket_vectors,
                f"{prefix}/index.faiss",
                str(settings.faiss_index_path),
            )
            print("[s3] downloaded artifacts from prefix", prefix)
            return
        except Exception as exc:
            print(f"[s3] failed to download artifacts from prefix {prefix}: {exc}")

    try:
        s3.download_file(settings.s3_bucket_vectors, "meta.json", str(settings.meta_json_path))
        s3.download_file(
            settings.s3_bucket_vectors,
            "index.faiss",
            str(settings.faiss_index_path),
        )
        print("[s3] downloaded artifacts from bucket root")
    except Exception as exc:
        print(f"[s3] failed to download artifacts from bucket root: {exc}")

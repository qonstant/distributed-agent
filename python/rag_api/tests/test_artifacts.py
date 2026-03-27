from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_service.infrastructure.artifacts import ensure_local_artifacts
from rag_service.infrastructure.config import Settings


class FakeS3:
    def __init__(
        self,
        downloads: dict[str, str] | None = None,
        fail_keys: set[str] | None = None,
        download_failures: dict[str, Exception] | None = None,
        get_object_failures: dict[str, Exception] | None = None,
    ) -> None:
        self.downloads = downloads or {}
        self.fail_keys = fail_keys or set()
        self.download_failures = download_failures or {}
        self.get_object_failures = get_object_failures or {}
        self.download_calls: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - matches boto3 signature
        raise self.get_object_failures.get(Key, RuntimeError(f"missing key: {Key}"))

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.download_calls.append(key)
        if key in self.download_failures:
            raise self.download_failures[key]
        if key in self.fail_keys:
            raise RuntimeError(f"download failed for {key}")
        content = self.downloads.get(key)
        if content is None:
            raise RuntimeError(f"missing object {key}")
        Path(filename).write_text(content, encoding="utf-8")


class ArtifactsTests(unittest.TestCase):
    def make_settings(self, out_dir: Path, *, release_prefix: str | None = None) -> Settings:
        return Settings(
            openai_api_key="test-key",
            s3_endpoint="s3.example.internal",
            s3_access_key="key",
            s3_secret="secret",
            s3_bucket_vectors="vectordb",
            s3_use_ssl=False,
            s3_verify=False,
            release_prefix=release_prefix,
            out_dir=out_dir,
            meta_json_path=out_dir / "meta.json",
            faiss_index_path=out_dir / "index.faiss",
            optional_artifacts=[],
        )

    def test_refreshes_existing_local_required_artifacts_from_bucket_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            settings = self.make_settings(out_dir)
            settings.meta_json_path.write_text("old-meta", encoding="utf-8")
            settings.faiss_index_path.write_text("old-index", encoding="utf-8")

            fake_s3 = FakeS3(
                downloads={
                    "meta.json": "new-meta",
                    "index.faiss": "new-index",
                }
            )

            with patch("rag_service.infrastructure.artifacts._create_s3_client", return_value=fake_s3):
                with patch("rag_service.infrastructure.artifacts._get_release_prefix_from_s3", return_value=None):
                    ensure_local_artifacts(settings)

            self.assertEqual(settings.meta_json_path.read_text(encoding="utf-8"), "new-meta")
            self.assertEqual(settings.faiss_index_path.read_text(encoding="utf-8"), "new-index")
            self.assertEqual(fake_s3.download_calls, ["meta.json", "index.faiss"])

    def test_prefers_configured_release_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            settings = self.make_settings(out_dir, release_prefix="releases/2026-03-27")

            fake_s3 = FakeS3(
                downloads={
                    "releases/2026-03-27/meta.json": "prefixed-meta",
                    "releases/2026-03-27/index.faiss": "prefixed-index",
                    "meta.json": "root-meta",
                    "index.faiss": "root-index",
                }
            )

            with patch("rag_service.infrastructure.artifacts._create_s3_client", return_value=fake_s3):
                ensure_local_artifacts(settings)

            self.assertEqual(settings.meta_json_path.read_text(encoding="utf-8"), "prefixed-meta")
            self.assertEqual(settings.faiss_index_path.read_text(encoding="utf-8"), "prefixed-index")
            self.assertEqual(
                fake_s3.download_calls,
                ["releases/2026-03-27/meta.json", "releases/2026-03-27/index.faiss"],
            )

    def test_falls_back_to_existing_local_artifacts_when_s3_download_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            settings = self.make_settings(out_dir)
            settings.meta_json_path.write_text("local-meta", encoding="utf-8")
            settings.faiss_index_path.write_text("local-index", encoding="utf-8")

            fake_s3 = FakeS3(fail_keys={"meta.json", "index.faiss"})

            with patch("rag_service.infrastructure.artifacts._create_s3_client", return_value=fake_s3):
                with patch("rag_service.infrastructure.artifacts._get_release_prefix_from_s3", return_value=None):
                    ensure_local_artifacts(settings)

            self.assertEqual(settings.meta_json_path.read_text(encoding="utf-8"), "local-meta")
            self.assertEqual(settings.faiss_index_path.read_text(encoding="utf-8"), "local-index")

    def test_missing_release_pointer_logs_friendly_fallback_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            settings = self.make_settings(out_dir)

            fake_s3 = FakeS3(
                downloads={
                    "meta.json": "new-meta",
                    "index.faiss": "new-index",
                },
                get_object_failures={
                    "releases/current": RuntimeError(
                        "An error occurred (NoSuchKey) when calling the GetObject operation: Key not found"
                    )
                },
            )

            with patch("builtins.print") as mock_print:
                with patch("rag_service.infrastructure.artifacts._create_s3_client", return_value=fake_s3):
                    ensure_local_artifacts(settings)

            log_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
            self.assertIn(
                "[s3] release pointer releases/current not found; using bucket root artifacts",
                log_lines,
            )
            self.assertFalse(any("NoSuchKey" in line for line in log_lines))

    def test_missing_optional_artifact_logs_friendly_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            settings = self.make_settings(out_dir)
            settings = Settings(
                **{
                    **settings.__dict__,
                    "optional_artifacts": ["manifest.json"],
                }
            )

            fake_s3 = FakeS3(
                downloads={
                    "meta.json": "new-meta",
                    "index.faiss": "new-index",
                },
                download_failures={
                    "manifest.json": RuntimeError(
                        "An error occurred (404) when calling the HeadObject operation: Not Found"
                    )
                },
            )

            with patch("builtins.print") as mock_print:
                with patch("rag_service.infrastructure.artifacts._create_s3_client", return_value=fake_s3):
                    with patch("rag_service.infrastructure.artifacts._get_release_prefix_from_s3", return_value=None):
                        ensure_local_artifacts(settings)

            log_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
            self.assertIn("[s3] optional artifact missing at manifest.json; continuing", log_lines)
            self.assertFalse(any("(404)" in line for line in log_lines))


if __name__ == "__main__":
    unittest.main()

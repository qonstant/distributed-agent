from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .lightrag_artifacts import upload_lightrag_release


JobMode = Literal["full", "continue"]
JobStatus = Literal["idle", "running", "succeeded", "failed"]


@dataclass
class LightRAGJob:
    status: JobStatus = "idle"
    mode: JobMode | None = None
    command: str = ""
    cwd: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    error: str = ""
    release: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)


class LightRAGJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job = LightRAGJob()
        self._thread: threading.Thread | None = None

    @staticmethod
    def repo_dir() -> Path:
        return Path(os.getenv("LIGHTRAG_REPO_DIR", "/workspace/distributed-agent"))

    @staticmethod
    def working_dir() -> Path:
        return Path(os.getenv("LIGHTRAG_WORK_DIR", "python/RAG/out/lightrag"))

    @classmethod
    def resolved_working_dir(cls) -> Path:
        working_dir = cls.working_dir()
        if working_dir.is_absolute():
            return working_dir
        return cls.repo_dir() / working_dir

    @staticmethod
    def upload_enabled() -> bool:
        value = os.getenv("LIGHTRAG_UPLOAD_ENABLED", "true").strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def full_command() -> str:
        return os.getenv("LIGHTRAG_BUILD_FULL_COMMAND", "make lightrag-s3-full PYTHON=/opt/lightrag-venv/bin/python")

    @staticmethod
    def continue_command() -> str:
        return os.getenv("LIGHTRAG_BUILD_CONTINUE_COMMAND", "make lightrag-s3-continue PYTHON=/opt/lightrag-venv/bin/python")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._job)
        payload["config"] = {
            "repo_dir": str(self.repo_dir()),
            "working_dir": str(self.working_dir()),
            "resolved_working_dir": str(self.resolved_working_dir()),
            "upload_enabled": self.upload_enabled(),
            "s3_prefix": os.getenv("LIGHTRAG_S3_PREFIX", "lightrag"),
            "source_prefix": os.getenv("LIGHTRAG_SOURCE_PREFIX", "italy"),
            "markdown_prefix": os.getenv("LIGHTRAG_MARKDOWN_S3_PREFIX", "markdowns"),
            "full_command": self.full_command(),
            "continue_command": self.continue_command(),
        }
        return payload

    def start(self, mode: JobMode) -> bool:
        with self._lock:
            if self._job.status == "running":
                return False
            command = self.full_command() if mode == "full" else self.continue_command()
            cwd = str(self.repo_dir())
            self._job = LightRAGJob(
                status="running",
                mode=mode,
                command=command,
                cwd=cwd,
                started_at=self._now(),
                logs=[f"[admin] starting LightRAG {mode} build"],
            )
            self._thread = threading.Thread(target=self._run_job, args=(command, cwd), daemon=True)
            self._thread.start()
            return True

    def _run_job(self, command: str, cwd: str) -> None:
        try:
            self._append_log(f"[admin] cwd={cwd}")
            self._append_log(f"[admin] command={command}")
            process = subprocess.Popen(
                command,
                cwd=cwd if Path(cwd).exists() else None,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(line.rstrip())
            returncode = process.wait()
            self._set_returncode(returncode)
            if returncode != 0:
                self._finish("failed", error=f"build command exited with code {returncode}")
                return

            if self.upload_enabled():
                self._append_log("[admin] build finished; uploading staged LightRAG release to S3")
                release = upload_lightrag_release(self.resolved_working_dir())
                self._set_release(release)
                self._append_log(
                    "[admin] promoted LightRAG release "
                    f"{release['release_prefix']} via {release['pointer_key']}"
                )
            else:
                self._append_log("[admin] upload disabled; leaving artifacts local only")

            self._finish("succeeded")
        except Exception as exc:
            self._finish("failed", error=str(exc))

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._job.logs.append(line)
            self._job.logs = self._job.logs[-500:]

    def _set_returncode(self, returncode: int) -> None:
        with self._lock:
            self._job.returncode = returncode

    def _set_release(self, release: dict[str, Any]) -> None:
        with self._lock:
            self._job.release = release

    def _finish(self, status: JobStatus, *, error: str = "") -> None:
        with self._lock:
            self._job.status = status
            self._job.error = error
            self._job.finished_at = self._now()
            if error:
                self._job.logs.append(f"[admin] error: {error}")
            self._job.logs.append(f"[admin] job {status}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


_manager = LightRAGJobManager()


def get_lightrag_job_manager() -> LightRAGJobManager:
    return _manager

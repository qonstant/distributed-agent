from __future__ import annotations

import asyncio
import json
import re
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import threading

import numpy as np

from rag_service.domain.models import RetrievedHit, normalize_language
from rag_service.infrastructure.artifacts import ensure_local_lightrag_artifacts
from rag_service.infrastructure.config import Settings
from rag_service.infrastructure.openai_gateway import OpenAIGateway


SOURCE_RE = re.compile(r"SOURCE_FILE:\s*([^\s`]+)", flags=re.IGNORECASE)
PAGE_RE = re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->", flags=re.IGNORECASE)
DOC_TYPE_RE = re.compile(r"DOC_TYPE:\s*([^\n`]+)", flags=re.IGNORECASE)
COUNTRY_RE = re.compile(r"COUNTRY:\s*([^\n`]+)", flags=re.IGNORECASE)
LANGUAGE_RE = re.compile(r"LANGUAGE:\s*([^\n`]+)", flags=re.IGNORECASE)


class SimpleCharTokenizer:
    """Small offline tokenizer so LightRAG does not try to fetch tokenizer data."""

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(char) for char in str(text or "")]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "".join(chr(int(token)) for token in tokens)


class AsyncLoopRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _language_from_source_file(source_file: str) -> str:
    match = re.search(r"_(kk|ru|en)\.pdf$", source_file.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _page_from_text(text: str) -> str:
    match = PAGE_RE.search(text or "")
    return match.group(1) if match else ""


def _source_from_text(text: str) -> str:
    match = SOURCE_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _strip_lightrag_markup(text: str) -> str:
    cleaned = SOURCE_RE.sub("", text or "")
    cleaned = DOC_TYPE_RE.sub("", cleaned)
    cleaned = COUNTRY_RE.sub("", cleaned)
    cleaned = LANGUAGE_RE.sub("", cleaned)
    cleaned = PAGE_RE.sub("", cleaned)
    cleaned = re.sub(r"TITLE:\s*[^\n`]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"The SOURCE_FILE line identifies[^\n]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except Exception:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = start + max(end, 1)
    return objects


def _chunk_contents_from_context(context: str) -> List[str]:
    objects = _extract_json_objects(context)
    contents = [str(item.get("content") or "").strip() for item in objects if str(item.get("content") or "").strip()]
    if contents:
        return contents

    # Fallback for non-JSON LightRAG contexts: split at source markers.
    parts = re.split(r"(?=SOURCE_FILE:\s*)", context or "")
    return [part.strip() for part in parts if _source_from_text(part)]


def _source_files_from_chunks(chunks_path: Path) -> List[str]:
    if not chunks_path.exists():
        return []
    try:
        parsed = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    files: List[str] = []
    seen = set()
    for payload in parsed.values() if isinstance(parsed, dict) else []:
        if not isinstance(payload, dict):
            continue
        source_file = _source_from_text(str(payload.get("content") or ""))
        if source_file and source_file not in seen:
            files.append(source_file)
            seen.add(source_file)
    return files


class LightRAGMetadataStore:
    requires_query_embedding = False

    def __init__(self, rag: Any, working_dir: Path, mode: str, source_files: List[str], runner: AsyncLoopRunner) -> None:
        self._rag = rag
        self._working_dir = working_dir
        self._mode = mode
        self._source_files = source_files
        self._runner = runner

    @classmethod
    def load(cls, settings: Settings) -> "LightRAGMetadataStore":
        try:
            ensure_local_lightrag_artifacts(settings)
        except Exception as exc:
            print("[startup] LightRAG S3 fetch attempt raised an unexpected error:", exc)
            traceback.print_exc()

        missing = [
            filename
            for filename in ("graph_chunk_entity_relation.graphml", "kv_store_text_chunks.json", "vdb_chunks.json")
            if not (settings.lightrag_dir / filename).exists()
        ]
        if missing:
            raise RuntimeError(
                f"LightRAG artifacts missing in {settings.lightrag_dir.resolve()}: {', '.join(missing)}"
            )

        try:
            from lightrag import LightRAG
            from lightrag.utils import wrap_embedding_func_with_attrs
        except ImportError as exc:
            missing_name = getattr(exc, "name", "") or str(exc)
            raise RuntimeError(f"LightRAG dependency missing ({missing_name})") from exc

        gateway = OpenAIGateway(
            SimpleNamespace(
                openai_api_key=settings.openai_api_key,
                class_model=settings.class_model,
                llm_model=settings.lightrag_llm_model,
                embed_model=settings.lightrag_embed_model,
            )
        )

        @wrap_embedding_func_with_attrs(
            embedding_dim=settings.lightrag_embed_dim,
            max_token_size=8192,
            model_name=settings.lightrag_embed_model,
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            response = await asyncio.to_thread(
                gateway._client.embeddings.create,
                model=settings.lightrag_embed_model,
                input=texts,
            )
            return np.array([item.embedding for item in response.data], dtype=np.float32)

        async def llm_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: list[dict[str, str]] = [],
            keyword_extraction: bool = False,
            **kwargs: Any,
        ) -> str:
            parts = []
            if system_prompt:
                parts.append(f"System instructions:\n{system_prompt}")
            for message in history_messages or []:
                content = str(message.get("content") or message.get("text") or "").strip()
                role = str(message.get("role") or "user").strip()
                if content:
                    parts.append(f"{role}: {content}")
            if keyword_extraction:
                parts.append("Task note: return the keyword extraction result requested by LightRAG.")
            parts.append(prompt)
            response = await asyncio.to_thread(
                gateway._client.responses.create,
                model=settings.lightrag_llm_model,
                input="\n\n".join(parts),
                max_output_tokens=int(kwargs.get("max_tokens") or kwargs.get("max_output_tokens") or 2048),
                temperature=float(kwargs.get("temperature", 0.0) or 0.0),
            )
            return gateway._resp_to_text(response) or ""

        rag = LightRAG(
            working_dir=str(settings.lightrag_dir),
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            tokenizer=SimpleCharTokenizer(),
        )
        runner = AsyncLoopRunner()
        runner.run(rag.initialize_storages())
        source_files = _source_files_from_chunks(settings.lightrag_dir / "kv_store_text_chunks.json")
        print(
            f"[lightrag-runtime] loaded LightRAG store dir={settings.lightrag_dir} "
            f"mode={settings.lightrag_query_mode} files={len(source_files)}"
        )
        return cls(
            rag=rag,
            working_dir=settings.lightrag_dir,
            mode=settings.lightrag_query_mode,
            source_files=source_files,
            runner=runner,
        )

    def search(
        self,
        query_embedding: Any = None,
        k: int = 64,
        *,
        country: str = "",
        language: str = "",
        query_text: str = "",
        verbose: bool = False,
    ) -> List[RetrievedHit]:
        del query_embedding, country, verbose
        query = (query_text or "").strip()
        if not query:
            return []

        try:
            from lightrag import QueryParam
        except ImportError as exc:
            raise RuntimeError("LightRAG dependency missing at query time") from exc

        context = self._runner.run(
            self._rag.aquery(
                query,
                param=QueryParam(
                    mode=self._mode,
                    only_need_context=True,
                    top_k=max(int(k or 1), 10),
                    enable_rerank=False,
                ),
            )
        )
        return self._hits_from_context(str(context or ""), k=max(1, int(k or 1)), language=language)

    def _hits_from_context(self, context: str, *, k: int, language: str) -> List[RetrievedHit]:
        target_language = normalize_language(language)
        target_language = "" if target_language == "other" else target_language
        chunks = _chunk_contents_from_context(context)
        hits: List[RetrievedHit] = []
        filtered_hits: List[RetrievedHit] = []
        seen = set()
        for index, content in enumerate(chunks, start=1):
            source_file = _source_from_text(content)
            if not source_file:
                continue
            page = _page_from_text(content)
            key = (source_file, page, _strip_lightrag_markup(content)[:120])
            if key in seen:
                continue
            seen.add(key)
            language_hint = normalize_language(_first_match(LANGUAGE_RE, content) or _language_from_source_file(source_file))
            score = 1.0 / float(index)
            if target_language and language_hint == target_language:
                score += 0.25
            meta: Dict[str, Any] = {
                "source_file": source_file,
                "filename": source_file,
                "page": page,
                "text": _strip_lightrag_markup(content),
                "country": _first_match(COUNTRY_RE, content),
                "language": language_hint,
                "doc_type": _first_match(DOC_TYPE_RE, content).strip().lower(),
                "retrieval_backend": "lightrag",
                "lightrag_mode": self._mode,
            }
            hit = RetrievedHit(score=score, nid=-index, meta=meta)
            hits.append(hit)
            if target_language and language_hint == target_language:
                filtered_hits.append(hit)

        selected_hits = filtered_hits if target_language else hits
        selected_hits.sort(key=lambda hit: hit.score, reverse=True)
        return selected_hits[:k]

from __future__ import annotations

import asyncio
import json
import re
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
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
TOKEN_RE = re.compile(r"[0-9a-zа-яёәғқңөұүһі]{2,}", flags=re.IGNORECASE)
FILE_SUMMARIES_FILENAME = "file_summaries.json"


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
    match = re.search(r"_(kk|kz|ru|en|eng)\.pdf$", source_file.strip(), flags=re.IGNORECASE)
    if not match:
        return ""
    suffix = match.group(1).lower()
    aliases = {
        "eng": "en",
        "kz": "kk",
    }
    return aliases.get(suffix, suffix)


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


def _summary_preview(text: str, *, limit: int = 320) -> str:
    cleaned = _strip_lightrag_markup(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _summary_is_informative(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    collapsed = cleaned.replace(".", "").replace(" ", "")
    if not collapsed:
        return False
    return len(cleaned) >= 24 and len(_metadata_tokens(cleaned)) >= 4


def _metadata_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(text or ""))}


def _text_match_score(query: str, text: str) -> float:
    query_tokens = _metadata_tokens(query)
    if not query_tokens:
        return 0.0

    normalized_query = " ".join(query_tokens)
    normalized_text = _strip_lightrag_markup(text).lower()
    if normalized_query and normalized_query in normalized_text:
        return 1.0

    text_tokens = _metadata_tokens(text)
    if not text_tokens:
        return 0.0

    overlap = len(query_tokens & text_tokens)
    if overlap == 0:
        return 0.0

    denominator = max(1, min(len(query_tokens), len(text_tokens)))
    return min(1.0, overlap / denominator)


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


def _load_chunks_by_source_file(chunks_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not chunks_path.exists():
        return {}
    try:
        parsed = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    chunks_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in parsed.values() if isinstance(parsed, dict) else []:
        if not isinstance(payload, dict):
            continue
        content = str(payload.get("content") or "").strip()
        source_file = _source_from_text(content) or str(payload.get("file_path") or "").strip()
        if not source_file:
            continue
        text = _strip_lightrag_markup(content)
        chunk = {
            "source_file": source_file,
            "page": _page_from_text(content),
            "text": text,
            "language": normalize_language(_first_match(LANGUAGE_RE, content) or _language_from_source_file(source_file)),
            "doc_type": _first_match(DOC_TYPE_RE, content).strip().lower(),
            "country": _first_match(COUNTRY_RE, content),
            "chunk_order_index": int(payload.get("chunk_order_index") or 0),
        }
        chunks_by_file[source_file].append(chunk)

    for chunks in chunks_by_file.values():
        chunks.sort(key=lambda item: (int(item.get("chunk_order_index") or 0), str(item.get("page") or "")))
    return dict(chunks_by_file)


def _load_file_catalog(
    working_dir: Path,
    chunks_by_source_file: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    generated_summary_path = working_dir / FILE_SUMMARIES_FILENAME
    if generated_summary_path.exists():
        try:
            parsed = json.loads(generated_summary_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = {}
        files = parsed.get("files") if isinstance(parsed, dict) else {}
        if isinstance(files, dict):
            for source_file, payload in files.items():
                if not isinstance(payload, dict):
                    continue
                normalized_source_file = str(payload.get("source_file") or source_file or "").strip()
                if not normalized_source_file:
                    continue
                catalog[normalized_source_file] = {
                    "source_file": normalized_source_file,
                    "title": str(payload.get("title") or Path(normalized_source_file).name),
                    "summary": str(payload.get("summary") or "").strip(),
                    "language": normalize_language(str(payload.get("language") or "")),
                    "doc_type": str(payload.get("doc_type") or "").strip().lower(),
                    "country": str(payload.get("country") or "").strip().lower(),
                    "pages": [
                        str(item.get("page") or "").strip()
                        for item in (payload.get("page_snippets") or [])
                        if isinstance(item, dict) and str(item.get("page") or "").strip()
                    ],
                }

    doc_status_path = working_dir / "kv_store_doc_status.json"
    if doc_status_path.exists():
        try:
            parsed = json.loads(doc_status_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            for payload in parsed.values():
                if not isinstance(payload, dict):
                    continue
                source_file = str(payload.get("file_path") or "").strip()
                if not source_file:
                    continue
                summary_text = str(payload.get("content_summary") or "").strip()
                summary = _summary_preview(summary_text)
                entry = catalog.setdefault(
                    source_file,
                    {
                        "source_file": source_file,
                        "title": Path(source_file).name,
                        "summary": "",
                        "language": "",
                        "doc_type": "",
                        "country": "",
                        "pages": [],
                    },
                )
                if not _summary_is_informative(str(entry.get("summary") or "")):
                    entry["summary"] = summary
                if not entry.get("language"):
                    entry["language"] = normalize_language(_first_match(LANGUAGE_RE, summary_text) or _language_from_source_file(source_file))
                if not entry.get("doc_type"):
                    entry["doc_type"] = _first_match(DOC_TYPE_RE, summary_text).strip().lower()
                if not entry.get("country"):
                    entry["country"] = _first_match(COUNTRY_RE, summary_text)

    for source_file, chunks in chunks_by_source_file.items():
        entry = catalog.setdefault(
            source_file,
            {
                "source_file": source_file,
                "title": Path(source_file).name,
                "summary": "",
                "language": "",
                "doc_type": "",
                "country": "",
                "pages": [],
            },
        )
        if not _summary_is_informative(str(entry.get("summary") or "")):
            entry["summary"] = _summary_preview(" ".join(chunk.get("text") or "" for chunk in chunks[:2]))
        if not entry.get("language"):
            entry["language"] = next((chunk.get("language") or "" for chunk in chunks if chunk.get("language")), "")
        if not entry.get("doc_type"):
            entry["doc_type"] = next((chunk.get("doc_type") or "" for chunk in chunks if chunk.get("doc_type")), "")
        if not entry.get("country"):
            entry["country"] = next((chunk.get("country") or "" for chunk in chunks if chunk.get("country")), "")
        pages = []
        seen_pages = set()
        for chunk in chunks:
            page = str(chunk.get("page") or "").strip()
            if page and page not in seen_pages:
                pages.append(page)
                seen_pages.add(page)
        entry["pages"] = pages

    return catalog


def _diversify_hits_by_file(hits: List[RetrievedHit], limit: int) -> List[RetrievedHit]:
    if limit <= 0:
        return []

    ordered = sorted(hits, key=lambda item: item.score, reverse=True)
    selected: List[RetrievedHit] = []
    seen_chunks = set()
    seen_files = set()

    for hit in ordered:
        source_file = str(hit.meta.get("source_file") or hit.meta.get("filename") or "").strip()
        page = str(hit.meta.get("page") or "").strip()
        text = str(hit.meta.get("text") or hit.meta.get("md") or "").strip()
        chunk_key = (source_file, page, text[:160])
        if not source_file or source_file in seen_files or chunk_key in seen_chunks:
            continue
        selected.append(hit)
        seen_files.add(source_file)
        seen_chunks.add(chunk_key)
        if len(selected) >= limit:
            return selected

    for hit in ordered:
        source_file = str(hit.meta.get("source_file") or hit.meta.get("filename") or "").strip()
        page = str(hit.meta.get("page") or "").strip()
        text = str(hit.meta.get("text") or hit.meta.get("md") or "").strip()
        chunk_key = (source_file, page, text[:160])
        if chunk_key in seen_chunks:
            continue
        selected.append(hit)
        seen_chunks.add(chunk_key)
        if len(selected) >= limit:
            break

    return selected


class LightRAGMetadataStore:
    requires_query_embedding = False

    def __init__(
        self,
        rag: Any,
        working_dir: Path,
        mode: str,
        source_files: List[str],
        runner: AsyncLoopRunner,
        chunks_by_source_file: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        file_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._rag = rag
        self._working_dir = working_dir
        self._mode = mode
        self._source_files = source_files
        self._runner = runner
        self._chunks_by_source_file = chunks_by_source_file or {}
        self._file_catalog = file_catalog or {}

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
        chunks_by_source_file = _load_chunks_by_source_file(settings.lightrag_dir / "kv_store_text_chunks.json")
        source_files = list(chunks_by_source_file) or _source_files_from_chunks(settings.lightrag_dir / "kv_store_text_chunks.json")
        file_catalog = _load_file_catalog(settings.lightrag_dir, chunks_by_source_file)
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
            chunks_by_source_file=chunks_by_source_file,
            file_catalog=file_catalog,
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
                score += 0.75
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

        # Prefer same-language chunks via score boost, but do not hard-drop other
        # languages. Some topics may exist only in one language, and returning a
        # useful cross-language chunk is better than returning nothing.
        #
        # Also diversify by file before trimming so several pages from one file
        # do not crowd out other relevant documents in the top-k passed onward
        # to sufficiency and answer generation.
        return _diversify_hits_by_file(hits, k)

    def refine_results(
        self,
        *,
        query_text: str,
        initial_hits: List[RetrievedHit],
        k: int,
        language: str,
        preferred_source: Optional[str] = None,
    ) -> Tuple[List[RetrievedHit], Dict[str, Any]]:
        if not initial_hits:
            return initial_hits, {}

        target_language = normalize_language(language)
        target_language = "" if target_language == "other" else target_language
        preferred_source = str(preferred_source or "").strip()

        file_scores: Dict[str, Dict[str, Any]] = {}
        for hit in initial_hits:
            source_file = str(hit.meta.get("source_file") or hit.meta.get("filename") or "").strip()
            if not source_file:
                continue
            state = file_scores.setdefault(
                source_file,
                {
                    "source_file": source_file,
                    "sum_score": 0.0,
                    "best_score": 0.0,
                    "pages": [],
                    "snippets": [],
                },
            )
            state["sum_score"] += float(hit.score)
            state["best_score"] = max(float(state["best_score"]), float(hit.score))
            page = str(hit.meta.get("page") or "").strip()
            if page and page not in state["pages"]:
                state["pages"].append(page)
            snippet = str(hit.meta.get("text") or hit.meta.get("md") or "").strip()
            if snippet:
                state["snippets"].append(snippet)

        all_source_files = set(file_scores)
        all_source_files.update(self._file_catalog)
        all_source_files.update(self._chunks_by_source_file)

        candidates: List[Dict[str, Any]] = []
        for source_file in sorted(all_source_files):
            state = dict(
                file_scores.get(source_file)
                or {
                    "source_file": source_file,
                    "sum_score": 0.0,
                    "best_score": 0.0,
                    "pages": [],
                    "snippets": [],
                }
            )
            catalog = dict(self._file_catalog.get(source_file) or {})
            summary = str(catalog.get("summary") or "")
            file_language = str(catalog.get("language") or "")
            if not file_language:
                file_language = str(
                    next(
                        (
                            chunk.get("language") or ""
                            for chunk in self._chunks_by_source_file.get(source_file, [])
                            if chunk.get("language")
                        ),
                        "",
                    )
                )
            snippet_parts = [str(part) for part in state.get("snippets") or [] if str(part).strip()]
            if not snippet_parts:
                snippet_parts.extend(
                    str(chunk.get("text") or "").strip()
                    for chunk in self._chunks_by_source_file.get(source_file, [])[:2]
                    if str(chunk.get("text") or "").strip()
                )
            snippet_text = " ".join(snippet_parts)
            summary_match = _text_match_score(query_text, summary)
            snippet_match = _text_match_score(query_text, snippet_text)
            name_match = _text_match_score(query_text, source_file)
            summary_score = (summary_match * 2.5) + (snippet_match * 1.2) + (name_match * 0.6)
            score = (float(state["sum_score"]) * 0.85) + (float(state["best_score"]) * 0.25)
            score += (summary_match * 2.2) + (snippet_match * 1.4) + (name_match * 0.5)
            if target_language and file_language == target_language:
                score += 0.35
                summary_score += 0.2
            if preferred_source and source_file == preferred_source:
                score += 2.0
                summary_score += 2.0
            if (
                float(state["sum_score"]) <= 0.0
                and summary_score <= 0.0
                and (not preferred_source or source_file != preferred_source)
            ):
                continue
            candidates.append(
                {
                    **catalog,
                    **state,
                    "candidate_score": score,
                    "summary_score": summary_score,
                }
            )

        candidates.sort(key=lambda item: float(item.get("candidate_score") or 0.0), reverse=True)
        initial_ranked_files = [
            str(item.get("source_file") or "")
            for item in sorted(
                candidates,
                key=lambda item: (
                    float(item.get("sum_score") or 0.0),
                    float(item.get("best_score") or 0.0),
                    float(item.get("candidate_score") or 0.0),
                ),
                reverse=True,
            )
            if float(item.get("sum_score") or 0.0) > 0.0 and str(item.get("source_file") or "")
        ]
        summary_ranked_files = [
            str(item.get("source_file") or "")
            for item in sorted(candidates, key=lambda item: float(item.get("summary_score") or 0.0), reverse=True)
            if float(item.get("summary_score") or 0.0) > 0.0 and str(item.get("source_file") or "")
        ]
        selected_files: List[str] = []
        if preferred_source and preferred_source in all_source_files:
            selected_files.append(preferred_source)
        for source_file in initial_ranked_files[:2]:
            if source_file not in selected_files:
                selected_files.append(source_file)
        for source_file in summary_ranked_files[:3]:
            if source_file not in selected_files:
                selected_files.append(source_file)
        for item in candidates:
            source_file = str(item.get("source_file") or "")
            if source_file and source_file not in selected_files:
                selected_files.append(source_file)
            if len(selected_files) >= 4:
                break
        selected_files = selected_files[:4]
        focus_files: List[str] = []
        if preferred_source and preferred_source in all_source_files:
            focus_files.append(preferred_source)
        for source_file in summary_ranked_files[:2]:
            if source_file not in focus_files:
                focus_files.append(source_file)
        if not focus_files:
            focus_files = selected_files[:2]
        if not selected_files:
            return initial_hits, {}

        focused_hits: List[RetrievedHit] = []
        for rank, source_file in enumerate(focus_files, start=1):
            before_count = len(focused_hits)
            candidate = next((item for item in candidates if item.get("source_file") == source_file), {})
            initial_pages = set(candidate.get("pages") or [])
            summary_match = _text_match_score(query_text, str(candidate.get("summary") or ""))
            for chunk in self._chunks_by_source_file.get(source_file, []):
                chunk_text = str(chunk.get("text") or "")
                lexical = _text_match_score(query_text, chunk_text)
                page = str(chunk.get("page") or "").strip()
                score = (lexical * 1.35) + (summary_match * 1.4) + (float(candidate.get("summary_score") or 0.0) * 0.2)
                if target_language and str(chunk.get("language") or "") == target_language:
                    score += 0.15
                if page and page in initial_pages:
                    score += 0.1
                if not page:
                    score -= 0.25
                if preferred_source and source_file == preferred_source:
                    score += 0.15
                if lexical <= 0 and page not in initial_pages:
                    continue
                focused_hits.append(
                    RetrievedHit(
                        score=score,
                        nid=-(rank * 1000 + len(focused_hits) + 1),
                        meta={
                            "source_file": source_file,
                            "filename": source_file,
                            "page": page,
                            "text": chunk_text,
                            "country": str(chunk.get("country") or ""),
                            "language": str(chunk.get("language") or ""),
                            "doc_type": str(chunk.get("doc_type") or ""),
                            "retrieval_backend": "lightrag",
                            "lightrag_mode": self._mode,
                        },
                    )
                )
            if len(focused_hits) > before_count:
                break

        if not focused_hits:
            return initial_hits, {
                "selected_files": selected_files,
                "focus_files": focus_files,
                "candidates": candidates[:3],
                "focused": False,
            }

        refined_hits = _diversify_hits_by_file(focused_hits, max(1, int(k or 1)))
        return refined_hits, {
            "selected_files": selected_files,
            "focus_files": focus_files,
            "candidates": candidates[:3],
            "focused": True,
        }

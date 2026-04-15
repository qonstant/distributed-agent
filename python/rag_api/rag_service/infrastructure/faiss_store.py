from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set

import faiss
import numpy as np

from rag_service.domain.models import RetrievedHit, normalize_language
from rag_service.infrastructure.artifacts import ensure_local_artifacts
from rag_service.infrastructure.config import Settings


# Current corpus contains only Italy. Keep this as one small switch so the
# resolver can later be replaced by a DB/user-profile country selector.
DEFAULT_COUNTRY_FILTER = "italy"


def _normalize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _normalize_filter_language(value: str) -> str:
    normalized = normalize_language(value)
    return "" if normalized == "other" else normalized


def _source_file(meta: Dict[str, Any]) -> str:
    return str(meta.get("source_file") or meta.get("filename") or "")


def _meta_country(meta: Dict[str, Any]) -> str:
    country = _normalize_slug(str(meta.get("country") or ""))
    if country:
        return country

    source_file = _source_file(meta)
    if not source_file:
        return ""
    first_part = Path(source_file).parts[0] if Path(source_file).parts else ""
    return _normalize_slug(first_part)


def _meta_language(meta: Dict[str, Any]) -> str:
    language = _normalize_filter_language(str(meta.get("language") or meta.get("lang") or ""))
    if language:
        return language

    source_file = _source_file(meta)
    stem = Path(source_file).stem.lower() if source_file else ""
    match = re.search(r"_(kk|ru|en)$", stem)
    return match.group(1) if match else ""


def _chunk_key(hit: RetrievedHit) -> str:
    meta = hit.meta
    source_file = _source_file(meta)
    page = meta.get("page", "")
    chunk_index = meta.get("chunk_index", "")
    return str(meta.get("id") or f"{source_file}:{page}:{chunk_index}" or hit.nid)


def _build_search_stages(country: str, language: str) -> List[Dict[str, str]]:
    candidates = [
        {"country": country, "language": language},
        {"country": country, "language": ""},
        {"country": "", "language": language},
        {"country": "", "language": ""},
    ]
    stages: List[Dict[str, str]] = []
    seen = set()
    for filters in candidates:
        key = (filters["country"], filters["language"])
        if key in seen:
            continue
        seen.add(key)
        stages.append(filters)
    return stages


def _matches_filters(meta: Dict[str, Any], filters: Dict[str, str]) -> bool:
    country = _meta_country(meta)
    language = _meta_language(meta)
    if filters.get("country") and country != filters["country"]:
        return False
    if filters.get("language") and language != filters["language"]:
        return False
    return True


def _format_stage(filters: Dict[str, str]) -> str:
    parts = [f"{key}={value}" for key, value in filters.items() if value]
    return ", ".join(parts) if parts else "global"


def _diversify_by_file(hits: List[RetrievedHit], limit: int) -> List[RetrievedHit]:
    if limit <= 0:
        return []

    ordered = sorted(hits, key=lambda item: item.score, reverse=True)
    selected: List[RetrievedHit] = []
    seen_chunks: Set[str] = set()
    seen_files: Set[str] = set()

    for hit in ordered:
        source_file = _source_file(hit.meta)
        chunk_key = _chunk_key(hit)
        if not source_file or source_file in seen_files or chunk_key in seen_chunks:
            continue
        selected.append(hit)
        seen_files.add(source_file)
        seen_chunks.add(chunk_key)
        if len(selected) >= limit:
            return selected

    for hit in ordered:
        chunk_key = _chunk_key(hit)
        if chunk_key in seen_chunks:
            continue
        selected.append(hit)
        seen_chunks.add(chunk_key)
        if len(selected) >= limit:
            break

    return selected


class FaissMetadataStore:
    def __init__(self, meta: Dict[str, Any], index) -> None:
        self._meta = meta
        self._index = index

    @classmethod
    def load(cls, settings: Settings) -> "FaissMetadataStore":
        try:
            ensure_local_artifacts(settings)
        except Exception as exc:
            print("[startup] S3 fetch attempt raised an unexpected error:", exc)
            traceback.print_exc()

        if not settings.meta_json_path.exists():
            raise RuntimeError(f"meta.json not found at {settings.meta_json_path.resolve()}")
        if not settings.faiss_index_path.exists():
            raise RuntimeError(f"FAISS index not found at {settings.faiss_index_path.resolve()}")

        try:
            meta: Dict[str, Any] = json.loads(settings.meta_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("[error] failed to parse meta.json:", exc)
            raise

        try:
            index = faiss.read_index(str(settings.faiss_index_path))
        except Exception as exc:
            print("[error] failed to load FAISS index:", exc)
            traceback.print_exc()
            raise

        return cls(meta=meta, index=index)

    def available_countries(self) -> List[str]:
        countries = {_meta_country(item) for item in self._meta.values() if _meta_country(item)}
        return sorted(countries)

    def resolve_country_filter(self, country_hint: str = "") -> str:
        hinted_country = _normalize_slug(country_hint)
        if hinted_country:
            return hinted_country

        countries = self.available_countries()
        if DEFAULT_COUNTRY_FILTER in countries:
            return DEFAULT_COUNTRY_FILTER
        if len(countries) == 1:
            return countries[0]
        return ""

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        country: str = "",
        language: str = "",
        query_text: str = "",
        verbose: bool = False,
    ) -> List[RetrievedHit]:
        q_arr = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_arr)
        k = max(1, int(k))
        ntotal = int(self._index.ntotal)
        if ntotal <= 0:
            return []
        k = min(k, ntotal)
        resolved_country = self.resolve_country_filter(country_hint=country)
        resolved_language = _normalize_filter_language(language)
        search_k = ntotal if resolved_country or resolved_language else k
        distances, ids = self._index.search(q_arr, search_k)

        ranked_results: List[RetrievedHit] = []
        for score, nid in zip(distances[0].tolist(), ids[0].tolist()):
            if int(nid) == -1:
                continue
            meta = self._meta.get(str(int(nid)))
            if not meta:
                print(f"[search] warn: missing meta for id {nid}")
                continue
            ranked_results.append(RetrievedHit(score=float(score), nid=int(nid), meta=meta))

        selected: List[RetrievedHit] = []
        seen_chunks: Set[str] = set()
        stages = _build_search_stages(resolved_country, resolved_language)

        for filters in stages:
            stage_hits = [
                hit
                for hit in ranked_results
                if _chunk_key(hit) not in seen_chunks and _matches_filters(hit.meta, filters)
            ]
            if verbose:
                stage_files = {_source_file(hit.meta) for hit in stage_hits if _source_file(hit.meta)}
                print(
                    f"[search] stage {_format_stage(filters)} -> "
                    f"candidate_pool={len(stage_hits)} chunks, files={len(stage_files)}"
                )

            for hit in _diversify_by_file(stage_hits, k - len(selected)):
                selected.append(hit)
                seen_chunks.add(_chunk_key(hit))
                if len(selected) >= k:
                    return selected

        return selected

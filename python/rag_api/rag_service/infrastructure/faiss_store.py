from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import faiss
import numpy as np

from rag_service.domain.models import RetrievedHit, normalize_language
from rag_service.infrastructure.artifacts import ensure_local_artifacts
from rag_service.infrastructure.config import Settings


# Current corpus contains only Italy. Keep this as one small switch so the
# resolver can later be replaced by a DB/user-profile country selector.
DEFAULT_COUNTRY_FILTER = "italy"
FAQ_ARTIFACT = "faq.jsonl"
METADATA_HINT_BOOST = 0.20
FAQ_BASE_SCORE = 0.68
FAQ_MAX_SCORE = 0.98


def _normalize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _tokenize_metadata(value: str) -> Set[str]:
    normalized = re.sub(r"[_/.-]+", " ", str(value or "").lower())
    return {
        token
        for token in re.findall(r"[0-9a-zа-яёәғқңөұүһі]+", normalized, flags=re.IGNORECASE)
        if len(token) >= 2
    }


def _normalize_phrase(value: str) -> str:
    normalized = re.sub(r"[_/.-]+", " ", str(value or "").lower())
    normalized = re.sub(r"[^0-9a-zа-яёәғқңөұүһі]+", " ", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _iter_metadata_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values: List[str] = []
        for item in value:
            values.extend(_iter_metadata_values(item))
        return values
    if isinstance(value, dict):
        values: List[str] = []
        for item in value.values():
            values.extend(_iter_metadata_values(item))
        return values
    return [str(value)]


def _normalize_filter_language(value: str) -> str:
    normalized = normalize_language(value)
    return "" if normalized == "other" else normalized


def _source_file(meta: Dict[str, Any]) -> str:
    return str(meta.get("source_file") or meta.get("filename") or "")


def _is_faq_meta(meta: Dict[str, Any]) -> bool:
    kind = str(meta.get("kind") or meta.get("type") or "").strip().lower()
    source_file = _source_file(meta).strip().lower()
    return bool(meta.get("is_faq")) or kind == "faq" or source_file.startswith("faq://")


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
    match = re.search(r"_(kk|kz|ru|en|eng)$", stem)
    if not match:
        return ""
    suffix = match.group(1)
    aliases = {
        "eng": "en",
        "kz": "kk",
    }
    return aliases.get(suffix, suffix)


def _chunk_key(hit: RetrievedHit) -> str:
    meta = hit.meta
    source_file = _source_file(meta)
    page = meta.get("page", "")
    chunk_index = meta.get("chunk_index", "")
    return str(meta.get("id") or f"{source_file}:{page}:{chunk_index}" or hit.nid)


def _metadata_lookup_values(
    meta: Dict[str, Any],
    *,
    include_location_fields: bool = True,
) -> List[str]:
    fields = [
        "canonical_doc_id",
        "doc_type",
        "topic",
        "topics",
        "title",
        "question",
        "questions",
        "aliases",
        "keywords",
        "tags",
    ]
    if include_location_fields:
        fields.extend(
            [
                "country",
                "language",
                "lang",
                "source_file",
                "filename",
            ]
        )
    if _is_faq_meta(meta):
        fields.extend(["answer", "text", "body"])

    values: List[str] = []
    for field in fields:
        values.extend(_iter_metadata_values(meta.get(field)))
    return [value for value in values if str(value).strip()]


def _metadata_match_score(
    query_text: str,
    meta: Dict[str, Any],
    *,
    include_location_fields: bool = True,
) -> float:
    query_tokens = _tokenize_metadata(query_text)
    if not query_tokens:
        return 0.0

    normalized_query = _normalize_phrase(query_text)
    metadata_values = _metadata_lookup_values(
        meta,
        include_location_fields=include_location_fields,
    )
    for value in metadata_values:
        phrase = _normalize_phrase(value)
        if len(phrase.split()) >= 2 and phrase in normalized_query:
            return 1.0

    metadata_text = " ".join(metadata_values)
    metadata_tokens = _tokenize_metadata(metadata_text)
    if not metadata_tokens:
        return 0.0

    overlap = len(query_tokens & metadata_tokens)
    if overlap == 0:
        return 0.0

    # Ratio against the shorter side rewards compact metadata such as aliases.
    denominator = max(1, min(len(query_tokens), len(metadata_tokens)))
    return min(1.0, overlap / denominator)


def _with_metadata_boost(hit: RetrievedHit, query_text: str) -> RetrievedHit:
    match_score = _metadata_match_score(query_text, hit.meta)
    if match_score <= 0:
        return hit

    meta = dict(hit.meta)
    meta["semantic_score"] = float(hit.score)
    meta["metadata_score"] = match_score
    meta["metadata_boost"] = METADATA_HINT_BOOST * match_score
    return RetrievedHit(
        score=float(hit.score) + float(meta["metadata_boost"]),
        nid=hit.nid,
        meta=meta,
    )


def _load_faq_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                print(f"[faq] skipping invalid JSON at {path}:{line_number}: {exc}")
                continue
            if not isinstance(parsed, dict):
                continue

            question = str(parsed.get("question") or parsed.get("title") or "").strip()
            answer = str(parsed.get("answer") or parsed.get("text") or "").strip()
            if not question and not answer:
                continue

            entry = dict(parsed)
            entry.setdefault("kind", "faq")
            entry.setdefault("is_faq", True)
            entry.setdefault("source_file", f"faq://{entry.get('id') or line_number}")
            entry.setdefault("page", "")
            entry["text"] = " ".join(part for part in (question, answer) if part).strip()
            entries.append(entry)
    return entries


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
    def __init__(
        self,
        meta: Dict[str, Any],
        index,
        faq_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._meta = meta
        self._index = index
        self._faq_entries = faq_entries or []

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

        faq_entries = _load_faq_entries(settings.out_dir / FAQ_ARTIFACT)
        if faq_entries:
            print(f"[faq] loaded {len(faq_entries)} FAQ entries from {settings.out_dir / FAQ_ARTIFACT}")

        return cls(meta=meta, index=index, faq_entries=faq_entries)

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
            ranked_results.append(
                _with_metadata_boost(
                    RetrievedHit(score=float(score), nid=int(nid), meta=meta),
                    query_text,
                )
            )
        ranked_results.extend(self._faq_hits(query_text))

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

    def _faq_hits(self, query_text: str) -> List[RetrievedHit]:
        hits: List[RetrievedHit] = []
        for index, meta in enumerate(self._faq_entries, start=1):
            score = _metadata_match_score(query_text, meta, include_location_fields=False)
            if score <= 0:
                continue
            enriched_meta = dict(meta)
            enriched_meta["metadata_score"] = score
            enriched_meta["metadata_boost"] = score
            hits.append(
                RetrievedHit(
                    score=min(FAQ_MAX_SCORE, FAQ_BASE_SCORE + (score * (FAQ_MAX_SCORE - FAQ_BASE_SCORE))),
                    nid=-index,
                    meta=enriched_meta,
                )
            )
        return hits

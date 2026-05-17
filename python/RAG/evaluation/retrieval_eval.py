#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv


EVAL_DIR = Path(__file__).resolve().parent
RAG_DIR = EVAL_DIR.parent
REPO_ROOT = RAG_DIR.parents[1]
RAG_API_DIR = REPO_ROOT / "python" / "rag_api"
OUT_DIR = RAG_DIR / "out"

if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

from rag_service.domain.models import Classification, RetrievalClarity, normalize_language
from rag_service.infrastructure.config import Settings
from rag_service.infrastructure.faiss_store import FaissMetadataStore
from rag_service.infrastructure.openai_gateway import OpenAIGateway


QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
FILENAME_COLUMNS = ("filename", "file", "expected_file", "relevant_file", "relevant_files")
LANGUAGE_COLUMNS = ("expected_language", "language", "lang", "expected_lang")
INTENT_COLUMNS = ("expected_intent", "intent", "expected_class", "class", "classification", "label")
RETRIEVAL_INTENTS = {"FACTUAL_QUESTION", "PROCEDURE", "COMPARISON", "GUIDANCE", "DOCUMENT_REQUEST"}
COLOR_CODES = {
    "green": "\033[32m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def should_colorize(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{COLOR_CODES[color]}{text}{COLOR_CODES['reset']}"


def normalized_header_map(fieldnames: Iterable[str]) -> Dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames if name is not None}


def first_existing_column(fieldnames: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in fieldnames:
            return fieldnames[candidate]
    return None


def split_files(value: str) -> List[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.replace("|", ";").split(";") if part.strip()]


def load_cases(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not csv_path.exists():
        raise RuntimeError(f"retrieval CSV not found: {csv_path}")

    cases: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("retrieval CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = first_existing_column(fieldnames, QUESTION_COLUMNS)
        filename_col = first_existing_column(fieldnames, FILENAME_COLUMNS)
        language_col = first_existing_column(fieldnames, LANGUAGE_COLUMNS)
        intent_col = first_existing_column(fieldnames, INTENT_COLUMNS)

        if not question_col:
            raise RuntimeError("CSV must contain a question/query column")
        if not filename_col:
            raise RuntimeError("CSV must contain a filename/file/relevant_files column")

        for row_number, row in enumerate(reader, start=2):
            query = (row.get(question_col) or "").strip()
            expected_files = split_files(row.get(filename_col) or "")
            if not query:
                continue
            if not expected_files:
                skipped.append({"row": row_number, "query": query, "reason": "missing expected filename"})
                continue
            cases.append(
                {
                    "query": query,
                    "expected_files": expected_files,
                    "expected_language": normalize_language(row.get(language_col) or "") if language_col else "",
                    "expected_intent": (row.get(intent_col) or "").strip().upper() if intent_col else "",
                }
            )

    if not cases:
        raise RuntimeError(f"retrieval CSV has no rows with expected files: {csv_path}")
    return cases, skipped


def available_source_files(store: FaissMetadataStore) -> List[str]:
    meta = getattr(store, "_meta", {})
    return sorted(
        {
            str(item.get("source_file") or item.get("filename") or "")
            for item in meta.values()
            if str(item.get("source_file") or item.get("filename") or "")
        }
    )


def filter_cases_by_corpus(
    cases: List[Dict[str, Any]],
    store: FaissMetadataStore,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus_files = set(available_source_files(store))
    valid: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for case in cases:
        relevant_files = [source_file for source_file in case["expected_files"] if source_file in corpus_files]
        missing_files = [source_file for source_file in case["expected_files"] if source_file not in corpus_files]
        if missing_files:
            skipped.append({"query": case["query"], "reason": "expected file missing from corpus", "files": missing_files})
        if relevant_files:
            valid.append({**case, "expected_files": relevant_files})
    return valid, skipped


def settings_for_eval(class_model: str, llm_model: str, embed_model: str, out_dir: Path) -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env", override=True)
    load_dotenv(EVAL_DIR / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in python/RAG/.env, repo .env, or environment")

    resolved_llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    resolved_class_model = class_model or os.getenv("CLASS_MODEL") or resolved_llm_model
    return Settings(
        openai_api_key=api_key,
        redis_url=None,
        s3_endpoint=None,
        s3_access_key=None,
        s3_secret=None,
        s3_bucket_vectors=None,
        s3_use_ssl=False,
        s3_verify=False,
        release_prefix=None,
        out_dir=out_dir,
        meta_json_path=out_dir / "meta.json",
        faiss_index_path=out_dir / "index.faiss",
        optional_artifacts=[],
        embed_model=embed_model or os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        llm_model=resolved_llm_model,
        class_model=resolved_class_model,
    )


def cache_key(model: str, embed_model: str, query: str) -> str:
    return json.dumps(
        {"model": model, "embed_model": embed_model, "query": query},
        ensure_ascii=False,
        sort_keys=True,
    )


def load_cache(cache_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_cache(cache_path: Optional[Path], cache: Dict[str, Dict[str, Any]]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_with_retry(
    gateway: OpenAIGateway,
    query: str,
    attempts: int,
    retry_backoff: float,
) -> Classification:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            classification, _usage = gateway.classify_query(query)
            return classification
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))
    raise RuntimeError(f"classification failed after {attempts} attempt(s): {last_error}")


def clarity_with_retry(
    gateway: OpenAIGateway,
    query: str,
    language: str,
    intent: str,
    attempts: int,
    retry_backoff: float,
) -> RetrievalClarity:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            clarity, _usage = gateway.clarify_or_rewrite_query(query, language, intent, history=[])
            return clarity
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))
    raise RuntimeError(f"clarity failed after {attempts} attempt(s): {last_error}")


def embed_with_retry(
    gateway: OpenAIGateway,
    text: str,
    attempts: int,
    retry_backoff: float,
) -> List[float]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            embedding, _usage = gateway.embed_text(text)
            return embedding.astype(float).tolist()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))
    raise RuntimeError(f"embedding failed after {attempts} attempt(s): {last_error}")


def pipeline_prediction(
    gateway: OpenAIGateway,
    settings: Settings,
    case: Dict[str, Any],
    attempts: int,
    retry_backoff: float,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
) -> Tuple[Dict[str, Any], bool]:
    query = case["query"]
    key = cache_key(settings.class_model, settings.embed_model, query)
    cached = None if refresh_cache else cache.get(key)
    cache_hit = cached is not None
    if cached is not None:
        return cached, cache_hit

    classification = classify_with_retry(gateway, query, attempts, retry_backoff)
    intent = classification.intent
    language = classification.language or case.get("expected_language") or ""
    route = classification.route or ""
    should_retrieve = (
        classification.needs_rag
        or route == "RAG_SEARCH"
        or intent in RETRIEVAL_INTENTS
    )

    clarity_payload = {
        "is_retrieval_related": should_retrieve,
        "is_clear": True,
        "standalone_query": classification.rewritten_query or query,
        "clarifying_question": "",
        "target_language": "",
        "reason": "Classifier did not require retrieval.",
    }
    if should_retrieve:
        clarity = clarity_with_retry(gateway, query, language, intent, attempts, retry_backoff)
        clarity_payload = {
            "is_retrieval_related": clarity.is_retrieval_related,
            "is_clear": clarity.is_clear,
            "standalone_query": clarity.standalone_query,
            "clarifying_question": clarity.clarifying_question,
            "target_language": clarity.target_language,
            "reason": clarity.reason,
        }

    retrieval_query = (
        classification.rewritten_query
        or str(clarity_payload.get("standalone_query") or "").strip()
        or query
    )
    can_embed = bool(should_retrieve and clarity_payload["is_retrieval_related"] and clarity_payload["is_clear"])
    embedding = embed_with_retry(gateway, retrieval_query, attempts, retry_backoff) if can_embed else []
    prediction = {
        "intent": classification.intent,
        "confidence": classification.confidence,
        "needs_rag": classification.needs_rag,
        "route": classification.route,
        "language": classification.language,
        "rewritten_query": classification.rewritten_query,
        "explain": classification.explain,
        "clarity": clarity_payload,
        "retrieval_query": retrieval_query,
        "embedding": embedding,
    }
    cache[key] = prediction
    return prediction, cache_hit


def source_file_from_hit(hit: Any) -> str:
    meta = getattr(hit, "meta", {}) or {}
    return str(meta.get("source_file") or meta.get("filename") or "")


def hit_page(hit: Any) -> str:
    meta = getattr(hit, "meta", {}) or {}
    page = meta.get("page")
    return "" if page is None else str(page)


def dedupe_hits_by_file(hits: List[Any]) -> List[Any]:
    selected = []
    seen = set()
    for hit in hits:
        source_file = source_file_from_hit(hit)
        if not source_file or source_file in seen:
            continue
        selected.append(hit)
        seen.add(source_file)
    return selected


def dcg_at_k(binary_relevance: List[int], top_k: int) -> float:
    dcg = 0.0
    for rank, rel in enumerate(binary_relevance[:top_k], start=1):
        if rel:
            dcg += float(rel) / float(np.log2(rank + 1))
    return dcg


def ndcg_at_k(binary_relevance: List[int], total_relevant: int, top_k: int) -> float:
    dcg = dcg_at_k(binary_relevance, top_k)
    ideal = [1] * min(total_relevant, top_k)
    idcg = dcg_at_k(ideal, top_k)
    return dcg / idcg if idcg else 0.0


def evaluate_cases(
    cases: List[Dict[str, Any]],
    gateway: OpenAIGateway,
    store: FaissMetadataStore,
    settings: Settings,
    raw_k: int,
    top_k: int,
    attempts: int,
    retry_backoff: float,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
    color_enabled: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prediction, cache_hit = pipeline_prediction(
            gateway,
            settings,
            case,
            attempts,
            retry_backoff,
            cache,
            refresh_cache,
        )

        hits: List[Any] = []
        if prediction.get("embedding"):
            embedding = np.array(prediction["embedding"], dtype=np.float32)
            hits = store.search(
                embedding,
                k=max(raw_k, top_k),
                language=prediction.get("language") or case.get("expected_language") or "",
                query_text=prediction.get("retrieval_query") or case["query"],
            )
        ranked_hits = dedupe_hits_by_file(hits)[:top_k]
        ranked_files = [source_file_from_hit(hit) for hit in ranked_hits]
        ranked_pages = [hit_page(hit) for hit in ranked_hits]
        expected_files = set(case["expected_files"])
        binary_relevance = [1 if source_file in expected_files else 0 for source_file in ranked_files]
        relevant_retrieved = len(set(ranked_files) & expected_files)
        first_relevant_rank = next((idx for idx, rel in enumerate(binary_relevance, start=1) if rel), None)
        row = {
            **case,
            "intent": prediction.get("intent") or "",
            "route": prediction.get("route") or "",
            "needs_rag": bool(prediction.get("needs_rag")),
            "language": prediction.get("language") or "",
            "confidence": prediction.get("confidence"),
            "retrieval_query": prediction.get("retrieval_query") or "",
            "clarity": prediction.get("clarity") or {},
            "retrieved_files": ranked_files,
            "retrieved_pages": ranked_pages,
            "hit": relevant_retrieved > 0,
            "top1_correct": first_relevant_rank == 1,
            "precision_at_k": relevant_retrieved / float(top_k) if top_k else 0.0,
            "recall_at_k": relevant_retrieved / float(len(expected_files)) if expected_files else 0.0,
            "mrr": 1.0 / float(first_relevant_rank) if first_relevant_rank else 0.0,
            "ndcg_at_k": ndcg_at_k(binary_relevance, len(expected_files), top_k),
            "first_relevant_rank": first_relevant_rank,
            "cache_hit": cache_hit,
        }
        rows.append(row)
        status = "ok" if row["hit"] else "MISS"
        status_display = colorize(status, "green", color_enabled) if row["hit"] else status
        top_file = ranked_files[0] if ranked_files else "-"
        line = (
            f"[{index:03d}/{len(cases):03d}] "
            f"{status_display} "
            f"rank={row['first_relevant_rank'] or '-'} "
            f"intent={row['intent']} "
            f"expected={','.join(case['expected_files'])} top={top_file} "
            f"query={case['query'][:70]}"
        )
        print(line if row["hit"] else colorize(line, "red", color_enabled))
    return rows


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(rows: List[Dict[str, Any]], top_k: int, skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "queries": len(rows),
        "top_k": top_k,
        "skipped": len(skipped),
        "accuracy_at_1": safe_mean([1.0 if row["top1_correct"] else 0.0 for row in rows]),
        "hit_rate_at_k": safe_mean([1.0 if row["hit"] else 0.0 for row in rows]),
        "precision_at_k": safe_mean([row["precision_at_k"] for row in rows]),
        "recall_at_k": safe_mean([row["recall_at_k"] for row in rows]),
        "ndcg_at_k": safe_mean([row["ndcg_at_k"] for row in rows]),
        "mrr": safe_mean([row["mrr"] for row in rows]),
        "cache_hits": sum(1 for row in rows if row.get("cache_hit")),
        "api_calls": sum(1 for row in rows if not row.get("cache_hit")),
    }


def print_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
) -> None:
    top_k = int(summary["top_k"])
    print("\n" + "=" * 80)
    print(f"Retrieval evaluation on {summary['queries']} labeled queries (file-level, top_k={top_k})")
    print(f"Accuracy@1  : {summary['accuracy_at_1']:.4f}")
    print(f"HitRate@{top_k:<2} : {summary['hit_rate_at_k']:.4f}")
    print(f"Precision@{top_k:<2}: {summary['precision_at_k']:.4f}")
    print(f"Recall@{top_k:<2}   : {summary['recall_at_k']:.4f}")
    print(f"NDCG@{top_k:<2}     : {summary['ndcg_at_k']:.4f}")
    print(f"MRR         : {summary['mrr']:.4f}")
    print(f"Cache/API   : {summary['cache_hits']} cached, {summary['api_calls']} API calls")
    if skipped:
        print(f"Skipped     : {len(skipped)} rows without usable expected files")

    misses = [row for row in rows if not row["hit"]]
    if misses:
        print("\n" + colorize("Retrieval misses:", "red", color_enabled))
        for row in misses[:max_examples]:
            print(colorize(f"- {row['query']}", "red", color_enabled))
            print(f"  expected : {row['expected_files']}")
            print(f"  retrieved: {row['retrieved_files']}")
            print(f"  rewrite  : {row['retrieval_query']}")
            print(f"  intent   : {row['intent']} route={row['route']} clarity={row['clarity'].get('reason', '')}")


def save_report(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "rows": rows,
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote report -> {path}")


def save_predictions_csv(path: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "expected_files",
        "intent",
        "route",
        "language",
        "confidence",
        "retrieval_query",
        "retrieved_files",
        "hit",
        "top1_correct",
        "first_relevant_rank",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "cache_hit",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fieldnames}
            out["expected_files"] = ";".join(row.get("expected_files") or [])
            out["retrieved_files"] = ";".join(row.get("retrieved_files") or [])
            writer.writerow(out)
    print(f"[eval] wrote predictions -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production RAG retrieval pipeline locally: classifier, "
            "clarity rewrite, embedding, and FAISS search over python/RAG/out artifacts."
        )
    )
    parser.add_argument("--csv", default=str(EVAL_DIR / "query_mappings.csv"), help="CSV with question and expected filename columns")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Local artifact directory containing meta.json and index.faiss")
    parser.add_argument("--top-k", type=int, default=5, help="File-level top K for metrics")
    parser.add_argument("--raw-k", type=int, default=64, help="Raw production FAISS search K before file dedupe")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N usable rows")
    parser.add_argument("--attempts", type=int, default=2, help="LLM/embed call attempts per uncached query")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Seconds between retry attempts")
    parser.add_argument("--class-model", default="", help="Override CLASS_MODEL for classification/clarity")
    parser.add_argument("--llm-model", default="", help="Override LLM_MODEL setting")
    parser.add_argument("--embed-model", default="", help="Override EMBED_MODEL for retrieval embeddings")
    parser.add_argument("--no-cache", action="store_true", help="Disable prediction cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached predictions and overwrite cache")
    parser.add_argument("--cache", default=str(OUT_DIR / "retrieval_eval_cache.json"), help="Prediction cache path")
    parser.add_argument("--output", default=str(OUT_DIR / "retrieval_eval_report.json"), help="JSON report path")
    parser.add_argument("--predictions-csv", default="", help="Optional detailed predictions CSV path")
    parser.add_argument("--max-examples", type=int, default=10, help="Max mistake examples to print")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Color terminal output. auto enables colors only when stdout is a TTY",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    settings = settings_for_eval(args.class_model, args.llm_model, args.embed_model, out_dir)
    store = FaissMetadataStore.load(settings)
    gateway = OpenAIGateway(settings)
    color_enabled = should_colorize(args.color)

    loaded_cases, skipped_no_file = load_cases(Path(args.csv))
    cases, skipped_missing = filter_cases_by_corpus(loaded_cases, store)
    skipped = skipped_no_file + skipped_missing
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    cache_path = None if args.no_cache else Path(args.cache)
    cache = load_cache(cache_path)
    rows = evaluate_cases(
        cases,
        gateway,
        store,
        settings,
        raw_k=max(1, args.raw_k),
        top_k=max(1, args.top_k),
        attempts=args.attempts,
        retry_backoff=args.retry_backoff,
        cache=cache,
        refresh_cache=args.refresh_cache,
        color_enabled=color_enabled,
    )
    summary = summarize(rows, top_k=max(1, args.top_k), skipped=skipped)
    print_report(summary, rows, skipped, max_examples=args.max_examples, color_enabled=color_enabled)
    save_report(Path(args.output), summary, rows, skipped)
    save_predictions_csv(Path(args.predictions_csv) if args.predictions_csv else None, rows)
    save_cache(cache_path, cache)


if __name__ == "__main__":
    main()

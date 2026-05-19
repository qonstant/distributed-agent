#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv


EVAL_DIR = Path(__file__).resolve().parent
RAG_DIR = EVAL_DIR.parent
REPO_ROOT = RAG_DIR.parents[1]
RAG_API_DIR = REPO_ROOT / "python" / "rag_api"
OUT_DIR = RAG_DIR / "out"

if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

from rag_service.domain.models import ConversationMessage, RetrievedHit, normalize_language
from rag_service.infrastructure.openai_gateway import OpenAIGateway


QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
INTENT_COLUMNS = ("intent", "classifier_intent", "expected_intent", "class", "classification")
LANGUAGE_COLUMNS = ("language", "language_hint", "lang", "expected_language")
CHUNKS_COLUMNS = ("chunks", "retrieved_chunks", "top_chunks", "context")
HISTORY_COLUMNS = ("history", "history_text", "conversation")
EXPECTED_SUFFICIENT_COLUMNS = ("expected_sufficient", "is_sufficient", "sufficient")
EXPECTED_QUESTION_TERMS_COLUMNS = ("expected_question_terms", "question_terms", "expected_message_terms")
FORBIDDEN_QUESTION_TERMS_COLUMNS = ("forbidden_question_terms", "forbidden_terms", "must_not_contain")

COLOR_CODES = {
    "green": "\033[32m",
    "red": "\033[31m",
    "reset": "\033[0m",
}
CACHE_VERSION = "sufficiency-eval-v2"


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


def normalize_bool(value: str, default: Optional[bool] = None) -> Optional[bool]:
    normalized = (value or "").strip().lower()
    if normalized == "":
        return default
    if normalized in {"1", "true", "yes", "y", "sufficient"}:
        return True
    if normalized in {"0", "false", "no", "n", "insufficient", "weak"}:
        return False
    raise ValueError(f"unsupported boolean value {value!r}")


def normalize_intent_hint(value: str) -> str:
    normalized = (value or "").strip().upper()
    aliases = {
        "CHIT_CHAT": "CHITCHAT",
        "GUIDANCE": "PROCEDURE",
        "DOCUMENT_REQUEST": "FACTUAL_QUESTION",
        "OTHER": "OUT_OF_DOMAIN",
        "": "FACTUAL_QUESTION",
    }
    return aliases.get(normalized, normalized)


def parse_history_text(value: str) -> List[ConversationMessage]:
    text = (value or "").strip()
    if not text:
        return []
    messages: List[ConversationMessage] = []
    for index, chunk in enumerate(text.split("||"), start=1):
        part = chunk.strip()
        if not part:
            continue
        role = "user"
        body = part
        if ":" in part:
            possible_role, possible_body = part.split(":", 1)
            normalized_role = possible_role.strip().lower()
            if normalized_role in {"user", "assistant"}:
                role = normalized_role
                body = possible_body
        body = body.strip()
        if body:
            messages.append(ConversationMessage(role=role, text=body, ts=index))
    return messages


def parse_chunks(value: str) -> List[RetrievedHit]:
    text = (value or "").strip()
    if not text:
        return []

    chunks: List[RetrievedHit] = []
    for index, raw_chunk in enumerate(text.split("||"), start=1):
        raw_chunk = raw_chunk.strip()
        if not raw_chunk:
            continue
        source_file = "eval/mock.pdf"
        page = index
        body = raw_chunk
        parts = raw_chunk.split("#", 2)
        if len(parts) == 3:
            source_file = parts[0].strip() or source_file
            try:
                page = int((parts[1] or "").strip() or index)
            except ValueError:
                page = index
            body = parts[2].strip()
        chunks.append(
            RetrievedHit(
                score=max(0.1, 1.0 - (index * 0.05)),
                nid=index,
                meta={
                    "source_file": source_file,
                    "page": page,
                    "text": body,
                },
            )
        )
    return chunks


def row_value(row: Dict[str, str], column: Optional[str]) -> str:
    if not column:
        return ""
    return row.get(column) or ""


def load_cases(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise RuntimeError(f"sufficiency CSV not found: {csv_path}")

    cases: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("sufficiency CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = first_existing_column(fieldnames, QUESTION_COLUMNS)
        intent_col = first_existing_column(fieldnames, INTENT_COLUMNS)
        language_col = first_existing_column(fieldnames, LANGUAGE_COLUMNS)
        chunks_col = first_existing_column(fieldnames, CHUNKS_COLUMNS)
        history_col = first_existing_column(fieldnames, HISTORY_COLUMNS)
        sufficient_col = first_existing_column(fieldnames, EXPECTED_SUFFICIENT_COLUMNS)
        question_terms_col = first_existing_column(fieldnames, EXPECTED_QUESTION_TERMS_COLUMNS)
        forbidden_terms_col = first_existing_column(fieldnames, FORBIDDEN_QUESTION_TERMS_COLUMNS)

        if not question_col:
            raise RuntimeError("CSV must contain a query/question column")
        if not chunks_col:
            raise RuntimeError("CSV must contain a chunks/retrieved_chunks column")
        if not sufficient_col:
            raise RuntimeError("CSV must contain expected_sufficient")

        for row_number, row in enumerate(reader, start=2):
            query = row_value(row, question_col).strip()
            if not query:
                continue
            expected_sufficient = normalize_bool(row_value(row, sufficient_col))
            if expected_sufficient is None:
                raise RuntimeError(f"expected_sufficient is empty at row {row_number}")
            cases.append(
                {
                    "query": query,
                    "intent": normalize_intent_hint(row_value(row, intent_col)),
                    "language": normalize_language(row_value(row, language_col) or "en"),
                    "chunks": parse_chunks(row_value(row, chunks_col)),
                    "history": parse_history_text(row_value(row, history_col)),
                    "expected_sufficient": expected_sufficient,
                    "expected_question_terms": parse_term_groups(row_value(row, question_terms_col)),
                    "forbidden_question_terms": parse_term_groups(row_value(row, forbidden_terms_col)),
                }
            )

    if not cases:
        raise RuntimeError(f"sufficiency CSV is empty after filtering: {csv_path}")
    return cases


def parse_term_groups(value: str) -> List[List[str]]:
    raw = (value or "").strip()
    if not raw:
        return []
    groups: List[List[str]] = []
    for group in raw.split(";"):
        alternatives = [part.strip().lower() for part in group.split("|") if part.strip()]
        if alternatives:
            groups.append(alternatives)
    return groups


def contains_term(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def expected_terms_match(text: str, groups: List[List[str]]) -> bool:
    return all(any(contains_term(text, term) for term in group) for group in groups)


def forbidden_terms_match(text: str, groups: List[List[str]]) -> bool:
    return not any(any(contains_term(text, term) for term in group) for group in groups)


def settings_for_sufficiency(class_model: str) -> SimpleNamespace:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env", override=True)
    load_dotenv(EVAL_DIR / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in python/RAG/.env, repo .env, or environment")

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return SimpleNamespace(
        openai_api_key=api_key,
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        llm_model=llm_model,
        class_model=class_model or os.getenv("CLASS_MODEL") or llm_model,
    )


def cache_key(model: str, case: Dict[str, Any]) -> str:
    chunks = [
        {
            "source_file": hit.meta.get("source_file"),
            "page": hit.meta.get("page"),
            "text": hit.meta.get("text"),
        }
        for hit in case["chunks"]
    ]
    history = [
        {
            "role": message.role,
            "text": message.text,
        }
        for message in case["history"]
    ]
    return json.dumps(
        {
            "version": CACHE_VERSION,
            "model": model,
            "query": case["query"],
            "intent": case["intent"],
            "language": case["language"],
            "chunks": chunks,
            "history": history,
        },
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


def predict_sufficiency(
    gateway: OpenAIGateway,
    case: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
    attempts: int,
    retry_backoff: float,
) -> tuple[Dict[str, Any], bool]:
    key = cache_key(gateway._settings.class_model, case)
    if not refresh_cache and key in cache:
        return cache[key], True

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            sufficiency, _usage = gateway.assess_retrieval_sufficiency(
                case["query"],
                case["language"],
                case["intent"],
                case["chunks"],
                history=case["history"],
            )
            if "sufficiency check failed" in (sufficiency.reason or "").lower():
                raise RuntimeError(sufficiency.reason)
            prediction = {
                "is_sufficient": sufficiency.is_sufficient,
                "clarifying_question": sufficiency.clarifying_question,
                "reason": sufficiency.reason,
            }
            cache[key] = prediction
            return prediction, False
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))
    raise RuntimeError(f"sufficiency failed after {attempts} attempt(s): {last_error}")


def evaluate_case(case: Dict[str, Any], prediction: Dict[str, Any], cache_hit: bool) -> Dict[str, Any]:
    question = str(prediction.get("clarifying_question") or "")
    sufficient_ok = bool(prediction.get("is_sufficient")) == case["expected_sufficient"]
    expected_terms_ok = expected_terms_match(question, case["expected_question_terms"])
    forbidden_terms_ok = forbidden_terms_match(question, case["forbidden_question_terms"])
    ok = sufficient_ok and expected_terms_ok and forbidden_terms_ok
    return {
        **case,
        "prediction": prediction,
        "cache_hit": cache_hit,
        "sufficient_ok": sufficient_ok,
        "expected_terms_ok": expected_terms_ok,
        "forbidden_terms_ok": forbidden_terms_ok,
        "ok": ok,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}
    return {
        "cases": total,
        "decision_accuracy": sum(1 for row in rows if row["ok"]) / total,
        "sufficient_accuracy": sum(1 for row in rows if row["sufficient_ok"]) / total,
        "expected_terms_accuracy": sum(1 for row in rows if row["expected_terms_ok"]) / total,
        "forbidden_terms_accuracy": sum(1 for row in rows if row["forbidden_terms_ok"]) / total,
        "cache_hits": sum(1 for row in rows if row["cache_hit"]),
        "api_calls": sum(1 for row in rows if not row["cache_hit"]),
    }


def print_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"Sufficiency evaluation on {summary['cases']} labeled queries")
    print(f"Decision Acc       : {summary['decision_accuracy']:.4f}")
    print(f"Sufficient Acc     : {summary['sufficient_accuracy']:.4f}")
    print(f"Expected Terms Acc : {summary['expected_terms_accuracy']:.4f}")
    print(f"Forbidden Terms Acc: {summary['forbidden_terms_accuracy']:.4f}")
    print(f"Cache/API          : {summary['cache_hits']} cached, {summary['api_calls']} API calls")

    misses = [row for row in rows if not row["ok"]]
    if not misses:
        return

    print("\n" + colorize("Sufficiency misses:", "red", color_enabled))
    for row in misses[:max_examples]:
        prediction = row["prediction"]
        print(f"- {row['query']}")
        print(f"  expected : sufficient={row['expected_sufficient']}")
        print(
            "  predicted: "
            f"sufficient={prediction.get('is_sufficient')} "
            f"({prediction.get('reason') or ''})"
        )
        if prediction.get("clarifying_question"):
            print(f"  message  : {prediction.get('clarifying_question')}")
        if not row["expected_terms_ok"]:
            print(f"  missing expected message terms: {row['expected_question_terms']}")
        if not row["forbidden_terms_ok"]:
            print(f"  contains forbidden message terms: {row['forbidden_question_terms']}")


def save_report(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    payload = {
        "summary": summary,
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[eval] wrote report -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the production retrieval-sufficiency LLM against labeled weak/strong retrieval cases."
    )
    parser.add_argument("--csv", default=str(EVAL_DIR / "sufficiency_mappings.csv"), help="CSV with sufficiency labels")
    parser.add_argument("--class-model", default="", help="Override CLASS_MODEL for the evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows")
    parser.add_argument("--attempts", type=int, default=2, help="LLM call attempts per uncached query")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Seconds between retry attempts")
    parser.add_argument("--no-cache", action="store_true", help="Disable prediction cache")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cached predictions and overwrite cache with fresh LLM results",
    )
    parser.add_argument("--cache", default=str(OUT_DIR / "sufficiency_eval_cache.json"), help="Prediction cache path")
    parser.add_argument("--output", default=str(OUT_DIR / "sufficiency_eval_report.json"), help="JSON report path")
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
    cases = load_cases(Path(args.csv))
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    settings = settings_for_sufficiency(args.class_model)
    gateway = OpenAIGateway(settings)
    cache_path = None if args.no_cache else Path(args.cache)
    cache = load_cache(cache_path)
    color_enabled = should_colorize(args.color)

    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prediction, cache_hit = predict_sufficiency(
            gateway,
            case,
            cache,
            refresh_cache=args.refresh_cache,
            attempts=args.attempts,
            retry_backoff=args.retry_backoff,
        )
        row = evaluate_case(case, prediction, cache_hit)
        rows.append(row)
        label = "ok" if row["ok"] else "MISS"
        label = colorize(label, "green" if row["ok"] else "red", color_enabled)
        print(
            f"[{index:03d}/{len(cases):03d}] {label} "
            f"expected=(sufficient={case['expected_sufficient']}) "
            f"predicted=(sufficient={prediction.get('is_sufficient')}) "
            f"query={case['query']}"
        )

    save_cache(cache_path, cache)
    summary = summarize(rows)
    print_report(summary, rows, max_examples=args.max_examples, color_enabled=color_enabled)
    save_report(Path(args.output), summary, rows)


if __name__ == "__main__":
    main()

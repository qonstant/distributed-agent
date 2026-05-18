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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv


EVAL_DIR = Path(__file__).resolve().parent
RAG_DIR = EVAL_DIR.parent
REPO_ROOT = RAG_DIR.parents[1]
RAG_API_DIR = REPO_ROOT / "python" / "rag_api"
OUT_DIR = RAG_DIR / "out"

if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

from rag_service.domain.models import ConversationMessage, normalize_language
from rag_service.infrastructure.openai_gateway import OpenAIGateway


QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
HISTORY_COLUMNS = ("history", "history_text", "conversation", "context")
HISTORY_JSON_COLUMNS = ("history_json", "conversation_json", "context_json")
INTENT_COLUMNS = ("intent", "classifier_intent", "expected_intent", "class", "classification")
LANGUAGE_COLUMNS = ("language", "language_hint", "lang", "expected_language")
EXPECTED_RELATED_COLUMNS = ("expected_retrieval_related", "retrieval_related", "expected_related", "related")
EXPECTED_CLEAR_COLUMNS = ("expected_clear", "is_clear", "clear")
EXPECTED_TARGET_LANGUAGE_COLUMNS = ("expected_target_language", "target_language")
EXPECTED_QUERY_TERMS_COLUMNS = (
    "expected_query_terms",
    "standalone_query_terms",
    "expected_standalone_terms",
)
EXPECTED_QUESTION_TERMS_COLUMNS = (
    "expected_question_terms",
    "clarifying_question_terms",
    "expected_clarification_terms",
)
COLOR_CODES = {
    "green": "\033[32m",
    "red": "\033[31m",
    "reset": "\033[0m",
}
CACHE_VERSION = "clarity-eval-v7"


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
    if normalized in {"1", "true", "yes", "y", "related", "clear"}:
        return True
    if normalized in {"0", "false", "no", "n", "unrelated", "unclear"}:
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


def parse_history_json(value: str) -> List[ConversationMessage]:
    text = (value or "").strip()
    if not text:
        return []

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("history_json must be a JSON list")

    messages: List[ConversationMessage] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        try:
            ts = int(item.get("ts") or index)
        except Exception:
            ts = index
        messages.append(ConversationMessage(role=role, text=body, ts=ts))
    return messages


def row_value(row: Dict[str, str], column: Optional[str]) -> str:
    if not column:
        return ""
    return row.get(column) or ""


def load_cases(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise RuntimeError(f"clarity CSV not found: {csv_path}")

    cases: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("clarity CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = first_existing_column(fieldnames, QUESTION_COLUMNS)
        history_col = first_existing_column(fieldnames, HISTORY_COLUMNS)
        history_json_col = first_existing_column(fieldnames, HISTORY_JSON_COLUMNS)
        intent_col = first_existing_column(fieldnames, INTENT_COLUMNS)
        language_col = first_existing_column(fieldnames, LANGUAGE_COLUMNS)
        related_col = first_existing_column(fieldnames, EXPECTED_RELATED_COLUMNS)
        clear_col = first_existing_column(fieldnames, EXPECTED_CLEAR_COLUMNS)
        target_language_col = first_existing_column(fieldnames, EXPECTED_TARGET_LANGUAGE_COLUMNS)
        query_terms_col = first_existing_column(fieldnames, EXPECTED_QUERY_TERMS_COLUMNS)
        question_terms_col = first_existing_column(fieldnames, EXPECTED_QUESTION_TERMS_COLUMNS)

        if not question_col:
            raise RuntimeError("CSV must contain a query/question column")
        if not related_col:
            raise RuntimeError("CSV must contain expected_retrieval_related")
        if not clear_col:
            raise RuntimeError("CSV must contain expected_clear")

        for row_number, row in enumerate(reader, start=2):
            query = row_value(row, question_col).strip()
            if not query:
                continue

            history = parse_history_json(row_value(row, history_json_col)) if history_json_col else []
            if not history:
                history = parse_history_text(row_value(row, history_col))

            expected_related = normalize_bool(row_value(row, related_col))
            expected_clear = normalize_bool(row_value(row, clear_col))
            if expected_related is None:
                raise RuntimeError(f"expected_retrieval_related is empty at row {row_number}")
            if expected_clear is None:
                raise RuntimeError(f"expected_clear is empty at row {row_number}")

            expected_target_language_raw = row_value(row, target_language_col).strip()
            cases.append(
                {
                    "query": query,
                    "history": history,
                    "intent": normalize_intent_hint(row_value(row, intent_col)),
                    "language": normalize_language(row_value(row, language_col) or "en"),
                    "expected_retrieval_related": expected_related,
                    "expected_clear": expected_clear,
                    "expected_target_language": (
                        normalize_language(expected_target_language_raw)
                        if expected_target_language_raw
                        else ""
                    ),
                    "expected_query_terms": row_value(row, query_terms_col).strip(),
                    "expected_question_terms": row_value(row, question_terms_col).strip(),
                }
            )

    if not cases:
        raise RuntimeError(f"clarity CSV is empty after filtering: {csv_path}")
    return cases


def settings_for_clarity(class_model: str) -> SimpleNamespace:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env", override=True)
    load_dotenv(EVAL_DIR / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in python/RAG/.env, repo .env, or environment")

    model = class_model or os.getenv("CLASS_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    return SimpleNamespace(
        openai_api_key=api_key,
        class_model=model,
        llm_model=os.getenv("LLM_MODEL", model),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    )


def history_payload(history: List[ConversationMessage]) -> List[Dict[str, Any]]:
    return [{"role": item.role, "text": item.text, "ts": item.ts} for item in history]


def cache_key(model: str, query: str, language: str, intent: str, history: List[ConversationMessage]) -> str:
    return json.dumps(
        {
            "model": model,
            "version": CACHE_VERSION,
            "query": query,
            "language": language,
            "intent": intent,
            "history": history_payload(history),
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


def clarity_with_retry(
    gateway: OpenAIGateway,
    query: str,
    language: str,
    intent: str,
    history: List[ConversationMessage],
    attempts: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    attempts = max(1, attempts)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            clarity, _usage = gateway.clarify_or_rewrite_query(query, language, intent, history=history)
            if (clarity.reason or "").startswith("clarity check failed:"):
                raise RuntimeError(clarity.reason)
            return {
                "is_retrieval_related": clarity.is_retrieval_related,
                "is_clear": clarity.is_clear,
                "standalone_query": clarity.standalone_query,
                "clarifying_question": clarity.clarifying_question,
                "target_language": clarity.target_language,
                "reason": clarity.reason,
            }
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))

    raise RuntimeError(f"clarity failed after {attempts} attempt(s): {last_error}")


def predict_clarity(
    gateway: OpenAIGateway,
    model: str,
    case: Dict[str, Any],
    attempts: int,
    retry_backoff: float,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
) -> Tuple[Dict[str, Any], bool]:
    key = cache_key(model, case["query"], case["language"], case["intent"], case["history"])
    prediction = None if refresh_cache else cache.get(key)
    cache_hit = prediction is not None
    if prediction is None:
        prediction = clarity_with_retry(
            gateway,
            case["query"],
            case["language"],
            case["intent"],
            case["history"],
            attempts,
            retry_backoff,
        )
        cache[key] = prediction
    return prediction, cache_hit


def parse_term_groups(value: str) -> List[List[str]]:
    groups: List[List[str]] = []
    for group in (value or "").split(";"):
        alternatives = [part.strip().casefold() for part in group.split("|") if part.strip()]
        if alternatives:
            groups.append(alternatives)
    return groups


def missing_term_groups(text: str, expected_terms: str) -> List[str]:
    normalized = (text or "").casefold()
    missing: List[str] = []
    for alternatives in parse_term_groups(expected_terms):
        if not any(alternative in normalized for alternative in alternatives):
            missing.append("|".join(alternatives))
    return missing


def term_check(text: str, expected_terms: str) -> Optional[bool]:
    if not (expected_terms or "").strip():
        return None
    return not missing_term_groups(text, expected_terms)


def evaluate_cases(
    cases: List[Dict[str, Any]],
    gateway: OpenAIGateway,
    model: str,
    attempts: int,
    retry_backoff: float,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
    color_enabled: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prediction, cache_hit = predict_clarity(
            gateway,
            model,
            case,
            attempts,
            retry_backoff,
            cache,
            refresh_cache,
        )
        predicted_related = bool(prediction.get("is_retrieval_related"))
        predicted_clear = bool(prediction.get("is_clear"))
        predicted_target_language = normalize_language(str(prediction.get("target_language") or "")) if prediction.get("target_language") else ""
        standalone_query = str(prediction.get("standalone_query") or "")
        clarifying_question = str(prediction.get("clarifying_question") or "")
        query_terms_correct = term_check(standalone_query, case["expected_query_terms"])
        question_terms_correct = term_check(clarifying_question, case["expected_question_terms"])
        target_language_correct = (
            None
            if not case["expected_target_language"]
            else predicted_target_language == case["expected_target_language"]
        )
        row_case = {
            **case,
            "history": history_payload(case["history"]),
        }
        row = {
            **row_case,
            "predicted_retrieval_related": predicted_related,
            "predicted_clear": predicted_clear,
            "predicted_target_language": predicted_target_language,
            "standalone_query": standalone_query,
            "clarifying_question": clarifying_question,
            "reason": str(prediction.get("reason") or ""),
            "cache_hit": cache_hit,
            "retrieval_related_correct": predicted_related == case["expected_retrieval_related"],
            "clear_correct": predicted_clear == case["expected_clear"],
            "target_language_correct": target_language_correct,
            "query_terms_correct": query_terms_correct,
            "question_terms_correct": question_terms_correct,
            "missing_query_terms": missing_term_groups(standalone_query, case["expected_query_terms"]),
            "missing_question_terms": missing_term_groups(clarifying_question, case["expected_question_terms"]),
        }
        content_checks = [
            value
            for value in (
                row["target_language_correct"],
                row["query_terms_correct"],
                row["question_terms_correct"],
            )
            if value is not None
        ]
        row["decision_correct"] = row["retrieval_related_correct"] and row["clear_correct"]
        row["strict_correct"] = row["decision_correct"] and all(content_checks)
        rows.append(row)

        status = "ok" if row["strict_correct"] else "MISS"
        status_display = colorize(status, "green", color_enabled) if row["strict_correct"] else status
        line = (
            f"[{index:03d}/{len(cases):03d}] "
            f"{status_display} "
            f"expected=(related={case['expected_retrieval_related']} clear={case['expected_clear']} target={case['expected_target_language'] or '-'}) "
            f"predicted=(related={predicted_related} clear={predicted_clear} target={predicted_target_language or '-'}) "
            f"query={case['query'][:80]}"
        )
        print(line if row["strict_correct"] else colorize(line, "red", color_enabled))
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_for_boolean(rows: List[Dict[str, Any]], expected_key: str, predicted_key: str) -> Dict[str, float]:
    tp = sum(1 for row in rows if row[expected_key] is True and row[predicted_key] is True)
    fp = sum(1 for row in rows if row[expected_key] is False and row[predicted_key] is True)
    fn = sum(1 for row in rows if row[expected_key] is True and row[predicted_key] is False)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def optional_accuracy(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    selected = [row for row in rows if row.get(key) is not None]
    if not selected:
        return None
    return safe_divide(sum(1 for row in selected if row[key]), len(selected))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cases": len(rows),
        "decision_accuracy": safe_divide(sum(1 for row in rows if row["decision_correct"]), len(rows)),
        "strict_accuracy": safe_divide(sum(1 for row in rows if row["strict_correct"]), len(rows)),
        "retrieval_related_accuracy": safe_divide(sum(1 for row in rows if row["retrieval_related_correct"]), len(rows)),
        "clear_accuracy": safe_divide(sum(1 for row in rows if row["clear_correct"]), len(rows)),
        "target_language_accuracy": optional_accuracy(rows, "target_language_correct"),
        "query_terms_accuracy": optional_accuracy(rows, "query_terms_correct"),
        "question_terms_accuracy": optional_accuracy(rows, "question_terms_correct"),
        "retrieval_related_f1": f1_for_boolean(
            rows,
            "expected_retrieval_related",
            "predicted_retrieval_related",
        ),
        "clear_f1": f1_for_boolean(rows, "expected_clear", "predicted_clear"),
        "cache_hits": sum(1 for row in rows if row.get("cache_hit")),
        "api_calls": sum(1 for row in rows if not row.get("cache_hit")),
    }


def print_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"Clarity evaluation on {summary['cases']} labeled queries")
    print(f"Decision Acc       : {summary['decision_accuracy']:.4f}")
    print(f"Strict Acc         : {summary['strict_accuracy']:.4f}")
    print(f"Retrieval Rel Acc  : {summary['retrieval_related_accuracy']:.4f}")
    print(f"Clear Acc          : {summary['clear_accuracy']:.4f}")
    print(
        "Retrieval Rel F1   : "
        f"{summary['retrieval_related_f1']['f1']:.4f} "
        f"(P={summary['retrieval_related_f1']['precision']:.4f} R={summary['retrieval_related_f1']['recall']:.4f})"
    )
    print(
        "Clear F1           : "
        f"{summary['clear_f1']['f1']:.4f} "
        f"(P={summary['clear_f1']['precision']:.4f} R={summary['clear_f1']['recall']:.4f})"
    )
    if summary["target_language_accuracy"] is not None:
        print(f"Target Lang Acc    : {summary['target_language_accuracy']:.4f}")
    if summary["query_terms_accuracy"] is not None:
        print(f"Query Terms Acc    : {summary['query_terms_accuracy']:.4f}")
    if summary["question_terms_accuracy"] is not None:
        print(f"Question Terms Acc : {summary['question_terms_accuracy']:.4f}")
    print(f"Cache/API          : {summary['cache_hits']} cached, {summary['api_calls']} API calls")

    misses = [row for row in rows if not row["strict_correct"]]
    if misses:
        print("\n" + colorize("Clarity misses:", "red", color_enabled))
        for row in misses[:max_examples]:
            print(colorize(f"- {row['query']}", "red", color_enabled))
            print(
                "  expected : "
                f"related={row['expected_retrieval_related']} clear={row['expected_clear']} "
                f"target={row['expected_target_language'] or '-'}"
            )
            print(
                "  predicted: "
                f"related={row['predicted_retrieval_related']} clear={row['predicted_clear']} "
                f"target={row['predicted_target_language'] or '-'} ({row['reason']})"
            )
            if row["standalone_query"]:
                print(f"  rewrite  : {row['standalone_query']}")
            if row["clarifying_question"]:
                print(f"  clarify  : {row['clarifying_question']}")
            if row["missing_query_terms"]:
                print(f"  missing rewrite terms: {row['missing_query_terms']}")
            if row["missing_question_terms"]:
                print(f"  missing question terms: {row['missing_question_terms']}")


def save_report(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote report -> {path}")


def save_predictions_csv(path: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "intent",
        "language",
        "expected_retrieval_related",
        "predicted_retrieval_related",
        "retrieval_related_correct",
        "expected_clear",
        "predicted_clear",
        "clear_correct",
        "expected_target_language",
        "predicted_target_language",
        "target_language_correct",
        "standalone_query",
        "clarifying_question",
        "query_terms_correct",
        "question_terms_correct",
        "decision_correct",
        "strict_correct",
        "cache_hit",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"[eval] wrote predictions -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production RAG clarity gate: whether a query should run retrieval, "
            "ask clarification, or become a standalone rewritten search query."
        )
    )
    parser.add_argument("--csv", default=str(EVAL_DIR / "clarity_mappings.csv"), help="CSV with clarity labels")
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
    parser.add_argument("--cache", default=str(OUT_DIR / "clarity_eval_cache.json"), help="Prediction cache path")
    parser.add_argument("--output", default=str(OUT_DIR / "clarity_eval_report.json"), help="JSON report path")
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
    cases = load_cases(Path(args.csv))
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    settings = settings_for_clarity(args.class_model)
    gateway = OpenAIGateway(settings)
    cache_path = None if args.no_cache else Path(args.cache)
    cache = load_cache(cache_path)
    color_enabled = should_colorize(args.color)

    rows = evaluate_cases(
        cases,
        gateway,
        model=settings.class_model,
        attempts=args.attempts,
        retry_backoff=args.retry_backoff,
        cache=cache,
        refresh_cache=args.refresh_cache,
        color_enabled=color_enabled,
    )
    summary = summarize(rows)
    print_report(summary, rows, max_examples=args.max_examples, color_enabled=color_enabled)
    save_report(Path(args.output), summary, rows)
    save_predictions_csv(Path(args.predictions_csv) if args.predictions_csv else None, rows)
    save_cache(cache_path, cache)


if __name__ == "__main__":
    main()

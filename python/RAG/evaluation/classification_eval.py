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

from rag_service.domain.models import normalize_language, normalize_profile_action
from rag_service.infrastructure.openai_gateway import OpenAIGateway


INTENTS = [
    "GREETING",
    "CHITCHAT",
    "FACTUAL_QUESTION",
    "PROCEDURE",
    "COMPARISON",
    "OUT_OF_DOMAIN",
]

QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
INTENT_COLUMNS = (
    "expected_intent",
    "intent",
    "expected_class",
    "class",
    "classification",
    "label",
    "expected_label",
)
LANGUAGE_COLUMNS = ("expected_language", "language", "lang", "expected_lang")
PROFILE_ACTION_COLUMNS = ("expected_profile_action", "profile_action")
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


def normalize_intent(value: str) -> str:
    normalized = (value or "").strip().upper()
    aliases = {
        "CHIT_CHAT": "CHITCHAT",
        "GUIDANCE": "PROCEDURE",
        "DOCUMENT_REQUEST": "FACTUAL_QUESTION",
        "OTHER": "OUT_OF_DOMAIN",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INTENTS:
        raise ValueError(f"unsupported intent {value!r}; expected one of {INTENTS}")
    return normalized


def load_cases(
    csv_path: Path,
    question_column: str = "",
    intent_column: str = "",
    language_column: str = "",
    profile_action_column: str = "",
    default_intent: str = "",
) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise RuntimeError(f"classification CSV not found: {csv_path}")

    cases: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("classification CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = question_column or first_existing_column(fieldnames, QUESTION_COLUMNS)
        intent_col = intent_column or first_existing_column(fieldnames, INTENT_COLUMNS)
        language_col = language_column or first_existing_column(fieldnames, LANGUAGE_COLUMNS)
        profile_action_col = profile_action_column or first_existing_column(fieldnames, PROFILE_ACTION_COLUMNS)

        if not question_col:
            raise RuntimeError(
                "CSV must contain a question column. Supported names: "
                + ", ".join(QUESTION_COLUMNS)
            )
        if not intent_col and not default_intent:
            raise RuntimeError(
                "CSV must contain an expected intent column or pass --default-intent. "
                "Supported names: " + ", ".join(INTENT_COLUMNS)
            )

        normalized_default_intent = normalize_intent(default_intent) if default_intent else ""
        for row in reader:
            query = (row.get(question_col) or "").strip()
            if not query:
                continue

            raw_intent = (row.get(intent_col) or "").strip() if intent_col else normalized_default_intent
            if not raw_intent:
                continue

            case = {
                "query": query,
                "expected_intent": normalize_intent(raw_intent),
                "expected_language": normalize_language(row.get(language_col) or "") if language_col else "",
                "expected_profile_action": normalize_profile_action(row.get(profile_action_col) or "")
                if profile_action_col
                else "",
            }
            cases.append(case)

    if not cases:
        raise RuntimeError(f"classification CSV is empty after filtering: {csv_path}")
    return cases


def settings_for_classifier(class_model: str) -> SimpleNamespace:
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


def cache_key(model: str, query: str) -> str:
    return json.dumps({"model": model, "query": query}, ensure_ascii=False, sort_keys=True)


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
) -> Dict[str, str]:
    attempts = max(1, attempts)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            classification, _usage = gateway.classify_query(query)
            if (classification.explain or "").startswith("classifier error:"):
                raise RuntimeError(classification.explain)
            return {
                "intent": classification.intent,
                "explain": classification.explain,
                "language": classification.language,
                "profile_action": classification.profile_action,
                "preferred_name": classification.preferred_name,
                "confidence": classification.confidence,
                "needs_rag": classification.needs_rag,
                "route": classification.route,
                "rewritten_query": classification.rewritten_query,
            }
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))

    raise RuntimeError(f"classification failed after {attempts} attempt(s): {last_error}")


def evaluate_cases(
    cases: List[Dict[str, str]],
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
        query = case["query"]
        key = cache_key(model, query)
        prediction = None if refresh_cache else cache.get(key)
        cache_hit = prediction is not None
        if prediction is None:
            prediction = classify_with_retry(gateway, query, attempts=attempts, retry_backoff=retry_backoff)
            cache[key] = prediction

        predicted_intent = normalize_intent(str(prediction.get("intent") or "OTHER"))
        predicted_language = normalize_language(str(prediction.get("language") or ""))
        predicted_profile_action = normalize_profile_action(str(prediction.get("profile_action") or ""))
        row = {
            **case,
            "predicted_intent": predicted_intent,
            "predicted_language": predicted_language,
            "predicted_profile_action": predicted_profile_action,
            "explain": str(prediction.get("explain") or ""),
            "cache_hit": cache_hit,
            "intent_correct": predicted_intent == case["expected_intent"],
            "language_correct": (
                None
                if not case.get("expected_language")
                else predicted_language == case["expected_language"]
            ),
            "profile_action_correct": (
                None
                if not case.get("expected_profile_action")
                else predicted_profile_action == case["expected_profile_action"]
            ),
            "confidence": prediction.get("confidence"),
            "needs_rag": prediction.get("needs_rag"),
            "route": prediction.get("route"),
            "rewritten_query": str(prediction.get("rewritten_query") or ""),
        }
        rows.append(row)
        status = "ok" if row["intent_correct"] else "MISS"
        status_display = colorize(status, "green", color_enabled) if row["intent_correct"] else status
        line = (
            f"[{index:03d}/{len(cases):03d}] "
            f"{status_display} "
            f"expected={row['expected_intent']} predicted={row['predicted_intent']} "
            f"query={query[:80]}"
        )
        print(line if row["intent_correct"] else colorize(line, "red", color_enabled))
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    labels = [label for label in INTENTS if any(row["expected_intent"] == label or row["predicted_intent"] == label for row in rows)]
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for row in rows:
        confusion.setdefault(row["expected_intent"], {predicted: 0 for predicted in labels})
        confusion[row["expected_intent"]].setdefault(row["predicted_intent"], 0)
        confusion[row["expected_intent"]][row["predicted_intent"]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    total = len(rows)
    correct = sum(1 for row in rows if row["intent_correct"])
    for label in labels:
        tp = sum(1 for row in rows if row["expected_intent"] == label and row["predicted_intent"] == label)
        fp = sum(1 for row in rows if row["expected_intent"] != label and row["predicted_intent"] == label)
        fn = sum(1 for row in rows if row["expected_intent"] == label and row["predicted_intent"] != label)
        support = sum(1 for row in rows if row["expected_intent"] == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = safe_divide(sum(item["f1"] for item in per_class.values()), len(per_class))
    weighted_f1 = safe_divide(
        sum(item["f1"] * item["support"] for item in per_class.values()),
        sum(item["support"] for item in per_class.values()),
    )
    expected_language_rows = [row for row in rows if row.get("language_correct") is not None]
    expected_profile_action_rows = [row for row in rows if row.get("profile_action_correct") is not None]

    summary = {
        "cases": total,
        "accuracy": safe_divide(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cache_hits": sum(1 for row in rows if row.get("cache_hit")),
        "api_calls": sum(1 for row in rows if not row.get("cache_hit")),
        "per_class": per_class,
        "language_accuracy": safe_divide(
            sum(1 for row in expected_language_rows if row["language_correct"]),
            len(expected_language_rows),
        )
        if expected_language_rows
        else None,
        "profile_action_accuracy": safe_divide(
            sum(1 for row in expected_profile_action_rows if row["profile_action_correct"]),
            len(expected_profile_action_rows),
        )
        if expected_profile_action_rows
        else None,
    }
    return summary, confusion


def print_report(
    summary: Dict[str, Any],
    confusion: Dict[str, Dict[str, int]],
    rows: List[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"Classification evaluation on {summary['cases']} labeled queries")
    print(f"Accuracy    : {summary['accuracy']:.4f}")
    print(f"Macro F1    : {summary['macro_f1']:.4f}")
    print(f"Weighted F1 : {summary['weighted_f1']:.4f}")
    print(f"Cache/API   : {summary['cache_hits']} cached, {summary['api_calls']} API calls")
    if summary["language_accuracy"] is not None:
        print(f"Language Acc: {summary['language_accuracy']:.4f}")
    if summary["profile_action_accuracy"] is not None:
        print(f"Profile Acc : {summary['profile_action_accuracy']:.4f}")

    print("\nPer-class metrics:")
    for label, metrics in summary["per_class"].items():
        print(
            f"- {label:<17} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"support={int(metrics['support'])}"
        )

    print("\nConfusion matrix (rows=expected, columns=predicted):")
    labels = list(confusion.keys())
    print("expected\\predicted".ljust(20) + " ".join(label[:8].rjust(8) for label in labels))
    for expected in labels:
        counts = " ".join(str(confusion[expected].get(predicted, 0)).rjust(8) for predicted in labels)
        print(expected[:19].ljust(20) + counts)

    misses = [row for row in rows if not row["intent_correct"]]
    if misses:
        print("\n" + colorize("Intent misses:", "red", color_enabled))
        for row in misses[:max_examples]:
            print(colorize(f"- {row['query']}", "red", color_enabled))
            print(f"  expected : {row['expected_intent']}")
            print(f"  predicted: {row['predicted_intent']} ({row['explain']})")


def save_report(path: Path, summary: Dict[str, Any], confusion: Dict[str, Dict[str, int]], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "confusion": confusion,
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
        "expected_intent",
        "predicted_intent",
        "intent_correct",
        "expected_language",
        "predicted_language",
        "language_correct",
        "expected_profile_action",
        "predicted_profile_action",
        "profile_action_correct",
        "cache_hit",
        "confidence",
        "needs_rag",
        "route",
        "rewritten_query",
        "explain",
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
            "Evaluate the production LLM classifier against a labeled CSV. "
            "Uses rag_service.infrastructure.openai_gateway.OpenAIGateway.classify_query."
        )
    )
    parser.add_argument("--csv", default=str(EVAL_DIR / "query_mappings.csv"), help="CSV with question and expected intent columns")
    parser.add_argument("--question-column", default="", help="Override question column name")
    parser.add_argument("--intent-column", default="", help="Override expected intent column name")
    parser.add_argument("--language-column", default="", help="Optional expected language column name")
    parser.add_argument("--profile-action-column", default="", help="Optional expected profile action column name")
    parser.add_argument(
        "--default-intent",
        default="",
        help="Use this expected intent for rows without an intent column, useful for smoke-testing query_mappings.csv",
    )
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
    parser.add_argument("--cache", default=str(OUT_DIR / "classification_eval_cache.json"), help="Prediction cache path")
    parser.add_argument("--output", default=str(OUT_DIR / "classification_eval_report.json"), help="JSON report path")
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
    csv_path = Path(args.csv)
    cases = load_cases(
        csv_path,
        question_column=args.question_column,
        intent_column=args.intent_column,
        language_column=args.language_column,
        profile_action_column=args.profile_action_column,
        default_intent=args.default_intent,
    )
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    settings = settings_for_classifier(args.class_model)
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
    summary, confusion = classification_metrics(rows)
    print_report(summary, confusion, rows, max_examples=args.max_examples, color_enabled=color_enabled)
    save_report(Path(args.output), summary, confusion, rows)
    save_predictions_csv(Path(args.predictions_csv) if args.predictions_csv else None, rows)
    save_cache(cache_path, cache)


if __name__ == "__main__":
    main()

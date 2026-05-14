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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAG_API_DIR = REPO_ROOT / "python" / "rag_api"
OUT_DIR = SCRIPT_DIR / "out"

if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

from rag_service.domain.models import ConversationMessage, normalize_language
from rag_service.infrastructure.openai_gateway import OpenAIGateway


QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
HISTORY_COLUMNS = ("history", "history_text", "conversation", "context")
HISTORY_JSON_COLUMNS = ("history_json", "conversation_json", "context_json")
ALLOWED_COLUMNS = ("expected_allowed", "allowed")
NEEDS_CONTEXT_COLUMNS = ("expected_needs_context", "needs_context")
VIOLATION_COLUMNS = ("expected_violation", "violation")
LANGUAGE_COLUMNS = ("expected_language", "language", "lang", "expected_lang")
CONTEXT_ALLOWED_COLUMNS = ("expected_context_allowed", "context_allowed")
CONTEXT_NEEDS_CONTEXT_COLUMNS = ("expected_context_needs_context", "context_needs_context")
CONTEXT_VIOLATION_COLUMNS = ("expected_context_violation", "context_violation")
CONTEXT_LANGUAGE_COLUMNS = ("expected_context_language", "context_language")
VIOLATIONS = {"", "out_of_scope", "unsafe", "unsupported_language"}
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


def normalize_bool(value: str, default: Optional[bool] = None) -> Optional[bool]:
    normalized = (value or "").strip().lower()
    if normalized == "":
        return default
    if normalized in {"1", "true", "yes", "y", "allowed", "allow"}:
        return True
    if normalized in {"0", "false", "no", "n", "blocked", "block", "deny"}:
        return False
    raise ValueError(f"unsupported boolean value {value!r}")


def normalize_violation(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"none", "null", "safe", "allowed"}:
        normalized = ""
    if normalized not in VIOLATIONS:
        raise ValueError(f"unsupported violation {value!r}; expected one of {sorted(VIOLATIONS)}")
    return normalized


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
        raise RuntimeError(f"guardrail CSV not found: {csv_path}")

    cases: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("guardrail CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = first_existing_column(fieldnames, QUESTION_COLUMNS)
        allowed_col = first_existing_column(fieldnames, ALLOWED_COLUMNS)
        needs_context_col = first_existing_column(fieldnames, NEEDS_CONTEXT_COLUMNS)
        violation_col = first_existing_column(fieldnames, VIOLATION_COLUMNS)
        language_col = first_existing_column(fieldnames, LANGUAGE_COLUMNS)
        history_col = first_existing_column(fieldnames, HISTORY_COLUMNS)
        history_json_col = first_existing_column(fieldnames, HISTORY_JSON_COLUMNS)
        context_allowed_col = first_existing_column(fieldnames, CONTEXT_ALLOWED_COLUMNS)
        context_needs_context_col = first_existing_column(fieldnames, CONTEXT_NEEDS_CONTEXT_COLUMNS)
        context_violation_col = first_existing_column(fieldnames, CONTEXT_VIOLATION_COLUMNS)
        context_language_col = first_existing_column(fieldnames, CONTEXT_LANGUAGE_COLUMNS)

        if not question_col:
            raise RuntimeError("CSV must contain a query/question column")
        if not allowed_col:
            raise RuntimeError("CSV must contain expected_allowed")

        for row_number, row in enumerate(reader, start=2):
            query = row_value(row, question_col).strip()
            if not query:
                continue

            history = parse_history_json(row_value(row, history_json_col)) if history_json_col else []
            if not history:
                history = parse_history_text(row_value(row, history_col))

            expected_allowed = normalize_bool(row_value(row, allowed_col))
            if expected_allowed is None:
                raise RuntimeError(f"expected_allowed is empty at row {row_number}")
            expected_needs_context = bool(normalize_bool(row_value(row, needs_context_col), default=False))
            expected_violation = normalize_violation(row_value(row, violation_col))
            expected_language = normalize_language(row_value(row, language_col)) if language_col else ""

            context_allowed_raw = row_value(row, context_allowed_col)
            has_context_expected = context_allowed_raw.strip() != ""
            context_expected = None
            if has_context_expected:
                context_allowed = normalize_bool(context_allowed_raw)
                if context_allowed is None:
                    raise RuntimeError(f"expected_context_allowed is empty at row {row_number}")
                context_expected = {
                    "allowed": context_allowed,
                    "needs_context": bool(normalize_bool(row_value(row, context_needs_context_col), default=False)),
                    "violation": normalize_violation(row_value(row, context_violation_col)),
                    "language": normalize_language(row_value(row, context_language_col)) if context_language_col else "",
                }

            cases.append(
                {
                    "query": query,
                    "history": history,
                    "expected": {
                        "allowed": expected_allowed,
                        "needs_context": expected_needs_context,
                        "violation": expected_violation,
                        "language": expected_language,
                    },
                    "context_expected": context_expected,
                }
            )

    if not cases:
        raise RuntimeError(f"guardrail CSV is empty after filtering: {csv_path}")
    return cases


def settings_for_guardrail(class_model: str) -> SimpleNamespace:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env", override=True)

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


def cache_key(model: str, query: str, history: List[ConversationMessage]) -> str:
    return json.dumps(
        {"model": model, "query": query, "history": history_payload(history)},
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


def guard_with_retry(
    gateway: OpenAIGateway,
    query: str,
    history: List[ConversationMessage],
    attempts: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    attempts = max(1, attempts)
    last_error: Optional[Exception] = None
    history_arg = history if history else None
    for attempt in range(1, attempts + 1):
        try:
            guardrail, _usage = gateway.guard_query(query, history=history_arg)
            return {
                "allowed": guardrail.allowed,
                "needs_context": guardrail.needs_context,
                "violation": guardrail.violation,
                "language": guardrail.language,
                "reason": guardrail.reason,
            }
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, retry_backoff))

    raise RuntimeError(f"guardrail failed after {attempts} attempt(s): {last_error}")


def predict_guardrail(
    gateway: OpenAIGateway,
    model: str,
    query: str,
    history: List[ConversationMessage],
    attempts: int,
    retry_backoff: float,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
) -> Tuple[Dict[str, Any], bool]:
    key = cache_key(model, query, history)
    prediction = None if refresh_cache else cache.get(key)
    cache_hit = prediction is not None
    if prediction is None:
        prediction = guard_with_retry(gateway, query, history, attempts, retry_backoff)
        cache[key] = prediction
    return prediction, cache_hit


def stage_row(
    case: Dict[str, Any],
    stage: str,
    expected: Dict[str, Any],
    prediction: Dict[str, Any],
    cache_hit: bool,
) -> Dict[str, Any]:
    predicted_allowed = bool(prediction.get("allowed"))
    predicted_needs_context = bool(prediction.get("needs_context"))
    predicted_violation = normalize_violation(str(prediction.get("violation") or ""))
    predicted_language = normalize_language(str(prediction.get("language") or ""))
    expected_language = str(expected.get("language") or "")
    row = {
        "stage": stage,
        "query": case["query"],
        "has_history": bool(case.get("history")),
        "expected_allowed": bool(expected["allowed"]),
        "predicted_allowed": predicted_allowed,
        "allowed_correct": predicted_allowed == bool(expected["allowed"]),
        "expected_needs_context": bool(expected["needs_context"]),
        "predicted_needs_context": predicted_needs_context,
        "needs_context_correct": predicted_needs_context == bool(expected["needs_context"]),
        "expected_violation": str(expected.get("violation") or ""),
        "predicted_violation": predicted_violation,
        "violation_correct": predicted_violation == str(expected.get("violation") or ""),
        "expected_language": expected_language,
        "predicted_language": predicted_language,
        "language_correct": None if not expected_language else predicted_language == expected_language,
        "reason": str(prediction.get("reason") or ""),
        "cache_hit": cache_hit,
    }
    row["decision_correct"] = (
        row["allowed_correct"]
        and row["needs_context_correct"]
        and row["violation_correct"]
    )
    return row


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
        first_prediction, first_cache_hit = predict_guardrail(
            gateway,
            model,
            case["query"],
            [],
            attempts,
            retry_backoff,
            cache,
            refresh_cache,
        )
        first_row = stage_row(case, "first_pass", case["expected"], first_prediction, first_cache_hit)
        rows.append(first_row)

        context_prediction = None
        context_cache_hit = False
        if case.get("history") and case.get("context_expected") is not None:
            context_prediction, context_cache_hit = predict_guardrail(
                gateway,
                model,
                case["query"],
                case["history"],
                attempts,
                retry_backoff,
                cache,
                refresh_cache,
            )
            rows.append(
                stage_row(
                    case,
                    "contextual",
                    case["context_expected"],
                    context_prediction,
                    context_cache_hit,
                )
            )

        if first_prediction.get("needs_context") and case.get("history") and context_prediction is None:
            context_prediction, context_cache_hit = predict_guardrail(
                gateway,
                model,
                case["query"],
                case["history"],
                attempts,
                retry_backoff,
                cache,
                refresh_cache,
            )

        pipeline_prediction = context_prediction if first_prediction.get("needs_context") and context_prediction else first_prediction
        pipeline_expected = case["context_expected"] if case["expected"]["needs_context"] and case.get("context_expected") else case["expected"]
        pipeline_cache_hit = context_cache_hit if pipeline_prediction is context_prediction else first_cache_hit
        rows.append(stage_row(case, "pipeline", pipeline_expected, pipeline_prediction, pipeline_cache_hit))

        status = "ok" if first_row["decision_correct"] else "MISS"
        status_display = colorize(status, "green", color_enabled) if first_row["decision_correct"] else status
        line = (
            f"[{index:03d}/{len(cases):03d}] "
            f"first={status_display} "
            f"expected=(allowed={first_row['expected_allowed']} context={first_row['expected_needs_context']} violation={first_row['expected_violation'] or '-'}) "
            f"predicted=(allowed={first_row['predicted_allowed']} context={first_row['predicted_needs_context']} violation={first_row['predicted_violation'] or '-'}) "
            f"query={case['query'][:80]}"
        )
        print(line if first_row["decision_correct"] else colorize(line, "red", color_enabled))
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_for_boolean(rows: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    expected_key = f"expected_{field}"
    predicted_key = f"predicted_{field}"
    tp = sum(1 for row in rows if row[expected_key] is True and row[predicted_key] is True)
    fp = sum(1 for row in rows if row[expected_key] is False and row[predicted_key] is True)
    fn = sum(1 for row in rows if row[expected_key] is True and row[predicted_key] is False)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def violation_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted({row["expected_violation"] for row in rows} | {row["predicted_violation"] for row in rows})
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for row in rows if row["expected_violation"] == label and row["predicted_violation"] == label)
        fp = sum(1 for row in rows if row["expected_violation"] != label and row["predicted_violation"] == label)
        fn = sum(1 for row in rows if row["expected_violation"] == label and row["predicted_violation"] != label)
        support = sum(1 for row in rows if row["expected_violation"] == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[label or "none"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = safe_divide(sum(item["f1"] for item in per_class.values()), len(per_class))
    return {"macro_f1": macro_f1, "per_class": per_class}


def metrics_by_stage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stages = sorted({row["stage"] for row in rows})
    out: Dict[str, Any] = {}
    for stage in stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        language_rows = [row for row in stage_rows if row.get("language_correct") is not None]
        out[stage] = {
            "cases": len(stage_rows),
            "decision_accuracy": safe_divide(sum(1 for row in stage_rows if row["decision_correct"]), len(stage_rows)),
            "allowed_accuracy": safe_divide(sum(1 for row in stage_rows if row["allowed_correct"]), len(stage_rows)),
            "needs_context_accuracy": safe_divide(sum(1 for row in stage_rows if row["needs_context_correct"]), len(stage_rows)),
            "violation_accuracy": safe_divide(sum(1 for row in stage_rows if row["violation_correct"]), len(stage_rows)),
            "language_accuracy": (
                safe_divide(sum(1 for row in language_rows if row["language_correct"]), len(language_rows))
                if language_rows
                else None
            ),
            "allowed_f1": f1_for_boolean(stage_rows, "allowed"),
            "needs_context_f1": f1_for_boolean(stage_rows, "needs_context"),
            "violation": violation_metrics(stage_rows),
            "cache_hits": sum(1 for row in stage_rows if row.get("cache_hit")),
            "api_calls": sum(1 for row in stage_rows if not row.get("cache_hit")),
        }
    return out


def print_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
) -> None:
    print("\n" + "=" * 80)
    print("Guardrail evaluation")
    for stage, metrics in summary.items():
        print(f"\n[{stage}] cases={metrics['cases']} cache/api={metrics['cache_hits']}/{metrics['api_calls']}")
        print(f"Decision Acc     : {metrics['decision_accuracy']:.4f}")
        print(f"Allowed Acc      : {metrics['allowed_accuracy']:.4f}")
        print(f"Needs Context Acc: {metrics['needs_context_accuracy']:.4f}")
        print(f"Violation Acc    : {metrics['violation_accuracy']:.4f}")
        if metrics["language_accuracy"] is not None:
            print(f"Language Acc     : {metrics['language_accuracy']:.4f}")
        print(
            "Allowed F1       : "
            f"{metrics['allowed_f1']['f1']:.4f} "
            f"(P={metrics['allowed_f1']['precision']:.4f} R={metrics['allowed_f1']['recall']:.4f})"
        )
        print(
            "Needs Context F1 : "
            f"{metrics['needs_context_f1']['f1']:.4f} "
            f"(P={metrics['needs_context_f1']['precision']:.4f} R={metrics['needs_context_f1']['recall']:.4f})"
        )
        print(f"Violation Macro F1: {metrics['violation']['macro_f1']:.4f}")

    misses = [row for row in rows if not row["decision_correct"]]
    if misses:
        print("\n" + colorize("Decision misses:", "red", color_enabled))
        for row in misses[:max_examples]:
            print(colorize(f"- [{row['stage']}] {row['query']}", "red", color_enabled))
            print(
                "  expected : "
                f"allowed={row['expected_allowed']} needs_context={row['expected_needs_context']} "
                f"violation={row['expected_violation'] or '-'} language={row['expected_language'] or '-'}"
            )
            print(
                "  predicted: "
                f"allowed={row['predicted_allowed']} needs_context={row['predicted_needs_context']} "
                f"violation={row['predicted_violation'] or '-'} language={row['predicted_language'] or '-'} "
                f"({row['reason']})"
            )


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
        "stage",
        "query",
        "has_history",
        "expected_allowed",
        "predicted_allowed",
        "allowed_correct",
        "expected_needs_context",
        "predicted_needs_context",
        "needs_context_correct",
        "expected_violation",
        "predicted_violation",
        "violation_correct",
        "expected_language",
        "predicted_language",
        "language_correct",
        "decision_correct",
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
            "Evaluate the production LLM guardrail against a labeled CSV. "
            "Reports first-pass, contextual, and real pipeline metrics."
        )
    )
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "guardrail_mappings.csv"), help="CSV with guardrail labels")
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
    parser.add_argument("--cache", default=str(OUT_DIR / "guardrail_eval_cache.json"), help="Prediction cache path")
    parser.add_argument("--output", default=str(OUT_DIR / "guardrail_eval_report.json"), help="JSON report path")
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

    settings = settings_for_guardrail(args.class_model)
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
    summary = metrics_by_stage(rows)
    print_report(summary, rows, max_examples=args.max_examples, color_enabled=color_enabled)
    save_report(Path(args.output), summary, rows)
    save_predictions_csv(Path(args.predictions_csv) if args.predictions_csv else None, rows)
    save_cache(cache_path, cache)


if __name__ == "__main__":
    main()

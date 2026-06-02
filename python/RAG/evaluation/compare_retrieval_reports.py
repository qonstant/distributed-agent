#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METRICS = [
    ("accuracy_at_1", "Acc@1"),
    ("hit_rate_at_k", "Hit@K"),
    ("precision_at_k", "Prec@K"),
    ("recall_at_k", "Rec@K"),
    ("ndcg_at_k", "NDCG"),
    ("mrr", "MRR"),
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def metric_delta(value: Any, baseline: Any) -> str:
    try:
        delta = float(value) - float(baseline)
    except Exception:
        return "-"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def row_key(row: Dict[str, Any]) -> str:
    return str(row.get("query") or "").strip()


def indexed_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row_key(row): row for row in rows if row_key(row)}


def win_loss_tie(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    field: str,
) -> Tuple[int, int, int, int]:
    baseline_by_query = indexed_rows(baseline_rows)
    candidate_by_query = indexed_rows(candidate_rows)
    common_queries = sorted(set(baseline_by_query) & set(candidate_by_query))

    wins = losses = ties = 0
    for query in common_queries:
        baseline_ok = bool(baseline_by_query[query].get(field))
        candidate_ok = bool(candidate_by_query[query].get(field))
        if candidate_ok and not baseline_ok:
            wins += 1
        elif baseline_ok and not candidate_ok:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, len(common_queries)


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [
        max(len(str(item)) for item in [header] + [row[index] for row in rows])
        for index, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * width for width in widths)
    print(header_line)
    print(divider)
    for row in rows:
        print("  ".join(str(item).ljust(widths[index]) for index, item in enumerate(row)))


def build_summary_rows(
    baseline_summary: Dict[str, Any],
    lightrag_summary: Dict[str, Dict[str, Any]],
) -> List[List[str]]:
    rows: List[List[str]] = []
    baseline_top_k = baseline_summary.get("top_k", 5)
    rows.append(
        [
            "faiss_pipeline",
            str(baseline_summary.get("queries", "-")),
            str(baseline_top_k),
            *[format_float(baseline_summary.get(key)) for key, _label in METRICS],
            "baseline",
        ]
    )

    for mode, summary in lightrag_summary.items():
        rows.append(
            [
                f"lightrag:{mode}",
                str(summary.get("queries", "-")),
                str(summary.get("top_k", baseline_top_k)),
                *[
                    f"{format_float(summary.get(key))} ({metric_delta(summary.get(key), baseline_summary.get(key))})"
                    for key, _label in METRICS
                ],
                "vs faiss",
            ]
        )
    return rows


def print_overlap_report(
    baseline_rows: List[Dict[str, Any]],
    lightrag_rows_by_mode: Dict[str, List[Dict[str, Any]]],
) -> None:
    print("\nPer-query comparison vs faiss_pipeline")
    headers = ["Mode", "Common", "Top1 W/L/T", "Hit W/L/T"]
    rows: List[List[str]] = []
    for mode, rows_for_mode in lightrag_rows_by_mode.items():
        top1_w, top1_l, top1_t, common = win_loss_tie(baseline_rows, rows_for_mode, "top1_correct")
        hit_w, hit_l, hit_t, _common = win_loss_tie(baseline_rows, rows_for_mode, "hit")
        rows.append(
            [
                f"lightrag:{mode}",
                str(common),
                f"{top1_w}/{top1_l}/{top1_t}",
                f"{hit_w}/{hit_l}/{hit_t}",
            ]
        )
    print_table(headers, rows)


def print_query_set_report(
    baseline_rows: List[Dict[str, Any]],
    lightrag_rows_by_mode: Dict[str, List[Dict[str, Any]]],
) -> None:
    baseline_queries = set(indexed_rows(baseline_rows))
    for mode, rows_for_mode in lightrag_rows_by_mode.items():
        candidate_queries = set(indexed_rows(rows_for_mode))
        if candidate_queries == baseline_queries:
            print(f"Query set [{mode}]: same {len(baseline_queries)} queries")
            continue

        missing = sorted(baseline_queries - candidate_queries)
        extra = sorted(candidate_queries - baseline_queries)
        print(
            f"Query set [{mode}]: mismatch "
            f"missing={len(missing)} extra={len(extra)} common={len(baseline_queries & candidate_queries)}"
        )
        if missing:
            print(f"  first missing: {missing[0]}")
        if extra:
            print(f"  first extra   : {extra[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FAISS retrieval_eval and LightRAG eval reports.")
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--lightrag-report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_report = load_json(Path(args.retrieval_report))
    lightrag_report = load_json(Path(args.lightrag_report))

    baseline_summary = retrieval_report.get("summary") or {}
    lightrag_summary = lightrag_report.get("summary") or {}
    baseline_rows = retrieval_report.get("rows") or []
    lightrag_rows_by_mode = lightrag_report.get("rows_by_mode") or {}

    print("\n" + "=" * 80)
    print("Retrieval System Comparison")
    print_table(
        ["System", "Queries", "K", *[label for _key, label in METRICS], "Note"],
        build_summary_rows(baseline_summary, lightrag_summary),
    )
    print(
        "\nFAISS cache/API: "
        f"{baseline_summary.get('cache_hits', '-')} cached, "
        f"{baseline_summary.get('api_calls', '-')} API calls"
    )
    print_query_set_report(baseline_rows, lightrag_rows_by_mode)
    print_overlap_report(baseline_rows, lightrag_rows_by_mode)


if __name__ == "__main__":
    main()

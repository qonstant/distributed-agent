#!/usr/bin/env python3
"""Render a readable LightRAG retrieval report with top files and page refs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODE = "naive+meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print expanded per-query LightRAG retrieval results."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON report produced by lightrag_eval.py.",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        help=f"LightRAG result mode to render (default: {DEFAULT_MODE}).",
    )
    parser.add_argument(
        "--output",
        help="Optional text output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--query",
        help="Only include rows whose query contains this text, case-insensitive.",
    )
    parser.add_argument(
        "--only-misses",
        action="store_true",
        help="Only include rows where the expected file was not found in top-k.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also show raw candidate files before metadata filtering.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also show top chunk candidates before/after metadata filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of matching rows to render.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Maximum page refs to show per file (default: 8).",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON report: {path}")
    return data


def mode_rows(report: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    for key in ("rows_by_mode", "results", "lightrag_results"):
        results = report.get(key, {})
        if not isinstance(results, dict):
            continue
        rows = results.get(mode, [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def summary_for(report: dict[str, Any], mode: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    mode_summary = summary.get(mode, {})
    return mode_summary if isinstance(mode_summary, dict) else {}


def graph_summary(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    graph = metadata.get("graph", {})
    return graph if isinstance(graph, dict) else {}


def expected_files(row: dict[str, Any]) -> list[str]:
    expected = row.get("expected_files")
    if isinstance(expected, list):
        return [str(item) for item in expected]
    expected_file = row.get("expected_file")
    return [str(expected_file)] if expected_file else []


def retrieved_with_pages(row: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = row.get("retrieved_file_page_refs")
    if isinstance(enriched, list):
        items = [item for item in enriched if isinstance(item, dict)]
        if items:
            return items

    files = row.get("retrieved_files")
    if not isinstance(files, list):
        return []
    return [{"source_file": str(source_file), "page_refs": []} for source_file in files]


def format_pages(page_refs: Any, max_pages: int) -> str:
    if not isinstance(page_refs, list) or not page_refs:
        return "-"
    pages = [str(page) for page in page_refs]
    visible = pages[:max_pages]
    suffix = f", +{len(pages) - max_pages} more" if len(pages) > max_pages else ""
    return ", ".join(visible) + suffix


def format_chunk(chunk: Any, max_pages: int) -> str:
    if not isinstance(chunk, dict):
        return "-"
    source_file = chunk.get("source_file") or "-"
    pages = format_pages(chunk.get("page_refs"), max_pages)
    language = chunk.get("language") or "-"
    rank = chunk.get("rank") or "-"
    return f"raw_rank={rank} {source_file} lang={language} pages={pages}"


def row_matches(row: dict[str, Any], query_filter: str | None, only_misses: bool) -> bool:
    if query_filter:
        query = str(row.get("query", ""))
        if query_filter.casefold() not in query.casefold():
            return False
    if only_misses and row.get("hit"):
        return False
    return True


def render_report(
    report: dict[str, Any],
    *,
    mode: str,
    query_filter: str | None,
    only_misses: bool,
    show_raw: bool,
    show_chunks: bool,
    limit: int | None,
    max_pages: int,
) -> str:
    rows = [
        row
        for row in mode_rows(report, mode)
        if row_matches(row, query_filter, only_misses)
    ]
    if limit is not None:
        rows = rows[:limit]

    summary = summary_for(report, mode)
    graph = graph_summary(report)
    lines: list[str] = []
    lines.append("LightRAG expanded retrieval report")
    lines.append(f"Mode: {mode}")
    if summary:
        lines.append(
            "Metrics: "
            f"queries={summary.get('queries', '-')} "
            f"candidate_k={summary.get('candidate_k', '-')} "
            f"acc@1={float(summary.get('accuracy_at_1', 0)):.4f} "
            f"hit@k={float(summary.get('hit_rate_at_k', 0)):.4f} "
            f"ndcg={float(summary.get('ndcg_at_k', 0)):.4f} "
            f"mrr={float(summary.get('mrr', 0)):.4f}"
        )
    if graph:
        lines.append(
            "Graph: "
            f"nodes={graph.get('nodes', '-')} "
            f"edges={graph.get('edges', '-')} "
            f"page_refs={graph.get('page_ref_values', '-')}"
        )
    if query_filter:
        lines.append(f"Query filter: {query_filter}")
    if only_misses:
        lines.append("Rows: misses only")
    lines.append("")

    if not rows:
        lines.append("No rows matched.")
        return "\n".join(lines) + "\n"

    for idx, row in enumerate(rows, start=1):
        status = "HIT" if row.get("hit") else "MISS"
        rank = row.get("rank", row.get("first_relevant_rank"))
        rank_text = str(rank) if rank not in (None, "", 0) else "-"
        lines.append(
            f"[{idx:03d}] {status} rank={rank_text} query={row.get('query', '')}"
        )
        expected = expected_files(row)
        lines.append(f"expected: {', '.join(expected) if expected else '-'}")
        for file_idx, item in enumerate(retrieved_with_pages(row), start=1):
            source_file = item.get("source_file") or item.get("file") or "-"
            pages = format_pages(item.get("page_refs"), max_pages)
            lines.append(f"  {file_idx}. {source_file} pages={pages}")
        if show_raw:
            raw_files = row.get("raw_retrieved_files") or []
            if isinstance(raw_files, list) and raw_files:
                lines.append("raw candidates:")
                for file_idx, source_file in enumerate(raw_files, start=1):
                    lines.append(f"  {file_idx}. {source_file}")
        if show_chunks:
            retrieved_chunks = row.get("retrieved_chunks") or []
            if isinstance(retrieved_chunks, list) and retrieved_chunks:
                lines.append("filtered chunks:")
                for chunk_idx, chunk in enumerate(retrieved_chunks, start=1):
                    lines.append(f"  {chunk_idx}. {format_chunk(chunk, max_pages)}")
            raw_chunks = row.get("raw_retrieved_chunks") or []
            if isinstance(raw_chunks, list) and raw_chunks:
                lines.append("raw chunks:")
                for chunk_idx, chunk in enumerate(raw_chunks[:10], start=1):
                    lines.append(f"  {chunk_idx}. {format_chunk(chunk, max_pages)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = load_report(Path(args.input))
    text = render_report(
        report,
        mode=args.mode,
        query_filter=args.query,
        only_misses=args.only_misses,
        show_raw=args.show_raw,
        show_chunks=args.show_chunks,
        limit=args.limit,
        max_pages=args.max_pages,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"[lightrag-expanded] wrote {output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

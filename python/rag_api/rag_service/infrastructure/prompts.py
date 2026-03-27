from __future__ import annotations

from typing import List

from rag_service.domain.models import RetrievedHit


def _excerpt_from_hit(hit: RetrievedHit) -> str:
    text = (hit.meta.get("text") or hit.meta.get("md") or "").strip()
    return text[:1600].replace("\n", " ").strip()


def prepare_document_request_prompt(query: str, top_chunks: List[RetrievedHit]) -> str:
    lines = [
        "You are a strict document retriever. Use ONLY the excerpts below; do NOT invent or generalize beyond them.",
        "User query:",
        query,
        "",
        "Below are document excerpts (file + page + excerpt). If one of the documents is the requested template or sample, choose it.",
        "",
    ]
    for index, hit in enumerate(top_chunks, start=1):
        source_file = hit.meta.get("source_file") or hit.meta.get("filename") or "unknown"
        page = hit.meta.get("page")
        lines.append(f"[{index}] file: {source_file} page: {page}")
        lines.append(f"excerpt: {_excerpt_from_hit(hit)}")
        lines.append("")
    lines.extend(
        [
            "Output requirements:",
            "- Respond ONLY with valid JSON with exactly these keys: 'answer' and 'file'.",
            "- 'answer' must be a short sentence telling whether the requested document exists in the provided excerpts.",
            "- If a matching document exists, 'answer' must be a short note (<=60 words) and 'file' must be the path string of that document.",
            "- If no matching document exists in the excerpts, set 'answer' to: \"I don't know based on the provided documents.\" and 'file' to null.",
            "- Do NOT add any other keys, commentary, or explanation. Return JSON only.",
        ]
    )
    return "\n".join(lines)


def prepare_guidance_prompt(query: str, top_chunks: List[RetrievedHit]) -> str:
    lines = [
        "You are an assistant that gives practical guidance using ONLY the provided document excerpts. Do NOT invent facts.",
        "User query:",
        query,
        "",
        "Here are top document excerpts (file + page + excerpt):",
        "",
    ]
    for index, hit in enumerate(top_chunks, start=1):
        source_file = hit.meta.get("source_file") or hit.meta.get("filename") or "unknown"
        page = hit.meta.get("page")
        lines.append(f"[{index}] file: {source_file} page: {page}")
        lines.append(f"excerpt: {_excerpt_from_hit(hit)}")
        lines.append("")
    lines.extend(
        [
            "Output requirements:",
            "- Respond ONLY with valid JSON with exactly two keys: 'answer' and 'file'.",
            "- 'answer' should be a short, step-oriented guidance or summary (<=180 words) drawn only from the excerpts. If you cannot produce a guidance wholly supported by the excerpts, set 'answer' to: \"I don't know based on the provided documents.\"",
            "- 'file' should be the single best supporting source_file path from the excerpts (or null if none).",
            "- Do NOT invent, assume, or provide extra commentary. Return JSON only.",
        ]
    )
    return "\n".join(lines)

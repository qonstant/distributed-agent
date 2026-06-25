#!/usr/bin/env python3
# make lightrag EVAL_LIMIT=1 LIGHTRAG_MODES=naive
# make rag-compare COMPARE_LIMIT=25 

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv


EVAL_DIR = Path(__file__).resolve().parent
RAG_DIR = EVAL_DIR.parent
REPO_ROOT = RAG_DIR.parents[1]
RAG_API_DIR = REPO_ROOT / "python" / "rag_api"
OUT_DIR = RAG_DIR / "out"
DOCS_MD_DIR = RAG_DIR / "markdown" / "docs_md"

if str(RAG_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_API_DIR))

QUESTION_COLUMNS = ("question", "questions", "query", "user_input", "input")
FILENAME_COLUMNS = ("filename", "file", "expected_file", "relevant_file", "relevant_files")
LANGUAGE_COLUMNS = ("expected_language", "language", "lang", "expected_lang")

DEFAULT_MODES = ("naive", "local", "global", "hybrid", "mix")
LANGUAGE_SUFFIXES = {"en", "ru", "kk"}

COLOR_CODES = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}


@dataclass(frozen=True)
class Document:
    source_file: str
    path: Path
    text: str
    metadata: Dict[str, Any]


class SimpleCharTokenizer:
    """Tiny offline tokenizer for LightRAG eval chunking.

    LightRAG defaults to tiktoken, which may try to download tokenizer files on
    a fresh machine. For this evaluation harness we only need deterministic
    chunking, so Unicode codepoints are enough and keep the run fully local
    until the chosen LLM/embedding provider is called.
    """

    @staticmethod
    def encode(text: str) -> List[int]:
        return [ord(char) for char in text or ""]

    @staticmethod
    def decode(tokens: Sequence[int]) -> str:
        return "".join(chr(int(token)) for token in tokens)


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


def split_modes(value: str) -> List[str]:
    modes = [mode.strip() for mode in value.replace("|", ",").split(",") if mode.strip()]
    return modes or list(DEFAULT_MODES)


def normalize_language(value: str) -> str:
    language = (value or "").strip().lower()
    aliases = {
        "english": "en",
        "eng": "en",
        "russian": "ru",
        "rus": "ru",
        "kazakh": "kk",
        "kaz": "kk",
        "қазақ": "kk",
    }
    return aliases.get(language, language if language in LANGUAGE_SUFFIXES else "")


def language_from_source_file(source_file: str) -> str:
    stem = Path(source_file).stem.lower()
    if "_" not in stem:
        return ""
    suffix = stem.rsplit("_", 1)[-1]
    aliases = {
        "eng": "en",
        "kz": "kk",
    }
    suffix = aliases.get(suffix, suffix)
    return suffix if suffix in LANGUAGE_SUFFIXES else ""


def load_cases(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not csv_path.exists():
        raise RuntimeError(f"LightRAG eval CSV not found: {csv_path}")

    cases: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError("LightRAG eval CSV has no header")

        fieldnames = normalized_header_map(reader.fieldnames)
        question_col = first_existing_column(fieldnames, QUESTION_COLUMNS)
        filename_col = first_existing_column(fieldnames, FILENAME_COLUMNS)
        language_col = first_existing_column(fieldnames, LANGUAGE_COLUMNS)
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
            expected_language = normalize_language(row.get(language_col) or "") if language_col else ""
            if not expected_language and expected_files:
                expected_language = language_from_source_file(expected_files[0])
            cases.append(
                {
                    "query": query,
                    "expected_files": expected_files,
                    "expected_language": expected_language,
                }
            )

    if not cases:
        raise RuntimeError(f"LightRAG eval CSV has no usable rows: {csv_path}")
    return cases, skipped


def parse_markdown_front_matter(raw: str) -> Tuple[Dict[str, Any], str]:
    if not raw.startswith("---"):
        return {}, raw

    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw

    front_matter = parts[1].strip()
    body = parts[2].lstrip()
    try:
        parsed = json.loads(front_matter)
    except Exception:
        return {}, raw
    return parsed if isinstance(parsed, dict) else {}, body


def repeat_source_by_page(source_file: str, body: str) -> str:
    parts = body.split("<!-- PAGE ")
    if len(parts) == 1:
        return f"SOURCE_FILE: {source_file}\n\n{body.strip()}\n"

    rendered = [f"SOURCE_FILE: {source_file}\n"]
    for part in parts:
        if not part.strip():
            continue
        rendered.append(f"\nSOURCE_FILE: {source_file}\n<!-- PAGE {part.strip()}\n")
    return "\n".join(rendered).strip() + "\n"


def load_markdown_documents(docs_dir: Path, prefix: str) -> List[Document]:
    if not docs_dir.exists():
        raise RuntimeError(f"markdown docs directory not found: {docs_dir}")

    docs: List[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_markdown_front_matter(raw)
        source_file = str(metadata.get("source_file") or "").strip()
        if not source_file:
            source_file = f"{prefix}/{path.stem}.pdf"
        text = build_lightrag_document(source_file, metadata, body)
        docs.append(Document(source_file=source_file, path=path, text=text, metadata=metadata))

    if not docs:
        raise RuntimeError(f"no markdown docs found in {docs_dir}")
    return docs


def build_lightrag_document(source_file: str, metadata: Dict[str, Any], body: str) -> str:
    title = Path(source_file).name
    doc_type = metadata.get("doc_type") or metadata.get("topic") or ""
    language = metadata.get("language") or ""
    country = metadata.get("country") or ""
    header = [
        f"SOURCE_FILE: {source_file}",
        f"TITLE: {title}",
        f"DOC_TYPE: {doc_type}",
        f"COUNTRY: {country}",
        f"LANGUAGE: {language}",
        "",
        "The SOURCE_FILE line identifies the exact PDF file this content came from.",
        "",
    ]
    return "\n".join(header) + repeat_source_by_page(source_file, body)


def corpus_fingerprint(docs: Sequence[Document]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.source_file.encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def metadata_path(working_dir: Path) -> Path:
    return working_dir / "nomadmit_lightrag_eval_meta.json"


def is_index_current(
    working_dir: Path,
    fingerprint: str,
    provider: str,
    index_limit: int,
) -> bool:
    path = metadata_path(working_dir)
    if not path.exists():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("fingerprint") == fingerprint
        and meta.get("provider") == provider
        and int(meta.get("index_limit") or 0) == int(index_limit)
    )


def save_index_meta(
    working_dir: Path,
    fingerprint: str,
    provider: str,
    modes: Sequence[str],
    docs: Sequence[Document],
    index_limit: int,
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "provider": provider,
        "modes": list(modes),
        "index_limit": index_limit,
        "documents": [doc.source_file for doc in docs],
    }
    working_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(working_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_status_path(working_dir: Path) -> Path:
    return working_dir / "kv_store_doc_status.json"


def load_doc_status_by_source_file(working_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = doc_status_path(working_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    by_source_file: Dict[str, Dict[str, Any]] = {}
    for payload in raw.values():
        if not isinstance(payload, dict):
            continue
        source_file = str(payload.get("file_path") or "").strip()
        if not source_file or source_file == "unknown_source":
            continue
        by_source_file[source_file] = payload
    return by_source_file


def processed_source_files(working_dir: Path) -> set[str]:
    return {
        source_file
        for source_file, payload in load_doc_status_by_source_file(working_dir).items()
        if str(payload.get("status") or "").strip().lower() == "processed"
    }


def build_lightrag_openai_prompt(
    *,
    prompt: str,
    system_prompt: Optional[str],
    history_messages: Sequence[Dict[str, str]],
    keyword_extraction: bool,
) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(f"System instructions:\n{system_prompt}")
    if history_messages:
        rendered_history = []
        for message in history_messages:
            role = str(message.get("role") or "user").strip()
            content = str(message.get("content") or message.get("text") or "").strip()
            if content:
                rendered_history.append(f"{role}: {content}")
        if rendered_history:
            parts.append("Conversation history:\n" + "\n".join(rendered_history))
    if keyword_extraction:
        parts.append("Task note: return the keyword extraction result requested by LightRAG.")
    parts.append(prompt)
    return "\n\n".join(parts)


def lightrag_runtime_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": args.chunk_overlap_token_size,
        "entity_extract_max_gleaning": args.entity_extract_max_gleaning,
        "max_extract_input_tokens": args.max_extract_input_tokens,
        "default_llm_timeout": args.llm_timeout,
        "llm_model_max_async": args.llm_max_async,
        "max_parallel_insert": args.max_parallel_insert,
    }


def filter_lightrag_kwargs(lightrag_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(lightrag_cls.__init__)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    supported = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    skipped = sorted(set(kwargs) - set(filtered))
    if skipped:
        print(f"[lightrag] skipping unsupported constructor option(s): {', '.join(skipped)}", flush=True)
    return filtered


def configure_openai_rag(
    working_dir: Path,
    llm_model: str,
    embed_model: str,
    embed_dim: int,
    runtime_kwargs: Dict[str, Any],
):
    try:
        from lightrag import LightRAG
        from lightrag.utils import wrap_embedding_func_with_attrs
        from rag_service.infrastructure.openai_gateway import OpenAIGateway
    except ImportError as exc:
        missing = getattr(exc, "name", "") or str(exc)
        raise RuntimeError(
            f"LightRAG/OpenAI dependency missing ({missing}). Run `make lightrag-install` or "
            "`python -m pip install -r python/RAG/evaluation/lightrag_requirements.txt`."
        ) from exc

    gateway = OpenAIGateway(
        SimpleNamespace(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            class_model=os.getenv("CLASS_MODEL") or llm_model,
            llm_model=llm_model,
            embed_model=embed_model,
        )
    )

    @wrap_embedding_func_with_attrs(
        embedding_dim=embed_dim,
        max_token_size=8192,
        model_name=embed_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        response = await asyncio.to_thread(
            gateway._client.embeddings.create,
            model=embed_model,
            input=texts,
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list[dict[str, str]] = [],
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
        input_text = build_lightrag_openai_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
        )
        response = await asyncio.to_thread(
            gateway._client.responses.create,
            model=llm_model,
            input=input_text,
            max_output_tokens=int(kwargs.get("max_tokens") or kwargs.get("max_output_tokens") or 2048),
            temperature=float(kwargs.get("temperature", 0.0) or 0.0),
        )
        return gateway._resp_to_text(response) or ""

    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        tokenizer=SimpleCharTokenizer(),
        **filter_lightrag_kwargs(LightRAG, runtime_kwargs),
    )


def configure_ollama_rag(
    working_dir: Path,
    llm_model: str,
    embed_model: str,
    embed_dim: int,
    num_ctx: int,
    runtime_kwargs: Dict[str, Any],
):
    try:
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_embed, ollama_model_complete
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG is not installed. Run `make lightrag-install` or "
            "`python -m pip install -r python/RAG/evaluation/lightrag_requirements.txt`."
        ) from exc

    @wrap_embedding_func_with_attrs(
        embedding_dim=embed_dim,
        max_token_size=8192,
        model_name=embed_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(texts, embed_model=embed_model)

    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={"options": {"num_ctx": num_ctx}},
        embedding_func=embedding_func,
        tokenizer=SimpleCharTokenizer(),
        **filter_lightrag_kwargs(LightRAG, runtime_kwargs),
    )


def configure_lightrag(args: argparse.Namespace, working_dir: Path):
    provider = args.provider.strip().lower()
    runtime_kwargs = lightrag_runtime_kwargs(args)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai")
        return configure_openai_rag(
            working_dir,
            llm_model=args.llm_model or os.getenv("LIGHTRAG_LLM_MODEL", "gpt-4o-mini"),
            embed_model=args.embed_model or os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-small"),
            embed_dim=args.embed_dim,
            runtime_kwargs=runtime_kwargs,
        )
    if provider == "ollama":
        return configure_ollama_rag(
            working_dir,
            llm_model=args.llm_model or os.getenv("LIGHTRAG_LLM_MODEL", "qwen2.5:14b"),
            embed_model=args.embed_model or os.getenv("LIGHTRAG_EMBED_MODEL", "nomic-embed-text"),
            embed_dim=args.embed_dim,
            num_ctx=args.ollama_num_ctx,
            runtime_kwargs=runtime_kwargs,
        )
    raise RuntimeError(f"unsupported LightRAG provider: {args.provider}")


def source_file_aliases(source_file: str) -> List[str]:
    path = Path(source_file)
    stem = path.stem
    return [
        source_file,
        source_file.replace("/", os.sep),
        path.name,
        stem,
        f"SOURCE_FILE: {source_file}",
    ]


def extract_ranked_files(context: str, source_files: Sequence[str], top_k: int) -> List[str]:
    lowered = context.lower()
    positions: List[Tuple[int, str]] = []
    for source_file in source_files:
        found = -1
        for alias in source_file_aliases(source_file):
            index = lowered.find(alias.lower())
            if index >= 0:
                found = index if found < 0 else min(found, index)
        if found >= 0:
            positions.append((found, source_file))
    return [source_file for _position, source_file in sorted(positions)[:top_k]]


def source_metadata_by_file(docs: Sequence[Document]) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for doc in docs:
        raw = doc.metadata or {}
        language = normalize_language(str(raw.get("language") or "")) or language_from_source_file(doc.source_file)
        metadata[doc.source_file] = {
            "language": language,
            "canonical_doc_id": str(raw.get("canonical_doc_id") or "").strip().lower(),
            "doc_type": str(raw.get("doc_type") or raw.get("topic") or "").strip().lower(),
            "country": str(raw.get("country") or "").strip().lower(),
        }
    return metadata


def sibling_files_by_canonical_language(source_metadata: Dict[str, Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    siblings: Dict[Tuple[str, str], str] = {}
    for source_file, meta in source_metadata.items():
        canonical = meta.get("canonical_doc_id") or ""
        language = meta.get("language") or ""
        if canonical and language:
            siblings[(canonical, language)] = source_file
    return siblings


def metadata_terms(meta: Dict[str, str], source_file: str) -> List[str]:
    terms = [
        meta.get("doc_type") or "",
        meta.get("canonical_doc_id") or "",
        Path(source_file).stem.lower(),
    ]
    expanded: List[str] = []
    for term in terms:
        expanded.extend(part for part in term.replace("-", "_").split("_") if part)
    return expanded


def metadata_language_rerank(
    *,
    query: str,
    ranked_files: Sequence[str],
    target_language: str,
    source_metadata: Dict[str, Dict[str, str]],
    sibling_by_canonical_language: Dict[Tuple[str, str], str],
    top_k: int,
    hard_filter: bool = False,
) -> List[str]:
    if not target_language:
        return list(ranked_files[:top_k])

    expanded: List[Tuple[str, int, bool]] = []
    seen = set()
    for rank, source_file in enumerate(ranked_files):
        if source_file not in seen:
            expanded.append((source_file, rank, False))
            seen.add(source_file)

        canonical = (source_metadata.get(source_file) or {}).get("canonical_doc_id") or ""
        sibling = sibling_by_canonical_language.get((canonical, target_language)) if canonical else None
        if sibling and sibling not in seen:
            expanded.append((sibling, rank, True))
            seen.add(sibling)

    query_lc = query.lower()

    def sort_key(item: Tuple[str, int, bool]) -> Tuple[int, int, int, int, str]:
        source_file, original_rank, promoted = item
        meta = source_metadata.get(source_file) or {}
        language_match = 1 if meta.get("language") == target_language else 0
        lexical_match = 1 if any(term and term in query_lc for term in metadata_terms(meta, source_file)) else 0
        promoted_match = 1 if promoted else 0
        return (-language_match, -lexical_match, -promoted_match, original_rank, source_file)

    sorted_items = sorted(expanded, key=sort_key)
    if hard_filter:
        sorted_items = [
            item
            for item in sorted_items
            if (source_metadata.get(item[0]) or {}).get("language") == target_language
        ]
    return [source_file for source_file, _rank, _promoted in sorted_items[:top_k]]


def split_lightrag_source_ids(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split("<SEP>") if part.strip()]


def page_refs_from_chunk_content(content: str) -> List[str]:
    refs: List[str] = []
    current_source = ""
    pattern = re.compile(r"SOURCE_FILE:\s*([^\s]+)|<!--\s*PAGE\s+(\d+)\s*-->", re.IGNORECASE)
    for match in pattern.finditer(content or ""):
        source_file, page_number = match.groups()
        if source_file:
            current_source = source_file.strip()
            continue
        if page_number and current_source:
            refs.append(f"{current_source}:p{page_number}")
    return dedupe_preserve_order(refs)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def chunk_page_refs(working_dir: Path) -> Dict[str, List[str]]:
    chunks_path = working_dir / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        raise RuntimeError(f"LightRAG text chunk store not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    refs_by_chunk: Dict[str, List[str]] = {}
    chunks_by_doc: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
    for chunk_id, payload in chunks.items():
        if not isinstance(payload, dict):
            continue
        refs = page_refs_from_chunk_content(str(payload.get("content") or ""))
        if refs:
            refs_by_chunk[chunk_id] = refs
        full_doc_id = str(payload.get("full_doc_id") or "")
        if full_doc_id:
            chunks_by_doc.setdefault(full_doc_id, []).append(
                (int(payload.get("chunk_order_index") or 0), chunk_id, payload)
            )

    for doc_chunks in chunks_by_doc.values():
        current_refs: List[str] = []
        for _order, chunk_id, payload in sorted(doc_chunks):
            explicit_refs = page_refs_from_chunk_content(str(payload.get("content") or ""))
            if explicit_refs:
                current_refs = explicit_refs
                refs_by_chunk[chunk_id] = explicit_refs
            elif current_refs:
                refs_by_chunk[chunk_id] = current_refs
    return refs_by_chunk


def refs_for_source_ids(source_ids: str, refs_by_chunk: Dict[str, List[str]]) -> List[str]:
    refs: List[str] = []
    for chunk_id in split_lightrag_source_ids(source_ids):
        refs.extend(refs_by_chunk.get(chunk_id) or [])
    return dedupe_preserve_order(refs)


def source_file_from_page_ref(page_ref: str) -> str:
    if ":p" not in page_ref:
        return page_ref
    return page_ref.rsplit(":p", 1)[0]


def lightrag_context_contents(context: str) -> List[str]:
    decoder = json.JSONDecoder()
    contents: List[str] = []
    index = 0
    while index < len(context or ""):
        start = context.find("{", index)
        if start < 0:
            break
        try:
            parsed, end = decoder.raw_decode(context[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            content = str(parsed.get("content") or "").strip()
            if content:
                contents.append(content)
        index = start + max(end, 1)
    return contents


def source_file_from_chunk_content(content: str) -> str:
    match = re.search(r"SOURCE_FILE:\s*([^\s`]+)", content or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""


def page_refs_by_source_file_from_context(context: str) -> Dict[str, List[str]]:
    refs_by_file: Dict[str, List[str]] = {}
    refs = page_refs_from_chunk_content(context)
    for content in lightrag_context_contents(context):
        refs.extend(page_refs_from_chunk_content(content))
    for page_ref in dedupe_preserve_order(refs):
        source_file = source_file_from_page_ref(page_ref)
        refs_by_file.setdefault(source_file, []).append(page_ref)
    return {
        source_file: dedupe_preserve_order(refs)
        for source_file, refs in refs_by_file.items()
    }


def ranked_file_page_refs(
    ranked_files: Sequence[str],
    context: str,
) -> List[Dict[str, Any]]:
    refs_by_file = page_refs_by_source_file_from_context(context)
    return [
        {
            "source_file": source_file,
            "page_refs": refs_by_file.get(source_file, []),
        }
        for source_file in ranked_files
    ]


def context_chunk_candidates(
    context: str,
    source_metadata: Dict[str, Dict[str, str]],
    limit: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for index, content in enumerate(lightrag_context_contents(context), start=1):
        refs = page_refs_from_chunk_content(content)
        source_file = source_file_from_page_ref(refs[0]) if refs else source_file_from_chunk_content(content)
        if not source_file:
            continue
        meta = source_metadata.get(source_file) or {}
        chunks.append(
            {
                "rank": index,
                "source_file": source_file,
                "page_refs": refs,
                "language": meta.get("language") or language_from_source_file(source_file),
                "preview": re.sub(r"\s+", " ", content).strip()[:240],
            }
        )
        if len(chunks) >= limit:
            break
    return chunks


def chunk_files(chunks: Sequence[Dict[str, Any]], top_k: int) -> List[str]:
    return dedupe_preserve_order(str(chunk.get("source_file") or "") for chunk in chunks)[:top_k]


def metadata_filter_chunks(
    chunks: Sequence[Dict[str, Any]],
    target_language: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not target_language:
        return list(chunks[:top_k])
    return [
        chunk
        for chunk in chunks
        if str(chunk.get("language") or "") == target_language
    ][:top_k]


def metadata_values_for_refs(
    refs: Sequence[str],
    source_metadata: Dict[str, Dict[str, str]],
    key: str,
) -> List[str]:
    values: List[str] = []
    for source_file in dedupe_preserve_order(source_file_from_page_ref(ref) for ref in refs):
        value = (source_metadata.get(source_file) or {}).get(key) or ""
        if value:
            values.append(value)
    return dedupe_preserve_order(values)


def apply_graph_metadata(
    data: Dict[str, Any],
    refs: Sequence[str],
    source_metadata: Dict[str, Dict[str, str]],
) -> None:
    if refs:
        data["page_refs"] = ";".join(refs)
        source_files = dedupe_preserve_order(source_file_from_page_ref(ref) for ref in refs)
        data["source_files"] = ";".join(source_files)
        for graph_key, metadata_key in (
            ("languages", "language"),
            ("doc_types", "doc_type"),
            ("countries", "country"),
        ):
            values = metadata_values_for_refs(refs, source_metadata, metadata_key)
            if values:
                data[graph_key] = ";".join(values)
    if str(data.get("file_path") or "") == "unknown_source":
        data.pop("file_path", None)
    if str(data.get("truncate") or "") == "":
        data.pop("truncate", None)
    data.pop("canonical_doc_ids", None)


def enrich_graphml_with_page_refs(
    working_dir: Path,
    source_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        raise RuntimeError(f"LightRAG graph file not found: {graph_path}")

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to enrich LightRAG GraphML page refs") from exc

    refs_by_chunk = chunk_page_refs(working_dir)
    graph = nx.read_graphml(graph_path)
    source_metadata = source_metadata or {}
    node_refs: Dict[str, List[str]] = {}
    edge_refs: Dict[str, List[str]] = {}

    for node_id, data in graph.nodes(data=True):
        refs = refs_for_source_ids(str(data.get("source_id") or ""), refs_by_chunk)
        if refs:
            node_refs[str(node_id)] = refs
        apply_graph_metadata(graph.nodes[node_id], refs, source_metadata)

    for source, target, data in graph.edges(data=True):
        refs = refs_for_source_ids(str(data.get("source_id") or ""), refs_by_chunk)
        if refs:
            edge_refs[f"{source}<SEP>{target}"] = refs
        apply_graph_metadata(data, refs, source_metadata)

    nx.write_graphml(graph, graph_path)
    report_path = working_dir / "page_refs.json"
    report_path.write_text(
        json.dumps(
            {
                "chunks_with_pages": len(refs_by_chunk),
                "nodes_with_pages": len(node_refs),
                "edges_with_pages": len(edge_refs),
                "metadata_fields": [
                    "page_refs",
                    "source_files",
                    "languages",
                    "doc_types",
                    "countries",
                ],
                "node_page_refs": node_refs,
                "edge_page_refs": edge_refs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[lightrag] enriched graph page refs -> "
        f"{graph_path} ({len(node_refs)} nodes, {len(edge_refs)} edges)"
    )
    print(f"[lightrag] wrote page ref report -> {report_path}")


def graph_stats(working_dir: Path) -> Dict[str, Any]:
    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        return {
            "graph_path": str(graph_path),
            "exists": False,
            "nodes": 0,
            "edges": 0,
        }

    try:
        import networkx as nx

        graph = nx.read_graphml(graph_path)
        return {
            "graph_path": str(graph_path),
            "exists": True,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }
    except Exception as exc:
        return {
            "graph_path": str(graph_path),
            "exists": True,
            "error": str(exc),
        }


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


def score_row(
    case: Dict[str, Any],
    mode: str,
    ranked_files: List[str],
    top_k: int,
    context: str,
    raw_ranked_files: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    raw_retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    expected_files = set(case["expected_files"])
    binary_relevance = [1 if source_file in expected_files else 0 for source_file in ranked_files]
    relevant_retrieved = len(set(ranked_files) & expected_files)
    first_relevant_rank = next((idx for idx, rel in enumerate(binary_relevance, start=1) if rel), None)
    return {
        **case,
        "mode": mode,
        "retrieved_files": ranked_files,
        "raw_retrieved_files": raw_ranked_files or ranked_files,
        "retrieved_file_page_refs": ranked_file_page_refs(ranked_files, context),
        "retrieved_chunks": retrieved_chunks or [],
        "raw_retrieved_chunks": raw_retrieved_chunks or [],
        "target_language": case.get("expected_language") or "",
        "hit": relevant_retrieved > 0,
        "top1_correct": first_relevant_rank == 1,
        "precision_at_k": relevant_retrieved / float(top_k) if top_k else 0.0,
        "recall_at_k": relevant_retrieved / float(len(expected_files)) if expected_files else 0.0,
        "mrr": 1.0 / float(first_relevant_rank) if first_relevant_rank else 0.0,
        "ndcg_at_k": ndcg_at_k(binary_relevance, len(expected_files), top_k),
        "first_relevant_rank": first_relevant_rank,
        "context_chars": len(context),
        "context_preview": context[:800],
    }


async def build_or_load_index(
    rag: Any,
    working_dir: Path,
    docs: Sequence[Document],
    fingerprint: str,
    provider: str,
    modes: Sequence[str],
    index_limit: int,
    rebuild: bool,
    strict_status_check: bool = True,
) -> None:
    await rag.initialize_storages()

    if not rebuild and is_index_current(working_dir, fingerprint, provider, index_limit):
        print(f"[lightrag] using existing index: {working_dir}")
        return

    processed = processed_source_files(working_dir)
    remaining = [doc for doc in docs if doc.source_file not in processed]
    if not remaining:
        print(f"[lightrag] all {len(docs)} markdown documents are already processed")
        save_index_meta(working_dir, fingerprint, provider, modes, docs, index_limit)
        return

    status_by_file = load_doc_status_by_source_file(working_dir)
    incomplete_existing = [doc for doc in remaining if doc.source_file in status_by_file]
    if incomplete_existing:
        names = ", ".join(doc.source_file for doc in incomplete_existing[:5])
        if len(incomplete_existing) > 5:
            names += f", +{len(incomplete_existing) - 5} more"
        print(f"[lightrag] retrying {len(incomplete_existing)} incomplete document(s): {names}")
        await rag.apipeline_process_enqueue_documents()
        processed = processed_source_files(working_dir)
        remaining = [doc for doc in docs if doc.source_file not in processed]
        if not remaining:
            print(f"[lightrag] all {len(docs)} markdown documents are now processed")
            save_index_meta(working_dir, fingerprint, provider, modes, docs, index_limit)
            return

    status_by_file = load_doc_status_by_source_file(working_dir)
    docs_to_insert = [doc for doc in remaining if doc.source_file not in status_by_file]
    skipped_count = len(docs) - len(remaining)
    print(
        f"[lightrag] resume index in {working_dir}: "
        f"{skipped_count} processed, {len(remaining)} remaining, {len(docs_to_insert)} new"
    )
    for index, doc in enumerate(docs_to_insert, start=1):
        print(f"[{index:03d}/{len(docs_to_insert):03d}] insert {doc.source_file}")
        await rag.ainsert(doc.text, file_paths=doc.source_file)

    still_missing = [doc.source_file for doc in docs if doc.source_file not in processed_source_files(working_dir)]
    if still_missing:
        preview = ", ".join(still_missing[:5])
        if len(still_missing) > 5:
            preview += f", +{len(still_missing) - 5} more"
        if not strict_status_check:
            print(
                "[lightrag] warning: doc status not fully flushed yet; "
                f"continuing index-only build with generated artifacts: {preview}",
                flush=True,
            )
            save_index_meta(working_dir, fingerprint, provider, modes, docs, index_limit)
            return
        raise RuntimeError(f"LightRAG index is still incomplete after resume: {preview}")
    save_index_meta(working_dir, fingerprint, provider, modes, docs, index_limit)


async def evaluate_mode(
    rag: Any,
    cases: Sequence[Dict[str, Any]],
    source_files: Sequence[str],
    source_metadata: Dict[str, Dict[str, str]],
    sibling_by_canonical_language: Dict[Tuple[str, str], str],
    mode: str,
    top_k: int,
    candidate_k: int,
    metadata_rerank: str,
    allow_no_context: bool,
    color_enabled: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    from lightrag import QueryParam

    rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        context = await rag.aquery(
            case["query"],
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=max(top_k, candidate_k),
                enable_rerank=False,
            ),
        )
        context_text = str(context or "")
        raw_chunks = context_chunk_candidates(
            context_text,
            source_metadata,
            limit=max(candidate_k, top_k * 4, 20),
        )
        raw_ranked_files = chunk_files(raw_chunks, top_k=max(candidate_k, top_k * 4, 20))
        if not raw_ranked_files:
            raw_ranked_files = extract_ranked_files(context_text, source_files, top_k=max(candidate_k, top_k * 4, 20))
        ranked_files = raw_ranked_files[:top_k]
        if not raw_ranked_files and not allow_no_context and "[no-context]" in context_text:
            raise RuntimeError(
                "LightRAG returned no context. This usually means the provider call failed "
                "or the index is unusable; rerun with provider access or pass --allow-no-context "
                "if you intentionally want to score empty-context misses."
            )
        row = score_row(
            case,
            mode,
            ranked_files,
            top_k,
            context_text,
            retrieved_chunks=raw_chunks[:top_k],
            raw_retrieved_chunks=raw_chunks,
        )
        rows.append(row)

        status = "ok" if row["hit"] else "MISS"
        top_file = ranked_files[0] if ranked_files else "-"
        line = (
            f"[{index:03d}/{len(cases):03d}] {mode:<6} "
            f"{status} rank={row['first_relevant_rank'] or '-'} "
            f"expected={','.join(case['expected_files'])} top={top_file} "
            f"query={case['query'][:70]}"
        )
        print(colorize(line, "green" if row["hit"] else "red", color_enabled))

        if metadata_rerank != "off":
            hard_metadata_filter = metadata_rerank in {"filter", "filter-verbose"}
            metadata_ranked_files = metadata_language_rerank(
                query=case["query"],
                ranked_files=raw_ranked_files,
                target_language=case.get("expected_language") or "",
                source_metadata=source_metadata,
                sibling_by_canonical_language=sibling_by_canonical_language,
                top_k=top_k,
                hard_filter=hard_metadata_filter,
            )
            metadata_chunks = (
                metadata_filter_chunks(
                    raw_chunks,
                    case.get("expected_language") or "",
                    top_k=top_k,
                )
                if hard_metadata_filter
                else raw_chunks[:top_k]
            )
            metadata_row = score_row(
                case,
                f"{mode}+meta",
                metadata_ranked_files,
                top_k,
                context_text,
                raw_ranked_files=raw_ranked_files,
                retrieved_chunks=metadata_chunks,
                raw_retrieved_chunks=raw_chunks,
            )
            metadata_rows.append(metadata_row)
            if metadata_rerank in {"verbose", "filter-verbose"}:
                meta_status = "ok" if metadata_row["hit"] else "MISS"
                meta_top = metadata_ranked_files[0] if metadata_ranked_files else "-"
                meta_line = (
                    f"[{index:03d}/{len(cases):03d}] {mode + '+meta':<11} "
                    f"{meta_status} rank={metadata_row['first_relevant_rank'] or '-'} "
                    f"target_lang={metadata_row['target_language'] or '-'} top={meta_top}"
                )
                print(colorize(meta_line, "green" if metadata_row["hit"] else "red", color_enabled))

    result = {mode: rows}
    if metadata_rerank != "off":
        result[f"{mode}+meta"] = metadata_rows
    return result


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def configure_logging(verbose_lightrag: bool) -> None:
    if verbose_lightrag:
        logging.disable(logging.NOTSET)
        return
    logging.disable(logging.INFO)
    logging.getLogger("lightrag").setLevel(logging.WARNING)
    logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)


def summarize(rows: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    return {
        "queries": len(rows),
        "top_k": top_k,
        "accuracy_at_1": safe_mean([1.0 if row["top1_correct"] else 0.0 for row in rows]),
        "hit_rate_at_k": safe_mean([1.0 if row["hit"] else 0.0 for row in rows]),
        "precision_at_k": safe_mean([row["precision_at_k"] for row in rows]),
        "recall_at_k": safe_mean([row["recall_at_k"] for row in rows]),
        "ndcg_at_k": safe_mean([row["ndcg_at_k"] for row in rows]),
        "mrr": safe_mean([row["mrr"] for row in rows]),
    }


def print_report(
    summaries: Dict[str, Dict[str, Any]],
    rows_by_mode: Dict[str, List[Dict[str, Any]]],
    skipped: Sequence[Dict[str, Any]],
    max_examples: int,
    color_enabled: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    print("\n" + "=" * 80)
    print("LightRAG retrieval-context evaluation")
    if metadata:
        graph = metadata.get("graph") or {}
        print(
            "Model        : "
            f"{metadata.get('llm_model') or '-'} "
            f"(provider={metadata.get('provider') or '-'})"
        )
        print(f"Embedding    : {metadata.get('embed_model') or '-'}")
        if graph.get("exists"):
            print(f"Graph        : nodes={graph.get('nodes', 0)} edges={graph.get('edges', 0)}")
    for mode, summary in summaries.items():
        top_k = int(summary["top_k"])
        print(f"\n[{mode}] queries={summary['queries']} top_k={top_k}")
        print(f"Accuracy@1  : {summary['accuracy_at_1']:.4f}")
        print(f"HitRate@{top_k:<2} : {summary['hit_rate_at_k']:.4f}")
        print(f"Precision@{top_k:<2}: {summary['precision_at_k']:.4f}")
        print(f"Recall@{top_k:<2}   : {summary['recall_at_k']:.4f}")
        print(f"NDCG@{top_k:<2}     : {summary['ndcg_at_k']:.4f}")
        print(f"MRR         : {summary['mrr']:.4f}")

    if skipped:
        print(f"\nSkipped: {len(skipped)} rows without usable expected files")

    for mode, rows in rows_by_mode.items():
        misses = [row for row in rows if not row["hit"]]
        if not misses:
            continue
        print("\n" + colorize(f"LightRAG misses [{mode}]:", "red", color_enabled))
        for row in misses[:max_examples]:
            print(colorize(f"- {row['query']}", "red", color_enabled))
            print(f"  expected : {row['expected_files']}")
            print(f"  retrieved: {row['retrieved_files']}")
            if row["context_preview"]:
                preview = " ".join(row["context_preview"].split())
                print(f"  context  : {preview[:240]}")


def save_report(
    output: Path,
    summaries: Dict[str, Dict[str, Any]],
    rows_by_mode: Dict[str, List[Dict[str, Any]]],
    skipped: Sequence[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "summary": summaries,
        "rows_by_mode": rows_by_mode,
        "skipped": list(skipped),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lightrag] wrote report -> {output}")


async def run(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env", override=True)
    load_dotenv(EVAL_DIR / ".env", override=True)

    color_enabled = should_colorize(args.color)
    modes = split_modes(args.modes)
    resolved_llm_model = args.llm_model or os.getenv(
        "LIGHTRAG_LLM_MODEL",
        "gpt-4o-mini" if args.provider == "openai" else "qwen2.5:14b",
    )
    resolved_embed_model = args.embed_model or os.getenv(
        "LIGHTRAG_EMBED_MODEL",
        "text-embedding-3-small" if args.provider == "openai" else "nomic-embed-text",
    )
    docs = load_markdown_documents(Path(args.docs_dir), args.doc_prefix)
    if args.index_limit and args.index_limit > 0:
        docs = docs[: args.index_limit]
        print(f"[lightrag] index limit active: {len(docs)} document(s)", flush=True)
    source_files = sorted({doc.source_file for doc in docs})
    source_metadata = source_metadata_by_file(docs)

    working_dir = Path(args.working_dir)
    if args.page_refs_only:
        enrich_graphml_with_page_refs(working_dir, source_metadata)
        return

    if args.reset and working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = configure_lightrag(args, working_dir)
    fingerprint = corpus_fingerprint(docs)
    try:
        await build_or_load_index(
            rag,
            working_dir=working_dir,
            docs=docs,
            fingerprint=fingerprint,
            provider=args.provider,
            modes=modes,
            index_limit=max(0, args.index_limit),
            rebuild=args.rebuild,
            strict_status_check=not args.index_only,
        )
        if args.page_refs:
            enrich_graphml_with_page_refs(working_dir, source_metadata)
            args.page_refs = False

        report_metadata = {
            "provider": args.provider,
            "llm_model": resolved_llm_model,
            "embed_model": resolved_embed_model,
            "embed_dim": args.embed_dim,
            "working_dir": str(working_dir),
            "docs_dir": str(Path(args.docs_dir)),
            "document_count": len(docs),
            "source_file_count": len(source_files),
            "graph": graph_stats(working_dir),
        }
        if args.index_only:
            print("[lightrag] index-only mode: skipping retrieval evaluation")
            save_report(Path(args.output), {}, {}, [], report_metadata)
            return

        sibling_by_canonical_language = sibling_files_by_canonical_language(source_metadata)
        loaded_cases, skipped_no_file = load_cases(Path(args.csv))
        corpus_files = set(source_files)
        cases: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = list(skipped_no_file)
        for case in loaded_cases:
            expected_files = [source_file for source_file in case["expected_files"] if source_file in corpus_files]
            if expected_files:
                cases.append({**case, "expected_files": expected_files})
            else:
                skipped.append({"query": case["query"], "reason": "expected file missing from markdown corpus"})
        if args.limit and args.limit > 0:
            cases = cases[: args.limit]

        rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        for mode in modes:
            print("\n" + colorize(f"[lightrag] mode={mode}", "yellow", color_enabled))
            rows = await evaluate_mode(
                rag,
                cases=cases,
                source_files=source_files,
                source_metadata=source_metadata,
                sibling_by_canonical_language=sibling_by_canonical_language,
                mode=mode,
                top_k=max(1, args.top_k),
                candidate_k=max(max(1, args.top_k), args.candidate_k),
                metadata_rerank=args.metadata_rerank,
                allow_no_context=args.allow_no_context,
                color_enabled=color_enabled,
            )
            for result_mode, mode_rows in rows.items():
                rows_by_mode[result_mode] = mode_rows
                summaries[result_mode] = {
                    **summarize(mode_rows, top_k=max(1, args.top_k)),
                    "candidate_k": max(max(1, args.top_k), args.candidate_k),
                }

        print_report(summaries, rows_by_mode, skipped, args.max_examples, color_enabled, report_metadata)
        save_report(Path(args.output), summaries, rows_by_mode, skipped, report_metadata)
    finally:
        await rag.finalize_storages()
        if args.page_refs:
            enrich_graphml_with_page_refs(working_dir, source_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LightRAG graph/vector retrieval modes on the local Nomadmit markdown corpus."
    )
    parser.add_argument("--csv", default=str(EVAL_DIR / "query_mappings.csv"), help="CSV with query and expected file columns")
    parser.add_argument("--docs-dir", default=str(DOCS_MD_DIR), help="Markdown directory produced by make rag-chunks-md")
    parser.add_argument("--working-dir", default=str(OUT_DIR / "lightrag"), help="LightRAG persistent working directory")
    parser.add_argument("--output", default=str(OUT_DIR / "lightrag_eval_report.json"), help="JSON report output path")
    parser.add_argument("--doc-prefix", default="italy", help="Fallback source-file prefix when markdown lacks source metadata")
    parser.add_argument("--provider", choices=("openai", "ollama"), default=os.getenv("LIGHTRAG_PROVIDER", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("LIGHTRAG_LLM_MODEL", ""), help="LightRAG LLM model name")
    parser.add_argument("--embed-model", default=os.getenv("LIGHTRAG_EMBED_MODEL", ""), help="LightRAG embedding model name")
    parser.add_argument("--embed-dim", type=int, default=int(os.getenv("LIGHTRAG_EMBED_DIM", "1536")))
    parser.add_argument("--ollama-num-ctx", type=int, default=int(os.getenv("LIGHTRAG_OLLAMA_NUM_CTX", "32768")))
    parser.add_argument("--chunk-token-size", type=int, default=int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE", "700")))
    parser.add_argument("--chunk-overlap-token-size", type=int, default=int(os.getenv("LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE", "80")))
    parser.add_argument(
        "--entity-extract-max-gleaning",
        type=int,
        default=int(os.getenv("LIGHTRAG_ENTITY_EXTRACT_MAX_GLEANING", "0")),
        help="Extra entity-extraction passes. Default 0 avoids large gleaning prompts/timeouts.",
    )
    parser.add_argument("--max-extract-input-tokens", type=int, default=int(os.getenv("LIGHTRAG_MAX_EXTRACT_INPUT_TOKENS", "12000")))
    parser.add_argument("--llm-timeout", type=int, default=int(os.getenv("LIGHTRAG_LLM_TIMEOUT", "600")))
    parser.add_argument("--llm-max-async", type=int, default=int(os.getenv("LIGHTRAG_LLM_MAX_ASYNC", "2")))
    parser.add_argument("--max-parallel-insert", type=int, default=int(os.getenv("LIGHTRAG_MAX_PARALLEL_INSERT", "1")))
    parser.add_argument("--modes", default=os.getenv("LIGHTRAG_MODES", ",".join(DEFAULT_MODES)))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=int(os.getenv("LIGHTRAG_CANDIDATE_K", "64")),
        help="Retrieve this many LightRAG candidates before metadata filtering/reranking.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--index-limit",
        type=int,
        default=int(os.getenv("LIGHTRAG_INDEX_LIMIT", "0")),
        help="Index only the first N markdown docs for cheap smoke tests",
    )
    parser.add_argument("--index-only", action="store_true", help="Build/resume the LightRAG index without running eval queries")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Resume/recheck the LightRAG index and process only missing or incomplete documents.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing LightRAG working dir before indexing from scratch.",
    )
    parser.add_argument(
        "--page-refs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enrich graph_chunk_entity_relation.graphml with source PDF page references.",
    )
    parser.add_argument(
        "--page-refs-only",
        action="store_true",
        help="Only enrich the existing LightRAG GraphML with page_refs; do not query or rebuild.",
    )
    parser.add_argument(
        "--metadata-rerank",
        choices=("off", "append", "verbose", "filter", "filter-verbose"),
        default=os.getenv("LIGHTRAG_METADATA_RERANK", "append"),
        help=(
            "Add language/canonical-doc metadata reranked results as mode+meta. "
            "Use filter to hard-exclude non-target languages, off for raw LightRAG only, "
            "verbose/filter-verbose to print per-row metadata lines."
        ),
    )
    parser.add_argument(
        "--allow-no-context",
        action="store_true",
        help="Score LightRAG no-context responses as misses instead of failing fast",
    )
    parser.add_argument("--verbose-lightrag", action="store_true", help="Show LightRAG/nano-vectordb INFO logs")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Color terminal output. auto enables colors only when stdout is a TTY",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose_lightrag)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RAG retrieval with classification-first logic (GPT-only intent + language detection).
- Classification (intent + language) is done by the GPT classifier.
- No local stored greetings. If classifier says GREETING/CHIT_CHAT, we ask GPT to produce
  the short reply in the detected language.
- DOCUMENT_REQUEST -> retrieval + aggregation + (optional) synthesis (LLM instructed to answer in same language).

Requires: out/index.faiss, out/meta.json
Requires: OPENAI_API_KEY in environment or .env
"""
from __future__ import annotations
import os
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ----------------- USER EDITABLE -----------------
QUERY = "Отлично, я бы хотел узнать как подать на ВНЖ?"   # example; change as needed

RAW_TOP_K = 64
RETURN_TOP_K = 5

# Models
CLASS_MODEL = os.getenv("CLASS_MODEL", "gpt-4o-mini")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"

# Confidence thresholds (tune to your dataset)
MIN_FILE_SCORE_SUM = 0.9
MIN_SINGLE_CHUNK_SCORE = 0.20
RELATIVE_MARGIN = 0.25
# -------------------------------------------------

# Paths
OUT = Path("out")
META_JSON = OUT / "meta.json"
FAISS_INDEX_PATH = OUT / "index.faiss"

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
client = OpenAI(api_key=API_KEY)

# Load meta
if not META_JSON.exists():
    print(f"[error] meta.json not found at {META_JSON.resolve()}", file=sys.stderr)
    sys.exit(1)
meta_map: Dict[str, Any] = json.loads(META_JSON.read_text(encoding="utf-8"))

# Known files set for validation
_known_files = set()
for k, v in meta_map.items():
    if isinstance(v, dict):
        sf = v.get("source_file") or v.get("filename")
        if sf:
            _known_files.add(sf)

# Load FAISS index
if not FAISS_INDEX_PATH.exists():
    print(f"[error] FAISS index not found at {FAISS_INDEX_PATH.resolve()}", file=sys.stderr)
    sys.exit(1)
index = faiss.read_index(str(FAISS_INDEX_PATH))

# helper: robustly extract text from responses
def _resp_to_text(resp: Any) -> str:
    text = ""
    if hasattr(resp, "output_text") and resp.output_text:
        text = resp.output_text
    else:
        out = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
        if isinstance(out, list):
            parts = []
            for node in out:
                if isinstance(node, dict):
                    content = node.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
                            elif isinstance(c, str):
                                parts.append(c)
                    elif isinstance(content, str):
                        parts.append(content)
                elif isinstance(node, str):
                    parts.append(node)
            text = "".join(parts).strip()
        elif isinstance(out, str):
            text = out.strip()
    if not text:
        try:
            text = json.dumps(resp, default=str)
        except Exception:
            text = str(resp)
    return text

# Classification prompt: ask GPT to return JSON with intent + language (no local heuristics)
def classify_query(query: str) -> Dict[str, Any]:
    """
    Returns dict with keys:
      'intent' -> one of: GREETING, CHIT_CHAT, FACTUAL_QUESTION, DOCUMENT_REQUEST, OTHER
      'explain' -> short explanation string
      'language' -> language name or two-letter code (e.g. 'ru', 'English')
    Classification is done by GPT. Only fallback to minimal heuristic if the call fails.
    """
    prompt = (
        "You are a small intent classifier + language detector. Given a single user input, "
        "return a JSON object with three keys exactly: "
        "\"intent\" (one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"DOCUMENT_REQUEST\",\"OTHER\"]), "
        "\"explain\" (one short sentence), and "
        "\"language\" (a short language name or 2-letter code, e.g. \"Russian\" or \"ru\").\n\n"
        "Definitions:\n"
        "- GREETING: user just says hi / hello / a salutation.\n"
        "- CHIT_CHAT: casual conversational message (small talk) that does not request knowledge or a document.\n"
        "- FACTUAL_QUESTION: user asks a general knowledge / factual question that can be answered by the model without reading user's documents.\n"
        "- DOCUMENT_REQUEST: user requests assistance that requires reading your uploaded files (e.g., \"show me the application form\", \"which file contains visa rules\").\n"
        "- OTHER: none of the above.\n\n"
        "Respond ONLY with valid JSON. Example:\n"
        '{"intent":"GREETING","explain":"Short hello","language":"Russian"}\n\n'
        f"User input: {json.dumps(query)}\n"
    )
    try:
        resp = client.responses.create(model=CLASS_MODEL, input=prompt, max_output_tokens=120, temperature=0.0)
        txt = _resp_to_text(resp)
        s = txt.strip()
        # strip wrapper code fences if present and parse JSON inside
        if s.startswith("```"):
            # try to extract {...}
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                s = s[start:end+1]
        try:
            j = json.loads(s)
        except Exception:
            # try to find JSON inside the response text
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                try:
                    j = json.loads(candidate)
                except Exception:
                    j = {"intent": "OTHER", "explain": txt, "language": ""}
            else:
                j = {"intent": "OTHER", "explain": txt, "language": ""}
        intent = (j.get("intent") or "").strip().upper()
        explain = j.get("explain") or ""
        language = (j.get("language") or "").strip()
        # sanitize intent
        if intent not in {"GREETING","CHIT_CHAT","FACTUAL_QUESTION","DOCUMENT_REQUEST","OTHER"}:
            intent = "OTHER"
        return {"intent": intent, "explain": explain, "language": language}
    except Exception as e:
        # if classifier fails, fallback minimal heuristic (rare)
        lower = query.strip().lower()
        if re.match(r'^\s*(hi|hello|hey|hola|bonjour|hallo)\b', lower):
            return {"intent": "GREETING", "explain": "fallback regex matched greeting", "language": "en"}
        if re.search('[\u0400-\u04FF]', query):
            return {"intent": "OTHER", "explain": f"classifier error: {e}", "language": "ru"}
        return {"intent": "OTHER", "explain": f"classifier error: {e}", "language": ""}

# If classifier returns GREETING/CHIT_CHAT, generate a short reply in the same language using GPT (no local map)
def generate_greeting_reply(user_text: str, language: str) -> str:
    # Ask LLM to produce single-sentence friendly response in the detected language.
    # Keep it short and do not include file paths or extra commentary.
    lang_hint = f"in {language}" if language else "in the same language as the user"
    prompt = (
        f"You are an assistant. The user said: {json.dumps(user_text)}\n\n"
        f"Produce a single short friendly reply ({lang_hint}). Keep it short (one sentence, max ~20 words). "
        "Do NOT include file paths, document names, or extra commentary. Return only the reply text."
    )
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=50, temperature=0.0)
        text = _resp_to_text(resp).strip()
        # If LLM returns code fences or extra JSON, try to extract first non-empty line
        if text.startswith("```"):
            text = text.strip("` \n")
        # prefer first paragraph
        text = text.splitlines()[0].strip()
        return text
    except Exception as e:
        # fallback English short greeting (last resort)
        return "Hi — how can I help you today?"

# Embedding function
def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    item = resp.data[0]
    emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
    if emb is None:
        raise RuntimeError("Failed to parse embedding response")
    return np.array(emb, dtype=np.float32)

# FAISS search
def search_faiss(q_emb: np.ndarray, k: int = RAW_TOP_K):
    q_arr = q_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_arr)
    k = max(1, int(k))
    k = min(k, max(1, int(index.ntotal)))
    D, I = index.search(q_arr, k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    results = []
    for sc, nid in zip(scores, ids):
        if int(nid) == -1:
            continue
        m = meta_map.get(str(int(nid)))
        if not m:
            continue
        results.append({"score": float(sc), "nid": int(nid), "meta": m})
    return results

# aggregate by file
def aggregate_by_file(results: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, float]]:
    file_sum: Dict[str, float] = {}
    best_chunk_for_file: Dict[str, Dict[str, Any]] = {}
    for r in results:
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        file_sum[sf] = file_sum.get(sf, 0.0) + r["score"]
        if sf not in best_chunk_for_file or r["score"] > best_chunk_for_file[sf]["score"]:
            best_chunk_for_file[sf] = r
    if not file_sum:
        return None, None, {}
    best_file = max(file_sum.items(), key=lambda kv: kv[1])[0]
    return best_file, best_chunk_for_file[best_file], file_sum

def format_retrieved(results: List[Dict[str,Any]], top_n:int=RETURN_TOP_K):
    out = []
    for r in results[:top_n]:
        m = r["meta"]
        out.append({
            "file": m.get("source_file") or m.get("filename"),
            "page": m.get("page"),
            "score": float(r["score"]),
        })
    return out

# for document synthesis: prepare prompt that requests JSON response {answer,file}
def _prepare_synthesis_prompt(query: str, top_chunks: List[Dict[str,Any]]) -> str:
    lines = [
        "You are a helpful assistant that synthesizes multiple document excerpts and chooses the single most relevant file path.",
        "User query:",
        query,
        "",
        "Below are top document excerpts retrieved (each has file and page). Use them to produce a short helpful answer (concise, factual) and decide which file is most relevant.",
        ""
    ]
    for i, r in enumerate(top_chunks, start=1):
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        page = m.get("page")
        text = (m.get("text") or m.get("md") or "").strip()
        excerpt = text[:800].replace("\n", " ").strip()
        lines.append(f"[{i}] file: {sf}  page: {page}")
        lines.append(f"excerpt: {excerpt}")
        lines.append("")
    lines.extend([
        "Important instructions:",
        "- Produce output ONLY as a single JSON object with exactly two keys: 'answer' and 'file'.",
        "- 'answer' must be a short natural-language helpful reply to the user's query (in the same language).",
        "- 'file' must be the single most relevant source_file (the path) chosen from the excerpts above, or null if none are relevant.",
        "- Do not output any extra text, explanation, or commentary — output must be valid JSON only.",
        "",
        "Return the JSON now."
    ])
    return "\n".join(lines)

def call_llm(prompt: str, model: str = LLM_MODEL, max_tokens: int = 512) -> str:
    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens, temperature=0.0)
    return _resp_to_text(resp)

# MAIN flow
def main(query: str):
    q = (query or "").strip()
    if not q:
        print(json.dumps({"answer":"", "file":None, "page":None, "score":None, "retrieved":[]}, ensure_ascii=False, indent=2))
        return

    # 0) classify using GPT (intent + language)
    cls = classify_query(q)
    intent = cls.get("intent")
    lang = cls.get("language") or ""
    # if classifier didn't produce language, keep it empty (we won't use local greeting map)
    # but we'll still instruct LLM to answer in same language when synthesizing

    # GREETING / CHIT_CHAT -> ask GPT to generate a short reply in detected language (no local mapping)
    if intent in ("GREETING", "CHIT_CHAT"):
        greeting = generate_greeting_reply(q, lang)
        out = {
            "answer": greeting,
            "file": None,
            "page": None,
            "score": None,
            "retrieved": [],
            "meta_conf": {"classifier": cls}
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # FACTUAL_QUESTION -> ask LLM directly (no retrieval), instruct it to answer in same language
    if intent == "FACTUAL_QUESTION":
        prompt = (
            "You are a concise helpful assistant. Answer the user question briefly (1-2 short paragraphs). "
            "Answer in the same language as the question if possible. "
            "Do NOT include any file paths or suggest internal document locations.\n\n"
            f"Question: {q}\n\nAnswer:"
        )
        try:
            answer = call_llm(prompt, model=LLM_MODEL, max_tokens=300).strip()
        except Exception as e:
            answer = f"(LLM error: {e})"
        out = {
            "answer": answer,
            "file": None,
            "page": None,
            "score": None,
            "retrieved": [],
            "meta_conf": {"classifier": cls}
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # DOCUMENT_REQUEST or OTHER -> perform retrieval
    try:
        q_emb = embed_query(q)
    except Exception as e:
        raise RuntimeError(f"embedding failed: {e}")

    results = search_faiss(q_emb, k=RAW_TOP_K)
    if not results:
        out = {"answer": "", "file": None, "page": None, "score": None, "retrieved": [], "meta_conf": {"classifier": cls}}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    best_file, best_chunk, file_scores = aggregate_by_file(results)
    retrieved = format_retrieved(results, top_n=RETURN_TOP_K)
    if best_chunk is None:
        out = {"answer": "", "file": None, "page": None, "score": None, "retrieved": retrieved, "meta_conf": {"classifier": cls}}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # Confidence checks
    agg_score = file_scores.get(best_file, 0.0)
    all_scores = sorted(file_scores.values(), reverse=True)
    second_best = all_scores[1] if len(all_scores) > 1 else 0.0
    best_chunk_score = float(best_chunk["score"])

    allow_return_file = (
        agg_score >= MIN_FILE_SCORE_SUM and
        best_chunk_score >= MIN_SINGLE_CHUNK_SCORE and
        (agg_score - second_best) >= RELATIVE_MARGIN
    )

    # validate best_file exists in meta
    if isinstance(best_file, str) and best_file not in _known_files:
        allow_return_file = False

    # Synthesize final answer using LLM (in same language if known)
    top_for_synth = results[:min(8, len(results))]
    prompt = _prepare_synthesis_prompt(q, top_for_synth)
    # tell LLM to answer in detected language
    if lang:
        prompt = f"Answer in the same language as detected: {lang}\n\n" + prompt
    else:
        prompt = "Answer in the same language as the user's question if possible.\n\n" + prompt

    llm_text = None
    llm_json = None
    try:
        llm_text = call_llm(prompt, model=LLM_MODEL, max_tokens=512)
        s = (llm_text or "").strip()
        if s.startswith("```"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start:end+1]
        try:
            llm_json = json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                try:
                    llm_json = json.loads(candidate)
                except Exception:
                    llm_json = None
            else:
                llm_json = None
    except Exception:
        llm_text = None
        llm_json = None

    # Final answer selection
    if isinstance(llm_json, dict) and "answer" in llm_json:
        answer = llm_json.get("answer", "").strip()
        file_candidate = llm_json.get("file")
        if isinstance(file_candidate, str) and file_candidate not in _known_files:
            file_candidate = None
        file_chosen = file_candidate or (best_file if allow_return_file else None)
    else:
        chunk_meta = best_chunk["meta"]
        snippet = chunk_meta.get("text") or chunk_meta.get("md") or ""
        answer = snippet.strip()
        if len(answer) > 800:
            answer = answer[:800].rstrip() + "..."
        file_chosen = best_file if allow_return_file else None

    if answer is None:
        answer = ""

    score_out = float(best_chunk["score"]) if allow_return_file else None
    page_out = best_chunk["meta"].get("page") if allow_return_file else None

    out = {
        "answer": answer,
        "file": file_chosen,
        "page": page_out,
        "score": score_out,
        "retrieved": retrieved,
        "meta_conf": {
            "agg_score": agg_score,
            "best_chunk_score": best_chunk_score,
            "second_best": second_best,
            "allow_return_file": allow_return_file,
            "classifier": cls
        }
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    q = QUERY
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    main(q)

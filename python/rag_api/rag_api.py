# rag_file_only_api.py
from __future__ import annotations
import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# S3 / storage config
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET")
S3_BUCKET_VECTORS = os.getenv("S3_BUCKET_VECTORS")
S3_USE_SSL = os.getenv("S3_USE_SSL", "false").lower() in ("1", "true", "yes")
RELEASE_PREFIX = os.getenv("RELEASE_PREFIX")

OUT = Path(os.getenv("OUT_DIR", "out"))
OUT.mkdir(parents=True, exist_ok=True)

META_JSON = OUT / "meta.json"
FAISS_INDEX_PATH = OUT / "index.faiss"
OPTIONAL_ARTIFACTS = ["embeddings.npy", "ids.npy", "chunks.jsonl", "manifest.json"]

# Models
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CLASS_MODEL = os.getenv("CLASS_MODEL", LLM_MODEL)  # classifier model

# --- S3 helpers (unchanged; present so code can still pull artifacts if needed) ---
def _s3_client():
    try:
        import boto3
        from botocore.client import Config as BotoConfig
    except Exception:
        print("[s3] boto3 not installed; skipping S3 support")
        return None

    if not S3_ENDPOINT or not S3_BUCKET_VECTORS:
        print("[s3] S3_ENDPOINT or S3_BUCKET_VECTORS not set; skipping S3")
        return None

    ep = S3_ENDPOINT.strip()
    if not ep.startswith("http://") and not ep.startswith("https://"):
        scheme = "https" if S3_USE_SSL else "http"
        endpoint_url = f"{scheme}://{ep}"
    else:
        endpoint_url = ep

    s3_verify = os.getenv("S3_VERIFY", "").lower()
    verify = False if s3_verify in ("0", "false", "no") else S3_USE_SSL

    cfg = BotoConfig(signature_version="s3v4")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET,
            config=cfg,
            verify=verify,
        )
        print(f"[s3] boto3 client created endpoint={endpoint_url} verify={verify}")
        return s3
    except Exception as e:
        print("[s3] boto3 client creation failed:", e)
        return None

def _get_release_prefix_from_s3(s3) -> Optional[str]:
    if RELEASE_PREFIX:
        return RELEASE_PREFIX.rstrip("/")
    key = "releases/current"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_VECTORS, Key=key)
        body = resp["Body"].read().decode("utf-8")
        prefix = body.strip()
        return prefix.rstrip("/") if prefix else None
    except Exception:
        return None

def attempt_s3_fetch_if_needed():
    # simplified S3 fetch attempt (keeps previous behavior)
    need_meta = not META_JSON.exists()
    need_index = not FAISS_INDEX_PATH.exists()
    if not (need_meta or need_index):
        print("[startup] local artifacts present, skipping S3 download")
        return

    s3 = _s3_client()
    if s3 is None:
        print("[s3] S3 client not available; expecting local out/ to contain artifacts.")
        return

    prefix = _get_release_prefix_from_s3(s3)
    if prefix:
        try:
            s3.download_file(S3_BUCKET_VECTORS, f"{prefix}/meta.json", str(META_JSON))
            s3.download_file(S3_BUCKET_VECTORS, f"{prefix}/index.faiss", str(FAISS_INDEX_PATH))
            print("[s3] downloaded artifacts from prefix", prefix)
            return
        except Exception:
            pass

    # try root-level
    try:
        s3.download_file(S3_BUCKET_VECTORS, "meta.json", str(META_JSON))
        s3.download_file(S3_BUCKET_VECTORS, "index.faiss", str(FAISS_INDEX_PATH))
        print("[s3] downloaded artifacts from bucket root")
        return
    except Exception:
        pass

try:
    attempt_s3_fetch_if_needed()
except Exception as e:
    print("[startup] S3 fetch attempt raised an unexpected error:", e)
    traceback.print_exc()

if not META_JSON.exists():
    raise RuntimeError(f"meta.json not found at {META_JSON.resolve()}")

if not FAISS_INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH.resolve()}")

# load meta & faiss
try:
    _meta: Dict[str, Any] = json.loads(META_JSON.read_text(encoding="utf-8"))
except Exception as e:
    print("[error] failed to parse meta.json:", e)
    raise

try:
    _index = faiss.read_index(str(FAISS_INDEX_PATH))
except Exception as e:
    print("[error] failed to load FAISS index:", e)
    traceback.print_exc()
    raise

app = FastAPI(title="RAG — classification-driven prompt engineering")

class Req(BaseModel):
    query: str
    raw_k: Optional[int] = 64
    top_for_llm: Optional[int] = 8

# embeddings & search
def _embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    item = resp.data[0]
    emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
    if emb is None:
        raise RuntimeError("Failed to parse embedding response")
    return np.array(emb, dtype=np.float32)

def _search_faiss(q_emb: np.ndarray, k: int):
    q_arr = q_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_arr)
    k = max(1, int(k))
    k = min(k, max(1, int(_index.ntotal)))
    D, I = _index.search(q_arr, k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    results = []
    for sc, nid in zip(scores, ids):
        if int(nid) == -1:
            continue
        m = _meta.get(str(int(nid)))
        if not m:
            print(f"[search] warn: missing meta for id {nid}")
            continue
        results.append({"score": float(sc), "nid": int(nid), "meta": m})
    return results

def _aggregate_by_file(results: List[Dict[str, Any]]):
    file_sum: Dict[str, float] = {}
    best_chunk_for_file: Dict[str, Dict[str, Any]] = {}
    for r in results:
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        file_sum[sf] = file_sum.get(sf, 0.0) + r["score"]
        if sf not in best_chunk_for_file or r["score"] > best_chunk_for_file[sf]["score"]:
            best_chunk_for_file[sf] = r
    if not file_sum:
        return None, None
    best_file = max(file_sum.items(), key=lambda kv: kv[1])[0]
    return best_file, best_chunk_for_file[best_file]

# robust response extraction utilities (same as before)
def _resp_to_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
    except Exception:
        pass
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
        return "".join(parts).strip()
    if isinstance(out, str):
        return out.strip()
    try:
        return json.dumps(resp, default=str)
    except Exception:
        return str(resp)

# -------------------------
# CLASSIFIER: now includes GUIDANCE intent
# -------------------------
def classify_query(query: str) -> Dict[str, str]:
    """
    Returns JSON with keys:
     - intent: one of ["GREETING","CHIT_CHAT","FACTUAL_QUESTION","GUIDANCE","DOCUMENT_REQUEST","OTHER"]
     - explain: short explanation
     - language: language code or name (optional)
    """
    prompt = (
        "You are a compact intent classifier and language detector. Given the user's single input below, "
        "return a JSON object with EXACTLY three keys:\n"
        " - \"intent\": one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"GUIDANCE\",\"DOCUMENT_REQUEST\",\"OTHER\"]\n"
        " - \"explain\": one short sentence explaining why\n"
        " - \"language\": the detected language name or two-letter code (e.g. \"Russian\" or \"ru\")\n\n"
        "Definitions/examples:\n"
        " - GREETING: short hello/goodbye messages (no docs needed)\n"
        " - CHIT_CHAT: small talk / thanks / compliment (no docs)\n"
        " - FACTUAL_QUESTION: generic factual question where no document retrieval is needed (e.g., \"What is AI?\")\n"
        " - GUIDANCE: user asks for step-by-step guidance, procedures or how-to that should be answered using documents if available, but may be synthesized from top-K excerpts (do NOT invent facts)\n"
        " - DOCUMENT_REQUEST: user explicitly requests a document, template, sample file, or wants 'send X' / 'пример файла' (must prefer returning a file path from available docs)\n"
        " - OTHER: none of the above\n\n"
        "Respond ONLY with valid JSON (no extra text). Example:\n"
        "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to apply for residency\",\"language\":\"ru\"}\n\n"
        f"User input: {json.dumps(query)}\n"
    )
    try:
        resp = client.responses.create(model=CLASS_MODEL, input=prompt, max_output_tokens=120, temperature=0.0)
        txt = _resp_to_text(resp) or ""
        s = txt.strip()
        # try to extract JSON
        if s.startswith("```"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                s = s[start:end+1]
        try:
            j = json.loads(s)
        except Exception:
            # fallback: find first {...}
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
        if intent not in {"GREETING","CHIT_CHAT","FACTUAL_QUESTION","GUIDANCE","DOCUMENT_REQUEST","OTHER"}:
            intent = "OTHER"
        return {"intent": intent, "explain": explain, "language": language}
    except Exception as e:
        print("[classify] classifier error:", e)
        return {"intent": "OTHER", "explain": f"classifier error: {e}", "language": ""}

# -------------------------
# SIMPLE LLM helpers (greeting and direct factual answers)
# -------------------------
def generate_greeting_reply(user_text: str, language_hint: str) -> str:
    if language_hint:
        lang_instr = f"in {language_hint}"
    else:
        lang_instr = "in the same language as the user"
    prompt = (
        f"The user wrote: {json.dumps(user_text)}\n\n"
        f"Produce a single short friendly reply ({lang_instr}). Keep it to one short sentence (<=20 words). "
        "Do NOT include file paths or any extra commentary. Return only the reply text."
    )
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=50, temperature=0.0)
        txt = _resp_to_text(resp).strip()
        if txt.startswith("```"):
            txt = txt.strip("` \n")
        for line in txt.splitlines():
            l = line.strip()
            if l:
                return l
        return txt
    except Exception as e:
        print("[greeting] generation failed:", e)
        return "Hi — how can I help you today?"

def llm_answer_factual(query: str, language_hint: str) -> str:
    if language_hint:
        lang_instr = f"Answer in {language_hint}."
    else:
        lang_instr = "Answer in the same language as the user."
    prompt = (
        f"You are a concise helpful assistant. {lang_instr} "
        "Answer the user question briefly (1-2 short paragraphs). Do NOT include any file paths or suggest internal document locations.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=400, temperature=0.0)
        return _resp_to_text(resp).strip()
    except Exception as e:
        print("[factual] LLM error:", e)
        return f"(LLM error: {e})"

# -------------------------
# PROMPT ENGINEERING: two retrieval prompt styles
# -------------------------
def _prepare_document_request_prompt(query: str, top_chunks: List[Dict[str,Any]]) -> str:
    """
    DOCUMENT_REQUEST: user expects a document/template. We instruct the LLM to:
      - Use ONLY the provided excerpts
      - Return EXACT JSON: { "answer": "...", "file": "<path or null>" }
      - If a document (template) matching the request exists among excerpts, set 'file' to that path.
      - If nothing applies, answer must be: "I don't know based on the provided documents." and file null.
    """
    lines = [
        "You are a strict document retriever. Use ONLY the excerpts below; do NOT invent or generalize beyond them.",
        "User query:",
        query,
        "",
        "Below are document excerpts (file + page + excerpt). If one of the documents is the requested template or sample, choose it.",
        ""
    ]
    for i, r in enumerate(top_chunks, start=1):
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        page = m.get("page")
        text = (m.get("text") or m.get("md") or "").strip()
        excerpt = text[:1600].replace("\n", " ").strip()
        lines.append(f"[{i}] file: {sf} page: {page}")
        lines.append(f"excerpt: {excerpt}")
        lines.append("")
    lines.extend([
        "Output requirements:",
        "- Respond ONLY with valid JSON with exactly these keys: 'answer' and 'file'.",
        "- 'answer' must be a short sentence telling whether the requested document exists in the provided excerpts.",
        "- If a matching document exists, 'answer' must be a short note (<=60 words) and 'file' must be the path string of that document.",
        "- If no matching document exists in the excerpts, set 'answer' to: \"I don't know based on the provided documents.\" and 'file' to null.",
        "- Do NOT add any other keys, commentary, or explanation. Return JSON only."
    ])
    return "\n".join(lines)

def _prepare_guidance_prompt(query: str, top_chunks: List[Dict[str,Any]]) -> str:
    """
    GUIDANCE: user asks for how-to / guidance. We want a concise synthesized answer based ON the excerpts.
      - Use only excerpts; do NOT invent
      - Return JSON { "answer": "...", "file": "<best supporting file or null>" }
      - If the excerpts don't contain sufficient info, answer must be "I don't know based on the provided documents."
    """
    lines = [
        "You are an assistant that gives practical guidance using ONLY the provided document excerpts. Do NOT invent facts.",
        "User query:",
        query,
        "",
        "Here are top document excerpts (file + page + excerpt):",
        ""
    ]
    for i, r in enumerate(top_chunks, start=1):
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        page = m.get("page")
        text = (m.get("text") or m.get("md") or "").strip()
        excerpt = text[:1600].replace("\n", " ").strip()
        lines.append(f"[{i}] file: {sf} page: {page}")
        lines.append(f"excerpt: {excerpt}")
        lines.append("")
    lines.extend([
        "Output requirements:",
        "- Respond ONLY with valid JSON with exactly two keys: 'answer' and 'file'.",
        "- 'answer' should be a short, step-oriented guidance or summary (<=180 words) drawn only from the excerpts. If you cannot produce a guidance wholly supported by the excerpts, set 'answer' to: \"I don't know based on the provided documents.\"",
        "- 'file' should be the single best supporting source_file path from the excerpts (or null if none).",
        "- Do NOT invent, assume, or provide extra commentary. Return JSON only."
    ])
    return "\n".join(lines)

def _call_llm_for_answer(prompt: str, model: str = LLM_MODEL, max_tokens:int = 512) -> str:
    resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens, temperature=0.0)
    text = None
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
    if text is None:
        try:
            text = json.dumps(resp, default=str)
        except Exception:
            text = str(resp)
    return text

# -------------------------
# Main / routing logic — strictly follows classifier intent only (no keyword heuristics)
# -------------------------
@app.post("/query")
def query_endpoint(req: Req) -> Dict[str, Optional[Any]]:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")

    # 1) classify
    try:
        cls = classify_query(q)
    except Exception as e:
        print("[query] classification failed:", e)
        cls = {"intent": "OTHER", "explain": "classification failure", "language": ""}

    intent = cls.get("intent")
    lang = cls.get("language") or ""
    print(f"[query] classifier -> intent={intent} lang={lang} explain={cls.get('explain')}")

    # 2) short-circuit intents
    if intent in ("GREETING", "CHIT_CHAT"):
        greeting = generate_greeting_reply(q, lang)
        return {"answer": greeting, "file": None}

    if intent == "FACTUAL_QUESTION":
        answer = llm_answer_factual(q, lang)
        return {"answer": answer, "file": None}

    # 3) retrieval-driven intents: GUIDANCE and DOCUMENT_REQUEST
    if intent in ("GUIDANCE", "DOCUMENT_REQUEST"):
        # run embedding + search
        try:
            q_emb = _embed_text(q)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

        try:
            raw_k = max(1, int(req.raw_k or 64))
            results = _search_faiss(q_emb, k=raw_k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"search failed: {e}")

        # if nothing found -> must return IDK per prompt style
        if not results:
            return {"answer": "I don't know based on the provided documents.", "file": None}

        best_file_agg, best_chunk = _aggregate_by_file(results)
        top_n = max(1, int(req.top_for_llm or 8))
        top_chunks = results[:top_n]

        # choose prompt variant
        if intent == "DOCUMENT_REQUEST":
            prompt = _prepare_document_request_prompt(q, top_chunks)
        else:
            prompt = _prepare_guidance_prompt(q, top_chunks)

        # instruct language at top if known
        if lang:
            prompt = f"Answer in the same language as detected: {lang}\n\n" + prompt
        else:
            prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

        # call LLM
        llm_text = None
        llm_json = None
        try:
            llm_text = _call_llm_for_answer(prompt, model=LLM_MODEL, max_tokens=512)
            s = (llm_text or "").strip()
            # extract JSON blob if wrapped in fences
            if s.startswith("```"):
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    s = s[start:end+1]
            try:
                llm_json = json.loads(s)
            except Exception:
                # try to find first {...}
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
        except Exception as e:
            print("[synth] LLM synth failed:", e)
            llm_text = None
            llm_json = None

        # If LLM returned proper JSON with answer & file -> honor it
        if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
            answer = llm_json.get("answer", "").strip()
            file_chosen = llm_json.get("file")  # may be null
            # normalize null -> None, string stays
            if file_chosen is None:
                file_chosen = None
            else:
                file_chosen = str(file_chosen)
        else:
            # fallback behavior: return the best chunk text (retrieval-only) and best aggregated file
            chunk_meta = best_chunk["meta"]
            answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()
            file_chosen = chunk_meta.get("source_file") or chunk_meta.get("filename") or best_file_agg

        if not answer:
            answer = "I don't know based on the provided documents."
        if isinstance(answer, str) and len(answer) > 1600:
            answer = answer[:1600].rstrip() + "..."

        return {"answer": answer, "file": file_chosen}

    # 4) OTHER -> fallback to concise factual LLM reply (no file)
    answer = llm_answer_factual(q, lang)
    return {"answer": answer, "file": None}


# curl -v -X POST http://127.0.0.1:8080/query \
#   -H "Content-Type: application/json" \
#   -d '{"query":"Как податься внж?"}'

# docker build --no-cache -t "rag" . 
# docker rmi rag

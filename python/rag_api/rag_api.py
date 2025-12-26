# rag_file_only_api.py
from __future__ import annotations
import os
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # allow .env to populate env vars for local testing

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# S3 env names (support common variants)
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET")
S3_BUCKET_VECTORS = os.getenv("S3_BUCKET_VECTORS")
S3_USE_SSL = os.getenv("S3_USE_SSL", "false").lower() in ("1", "true", "yes")
RELEASE_PREFIX = os.getenv("RELEASE_PREFIX")  # optional override for release prefix

OUT = Path(os.getenv("OUT_DIR", "out"))
OUT.mkdir(parents=True, exist_ok=True)

# expected artifact names
META_JSON = OUT / "meta.json"
FAISS_INDEX_PATH = OUT / "index.faiss"

# optional artifacts (download if available)
OPTIONAL_ARTIFACTS = ["embeddings.npy", "ids.npy", "chunks.jsonl", "manifest.json"]

# embedding & LLM models
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CLASS_MODEL = os.getenv("CLASS_MODEL", LLM_MODEL)  # classifier model (defaults to LLM_MODEL)

# Attempt to download artifacts from S3 if local files are missing

def _s3_client():
    """
    Create and return a boto3 S3 client, or None if S3 not configured / boto3 unavailable.
    This normalizes S3_ENDPOINT to ensure it includes a scheme (http/https),
    because boto3 requires endpoint_url to contain a scheme.
    """
    try:
        import boto3
        from botocore.client import Config as BotoConfig
    except Exception:
        print("[s3] boto3 not installed; skipping S3 support")
        return None

    # must have at least endpoint and bucket to attempt
    if not S3_ENDPOINT or not S3_BUCKET_VECTORS:
        print("[s3] S3_ENDPOINT or S3_BUCKET_VECTORS not set; skipping S3")
        return None

    # Normalize endpoint: ensure it has http:// or https://
    ep = S3_ENDPOINT.strip()
    if not ep.startswith("http://") and not ep.startswith("https://"):
        scheme = "https" if S3_USE_SSL else "http"
        endpoint_url = f"{scheme}://{ep}"
    else:
        endpoint_url = ep

    # Optional: allow an env var to override verify behavior (e.g. disable for self-signed in dev)
    # If S3_VERIFY env is set to "false" or "0", we'll pass verify=False
    s3_verify = os.getenv("S3_VERIFY", "").lower()
    if s3_verify in ("0", "false", "no"):
        verify = False
    else:
        verify = S3_USE_SSL  # default: verify SSL only when using https

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
        # optional quick sanity check (comment out if you don't want list calls at startup)
        # s3.list_buckets()
        print(f"[s3] boto3 client created endpoint={endpoint_url} verify={verify}")
        return s3
    except Exception as e:
        print("[s3] boto3 client creation failed:", e)
        return None

def _get_release_prefix_from_s3(s3) -> Optional[str]:
    # 1) explicit RELEASE_PREFIX env var
    if RELEASE_PREFIX:
        print("[s3] Using RELEASE_PREFIX from env:", RELEASE_PREFIX)
        return RELEASE_PREFIX.rstrip("/")

    # 2) try reading releases/current object
    key = "releases/current"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_VECTORS, Key=key)
        body = resp["Body"].read().decode("utf-8")
        prefix = body.strip()
        if prefix:
            print(f"[s3] read {key} -> '{prefix}'")
            return prefix.rstrip("/")
        return None
    except Exception as e:
        # Not fatal — we'll try other methods
        print(f"[s3] cannot read {key} ({e})")
        return None

def _download_from_prefix(s3, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    required_files = ["meta.json", "index.faiss"]
    ok = True
    for fname in required_files:
        key = f"{prefix}/{fname}"
        dest = OUT / fname
        try:
            print(f"[s3] downloading s3://{S3_BUCKET_VECTORS}/{key} -> {dest}")
            s3.download_file(S3_BUCKET_VECTORS, key, str(dest))
        except Exception as e:
            print(f"[s3] failed to download {key}: {e}")
            ok = False
    # optional artifacts
    for fname in OPTIONAL_ARTIFACTS:
        key = f"{prefix}/{fname}"
        dest = OUT / fname
        try:
            s3.download_file(S3_BUCKET_VECTORS, key, str(dest))
            print(f"[s3] optional downloaded {key}")
        except Exception:
            # ignore optional failures
            pass
    return ok

def _download_root_files_or_search(s3) -> bool:
    """
    Try to download meta.json and index.faiss from the bucket root.
    If not present at root, attempt to list objects and try to find the first keys that end with those names.
    """
    required = {"meta.json": OUT / "meta.json", "index.faiss": OUT / "index.faiss"}
    success = True
    for key_name, dest in required.items():
        try:
            print(f"[s3] attempting to download s3://{S3_BUCKET_VECTORS}/{key_name} -> {dest}")
            s3.download_file(S3_BUCKET_VECTORS, key_name, str(dest))
            continue
        except Exception as e:
            print(f"[s3] direct download failed for {key_name}: {e}")
        # fallback: try to find a matching key by listing
        try:
            resp = s3.list_objects_v2(Bucket=S3_BUCKET_VECTORS, Prefix="", MaxKeys=1000)
            items = resp.get("Contents", []) or []
            candidate = None
            # prefer exact match anywhere in key suffix
            for obj in items:
                k = obj.get("Key", "")
                if k.endswith("/" + key_name) or k == key_name or k.endswith(key_name):
                    candidate = k
                    break
            if candidate:
                print(f"[s3] found candidate for {key_name}: {candidate}, downloading")
                s3.download_file(S3_BUCKET_VECTORS, candidate, str(dest))
            else:
                print(f"[s3] no candidate found for {key_name} in top-level listing")
                success = False
        except Exception as e:
            print(f"[s3] list_objects_v2 failed: {e}")
            success = False
    # try optional artifacts by searching for common names
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET_VECTORS, Prefix="", MaxKeys=1000)
        items = resp.get("Contents", []) or []
        keys = [obj.get("Key", "") for obj in items]
        for fname in OPTIONAL_ARTIFACTS:
            for k in keys:
                if k.endswith(fname):
                    try:
                        print(f"[s3] downloading optional {k} -> {OUT/fname}")
                        s3.download_file(S3_BUCKET_VECTORS, k, str(OUT / fname))
                    except Exception:
                        pass
                    break
    except Exception:
        pass
    return success

def attempt_s3_fetch_if_needed():
    # Only attempt if local artifacts missing and S3 configured
    need_meta = not META_JSON.exists()
    need_index = not FAISS_INDEX_PATH.exists()
    if not (need_meta or need_index):
        print("[startup] local artifacts present, skipping S3 download")
        return

    s3 = _s3_client()
    if s3 is None:
        print("[s3] S3 client not available or S3 env not configured; expecting local out/ to contain artifacts.")
        return

    # 1) try release prefix via releases/current or env
    prefix = _get_release_prefix_from_s3(s3)
    if prefix:
        ok = _download_from_prefix(s3, prefix)
        if ok:
            print(f"[s3] downloaded artifacts from prefix {prefix}")
            return
        else:
            print(f"[s3] failed to download all required artifacts from prefix {prefix}, will try root-level search")

    # 2) try root-level keys or search
    ok2 = _download_root_files_or_search(s3)
    if ok2:
        print("[s3] downloaded artifacts from bucket root or found matching keys")
        return

    # 3) If still missing, attempt to list prefixes under "releases/" and try the most recent
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET_VECTORS, Prefix="releases/", Delimiter="/", MaxKeys=1000)
        prefixes = []
        # boto3 returns CommonPrefixes when Delimiter set
        for p in resp.get("CommonPrefixes", []) or []:
            pref = p.get("Prefix")
            if pref:
                prefixes.append(pref.rstrip("/"))
        if prefixes:
            # try last prefix (not guaranteed chronological, but often created order)
            tried = 0
            for pref in reversed(prefixes):
                if tried >= 5:
                    break
                tried += 1
                # try downloading from this prefix
                candidate_prefix = pref
                print(f"[s3] attempting to download from discovered prefix: {candidate_prefix}")
                ok = _download_from_prefix(s3, candidate_prefix)
                if ok:
                    print("[s3] success with discovered prefix:", candidate_prefix)
                    return
    except Exception as e:
        print(f"[s3] listing releases/ failed: {e}")

    # Nothing worked — print bucket top keys to help debugging
    try:
        print("[s3] listing top 50 objects in bucket to help debugging:")
        resp = s3.list_objects_v2(Bucket=S3_BUCKET_VECTORS, MaxKeys=50)
        for obj in resp.get("Contents", []) or []:
            print(" -", obj.get("Key"))
    except Exception as e:
        print("[s3] failed to list bucket objects:", e)

# run S3 fetch attempt if needed
try:
    attempt_s3_fetch_if_needed()
except Exception as e:
    print("[startup] S3 fetch attempt raised an unexpected error:", e)
    traceback.print_exc()

# after S3 attempts, ensure required files present
if not META_JSON.exists():
    raise RuntimeError(f"meta.json not found at {META_JSON.resolve()} (S3 attempt done, check bucket and keys)")

if not FAISS_INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH.resolve()} (S3 attempt done, check bucket and keys)")

# load meta and index
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

app = FastAPI(title="RAG minimal file chooser + LLM synth (S3-capable)")

class Req(BaseModel):
    query: str
    raw_k: Optional[int] = 64
    top_for_llm: Optional[int] = 8

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

def _prepare_llm_prompt(query: str, top_chunks: List[Dict[str,Any]]) -> str:
    lines = [
        "You are a helpful assistant that synthesizes multiple document excerpts and decides which document is most relevant.",
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
        "- 'file' must be the single most relevant source_file (the path) chosen from the excerpts above.",
        "- Do not output any extra text, explanation, or commentary — output must be valid JSON only.",
        "",
        "Return the JSON now."
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

# helper: robustly extract text from responses (reusable)
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
    # fallback
    try:
        return json.dumps(resp, default=str)
    except Exception:
        return str(resp)

# GPT-only classifier (intent + language)
def classify_query(query: str) -> Dict[str, str]:
    """
    Ask GPT to return JSON with keys:
      - intent: GREETING, CHIT_CHAT, FACTUAL_QUESTION, DOCUMENT_REQUEST, OTHER
      - explain: short explanation
      - language: language name or two-letter code (e.g., 'ru', 'Russian', 'en')
    If GPT parsing fails, fallback to intent=OTHER with empty language.
    """
    prompt = (
        "You are a compact intent classifier and language detector. Given the user's single input below, "
        "return a JSON object with EXACTLY three keys:\n"
        " - \"intent\": one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"DOCUMENT_REQUEST\",\"OTHER\"]\n"
        " - \"explain\": one short sentence explaining why\n"
        " - \"language\": the detected language name or two-letter code (e.g. \"Russian\" or \"ru\")\n\n"
        "Respond ONLY with valid JSON (no extra text). Example:\n"
        "{\"intent\":\"GREETING\",\"explain\":\"short hello\",\"language\":\"ru\"}\n\n"
        f"User input: {json.dumps(query)}\n"
    )
    try:
        resp = client.responses.create(model=CLASS_MODEL, input=prompt, max_output_tokens=120, temperature=0.0)
        txt = _resp_to_text(resp)
        s = (txt or "").strip()
        # attempt to extract JSON object from response
        if s.startswith("```"):
            # find braces inside fenced block
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                s = s[start:end+1]
        try:
            j = json.loads(s)
        except Exception:
            # fallback: try to find first {...}
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
        if intent not in {"GREETING","CHIT_CHAT","FACTUAL_QUESTION","DOCUMENT_REQUEST","OTHER"}:
            intent = "OTHER"
        return {"intent": intent, "explain": explain, "language": language}
    except Exception as e:
        # last-resort fallback: treat as OTHER and empty language
        print("[classify] classifier error:", e)
        return {"intent": "OTHER", "explain": f"classifier error: {e}", "language": ""}

# Generate greeting reply using GPT (no local greeting map)
def generate_greeting_reply(user_text: str, language_hint: str) -> str:
    """
    Ask GPT to return a short one-sentence friendly reply in the detected language.
    If language_hint is empty, ask GPT to reply in the same language as user input.
    """
    if language_hint:
        lang_instr = f"in {language_hint}"
    else:
        lang_instr = "in the same language as the user"
    prompt = (
        f"The user wrote: {json.dumps(user_text)}\n\n"
        f"Produce a single short friendly reply ({lang_instr}). Keep it to one short sentence (<=20 words). "
        "Do NOT include file paths, document names, or any extra commentary. Return only the reply text."
    )
    try:
        resp = client.responses.create(model=LLM_MODEL, input=prompt, max_output_tokens=50, temperature=0.0)
        txt = _resp_to_text(resp).strip()
        # strip code fences and return first non-empty line
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

# Provide short answer from LLM for FACTUAL_QUESTION (no retrieval)
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

@app.post("/query")
def query_endpoint(req: Req) -> Dict[str, Optional[Any]]:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")

    # --- Classification-first step (GPT-only) ---
    try:
        cls = classify_query(q)
    except Exception as e:
        # classification failure; log and fall back to previous pipeline
        print("[query] classification failed:", e)
        cls = {"intent": "OTHER", "explain": "classification failure", "language": ""}

    intent = cls.get("intent")
    lang = cls.get("language") or ""

    # GREETING / CHIT_CHAT: generate greeting via GPT and return (no retrieval)
    if intent in ("GREETING", "CHIT_CHAT"):
        greeting = generate_greeting_reply(q, lang)
        return {"answer": greeting, "file": None}

    # FACTUAL_QUESTION: ask LLM directly (no retrieval)
    if intent == "FACTUAL_QUESTION":
        answer = llm_answer_factual(q, lang)
        return {"answer": answer, "file": None}

    # DOCUMENT_REQUEST or OTHER -> proceed to retrieval pipeline
    try:
        q_emb = _embed_text(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

    try:
        raw_k = max(1, int(req.raw_k or 64))
        results = _search_faiss(q_emb, k=raw_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    if not results:
        return {"answer": "", "file": None}

    best_file_agg, best_chunk = _aggregate_by_file(results)
    top_n = max(1, int(req.top_for_llm or 8))
    top_chunks = results[:top_n]

    # Tell the synthesizer explicitly to answer in same language if we have language hint
    prompt = _prepare_llm_prompt(q, top_chunks)
    if lang:
        # insert language instruction at the top (keeps original instructions intact)
        prompt = f"Answer in the same language as detected: {lang}\n\n" + prompt
    else:
        # instruct LLM to answer in same language as user if possible
        prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

    llm_text = None
    llm_json = None
    try:
        llm_text = _call_llm_for_answer(prompt, model=LLM_MODEL, max_tokens=512)
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

    if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
        answer = llm_json.get("answer", "").strip()
        file_chosen = llm_json.get("file") or best_file_agg
    else:
        chunk_meta = best_chunk["meta"]
        answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()
        file_chosen = chunk_meta.get("source_file") or chunk_meta.get("filename")

    if answer is None:
        answer = ""
    if isinstance(answer, str) and len(answer) > 1600:
        answer = answer[:1600].rstrip() + "..."

    return {"answer": answer, "file": file_chosen}


# curl -v -X POST http://127.0.0.1:8080/query \
#   -H "Content-Type: application/json" \
#   -d '{"query":"Как податься внж?"}'

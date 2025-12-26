#!/usr/bin/env python3
"""
Simple retrieval bot:
 - Edit QUERY below (or replace with your own input mechanism).
 - Requires: out/index.faiss, out/meta.json, out/embeddings.npy, out/ids.npy (created by your pipeline).
 - Expects OPENAI_API_KEY in environment or .env.

Output: prints JSON like:
{
  "answer": "snippet text ...",
  "file": "italy/application_ru.pdf",
  "page": 3,
  "score": 0.52,
  "retrieved": [
     {"file":"italy/application_ru.pdf","page":3,"score":0.52},
     ...
  ]
}
"""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ----------------- USER EDITABLE -----------------
# Put your query here (or replace this with input() or HTTP endpoint)
QUERY = "Как подать на ВНЖ?"   # example; change as needed

# top_k of raw retrieval from FAISS (we will aggregate by file afterwards)
RAW_TOP_K = 64
RETURN_TOP_K = 5  # how many final answers to show in "retrieved"
# -------------------------------------------------

# Paths
OUT = Path("out")
META_JSON = OUT / "meta.json"
FAISS_INDEX_PATH = OUT / "index.faiss"

EMBED_MODEL = "text-embedding-3-small"

# Load env
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

# Load FAISS index
if not FAISS_INDEX_PATH.exists():
    print(f"[error] FAISS index not found at {FAISS_INDEX_PATH.resolve()}", file=sys.stderr)
    sys.exit(1)
index = faiss.read_index(str(FAISS_INDEX_PATH))

# Utility: numeric -> meta lookup
def meta_for_nid(nid: int) -> Dict[str, Any]:
    return meta_map.get(str(int(nid)), {})

# Embed query
def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    item = resp.data[0]
    emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
    if emb is None:
        raise RuntimeError("Failed to parse embedding response")
    return np.array(emb, dtype=np.float32)

# Search
def search_faiss(q_emb: np.ndarray, k: int = RAW_TOP_K):
    q_arr = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_arr)
    D, I = index.search(q_arr, k)
    # D: (1, k) scores, I: (1, k) ids
    scores = D[0].tolist()
    ids = I[0].tolist()
    results = []
    for sc, nid in zip(scores, ids):
        if int(nid) == -1:
            continue
        m = meta_for_nid(int(nid))
        if not m:
            # missing meta for nid
            continue
        results.append({"score": float(sc), "nid": int(nid), "meta": m})
    return results

# Decide best file by aggregating scores per source_file
def choose_best_file(results: List[Dict[str, Any]]):
    # group by source_file: sum scores and track best chunk per file
    file_scores = {}
    best_chunk_per_file = {}
    for r in results:
        m = r["meta"]
        sf = m.get("source_file") or m.get("filename") or "unknown"
        if sf not in file_scores:
            file_scores[sf] = 0.0
            best_chunk_per_file[sf] = r
        file_scores[sf] += r["score"]
        # keep best-scoring chunk for this file
        if r["score"] > best_chunk_per_file[sf]["score"]:
            best_chunk_per_file[sf] = r
    if not file_scores:
        return None, None, None
    # pick file with highest aggregate score
    best_file = max(file_scores.items(), key=lambda x: x[1])[0]
    best_score_sum = file_scores[best_file]
    best_chunk = best_chunk_per_file[best_file]
    return best_file, best_score_sum, best_chunk

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

def main(query: str):
    # 1) embed query
    q_emb = embed_query(query)

    # 2) search
    results = search_faiss(q_emb, k=RAW_TOP_K)
    if not results:
        out = {
            "answer": "",
            "file": None,
            "page": None,
            "score": None,
            "retrieved": []
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # 3) choose best file
    best_file, agg_score, best_chunk = choose_best_file(results)
    retrieved = format_retrieved(results, top_n=RETURN_TOP_K)

    if best_chunk is None:
        out = {
            "answer": "",
            "file": None,
            "page": None,
            "score": None,
            "retrieved": retrieved
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # 4) produce answer text — use the chunk text (short)
    chunk_meta = best_chunk["meta"]
    snippet = chunk_meta.get("text") or chunk_meta.get("md") or ""
    # produce friendly answer: first 600 chars of snippet
    answer_text = snippet.strip()
    if len(answer_text) > 800:
        answer_text = answer_text[:800].rstrip() + "..."

    out = {
        "answer": answer_text,
        "file": best_file,
        "page": chunk_meta.get("page"),
        "score": float(best_chunk["score"]),
        "retrieved": retrieved
    }
    # print JSON (UTF-8 safe)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(QUERY)

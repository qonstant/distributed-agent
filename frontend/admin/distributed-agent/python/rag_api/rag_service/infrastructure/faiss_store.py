from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List

import faiss
import numpy as np

from rag_service.domain.models import RetrievedHit
from rag_service.infrastructure.artifacts import ensure_local_artifacts
from rag_service.infrastructure.config import Settings


class FaissMetadataStore:
    def __init__(self, meta: Dict[str, Any], index) -> None:
        self._meta = meta
        self._index = index

    @classmethod
    def load(cls, settings: Settings) -> "FaissMetadataStore":
        try:
            ensure_local_artifacts(settings)
        except Exception as exc:
            print("[startup] S3 fetch attempt raised an unexpected error:", exc)
            traceback.print_exc()

        if not settings.meta_json_path.exists():
            raise RuntimeError(f"meta.json not found at {settings.meta_json_path.resolve()}")
        if not settings.faiss_index_path.exists():
            raise RuntimeError(f"FAISS index not found at {settings.faiss_index_path.resolve()}")

        try:
            meta: Dict[str, Any] = json.loads(settings.meta_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("[error] failed to parse meta.json:", exc)
            raise

        try:
            index = faiss.read_index(str(settings.faiss_index_path))
        except Exception as exc:
            print("[error] failed to load FAISS index:", exc)
            traceback.print_exc()
            raise

        return cls(meta=meta, index=index)

    def search(self, query_embedding: np.ndarray, k: int) -> List[RetrievedHit]:
        q_arr = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_arr)
        k = max(1, int(k))
        k = min(k, max(1, int(self._index.ntotal)))
        distances, ids = self._index.search(q_arr, k)

        results: List[RetrievedHit] = []
        for score, nid in zip(distances[0].tolist(), ids[0].tolist()):
            if int(nid) == -1:
                continue
            meta = self._meta.get(str(int(nid)))
            if not meta:
                print(f"[search] warn: missing meta for id {nid}")
                continue
            results.append(RetrievedHit(score=float(score), nid=int(nid), meta=meta))
        return results
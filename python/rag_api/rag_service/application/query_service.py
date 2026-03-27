from __future__ import annotations

from typing import List, Optional, Tuple

from rag_service.domain.models import QueryResult, RetrievedHit
from rag_service.infrastructure.openai_gateway import OpenAIGateway
from rag_service.infrastructure.prompts import (
    prepare_document_request_prompt,
    prepare_guidance_prompt,
)


def _aggregate_by_file(results: List[RetrievedHit]) -> Tuple[Optional[str], Optional[RetrievedHit]]:
    file_sum: dict[str, float] = {}
    best_chunk_for_file: dict[str, RetrievedHit] = {}
    for hit in results:
        meta = hit.meta
        source_file = meta.get("source_file") or meta.get("filename") or "unknown"
        file_sum[source_file] = file_sum.get(source_file, 0.0) + hit.score
        if source_file not in best_chunk_for_file or hit.score > best_chunk_for_file[source_file].score:
            best_chunk_for_file[source_file] = hit
    if not file_sum:
        return None, None
    best_file = max(file_sum.items(), key=lambda item: item[1])[0]
    return best_file, best_chunk_for_file[best_file]


class QueryService:
    def __init__(self, gateway: OpenAIGateway, store) -> None:
        self._gateway = gateway
        self._store = store

    def handle_query(
        self,
        query: str,
        raw_k: Optional[int] = None,
        top_for_llm: Optional[int] = None,
    ) -> QueryResult:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("query is empty")

        classification = self._gateway.classify_query(normalized_query)
        intent = classification.intent
        language = classification.language or ""
        print(
            f"[query] classifier -> intent={intent} lang={language} "
            f"explain={classification.explain}"
        )

        if intent in ("GREETING", "CHIT_CHAT"):
            greeting = self._gateway.generate_greeting_reply(normalized_query, language)
            return QueryResult(answer=greeting, file=None)

        if intent == "FACTUAL_QUESTION":
            answer = self._gateway.answer_factual(normalized_query, language)
            return QueryResult(answer=answer, file=None)

        if intent in ("GUIDANCE", "DOCUMENT_REQUEST"):
            try:
                query_embedding = self._gateway.embed_text(normalized_query)
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"embedding failed: {exc}") from exc

            try:
                results = self._store.search(query_embedding, k=max(1, int(raw_k or 64)))
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"search failed: {exc}") from exc

            if not results:
                return QueryResult(
                    answer="I don't know based on the provided documents.",
                    file=None,
                )

            best_file_agg, best_chunk = _aggregate_by_file(results)
            top_n = max(1, int(top_for_llm or 8))
            top_chunks = results[:top_n]

            if intent == "DOCUMENT_REQUEST":
                prompt = prepare_document_request_prompt(normalized_query, top_chunks)
            else:
                prompt = prepare_guidance_prompt(normalized_query, top_chunks)

            if language:
                prompt = f"Answer in the same language as detected: {language}\n\n" + prompt
            else:
                prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

            llm_json = self._gateway.generate_json_response(prompt, max_tokens=512)

            if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
                answer = str(llm_json.get("answer", "")).strip()
                file_chosen = llm_json.get("file")
                if file_chosen is not None:
                    file_chosen = str(file_chosen)
            else:
                if best_chunk is None:
                    return QueryResult(
                        answer="I don't know based on the provided documents.",
                        file=None,
                    )
                chunk_meta = best_chunk.meta
                answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()
                file_chosen = (
                    chunk_meta.get("source_file")
                    or chunk_meta.get("filename")
                    or best_file_agg
                )

            if not answer:
                answer = "I don't know based on the provided documents."
            if len(answer) > 1600:
                answer = answer[:1600].rstrip() + "..."

            return QueryResult(answer=answer, file=file_chosen)

        answer = self._gateway.answer_factual(normalized_query, language)
        return QueryResult(answer=answer, file=None)

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_service.application.query_service import QueryService
from rag_service.infrastructure.config import load_settings
from rag_service.infrastructure.faiss_store import FaissMetadataStore
from rag_service.infrastructure.openai_gateway import OpenAIGateway


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    raw_k: Optional[int] = 64
    top_for_llm: Optional[int] = 8


class QueryResponse(BaseModel):
    answer: str
    file: Optional[str] = None


def create_app() -> FastAPI:
    settings = load_settings()
    gateway = OpenAIGateway(settings)
    store = FaissMetadataStore.load(settings)
    conversation_memory = None
    if settings.redis_url:
        from rag_service.infrastructure.conversation_memory import RedisConversationMemory

        try:
            conversation_memory = RedisConversationMemory(
                settings.redis_url,
                key_prefix=settings.conversation_key_prefix,
                max_items=settings.conversation_max_items,
            )
            print(
                f"[memory] redis conversation history enabled "
                f"(prefix={settings.conversation_key_prefix!r} max_items={settings.conversation_max_items})"
            )
        except Exception as exc:
            print(f"[memory] redis conversation history disabled: {exc}")

    query_service = QueryService(gateway, store, conversation_memory=conversation_memory)

    app = FastAPI(title="RAG — classification-driven prompt engineering")

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(req: QueryRequest) -> QueryResponse:
        try:
            result = query_service.handle_query(
                req.query,
                conversation_id=req.conversation_id,
                raw_k=req.raw_k,
                top_for_llm=req.top_for_llm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return QueryResponse(answer=result.answer, file=result.file)

    return app


app = create_app()

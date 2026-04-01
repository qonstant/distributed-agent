from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_service.application.query_service import QueryService
from rag_service.infrastructure.config import load_settings
from rag_service.infrastructure.faiss_store import FaissMetadataStore
from rag_service.infrastructure.openai_gateway import OpenAIGateway


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    preferred_name: Optional[str] = None
    raw_k: Optional[int] = 64
    top_for_llm: Optional[int] = 8


class ClassificationResponse(BaseModel):
    intent: str
    explain: str = ""
    language: str = ""
    model: str = ""
    version: str = ""
    profile_action: str = ""
    preferred_name: str = ""


class UsageEventResponse(BaseModel):
    event_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    file: Optional[str] = None
    classification: Optional[ClassificationResponse] = None
    usage_events: list[UsageEventResponse] = Field(default_factory=list)


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
                preferred_name=req.preferred_name,
                raw_k=req.raw_k,
                top_for_llm=req.top_for_llm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        classification = None
        if result.classification is not None:
            classification = ClassificationResponse(
                intent=result.classification.intent,
                explain=result.classification.explain,
                language=result.classification.language,
                model=result.classification.model,
                version=result.classification.version,
                profile_action=result.classification.profile_action,
                preferred_name=result.classification.preferred_name,
            )
        return QueryResponse(
            answer=result.answer,
            file=result.file,
            classification=classification,
            usage_events=[
                UsageEventResponse(
                    event_type=item.event_type,
                    input_tokens=item.input_tokens,
                    output_tokens=item.output_tokens,
                    total_tokens=item.total_tokens,
                    estimated_cost=item.estimated_cost,
                )
                for item in result.usage_events
            ],
        )

    return app


app = create_app()

from __future__ import annotations

import json
import math
from typing import List, Optional

from rag_service.domain.models import (
    Classification,
    ConversationMessage,
    GuardrailResult,
    ModelUsage,
    UsageEventRecord,
)
from rag_service.infrastructure.prompts import build_history_lines

# Hardcoded pricing based on OpenAI API pricing pages for the current models in use:
# - gpt-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens
# - text-embedding-3-small: $0.02 / 1M input tokens
GPT_4O_MINI_INPUT_PER_1M = 0.15
GPT_4O_MINI_OUTPUT_PER_1M = 0.60
TEXT_EMBEDDING_3_SMALL_INPUT_PER_1M = 0.02

# Rough approximation only. We are intentionally not using a tokenizer yet.
CHARS_PER_TOKEN = 4.0

CLASSIFICATION_PROMPT_OVERHEAD_TOKENS = 110
GREETING_PROMPT_OVERHEAD_TOKENS = 60
FACTUAL_PROMPT_OVERHEAD_TOKENS = 80


def estimate_tokens(text: str) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, math.ceil(len(value) / CHARS_PER_TOKEN))


def estimate_classification_event(
    query: str,
    history: Optional[List[ConversationMessage]],
    classification: Classification,
) -> UsageEventRecord:
    input_tokens = (
        CLASSIFICATION_PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(query)
        + _estimate_history_tokens(history)
    )
    output_tokens = estimate_tokens(
        json.dumps(
            {
                "intent": classification.intent,
                "confidence": classification.confidence,
                "needs_rag": classification.needs_rag,
                "route": classification.route,
                "explain": classification.explain,
                "rewritten_query": classification.rewritten_query,
                "language": classification.language,
            },
            ensure_ascii=False,
        )
    )
    return _gpt_4o_mini_event("classification", input_tokens, output_tokens)


def estimate_guardrail_event(
    query: str,
    history: Optional[List[ConversationMessage]],
    guardrail: GuardrailResult,
) -> UsageEventRecord:
    input_tokens = (
        CLASSIFICATION_PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(query)
        + _estimate_history_tokens(history)
    )
    output_tokens = estimate_tokens(
        json.dumps(
            {
                "allowed": guardrail.allowed,
                "violation": guardrail.violation,
                "needs_context": guardrail.needs_context,
                "reason": guardrail.reason,
                "language": guardrail.language,
            },
            ensure_ascii=False,
        )
    )
    return _gpt_4o_mini_event("guardrail", input_tokens, output_tokens)


def estimate_greeting_completion_event(
    query: str,
    history: Optional[List[ConversationMessage]],
    answer: str,
) -> UsageEventRecord:
    input_tokens = (
        GREETING_PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(query)
        + _estimate_history_tokens(history)
    )
    output_tokens = estimate_tokens(answer)
    return _gpt_4o_mini_event("chat_completion", input_tokens, output_tokens)


def estimate_factual_completion_event(
    query: str,
    history: Optional[List[ConversationMessage]],
    answer: str,
) -> UsageEventRecord:
    input_tokens = (
        FACTUAL_PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(query)
        + _estimate_history_tokens(history)
    )
    output_tokens = estimate_tokens(answer)
    return _gpt_4o_mini_event("chat_completion", input_tokens, output_tokens)


def estimate_prompt_completion_event(prompt: str, answer: str) -> UsageEventRecord:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(answer)
    return _gpt_4o_mini_event("chat_completion", input_tokens, output_tokens)


def estimate_embedding_event(text: str) -> UsageEventRecord:
    input_tokens = estimate_tokens(text)
    cost = _rounded_cost(input_tokens * TEXT_EMBEDDING_3_SMALL_INPUT_PER_1M / 1_000_000)
    return UsageEventRecord(
        event_type="embedding",
        input_tokens=input_tokens,
        output_tokens=0,
        total_tokens=input_tokens,
        estimated_cost=cost,
    )


def usage_event_from_model_usage(event_type: str, usage: ModelUsage) -> UsageEventRecord:
    input_tokens = max(0, int(usage.input_tokens or 0))
    output_tokens = max(0, int(usage.output_tokens or 0))
    total_tokens = max(0, int(usage.total_tokens or 0))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    cost = _rounded_cost(_cost_for_model(usage.model, input_tokens, output_tokens))
    return UsageEventRecord(
        event_type=event_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=cost,
    )


def _estimate_history_tokens(history: Optional[List[ConversationMessage]]) -> int:
    if not history:
        return 0
    return estimate_tokens(
        "\n".join(build_history_lines(history, "Recent conversation context (oldest to newest):"))
    )


def _gpt_4o_mini_event(event_type: str, input_tokens: int, output_tokens: int) -> UsageEventRecord:
    cost = _rounded_cost(_cost_for_model("gpt-4o-mini", input_tokens, output_tokens))
    return UsageEventRecord(
        event_type=event_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost=cost,
    )


def _cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    normalized = (model or "").strip().lower()
    if normalized == "gpt-4o-mini":
        return (
            (input_tokens * GPT_4O_MINI_INPUT_PER_1M / 1_000_000)
            + (output_tokens * GPT_4O_MINI_OUTPUT_PER_1M / 1_000_000)
        )
    if normalized == "text-embedding-3-small":
        return input_tokens * TEXT_EMBEDDING_3_SMALL_INPUT_PER_1M / 1_000_000
    return 0.0


def _rounded_cost(cost: float) -> float:
    rounded = round(cost, 6)
    if cost > 0 and rounded == 0:
        return 0.000001
    return rounded

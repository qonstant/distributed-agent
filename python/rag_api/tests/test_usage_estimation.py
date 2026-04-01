from __future__ import annotations

import unittest

from rag_service.application.usage_estimation import (
    estimate_classification_event,
    estimate_embedding_event,
    estimate_factual_completion_event,
    estimate_prompt_completion_event,
    rag_query_event,
    usage_event_from_model_usage,
)
from rag_service.domain.models import Classification, ConversationMessage, ModelUsage


class UsageEstimationTests(unittest.TestCase):
    def test_classification_event_has_non_zero_tokens_and_cost(self) -> None:
        history = [ConversationMessage(role="user", text="Hello there", ts=1)]
        classification = Classification(intent="GREETING", explain="short greeting", language="en")

        event = estimate_classification_event("hello", history, classification)

        self.assertEqual(event.event_type, "classification")
        self.assertGreater(event.input_tokens, 0)
        self.assertGreater(event.output_tokens, 0)
        self.assertGreater(event.total_tokens, 0)
        self.assertGreater(event.estimated_cost, 0)

    def test_embedding_event_rounds_small_positive_cost_up_to_visible_precision(self) -> None:
        event = estimate_embedding_event("short query")

        self.assertEqual(event.event_type, "embedding")
        self.assertGreater(event.input_tokens, 0)
        self.assertEqual(event.output_tokens, 0)
        self.assertGreater(event.estimated_cost, 0)

    def test_prompt_completion_event_has_non_zero_tokens_and_cost(self) -> None:
        event = estimate_prompt_completion_event("Prompt text here", "Short answer")

        self.assertEqual(event.event_type, "chat_completion")
        self.assertGreater(event.input_tokens, 0)
        self.assertGreater(event.output_tokens, 0)
        self.assertGreater(event.estimated_cost, 0)

    def test_factual_completion_event_uses_query_and_history(self) -> None:
        history = [ConversationMessage(role="assistant", text="Previous context", ts=1)]

        event = estimate_factual_completion_event("What now?", history, "Answer now.")

        self.assertEqual(event.event_type, "chat_completion")
        self.assertGreater(event.input_tokens, 0)
        self.assertGreater(event.output_tokens, 0)
        self.assertGreater(event.estimated_cost, 0)

    def test_rag_query_event_remains_non_billable(self) -> None:
        event = rag_query_event()

        self.assertEqual(event.event_type, "rag_query")
        self.assertEqual(event.input_tokens, 0)
        self.assertEqual(event.output_tokens, 0)
        self.assertEqual(event.total_tokens, 0)
        self.assertEqual(event.estimated_cost, 0.0)

    def test_usage_event_from_model_usage_uses_actual_token_counts(self) -> None:
        usage = ModelUsage(model="gpt-4o-mini", input_tokens=100, output_tokens=25, total_tokens=125)

        event = usage_event_from_model_usage("chat_completion", usage)

        self.assertEqual(event.event_type, "chat_completion")
        self.assertEqual(event.input_tokens, 100)
        self.assertEqual(event.output_tokens, 25)
        self.assertEqual(event.total_tokens, 125)
        self.assertGreater(event.estimated_cost, 0)


if __name__ == "__main__":
    unittest.main()

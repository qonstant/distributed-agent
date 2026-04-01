from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from rag_service.domain.models import (
    Classification,
    ConversationMessage,
    ModelUsage,
    normalize_intent,
    normalize_language,
)
from rag_service.infrastructure.config import Settings
from rag_service.infrastructure.prompts import build_history_lines


class OpenAIGateway:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_text(self, text: str) -> tuple[np.ndarray, Optional[ModelUsage]]:
        response = self._client.embeddings.create(model=self._settings.embed_model, input=[text])
        item = response.data[0]
        embedding = getattr(item, "embedding", None) or (
            item.get("embedding") if isinstance(item, dict) else None
        )
        if embedding is None:
            raise RuntimeError("Failed to parse embedding response")
        return np.array(embedding, dtype=np.float32), self._extract_usage(
            response,
            self._settings.embed_model,
        )

    def classify_query(
        self,
        query: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[Classification, Optional[ModelUsage]]:
        history_block = self._history_block(history)
        prompt = (
            "You are a compact intent classifier and language detector. Given the user's latest input and optional recent conversation context below, "
            "return a JSON object with EXACTLY three keys:\n"
            " - \"intent\": one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"GUIDANCE\",\"DOCUMENT_REQUEST\",\"OTHER\"]\n"
            " - \"explain\": one short sentence explaining why\n"
            " - \"language\": exactly one of [\"kk\",\"ru\",\"en\",\"other\"]\n\n"
            "Definitions/examples:\n"
            " - GREETING: short hello/goodbye messages (no docs needed)\n"
            " - CHIT_CHAT: small talk / thanks / compliment (no docs)\n"
            " - FACTUAL_QUESTION: generic factual question where no document retrieval is needed (e.g., \"What is AI?\")\n"
            " - GUIDANCE: user asks for step-by-step guidance, procedures or how-to that should be answered using documents if available, but may be synthesized from top-K excerpts (do NOT invent facts)\n"
            " - DOCUMENT_REQUEST: user explicitly requests a document, template, sample file, or wants 'send X' / 'пример файла' (must prefer returning a file path from available docs)\n"
            " - OTHER: none of the above\n\n"
            "Respond ONLY with valid JSON (no extra text). Example:\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to apply for residency\",\"language\":\"ru\"}\n"
            "{\"intent\":\"GREETING\",\"explain\":\"short greeting in Kazakh\",\"language\":\"kk\"}\n"
            "{\"intent\":\"OTHER\",\"explain\":\"language outside supported set\",\"language\":\"other\"}\n\n"
            f"{history_block}"
            f"Latest user input: {json.dumps(query)}\n"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.class_model,
                input=prompt,
                max_output_tokens=120,
                temperature=0.0,
            )
            raw_text = self._resp_to_text(response) or ""
            parsed = self._extract_json(raw_text) or {
                "intent": "OTHER",
                "explain": raw_text,
                "language": "",
            }
            return (
                Classification(
                    intent=normalize_intent(parsed.get("intent", "")),
                    explain=str(parsed.get("explain") or ""),
                    language=normalize_language(str(parsed.get("language") or "")),
                    model=self._settings.class_model,
                ),
                self._extract_usage(response, self._settings.class_model),
            )
        except Exception as exc:
            print("[classify] classifier error:", exc)
            return (
                Classification(
                    intent="OTHER",
                    explain=f"classifier error: {exc}",
                    language="other",
                    model=self._settings.class_model,
                ),
                None,
            )

    def generate_greeting_reply(
        self,
        user_text: str,
        language_hint: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        language_name = self._language_name(language_hint)
        lang_instruction = (
            f"in {language_name}"
            if language_name
            else "in the same language as the user"
        )
        history_block = self._history_block(history)
        prompt = (
            f"{history_block}"
            f"The user wrote: {json.dumps(user_text)}\n\n"
            f"Produce a single short friendly reply ({lang_instruction}). Keep it to one short sentence (<=20 words). "
            "Do NOT include file paths or any extra commentary. Return only the reply text."
        )
        try:
            response = self._client.responses.create(
                model=self._settings.llm_model,
                input=prompt,
                max_output_tokens=50,
                temperature=0.0,
            )
            text = self._resp_to_text(response).strip()
            if text.startswith("```"):
                text = text.strip("` \n")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped, self._extract_usage(response, self._settings.llm_model)
            return text, self._extract_usage(response, self._settings.llm_model)
        except Exception as exc:
            print("[greeting] generation failed:", exc)
            return "Hi — how can I help you today?", None

    def answer_factual(
        self,
        query: str,
        language_hint: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        language_name = self._language_name(language_hint)
        lang_instruction = (
            f"Answer in {language_name}."
            if language_name
            else "Answer in the same language as the user."
        )
        history_block = self._history_block(history)
        prompt = (
            f"{history_block}"
            f"You are a concise helpful assistant. {lang_instruction} "
            "Answer the user question briefly (1-2 short paragraphs). Use recent conversation context when it is relevant. Do NOT include any file paths or suggest internal document locations.\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.llm_model,
                input=prompt,
                max_output_tokens=400,
                temperature=0.0,
            )
            return self._resp_to_text(response).strip(), self._extract_usage(
                response,
                self._settings.llm_model,
            )
        except Exception as exc:
            print("[factual] LLM error:", exc)
            return f"(LLM error: {exc})", None

    def rewrite_query_with_history(
        self,
        query: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        if not history:
            return query, None

        history_block = self._history_block(history)
        prompt = (
            "Rewrite the latest user message into a standalone search query for document retrieval. "
            "Use the recent conversation only to resolve references like 'it', 'that form', 'the previous one'. "
            "If the latest user message is already standalone, return it unchanged. Return plain text only.\n\n"
            f"{history_block}"
            f"Latest user message: {json.dumps(query)}\n"
            "Standalone query:"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.class_model,
                input=prompt,
                max_output_tokens=120,
                temperature=0.0,
            )
            text = self._resp_to_text(response).strip().strip("` \n")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped, self._extract_usage(response, self._settings.class_model)
        except Exception as exc:
            print("[rewrite] query rewrite failed:", exc)
        return query, None

    def generate_json_response(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> tuple[Optional[Dict[str, Any]], Optional[ModelUsage]]:
        try:
            response = self._client.responses.create(
                model=self._settings.llm_model,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=0.0,
            )
            text = self._resp_to_text(response)
            return self._extract_json(text), self._extract_usage(
                response,
                self._settings.llm_model,
            )
        except Exception as exc:
            print("[synth] LLM synth failed:", exc)
            return None, None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                stripped = stripped[start : end + 1]
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(stripped[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
            return None

    @staticmethod
    def _resp_to_text(resp: Any) -> str:
        if isinstance(resp, str):
            return resp
        try:
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
        except Exception:
            pass
        output = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
        if isinstance(output, list):
            parts = []
            for node in output:
                if isinstance(node, dict):
                    content = node.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "output_text":
                                parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                parts.append(item)
                    elif isinstance(content, str):
                        parts.append(content)
                elif isinstance(node, str):
                    parts.append(node)
            return "".join(parts).strip()
        if isinstance(output, str):
            return output.strip()
        try:
            return json.dumps(resp, default=str)
        except Exception:
            return str(resp)

    @staticmethod
    def _history_block(history: Optional[List[ConversationMessage]]) -> str:
        if not history:
            return ""

        lines = build_history_lines(history, "Recent conversation context (oldest to newest):")
        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    @staticmethod
    def _language_name(language_hint: str) -> str:
        normalized = (language_hint or "").strip().lower()
        mapping = {
            "en": "English",
            "ru": "Russian",
            "kk": "Kazakh",
            "other": "",
        }
        return mapping.get(normalized, (language_hint or "").strip())

    @staticmethod
    def _extract_usage(response: Any, model: str) -> Optional[ModelUsage]:
        usage = getattr(response, "usage", None) or (
            response.get("usage") if isinstance(response, dict) else None
        )
        if usage is None:
            return None

        def get_value(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        input_tokens = get_value(usage, "input_tokens")
        if input_tokens is None:
            input_tokens = get_value(usage, "prompt_tokens")
        output_tokens = get_value(usage, "output_tokens") or 0
        total_tokens = get_value(usage, "total_tokens")
        if total_tokens is None:
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

        return ModelUsage(
            model=model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=int(total_tokens or 0),
        )

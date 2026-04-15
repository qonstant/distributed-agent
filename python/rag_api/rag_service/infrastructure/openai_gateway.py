from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from rag_service.domain.models import (
    Classification,
    ConversationMessage,
    ModelUsage,
    RetrievalClarity,
    normalize_attachment_action,
    normalize_intent,
    normalize_language,
    normalize_profile_action,
)
from rag_service.infrastructure.config import Settings
from rag_service.infrastructure.prompts import build_history_lines, build_personalization_lines


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
            "return a JSON object with EXACTLY five keys:\n"
            " - \"intent\": one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"GUIDANCE\",\"DOCUMENT_REQUEST\",\"OTHER\"]\n"
            " - \"explain\": one short sentence explaining why\n"
            " - \"language\": exactly one of [\"kk\",\"ru\",\"en\",\"other\"]\n\n"
            " - \"profile_action\": either \"set_preferred_name\" or \"\"\n"
            " - \"preferred_name\": extracted preferred name if the user is telling you what to call them, else \"\"\n\n"
            "Definitions/examples:\n"
            " - GREETING: short hello/goodbye messages (no docs needed)\n"
            " - CHIT_CHAT: small talk / thanks / compliment (no docs)\n"
            " - FACTUAL_QUESTION: generic factual question where no document retrieval is needed (e.g., \"What is AI?\")\n"
            " - GUIDANCE: user asks for step-by-step guidance, procedures or how-to that should be answered using documents if available, but may be synthesized from top-K excerpts (do NOT invent facts)\n"
            " - DOCUMENT_REQUEST: user explicitly requests a document, template, sample file, or wants 'send X' / 'пример файла' (must prefer returning a file path from available docs)\n"
            " - OTHER: none of the above\n\n"
            "Language rules:\n"
            " - Do NOT choose Kazakh just because the text is written in Cyrillic.\n"
            " - Prefer \"ru\" for standard Russian wording such as \"Как меня зовут?\", \"Как зовут меня?\", \"Вот меня зовут ...\", \"Зови меня ...\", \"Привет\", \"Спасибо\".\n"
            " - Choose \"kk\" only when there are clear Kazakh signals, for example distinct Kazakh letters (ә, ғ, қ, ң, ө, ұ, ү, һ, і) or clearly Kazakh wording such as \"қалай\", \"мені\", \"аты\", \"сәлем\".\n"
            " - If the text is short, Cyrillic, and ambiguous, prefer \"ru\" over \"kk\" unless there is a strong Kazakh marker.\n\n"
            "Set profile_action to \"set_preferred_name\" only when the user is explicitly telling you what name to use for them, for example "
            "\"call me Alex\", \"my name is Rocco\", \"зови меня Роман\", \"меня зовут Азамат\", or rename phrases like \"зовут меня теперь Heisenberg\". "
            "When you do that, put only the clean extracted name into preferred_name.\n\n"
            "Respond ONLY with valid JSON (no extra text). Example:\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to apply for residency\",\"language\":\"ru\",\"profile_action\":\"\",\"preferred_name\":\"\"}\n"
            "{\"intent\":\"GREETING\",\"explain\":\"short greeting in Kazakh\",\"language\":\"kk\",\"profile_action\":\"\",\"preferred_name\":\"\"}\n"
            "{\"intent\":\"CHIT_CHAT\",\"explain\":\"user sets a preferred name\",\"language\":\"ru\",\"profile_action\":\"set_preferred_name\",\"preferred_name\":\"Роман\"}\n"
            "{\"intent\":\"FACTUAL_QUESTION\",\"explain\":\"user asks what their name is in Russian\",\"language\":\"ru\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"Как меня зовут?\"\n"
            "{\"intent\":\"OTHER\",\"explain\":\"language outside supported set\",\"language\":\"other\",\"profile_action\":\"\",\"preferred_name\":\"\"}\n\n"
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
                "profile_action": "",
                "preferred_name": "",
            }
            return (
                Classification(
                    intent=normalize_intent(parsed.get("intent", "")),
                    explain=str(parsed.get("explain") or ""),
                    language=normalize_language(str(parsed.get("language") or "")),
                    model=self._settings.class_model,
                    profile_action=normalize_profile_action(str(parsed.get("profile_action") or "")),
                    preferred_name=str(parsed.get("preferred_name") or "").strip(),
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
                    profile_action="",
                    preferred_name="",
                ),
                None,
            )

    def generate_greeting_reply(
        self,
        user_text: str,
        language_hint: str,
        preferred_name: str = "",
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        language_name = self._language_name(language_hint)
        lang_instruction = (
            f"in {language_name}"
            if language_name
            else "in the same language as the user"
        )
        history_block = self._history_block(history)
        personalization_block = self._personalization_block(preferred_name)
        prompt = (
            f"{history_block}"
            f"{personalization_block}"
            f"The user wrote: {json.dumps(user_text)}\n\n"
            f"Produce a single short friendly reply ({lang_instruction}). Keep it to one short sentence (<=20 words). "
            "If a preferred user name is available, start the reply with it when natural. "
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
        preferred_name: str = "",
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        language_name = self._language_name(language_hint)
        lang_instruction = (
            f"Answer in {language_name}."
            if language_name
            else "Answer in the same language as the user."
        )
        history_block = self._history_block(history)
        personalization_block = self._personalization_block(preferred_name)
        prompt = (
            f"{history_block}"
            f"{personalization_block}"
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

    def clarify_or_rewrite_query(
        self,
        query: str,
        language_hint: str,
        intent: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[RetrievalClarity, Optional[ModelUsage]]:
        history_block = self._history_block(history)
        language_name = self._language_name(language_hint) or "the user's language"
        prompt = (
            "You are a clarification gate for a document-grounded RAG assistant.\n"
            "Your job is to decide whether the latest user message is related to document retrieval, "
            "whether it is specific enough to run retrieval now, or whether the assistant should ask exactly one clarifying question first.\n\n"
            "Current corpus scope:\n"
            " - Country defaults to Italy. Do NOT ask for country just because it is missing.\n"
            " - Supported topics include: Italian student visa, CV, DSU scholarship, motivation letter, and recommendation letter.\n\n"
            "Treat short/lazy queries as clear when the topic is identifiable. Examples of clear queries: "
            "\"visa docs\", \"cv help\", \"dsu money\", \"motivation letter structure\", \"recommendation letter who\".\n"
            "Ask a clarification only when the missing detail would change which document/topic should be searched. "
            "Examples of unclear queries: \"what documents do I need?\", \"how to apply?\", \"send file\", \"что нужно?\", \"қалай тапсырам?\".\n"
            "If the recent conversation contains an assistant clarification question, combine the latest user reply with that context. "
            "If the combined meaning is clear, produce a complete standalone search query.\n\n"
            "If the classifier intent is OTHER, use the recent conversation to decide whether the latest message is a continuation of a document clarification. "
            "If it is not a document request/guidance question and not a clarification follow-up, set is_retrieval_related to false and leave standalone_query and clarifying_question empty.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            ' - "is_retrieval_related": boolean\n'
            ' - "is_clear": boolean\n'
            ' - "standalone_query": string; if is_clear is true, this must be a complete retrieval query\n'
            ' - "clarifying_question": string; if is_clear is false, ask one concise question in '
            f"{language_name}\n"
            ' - "reason": one short sentence\n\n'
            "Do not answer the user. Do not mention internal retrieval, embeddings, metadata, or files unless the user asked for a file.\n\n"
            "Examples:\n"
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"What documents are needed for an Italian student visa?","clarifying_question":"","reason":"The visa document topic is clear."}\n'
            '{"is_retrieval_related":true,"is_clear":false,"standalone_query":"","clarifying_question":"Which topic do you mean: student visa, CV, DSU scholarship, motivation letter, or recommendation letter?","reason":"The user asks for documents but not the process."}\n'
            '{"is_retrieval_related":false,"is_clear":false,"standalone_query":"","clarifying_question":"","reason":"The user is not asking a document-grounded question."}\n\n'
            f"{history_block}"
            f"Classifier intent: {json.dumps(intent)}\n"
            f"Latest user input: {json.dumps(query)}\n"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.class_model,
                input=prompt,
                max_output_tokens=180,
                temperature=0.0,
            )
            raw_text = self._resp_to_text(response) or ""
            parsed = self._extract_json(raw_text) or {}
            clarity = RetrievalClarity(
                is_clear=self._json_bool(parsed.get("is_clear"), default=True),
                standalone_query=str(parsed.get("standalone_query") or "").strip(),
                clarifying_question=str(parsed.get("clarifying_question") or "").strip(),
                reason=str(parsed.get("reason") or "").strip(),
                is_retrieval_related=self._json_bool(parsed.get("is_retrieval_related"), default=True),
            )
            if not clarity.is_retrieval_related:
                return clarity, self._extract_usage(response, self._settings.class_model)
            if clarity.is_clear and not clarity.standalone_query:
                clarity = RetrievalClarity(
                    is_clear=True,
                    standalone_query=query,
                    clarifying_question="",
                    reason=clarity.reason or "Fallback to latest query.",
                    is_retrieval_related=True,
                )
            if not clarity.is_clear and not clarity.clarifying_question:
                clarity = RetrievalClarity(
                    is_clear=False,
                    standalone_query="",
                    clarifying_question=self._default_clarifying_question(language_hint),
                    reason=clarity.reason or "The request is ambiguous.",
                    is_retrieval_related=True,
                )
            return clarity, self._extract_usage(response, self._settings.class_model)
        except Exception as exc:
            print("[clarify] query clarity check failed:", exc)
            return (
                RetrievalClarity(
                    is_clear=True,
                    standalone_query=query,
                    reason=f"clarity check failed: {exc}",
                ),
                None,
            )

    def classify_attachment_follow_up(
        self,
        query: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[str, Optional[ModelUsage]]:
        if not history:
            return "", None

        history_block = self._history_block(history)
        prompt = (
            "You detect whether the latest user message is explicitly asking to resend the most recently sent assistant attachment from the recent conversation context.\n"
            "Return a JSON object with EXACTLY two keys:\n"
            ' - "attachment_action": either "resend_last_attachment" or ""\n'
            ' - "explain": one short sentence explaining why\n\n'
            'Choose "resend_last_attachment" only when the user is clearly asking to send the already-mentioned file again, even in short follow-ups like "again", '
            '"one more time", "еще раз", or similar context-dependent requests. '
            'If the user is asking what the file is about, asking a new question, or you are unsure, return "".\n\n'
            "Respond ONLY with valid JSON. Example:\n"
            '{"attachment_action":"resend_last_attachment","explain":"user asks to send the previously sent file again"}\n'
            '{"attachment_action":"","explain":"user asks about the file rather than requesting a resend"}\n\n'
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
                "attachment_action": "",
                "explain": raw_text,
            }
            return (
                normalize_attachment_action(str(parsed.get("attachment_action") or "")),
                self._extract_usage(response, self._settings.class_model),
            )
        except Exception as exc:
            print("[attach-classify] attachment follow-up classifier error:", exc)
            return "", None

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
    def _json_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
        return default

    @staticmethod
    def _default_clarifying_question(language_hint: str) -> str:
        normalized = normalize_language(language_hint)
        if normalized == "kk":
            return "Қай тақырып бойынша сұрап тұрсыз: студенттік виза, CV, DSU шәкіртақысы, мотивациялық хат немесе ұсыныс хат?"
        if normalized == "ru":
            return "По какой теме вы спрашиваете: студенческая виза, CV, стипендия DSU, мотивационное письмо или рекомендательное письмо?"
        return "Which topic do you mean: student visa, CV, DSU scholarship, motivation letter, or recommendation letter?"

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
    def _personalization_block(preferred_name: str) -> str:
        lines = build_personalization_lines(preferred_name)
        if not lines:
            return ""
        return "\n".join(lines)

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

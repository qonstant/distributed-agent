from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from rag_service.domain.models import (
    Classification,
    ConversationMessage,
    GuardrailResult,
    ModelUsage,
    RetrievalClarity,
    RetrievalSufficiency,
    RetrievedHit,
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

    def guard_query(
        self,
        query: str,
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[GuardrailResult, Optional[ModelUsage]]:
        history_block = self._history_block(history)
        context_block = history_block or "Recent conversation context: not provided in this pass.\n\n"
        prompt = (
            "You are a guardrail for an education/study-abroad assistant. "
            "Decide whether the latest user input is safe and related enough to pass to intent classification.\n\n"
            "This guardrail may be called in two passes. In the first pass, recent conversation context may be omitted to keep the prompt small. "
            "If the latest input is enough to decide safety and scope by itself, make the decision without context. "
            "In the first pass only, if the latest input is too short, elliptical, or only an answer to the assistant's previous question, set needs_context=true so the caller can retry with Redis conversation history. "
            "In the second pass, recent conversation context is already provided, so you MUST make a final safety/scope decision and MUST set needs_context=false.\n\n"
            "Return a JSON object with EXACTLY five keys:\n"
            ' - "allowed": boolean\n'
            ' - "needs_context": boolean\n'
            ' - "violation": one of ["","out_of_scope","unsafe","unsupported_language"]\n'
            ' - "reason": one short sentence explaining the decision\n'
            ' - "language": exactly one of ["kk","ru","en","other"]\n\n'
            "Decision rules:\n"
            " - If needs_context is true, set allowed=false and violation=\"\".\n"
            " - If recent conversation context is provided, use the whole Redis-loaded conversation before judging the latest input and set needs_context=false, always.\n"
            " - When recent conversation context is provided and the latest input is one short word or phrase, judge it by the immediate previous meaningful topic. Do not treat short replies as unsafe merely because the conversation contains visa, permit, or document words.\n"
            " - Do not request context for standalone messages like \"How to get DSU\", \"How to apply to uni\", \"visa docs\", \"How can I apply for residence permit in Italy?\", \"Как получить ВНЖ\", or clearly unsafe/out-of-scope requests.\n"
            " - Do not request context for complete unrelated questions. Standalone unrelated questions are out_of_scope, not needs_context. Example: \"What is the capital of France?\" is out_of_scope.\n\n"
            "Allowed scope:\n"
            " - Education/study-abroad support, including university admission abroad, university application guidance, admission documents, student visas for study/enrollment, student residence permits/permesso di soggiorno for study, study-related travel rights or constraints, scholarships, DSU scholarship/student financial aid in Italy, CV, motivation letter, and recommendation letter.\n"
            " - Greetings, thanks, small talk, user profile/name updates, and questions about recent conversation are allowed because they help the assistant conversation.\n"
            " - Short follow-ups are allowed when recent conversation makes them refer to an in-scope topic. Examples: \"yes\", \"how\", \"how to apply\", \"steps\", \"documents\", \"да\", \"как\", \"қалай\".\n"
            " - Short confirmations or refusals are allowed when recent conversation makes them meaningful. Examples: \"yes\", \"no\", \"yeah\", \"nope\", \"да\", \"нет\", \"иә\", \"жоқ\", \"Да да да\".\n"
            " - If the assistant just asked an in-scope clarification question and the latest user input confirms or rejects it, allow the message. Do not block it just because the latest input alone has no education keyword.\n"
            " - If the latest user input is only a short acknowledgement, confirmation, refusal, or correction, prefer allowed=true unless the recent conversation is clearly unsafe or out of scope.\n"
            " - If the user says \"uni\", \"university\", \"admission\", or \"apply\" in a study-abroad context, allow it even if the country is missing.\n"
            " - If the user says \"visa\" without saying tourist, work, business, family, travel, or another non-study context, assume student visa by default. Short phrases like \"visa docs\" or \"student visa documents Italy\" are standalone and allowed.\n"
            " - If the user mentions DSU without another explicit meaning, allow it as the Italian DSU student scholarship/financial aid.\n"
            " - If the user asks about an Italian residence permit, permesso di soggiorno, student residence permit, ВНЖ, внж, VNJ, or residence permit without saying work, tourist, family, permanent residence, asylum, or another non-study context, allow it as the student residence permit.\n\n"
            "Block as out_of_scope:\n"
            " - Tourist visas, travel visas unrelated to study, work visas, business visas, family visas, general immigration, permanent relocation, permanent residence, moving to Italy permanently, tourism, flights, hotels, travel itineraries, and unrelated general knowledge.\n"
            " - If recent conversation is clearly out of scope and the latest input is only a short continuation or confirmation, block as out_of_scope.\n\n"
            "Block as unsafe:\n"
            " - Requests to forge, fake, falsify, buy, sell, or misuse documents, certificates, bank statements, recommendation letters, identity documents, visas, permits, transcripts, or exam results.\n"
            " - Requests to lie in applications, evade immigration law, bypass official systems, hack accounts, steal data, or commit fraud.\n"
            " - If recent conversation explicitly contains a user request to fake, forge, falsify, bypass, buy, sell, lie, hack, or evade official systems, and the latest input is only a short continuation like \"how\", \"steps\", \"yes\", \"да\", or \"как\", block as unsafe.\n"
            " - Do NOT mark a short follow-up as unsafe when the previous topic is a normal legal visa/photo/document/DSU/residence-permit discussion.\n"
            " - If the user asks how to avoid these problems legally or how to correct a mistake honestly, allow it.\n\n"
            "Language rules:\n"
            " - Do NOT choose Kazakh just because the text is written in Cyrillic.\n"
            " - Prefer \"ru\" for standard Russian wording unless there are clear Kazakh signals such as ә, ғ, қ, ң, ө, ұ, ү, һ, і or words like \"қалай\".\n\n"
            " - Choose \"kk\" for Kazakh wording such as \"Италияда оқып жүріп саяхаттай аламын ба?\", \"DSU стипендиясына қалай өтінемін?\", or phrases with оқып, жүріп, аламын, өтінемін, стипендиясына.\n\n"
            "Respond ONLY with valid JSON. Examples:\n"
            '{"allowed":true,"needs_context":false,"violation":"","reason":"University admission guidance is in scope.","language":"en"} for input like "How to apply to uni"\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"DSU defaults to the Italian student scholarship topic.","language":"en"} for input like "How to get dsu"\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"Visa documents default to the student visa topic.","language":"en"} for input like "visa docs"\n'
            '{"allowed":false,"needs_context":true,"violation":"","reason":"The latest message is only a confirmation and needs recent conversation context.","language":"ru"} for input like "Да да да" when no context is provided\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user confirms a previous in-scope visa-photo file offer.","language":"en"} for input like "yes" after the assistant asks whether to send visa photo information\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user declines a previous in-scope visa-photo file offer.","language":"en"} for input like "no" after the assistant asks whether to send visa photo information\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user asks how to continue with the previous in-scope DSU topic.","language":"en"} for input like "How" after a DSU explanation\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user confirms the previous in-scope student residence permit clarification.","language":"ru"} for input like "Да да да" after the assistant asks whether they mean a student residence permit\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user rejects a previous in-scope student residence permit clarification, but the conversation is still safe.","language":"ru"} for input like "Нет" after the assistant asks whether they mean a student residence permit\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user asks how to continue with the previous in-scope DSU scholarship topic.","language":"ru"} for input like "как" after a DSU scholarship explanation\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"Residence permit defaults to student residence permit in Italy.","language":"en"} for input like "How can I apply for residence permit in Italy?"\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"Study-related travel while enrolled in Italy is in scope.","language":"kk"} for input like "Италияда оқып жүріп саяхаттай аламын ба?"\n'
            '{"allowed":true,"needs_context":false,"violation":"","reason":"ВНЖ defaults to the student residence permit topic.","language":"ru"} for input like "Как получить внж"\n'
            '{"allowed":false,"needs_context":false,"violation":"out_of_scope","reason":"General geography is outside the education-abroad scope.","language":"en"} for input like "What is the capital of France?"\n'
            '{"allowed":false,"needs_context":false,"violation":"out_of_scope","reason":"Permanent migration is outside the study-abroad scope.","language":"en"} for input like "How do I move to Italy permanently?"\n'
            '{"allowed":false,"needs_context":false,"violation":"out_of_scope","reason":"Tourist visa is outside the study-abroad scope.","language":"en"} for input like "How do I get an Italian tourist visa?"\n'
            '{"allowed":false,"needs_context":false,"violation":"unsafe","reason":"The user asks for help falsifying application documents.","language":"en"} for input like "How can I fake a bank statement for visa?"\n\n'
            '{"allowed":false,"needs_context":false,"violation":"unsafe","reason":"The latest message continues a previous explicit document-fraud request.","language":"en"} for input like "how" after a previous user request to fake a bank statement\n\n'
            f"{context_block}"
            f"Latest user input: {json.dumps(query, ensure_ascii=False)}\n"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.class_model,
                input=prompt,
                max_output_tokens=120,
                temperature=0.0,
            )
            raw_text = self._resp_to_text(response) or ""
            parsed = self._extract_json(raw_text) or {}
            usage = self._extract_usage(response, self._settings.class_model)

            if history_block and self._json_bool(parsed.get("needs_context"), default=False):
                repair_prompt = (
                    f"{prompt}\n"
                    "The previous guardrail JSON returned needs_context=true, but recent conversation "
                    "context was already provided. That is invalid in the second pass. "
                    "Re-evaluate the latest user input using the provided conversation and return the final JSON now. "
                    "Set needs_context=false. If the conversation makes the latest input continue an in-scope topic, set allowed=true. "
                    "If it is unsafe, set violation=\"unsafe\". If it is unrelated, set violation=\"out_of_scope\".\n\n"
                    f"Previous guardrail JSON: {json.dumps(parsed, ensure_ascii=False)}\n"
                    "Corrected JSON only:\n"
                )
                try:
                    repair_response = self._client.responses.create(
                        model=self._settings.class_model,
                        input=repair_prompt,
                        max_output_tokens=120,
                        temperature=0.0,
                    )
                    repaired = self._extract_json(self._resp_to_text(repair_response) or "") or {}
                    if repaired:
                        parsed = repaired
                    usage = self._merge_usage(
                        usage,
                        self._extract_usage(repair_response, self._settings.class_model),
                    )
                except Exception as repair_exc:
                    print("[guardrail] contextual repair error:", repair_exc)

            return (
                self._guardrail_result_from_payload(parsed),
                usage,
            )
        except Exception as exc:
            print("[guardrail] guardrail error:", exc)
            return (
                GuardrailResult(
                    allowed=False,
                    reason=f"guardrail error: {exc}",
                    language="other",
                    violation="unsafe",
                    model=self._settings.class_model,
                ),
                None,
            )

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
            " - \"intent\": one of [\"GREETING\",\"CHIT_CHAT\",\"FACTUAL_QUESTION\",\"GUIDANCE\",\"DOCUMENT_REQUEST\"]\n"
            " - \"explain\": one short sentence explaining why\n"
            " - \"language\": exactly one of [\"kk\",\"ru\",\"en\",\"other\"]\n\n"
            " - \"profile_action\": either \"set_preferred_name\" or \"\"\n"
            " - \"preferred_name\": extracted preferred name if the user is telling you what to call them, else \"\"\n\n"
            "The guardrail has already checked safety and assistant scope before this classifier runs. "
            "Do not reject the request for being out of scope here; choose the best conversational or retrieval intent.\n\n"
            "Conversation/context rules:\n"
            " - Use recent conversation to classify short follow-ups. If the previous in-scope topic was DSU, scholarship, visa, residence permit, CV, motivation letter, recommendation letter, or university admission, short replies like \"yes\", \"how\", \"how to apply\", \"steps\", \"documents\", \"да\", \"как\", or \"қалай\" are continuations of that topic.\n"
            " - Prefer making the helpful education-abroad assumption over asking for clarification when the topic is identifiable from the latest input or history.\n\n"
            "Definitions/examples:\n"
            " - GREETING: short hello/goodbye messages (no docs needed)\n"
            " - CHIT_CHAT: small talk / thanks / compliment (no docs)\n"
            " - FACTUAL_QUESTION: direct factual question. If it is about education-abroad documents or procedures, including student residence permits or study-related travel permissions, it will be answered using retrieval. If it is only about recent conversation or saved user info (e.g., \"What is my name?\"), no document retrieval is needed. Do NOT use for general world knowledge or out-of-scope travel/visa questions.\n"
            " - GUIDANCE: user asks for in-scope education-abroad step-by-step guidance, procedures or how-to that should be answered using documents if available, but may be synthesized from top-K excerpts (do NOT invent facts). Also use GUIDANCE when the user asks to repeat/continue a previous in-scope answer in another supported language.\n"
            " - DOCUMENT_REQUEST: user explicitly requests an in-scope education-abroad document, template, sample file, or wants 'send X' / 'пример файла' (must prefer returning a file path from available docs)\n\n"
            "Language rules:\n"
            " - Do NOT choose Kazakh just because the text is written in Cyrillic.\n"
            " - Prefer \"ru\" for standard Russian wording such as \"Как меня зовут?\", \"Как зовут меня?\", \"Вот меня зовут ...\", \"Зови меня ...\", \"Привет\", \"Спасибо\".\n"
            " - Choose \"kk\" only when there are clear Kazakh signals, for example distinct Kazakh letters (ә, ғ, қ, ң, ө, ұ, ү, һ, і) or clearly Kazakh wording such as \"қалай\", \"мені\", \"аты\", \"сәлем\".\n"
            " - If the text is short, Cyrillic, and ambiguous, prefer \"ru\" over \"kk\" unless there is a strong Kazakh marker.\n\n"
            "Set profile_action to \"set_preferred_name\" only when the user is explicitly telling you what name to use for them, for example "
            "\"call me Alex\", \"my name is Rocco\", \"зови меня Роман\", \"меня зовут Азамат\", or rename phrases like \"зовут меня теперь Heisenberg\". "
            "When you do that, put only the clean extracted name into preferred_name.\n\n"
            "Respond ONLY with valid JSON (no extra text). Example:\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks for university admission guidance\",\"language\":\"en\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"How to apply to uni\"\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to get the Italian DSU scholarship\",\"language\":\"en\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"How to get dsu\"\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to apply for an Italian student visa\",\"language\":\"ru\",\"profile_action\":\"\",\"preferred_name\":\"\"}\n"
            "{\"intent\":\"GUIDANCE\",\"explain\":\"user asks how to apply for an Italian student residence permit\",\"language\":\"en\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"How can I apply for residence permit in Italy?\"\n"
            "{\"intent\":\"FACTUAL_QUESTION\",\"explain\":\"user asks about travel permission while studying abroad\",\"language\":\"en\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"Can I travel while studying in Italy?\"\n"
            "{\"intent\":\"GREETING\",\"explain\":\"short greeting in Kazakh\",\"language\":\"kk\",\"profile_action\":\"\",\"preferred_name\":\"\"}\n"
            "{\"intent\":\"CHIT_CHAT\",\"explain\":\"user sets a preferred name\",\"language\":\"ru\",\"profile_action\":\"set_preferred_name\",\"preferred_name\":\"Роман\"}\n"
            "{\"intent\":\"FACTUAL_QUESTION\",\"explain\":\"user asks what their name is in Russian\",\"language\":\"ru\",\"profile_action\":\"\",\"preferred_name\":\"\"} for input like \"Как меня зовут?\"\n\n"
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
            f"You are a concise helpful assistant for an education/study-abroad bot. {lang_instruction} "
            "Answer only factual questions about the recent conversation or saved user preferences. "
            "For general world knowledge or non-education-abroad topics, politely say the assistant can help only with education-abroad questions. "
            "Answer briefly (1-2 short paragraphs). Use recent conversation context when it is relevant. Do NOT include any file paths or suggest internal document locations.\n\n"
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
            " - Supported topics include: Italian university admission/application guidance, Italian student visa, Italian student residence permit/permesso di soggiorno, study-related travel rights or constraints, CV, DSU scholarship/student financial aid, motivation letter, and recommendation letter.\n"
            " - Retrieval-related means education/study-abroad only.\n"
            " - DSU defaults to the Italian student scholarship/financial-aid topic unless the user explicitly gives another meaning.\n"
            " - A visa question is in scope only when it is about a student/study/enrollment visa.\n"
            " - A residence permit or permesso di soggiorno question is in scope by default as a student residence permit unless the user explicitly says work, tourist, family, permanent residence, asylum, or another non-study context.\n"
            " - Tourist visas, travel visas unrelated to study, work visas, business visas, family visas, general immigration, tourism, flights, hotels, and travel itineraries are out of scope. For these, set is_retrieval_related to false.\n\n"
            "Treat short/lazy queries as clear when the topic is identifiable. Examples of clear queries: "
            "\"how to apply to uni\", \"how to get dsu\", \"dsu money\", \"visa docs\", \"residence permit in Italy\", \"permesso di soggiorno\", \"can I travel while studying\", \"cv help\", \"scholarship money\", \"motivation letter structure\", \"recommendation letter who\".\n"
            "Ask a clarification only when the missing detail would change which supported education-abroad document/topic should be searched. "
            "Examples of unclear queries: \"what documents do I need?\", \"how to apply?\", \"send file\", \"что нужно?\", \"қалай тапсырам?\".\n"
            "Do NOT ask sub-aspect clarifications inside an already identified topic. If the user asks about a detail such as visa photo format, photo size, background, funds, appointment, CV structure, scholarship documents, or recommendation-letter requirements, treat it as clear and search that detail. "
            "For example, \"what photo format is needed for the visa\" is clear; do not ask whether they mean size, background, or something else.\n"
            "For a generic visa query without enough context, ask whether the user means the student visa; do not assume tourist or student.\n"
            "For a generic residence permit query, assume student residence permit unless a non-study context is explicit.\n"
            "If the recent conversation contains an assistant clarification question, combine the latest user reply with that context. "
            "If the combined meaning is clear, produce a complete standalone search query.\n\n"
            "Short follow-up handling:\n"
            " - If the assistant just asked a clarification question and the latest user reply is a confirmation like \"yes\", \"да\", \"иә\", treat it as confirming the assistant's proposed topic and produce a standalone query.\n"
            " - If the latest user asks a short continuation like \"how\", \"how?\", \"how to apply\", \"steps\", \"documents\", \"what docs\", \"как\", \"как подать\", or \"қалай\" after an in-scope answer or clarification, reuse the previous topic from history and produce a complete standalone query for that topic.\n"
            " - If the assistant offered choices like documents vs process, size vs background, or any other sub-aspects, and the user replies \"all\", \"both\", \"everything\", \"все\", \"все я сказал\", \"все вообще\", or similar, do NOT ask again; produce a broad standalone query covering all available details for the already identified topic.\n"
            " - If the user asks a short continuation like \"then?\" or \"потом?\" after an in-scope answer, keep the same topic from history and ask for the next step in the standalone query.\n\n"
            "Language-switch follow-up handling:\n"
            " - If the latest user asks to answer/send/explain the previous in-scope topic in another supported language, set is_retrieval_related true, is_clear true, and reuse the previous in-scope topic as standalone_query.\n"
            " - Set target_language to the requested language code when the user asks for another language: English -> en, Russian -> ru, Kazakh -> kk.\n"
            " - Examples: \"Can you do it in English?\", \"А можно на английском?\", \"а на русском?\", \"қазақша бола ма?\".\n\n"
            "If the classifier intent is OTHER, use the recent conversation to decide whether the latest message is a continuation of a document clarification. "
            "If it is not a document request/guidance question and not a clarification follow-up, set is_retrieval_related to false and leave standalone_query and clarifying_question empty.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            ' - "is_retrieval_related": boolean\n'
            ' - "is_clear": boolean\n'
            ' - "standalone_query": string; if is_clear is true, this must be a complete retrieval query\n'
            ' - "clarifying_question": string; if is_clear is false, ask one concise question in '
            f"{language_name}\n"
            ' - "target_language": string; one of ["en","ru","kk",""]; set only when the user explicitly asks to answer in another supported language\n'
            ' - "reason": one short sentence\n\n'
            "Do not answer the user. Do not mention internal retrieval, embeddings, metadata, or files unless the user asked for a file.\n\n"
            "Examples:\n"
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"How to apply to an Italian university as an international student?","clarifying_question":"","target_language":"","reason":"University admission guidance is in scope and country defaults to Italy."}\n'
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"How to apply for the Italian DSU student scholarship?","clarifying_question":"","target_language":"","reason":"DSU defaults to the Italian student scholarship topic."}\n'
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"What documents are needed for an Italian student visa?","clarifying_question":"","target_language":"","reason":"The visa document topic is clear."}\n'
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"How to apply for an Italian student residence permit?","clarifying_question":"","target_language":"","reason":"Residence permit defaults to the student residence permit context."}\n'
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"All available photo format requirements for an Italian student visa, including size, background, and ICAO standards if present in the documents.","clarifying_question":"","target_language":"","reason":"The visa photo detail is specific enough to search."}\n'
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"How to apply for an Italian student visa?","clarifying_question":"","target_language":"en","reason":"The user asks to continue the previous visa topic in English."}\n'
            '{"is_retrieval_related":true,"is_clear":false,"standalone_query":"","clarifying_question":"Which topic do you mean: student visa, student residence permit, CV, scholarship, motivation letter, or recommendation letter?","target_language":"","reason":"The user asks for documents but not the process."}\n'
            '{"is_retrieval_related":false,"is_clear":false,"standalone_query":"","clarifying_question":"","target_language":"","reason":"The user is not asking a document-grounded question."}\n\n'
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
            target_language_raw = str(parsed.get("target_language") or "").strip()
            clarity = RetrievalClarity(
                is_clear=self._json_bool(parsed.get("is_clear"), default=True),
                standalone_query=str(parsed.get("standalone_query") or "").strip(),
                clarifying_question=str(parsed.get("clarifying_question") or "").strip(),
                reason=str(parsed.get("reason") or "").strip(),
                is_retrieval_related=self._json_bool(parsed.get("is_retrieval_related"), default=True),
                target_language=normalize_language(target_language_raw) if target_language_raw else "",
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
                    target_language=clarity.target_language,
                )
            if not clarity.is_clear and not clarity.clarifying_question:
                clarity = RetrievalClarity(
                    is_clear=False,
                    standalone_query="",
                    clarifying_question=self._default_clarifying_question(language_hint),
                    reason=clarity.reason or "The request is ambiguous.",
                    is_retrieval_related=True,
                    target_language=clarity.target_language,
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

    def assess_retrieval_sufficiency(
        self,
        query: str,
        language_hint: str,
        intent: str,
        top_chunks: List[RetrievedHit],
        history: Optional[List[ConversationMessage]] = None,
    ) -> tuple[RetrievalSufficiency, Optional[ModelUsage]]:
        history_block = self._history_block(history)
        language_name = self._language_name(language_hint) or "the user's language"
        excerpts_block = self._retrieval_excerpts_block(top_chunks)
        prompt = (
            "You are a retrieval sufficiency judge for a document-grounded education-abroad RAG assistant.\n"
            "Your job is NOT to answer the user. Decide whether the retrieved excerpts are enough to answer the latest standalone query, "
            "or whether the assistant should ask exactly one more clarifying question first.\n\n"
            "Scope:\n"
            " - The corpus currently covers Italy education-abroad topics: university admission/application guidance, student visa, student residence permit/permesso di soggiorno, study-related travel rights or constraints, CV, DSU scholarship/student financial aid, motivation letter, and recommendation letter.\n"
            " - Sufficient means the excerpts directly discuss the requested topic and contain enough information to produce a grounded answer or send the requested document.\n"
            " - Insufficient means the excerpts are empty, mostly about the wrong document/topic, the query still lacks a detail that changes which document should be searched, or the requested information is not visible in the excerpts.\n"
            " - If the query asks a generic unsupported country/topic but the excerpts are only Italy docs, ask a clarification instead of guessing.\n"
            " - DSU is a supported Italian student scholarship/financial-aid topic by default; do not ask whether DSU means financial aid unless the excerpts are clearly about a different DSU.\n"
            " - Treat broad queries like \"everything\", \"all\", \"general overview\", \"все\", or \"все вообще\" as sufficient when the excerpts are on the right topic; the answer generator can summarize what is available.\n"
            " - Treat short topic queries like \"how to get DSU\", \"DSU money\", \"visa docs\", or \"cv help\" as sufficient when the excerpts are on the matching topic.\n"
            " - Do NOT ask the user to choose between documents and process when the query asks for a broad overview and the excerpts cover the same topic.\n"
            " - Do NOT ask the user to choose sub-aspects inside an already identified document topic. For example, if the query asks about visa photo format and the excerpts mention photo requirements, mark sufficient; do not ask whether they mean size, background, or another photo detail.\n"
            " - If the excerpts only cover part of a broad detail question, still mark sufficient when they directly address the topic. The answer generator must say only what is available in the excerpts and avoid inventing missing details.\n"
            " - Prefer asking one concise clarification only when the excerpts are weak, empty, mismatched, or cannot identify the requested education-abroad topic.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            ' - "is_sufficient": boolean\n'
            ' - "clarifying_question": string; if is_sufficient is false, ask one concise question in '
            f"{language_name}\n"
            ' - "reason": one short sentence\n\n'
            "Do not mention internal retrieval, embeddings, scores, metadata, or top-K.\n\n"
            "Examples:\n"
            '{"is_sufficient":true,"clarifying_question":"","reason":"The excerpts directly cover Italian student visa documents."}\n'
            '{"is_sufficient":true,"clarifying_question":"","reason":"The excerpts directly mention student visa photo requirements, so the answer can state the available details without asking for a sub-aspect."}\n'
            '{"is_sufficient":false,"clarifying_question":"Which topic do you mean: student visa, student residence permit, CV, scholarship, motivation letter, or recommendation letter?","reason":"The excerpts do not identify the requested document topic."}\n\n'
            f"{history_block}"
            f"Classifier intent: {json.dumps(intent)}\n"
            f"Standalone query: {json.dumps(query)}\n\n"
            f"{excerpts_block}"
        )
        try:
            response = self._client.responses.create(
                model=self._settings.class_model,
                input=prompt,
                max_output_tokens=160,
                temperature=0.0,
            )
            raw_text = self._resp_to_text(response) or ""
            parsed = self._extract_json(raw_text) or {}
            sufficiency = RetrievalSufficiency(
                is_sufficient=self._json_bool(parsed.get("is_sufficient"), default=True),
                clarifying_question=str(parsed.get("clarifying_question") or "").strip(),
                reason=str(parsed.get("reason") or "").strip(),
            )
            if not sufficiency.is_sufficient and not sufficiency.clarifying_question:
                sufficiency = RetrievalSufficiency(
                    is_sufficient=False,
                    clarifying_question=self._default_retrieval_follow_up_question(language_hint),
                    reason=sufficiency.reason or "The retrieved excerpts are not sufficient.",
                )
            return sufficiency, self._extract_usage(response, self._settings.class_model)
        except Exception as exc:
            print("[sufficiency] retrieval sufficiency check failed:", exc)
            return (
                RetrievalSufficiency(
                    is_sufficient=True,
                    reason=f"sufficiency check failed: {exc}",
                ),
                None,
            )

    def classify_attachment_follow_up(
        self,
        query: str,
        history: Optional[List[ConversationMessage]] = None,
        pending_file: str = "",
    ) -> tuple[str, Optional[ModelUsage]]:
        normalized_pending_file = (pending_file or "").strip()
        if not history and not normalized_pending_file:
            return "", None

        history_block = self._history_block(history)
        pending_block = (
            f"Pending offered attachment source: {json.dumps(normalized_pending_file)}\n"
            if normalized_pending_file
            else "Pending offered attachment source: none\n"
        )
        prompt = (
            "You detect whether the latest user message is asking for an attachment.\n"
            "Return a JSON object with EXACTLY two keys:\n"
            ' - "attachment_action": one of ["resend_last_attachment","send_pending_attachment",""]\n'
            ' - "explain": one short sentence explaining why\n\n'
            'Choose "send_pending_attachment" only when there is a pending offered attachment source above and the user clearly accepts or asks to receive that offered file, '
            'including short replies like "yes", "yes please", "send it", "да", "давай", "можно", "иә", or similar in context. '
            'Choose "resend_last_attachment" only when the user is clearly asking to send the already-mentioned file again, even in short follow-ups like "again", '
            '"one more time", "еще раз", or similar context-dependent requests. '
            'If the user is asking what the file is about, asking a new question, rejecting the offer, or you are unsure, return "".\n\n'
            "Respond ONLY with valid JSON. Example:\n"
            '{"attachment_action":"resend_last_attachment","explain":"user asks to send the previously sent file again"}\n'
            '{"attachment_action":"send_pending_attachment","explain":"user accepts the offered file"}\n'
            '{"attachment_action":"","explain":"user asks about the file rather than requesting a resend"}\n\n'
            f"{pending_block}"
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
            return "Қай тақырып бойынша сұрап тұрсыз: студенттік виза, студенттік тұруға рұқсат, CV, шәкіртақы, мотивациялық хат немесе ұсыныс хат?"
        if normalized == "ru":
            return "По какой теме вы спрашиваете: студенческая виза, студенческий ВНЖ, CV, стипендия, мотивационное письмо или рекомендательное письмо?"
        return "Which topic do you mean: student visa, student residence permit, CV, scholarship, motivation letter, or recommendation letter?"

    @staticmethod
    def _default_retrieval_follow_up_question(language_hint: str) -> str:
        normalized = normalize_language(language_hint)
        if normalized == "kk":
            return "Құжаттардан нақты жауап табу үшін тақырыпты нақтылай аласыз ба: студенттік виза, студенттік тұруға рұқсат, CV, шәкіртақы, мотивациялық хат немесе ұсыныс хат?"
        if normalized == "ru":
            return "Чтобы найти точный ответ в документах, уточните тему: студенческая виза, студенческий ВНЖ, CV, стипендия, мотивационное письмо или рекомендательное письмо?"
        return "To find the right answer in the documents, which topic do you mean: student visa, student residence permit, CV, scholarship, motivation letter, or recommendation letter?"

    @staticmethod
    def _retrieval_excerpts_block(top_chunks: List[RetrievedHit]) -> str:
        if not top_chunks:
            return "Retrieved excerpts: none\n"

        lines = ["Retrieved excerpts:"]
        for index, hit in enumerate(top_chunks, start=1):
            source_file = hit.meta.get("source_file") or hit.meta.get("filename") or "unknown"
            page = hit.meta.get("page")
            text = (hit.meta.get("text") or hit.meta.get("md") or "").strip()
            excerpt = text[:1000].replace("\n", " ").strip()
            lines.append(f"[{index}] file: {source_file} page: {page}")
            lines.append(f"excerpt: {excerpt}")
            lines.append("")
        return "\n".join(lines)

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

    def _guardrail_result_from_payload(self, parsed: Dict[str, Any]) -> GuardrailResult:
        violation = str(parsed.get("violation") or "").strip().lower()
        if violation not in {"", "out_of_scope", "unsafe", "unsupported_language"}:
            violation = "out_of_scope"

        needs_context = self._json_bool(parsed.get("needs_context"), default=False)
        allowed = self._json_bool(parsed.get("allowed"), default=False)
        if needs_context:
            allowed = False
            violation = ""
        elif allowed:
            violation = ""
        elif not violation:
            violation = "out_of_scope"

        return GuardrailResult(
            allowed=allowed,
            reason=str(parsed.get("reason") or "").strip(),
            language=normalize_language(str(parsed.get("language") or "")),
            violation=violation,
            model=self._settings.class_model,
            needs_context=needs_context,
        )

    @staticmethod
    def _merge_usage(first: Optional[ModelUsage], second: Optional[ModelUsage]) -> Optional[ModelUsage]:
        if first is None:
            return second
        if second is None:
            return first
        return ModelUsage(
            model=first.model,
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            total_tokens=first.total_tokens + second.total_tokens,
        )

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

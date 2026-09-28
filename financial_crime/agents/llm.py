from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from financial_crime.config import settings
from financial_crime.errors import LLMConfigurationError, LLMProviderError, LLMRateLimitError

logger = logging.getLogger("financial_crime.llm")


class NarrativeStatement(BaseModel):
    category: Literal["AGENT_HYPOTHESIS"]
    text: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)


class NarrativeDraft(BaseModel):
    statements: list[NarrativeStatement] = Field(default_factory=list, max_length=5)


class LLMProvider(Protocol):
    name: str

    def generate(self, evidence_packet: dict) -> ProviderResponse: ...


@dataclass
class ProviderResponse:
    draft: NarrativeDraft
    input_tokens: int
    output_tokens: int


def _prompt(packet: dict) -> str:
    return (
        "You support a human financial-crime investigator. Return only the requested JSON schema. "
        "Provide at most five AGENT_HYPOTHESIS statements that suggest follow-up questions. "
        "Every statement must cite one or more supplied evidence IDs. Do not state guilt, intent, "
        "or new observations. Do not calculate, repeat, or alter risk scores or SHAP values. "
        "If the evidence does not support a useful hypothesis, return an empty statements list.\n"
        + json.dumps(packet, separators=(",", ":"))
    )


class GroqProvider:
    name = "groq"

    def __init__(self):
        if not settings.groq_api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not configured")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    def generate(self, evidence_packet: dict) -> ProviderResponse:
        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": _prompt(evidence_packet)}],
            temperature=0,
            max_tokens=settings.llm_max_output_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("Groq returned an empty response")
        usage = response.usage
        return ProviderResponse(
            draft=NarrativeDraft.model_validate_json(content),
            input_tokens=int(usage.prompt_tokens) if usage else 0,
            output_tokens=int(usage.completion_tokens) if usage else 0,
        )


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        if not settings.gemini_api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )

    def generate(self, evidence_packet: dict) -> ProviderResponse:
        response = self.client.models.generate_content(
            model=settings.gemini_model,
            contents=_prompt(evidence_packet),
            config=self.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=settings.llm_max_output_tokens,
                response_mime_type="application/json",
                response_schema=NarrativeDraft,
            ),
        )
        if not response.text:
            raise LLMProviderError("Gemini returned an empty response")
        usage = response.usage_metadata
        return ProviderResponse(
            draft=NarrativeDraft.model_validate_json(response.text),
            input_tokens=int(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=int(usage.candidates_token_count or 0) if usage else 0,
        )


def _provider(name: str) -> LLMProvider:
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    raise LLMConfigurationError("LLM provider is disabled or unsupported")


def _retry(provider: LLMProvider, packet: dict) -> ProviderResponse:
    last_error: Exception | None = None
    attempts = max(0, settings.llm_max_retries) + 1
    for attempt in range(attempts):
        try:
            return provider.generate(packet)
        except Exception as error:
            last_error = error
            code = getattr(error, "status_code", None) or getattr(error, "code", None)
            error_text = str(error).casefold()
            if code == 429 or "rate limit" in error_text or "resource_exhausted" in error_text:
                last_error = LLMRateLimitError("provider rate limit reached")
            logger.warning(
                "LLM request failed provider=%s error=%s attempt=%s",
                provider.name,
                type(last_error).__name__,
                attempt + 1,
            )
            if attempt + 1 < attempts:
                # Non-security jitter prevents synchronized retries to an optional provider.
                delay = min(2**attempt + random.random() * 0.25, 4.0)  # nosec B311
                time.sleep(delay)
    raise LLMProviderError(type(last_error).__name__ if last_error else "request_failed")


def generate_grounded_hypotheses(report: dict) -> dict:
    if settings.llm_provider == "disabled":
        return {
            "status": "disabled",
            "statements": [],
            "provider": None,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    if not settings.llm_allow_evidence_export:
        return {
            "status": "evidence_export_disabled",
            "statements": [],
            "provider": None,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    evidence = report.get("observed_evidence", [])[:20]
    allowed_ids = {item["evidence_id"] for item in evidence}
    packet = {
        "case_id": report.get("case_id"),
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "category": item["category"],
                "source": item["source"],
                "claim": item["claim"][:500],
            }
            for item in evidence
        ],
        "allowed_evidence_ids": sorted(allowed_ids),
        "risk_score_available": report.get("risk_score") is not None,
        "shap_features_available": bool(report.get("top_shap_factors")),
    }
    max_chars = settings.llm_max_input_tokens * 4
    while evidence and len(json.dumps(packet, separators=(",", ":"))) > max_chars:
        evidence.pop()
        packet["evidence"].pop()
        packet["allowed_evidence_ids"] = [item["evidence_id"] for item in packet["evidence"]]
    providers = [settings.llm_provider, settings.llm_fallback_provider]
    errors = []
    for name in providers:
        if name == "disabled":
            continue
        try:
            provider_response = _retry(_provider(name), packet)
            draft = provider_response.draft
            accepted = [
                statement.model_dump()
                for statement in draft.statements
                if set(statement.evidence_ids) <= allowed_ids
            ]
            return {
                "status": "available",
                "provider": name,
                "statements": accepted,
                "unsupported_claims_rejected": len(draft.statements) - len(accepted),
                "input_tokens": provider_response.input_tokens,
                "output_tokens": provider_response.output_tokens,
            }
        except Exception as error:
            errors.append(type(error).__name__)
            logger.error(
                "LLM provider unavailable provider=%s error=%s", name, type(error).__name__
            )
    return {
        "status": "unavailable",
        "provider": None,
        "statements": [],
        "errors": errors,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def provider_health() -> dict[str, dict[str, str | bool]]:
    providers: dict[str, dict[str, str | bool]] = {}
    for name, configured in [
        ("groq", bool(settings.groq_api_key)),
        ("gemini", bool(settings.gemini_api_key)),
    ]:
        providers[name] = {
            "configured": configured,
            "status": "configured" if configured else "missing_credentials",
        }
    return {
        "primary": providers.get(
            settings.llm_provider, {"configured": False, "status": "disabled"}
        ),
        "fallback": providers.get(
            settings.llm_fallback_provider, {"configured": False, "status": "disabled"}
        ),
        "evidence_export_enabled": settings.llm_allow_evidence_export,
        "active_provider": settings.llm_provider,
    }

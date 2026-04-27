from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from smart_db_csv_builder.models.schemas import RecSystemType, SchemaResponse
from smart_db_csv_builder.services.planner_support import (
    _auto_alias_role_columns,
    _build_fallback_plan,
    _build_prompt,
    _sanitize_plan,
)
from smart_db_csv_builder.services.planner_types import CollectionFetch, MergePlan, TableQuery

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "codestral-2508")
DEFAULT_CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "")


def _post_json(url, payload, headers, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_text = ""
        try:
            response_text = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            response_text = ""
        detail = f"HTTP {exc.code} {exc.reason}"
        if response_text:
            detail = f"{detail}: {response_text[:1000]}"
        raise RuntimeError(detail) from exc


def _call_groq(prompt, api_key, model):
    data = _post_json(
        url="https://api.groq.com/openai/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    if "choices" not in data:
        raise RuntimeError(f"Groq invalid response: {data}")

    return data["choices"][0]["message"]["content"]


def _call_openai(prompt, api_key, model):
    data = _post_json(
        url="https://api.openai.com/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return data["choices"][0]["message"]["content"]


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text" and chunk.get("text"):
                parts.append(chunk["text"])
        if parts:
            return "\n".join(parts)
    raise RuntimeError(f"Unsupported chat content payload: {content!r}")


def _call_mistral(prompt, api_key, model):
    data = _post_json(
        url="https://api.mistral.ai/v1/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _extract_text_content(data["choices"][0]["message"]["content"])


def _parse_chat_model_name(model_name: str) -> tuple[str, str]:
    raw = (model_name or "").strip()
    if not raw:
        return "", ""
    if ":" not in raw:
        return "google_genai", raw
    provider, model = raw.split(":", 1)
    return provider.strip().lower(), model.strip()


def _call_google_genai(prompt, api_key, model):
    data = _post_json(
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        },
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Google GenAI invalid response: {data}")

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Google GenAI returned no text content: {data}")

    return "\n".join(text_parts)


def _parse_plan(raw_text):
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON returned from LLM")
    return json.loads(match.group(0))


def build_merge_plan(
    schemas: list[SchemaResponse],
    rec_type: RecSystemType,
    target_description: Optional[str] = None,
    mistral_api_key: Optional[str] = "",
    groq_api_key: Optional[str] = "",
    openai_api_key: Optional[str] = "",
    chat_api_key: Optional[str] = "",
    chat_model_name: str = DEFAULT_CHAT_MODEL_NAME,
    mistral_model: str = DEFAULT_MISTRAL_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> MergePlan:
    chat_api_key = chat_api_key or os.getenv("CHAT_API_KEY", "")
    chat_model_name = chat_model_name or os.getenv("CHAT_MODEL_NAME", "")
    mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
    mistral_model = mistral_model or os.getenv("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    chat_provider, chat_model = _parse_chat_model_name(chat_model_name)

    if (
        chat_model_name
        and not chat_api_key
        and not mistral_api_key
        and not groq_api_key
        and not openai_api_key
    ):
        raise RuntimeError(
            "CHAT_MODEL_NAME is configured but no API key is available. Set CHAT_API_KEY or MISTRAL_API_KEY in the backend environment."
        )

    if not mistral_api_key and not chat_api_key and not groq_api_key and not openai_api_key:
        raise RuntimeError(
            "No LLM API key configured. Set MISTRAL_API_KEY, CHAT_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in the backend "
            "environment or include one in the build request."
        )

    prompt = _build_prompt(schemas, rec_type, target_description)
    logger.info("Sending schema prompt to LLM")

    raw = None
    errors = []

    if mistral_api_key:
        try:
            logger.info("Trying Mistral model %s...", mistral_model)
            raw = _call_mistral(prompt, mistral_api_key, mistral_model)
            logger.info("Mistral success")
        except Exception as exc:
            logger.warning("Mistral failed: %s", exc)
            errors.append(f"Mistral: {exc}")

    if raw is None and chat_api_key and chat_model:
        try:
            if chat_provider == "google_genai":
                logger.info("Trying Google GenAI model %s...", chat_model)
                raw = _call_google_genai(prompt, chat_api_key, chat_model)
                logger.info("Google GenAI success")
            elif chat_provider == "mistral":
                logger.info("Trying Mistral via CHAT_MODEL_NAME model %s...", chat_model)
                raw = _call_mistral(prompt, chat_api_key, chat_model)
                logger.info("Mistral via CHAT_MODEL_NAME success")
            else:
                raise RuntimeError(
                    f"Unsupported CHAT_MODEL_NAME provider '{chat_provider}'. Expected 'google_genai:<model-name>' or 'mistral:<model-name>'."
                )
        except Exception as exc:
            logger.warning("CHAT_MODEL_NAME provider failed: %s", exc)
            errors.append(f"{chat_provider or 'CHAT_MODEL_NAME'}: {exc}")

    if groq_api_key:
        try:
            logger.info("Trying Groq...")
            raw = _call_groq(prompt, groq_api_key, groq_model)
            logger.info("Groq success")
        except Exception as exc:
            logger.warning("Groq failed: %s", exc)
            errors.append(f"Groq: {exc}")

    if raw is None and openai_api_key:
        try:
            logger.info("Falling back to OpenAI...")
            raw = _call_openai(prompt, openai_api_key, openai_model)
            logger.info("OpenAI success")
        except Exception as exc:
            logger.error("OpenAI failed: %s", exc)
            errors.append(f"OpenAI: {exc}")

    if raw is None:
        if errors:
            raise RuntimeError("All LLM providers failed: " + " | ".join(errors))
        raise RuntimeError(
            "No LLM response was produced. Check MISTRAL_API_KEY, CHAT_API_KEY, GROQ_API_KEY, and OPENAI_API_KEY configuration and provider availability."
        )

    parsed_plan = _parse_plan(raw)
    try:
        plan_dict = _sanitize_plan(parsed_plan, schemas, rec_type)
    except ValueError as exc:
        logger.warning("LLM plan validation failed; using fallback planner. Reason: %s", exc)
        plan_dict = _build_fallback_plan(
            schemas=schemas,
            rec_type=rec_type,
            target_description=target_description,
            reason=str(exc),
            raw_plan=parsed_plan,
            raw_text=raw,
        )

    table_queries = [
        TableQuery(
            connection_id=tq["connection_id"],
            table=tq["table"],
            columns=tq.get("columns", []),
            where=tq.get("where", ""),
            alias_map=tq.get("alias_map", {}),
        )
        for tq in plan_dict.get("table_queries", [])
    ]

    return MergePlan(
        table_queries=table_queries,
        collection_fetches=[],
        merge_keys=plan_dict.get("merge_keys", []),
        final_columns=plan_dict.get("final_columns", []),
        description=plan_dict.get("description", ""),
        raw_plan=plan_dict,
    )


__all__ = [
    "CollectionFetch",
    "MergePlan",
    "TableQuery",
    "_auto_alias_role_columns",
    "_call_google_genai",
    "_call_groq",
    "_call_mistral",
    "_call_openai",
    "_parse_chat_model_name",
    "build_merge_plan",
]

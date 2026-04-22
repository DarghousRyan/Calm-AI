"""
Groq chat assist service.
"""

from __future__ import annotations

import os

import requests

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _chat_provider() -> tuple[str, str] | None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        model = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        return (groq_key, model)

    return None


def get_chat_reply(user_message: str) -> str:
    provider = _chat_provider()
    if provider is None:
        raise RuntimeError("No chat provider configured. Set GROQ_API_KEY.")
    api_key, model_name = provider

    system_prompt = (
        "You are Calm AI's supportive assistant. Use warm, practical, non-clinical language. "
        "Avoid diagnosis or medical claims. Keep responses concise and meaningful."
    )
    try:
        resp = requests.post(
            f"{_GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise RuntimeError(f"groq chat request failed: {e}") from e

    text = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("The chat model returned an empty response.")
    return text

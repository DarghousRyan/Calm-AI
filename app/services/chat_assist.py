"""
Groq chat assist service.
"""

from __future__ import annotations

import os

import requests

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_ALLOWED_ROLES = {"user", "assistant"}


def _chat_provider() -> tuple[str, str] | None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        model = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        return (groq_key, model)

    return None


def _sanitize_history(history: list[dict[str, str]] | None, *, limit: int = 20) -> list[dict[str, str]]:
    if not history:
        return []

    cleaned: list[dict[str, str]] = []
    for item in history[-limit:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in _ALLOWED_ROLES or not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


def get_chat_reply(user_message: str, *, history: list[dict[str, str]] | None = None) -> str:
    provider = _chat_provider()
    if provider is None:
        raise RuntimeError("No chat provider configured. Set GROQ_API_KEY.")
    api_key, model_name = provider

    system_prompt = (
        "You are Calm AI's supportive assistant. Use warm, practical, non-clinical language. "
        "Avoid diagnosis or medical claims. Keep responses concise and meaningful."
    )
    um = user_message.strip()[:4000]
    if not um:
        raise RuntimeError("Empty user message.")

    chat_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    prior = _sanitize_history(history)
    # Frontend often includes the latest user turn in `history`; avoid sending it twice.
    while prior and prior[-1]["role"] == "user" and prior[-1]["content"] == um:
        prior = prior[:-1]
    chat_messages.extend(prior)
    # Always end with the current user message so the provider never gets a trailing assistant-only turn.
    chat_messages.append({"role": "user", "content": um})

    try:
        resp = requests.post(
            f"{_GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": chat_messages,
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

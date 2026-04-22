"""
Optional Groq assist for recommendations.

If GROQ_API_KEY is set, rule-based suggestions are sent to a chat model
for gentle personalization. On any failure or missing key, callers should use
the original rule-based list unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

import requests

from app.services.recommendation_service import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "medium", "high"]

_MAX_ITEMS = 5
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _provider_settings() -> tuple[str, str] | None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        return None
    model = os.environ.get("GROQ_RECOMMENDATIONS_MODEL", "llama-3.1-8b-instant").strip()
    return (groq_key, model or "llama-3.1-8b-instant")


def enrich_with_groq_if_configured(
    latest_log: dict[str, Any],
    risk_level: RiskLevel,
    base_recommendations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    When GROQ_API_KEY is present, ask the model to refine the baseline list.
    Otherwise return ``base_recommendations`` unchanged.
    """
    provider = _provider_settings()
    if provider is None:
        return base_recommendations
    api_key, model_name = provider

    # Strip per-item disclaimers from the payload; we re-attach the canonical one.
    slim_base: list[dict[str, str]] = []
    for item in base_recommendations:
        slim_base.append(
            {
                "title": str(item.get("title", "")),
                "suggestion": str(item.get("suggestion", "")),
                "explanation": str(item.get("explanation", "")),
            }
        )

    user_payload = {
        "latest_log": latest_log,
        "risk_level": risk_level,
        "baseline_recommendations": slim_base,
    }

    system = (
        "You help a wellbeing journaling app refine short, supportive suggestions. "
        "Rules: use warm, non-clinical language; do not diagnose or prescribe treatment; "
        "do not claim medical authority; keep suggestions practical and safe. "
        "Output a JSON object with key \"recommendations\" whose value is an array of "
        f"3-{_MAX_ITEMS} objects. Each object must have string fields: title, suggestion, "
        "explanation (no other keys). Base your output on the user's log and risk level; "
        "you may adapt or merge the baseline recommendations but stay in the same spirit."
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
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "temperature": 0.7,
                "max_tokens": 1200,
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("groq recommendation assist failed: %s", e)
        return base_recommendations

    raw = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    if not raw:
        return base_recommendations

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Groq returned non-JSON; using rule-based recommendations")
        return base_recommendations

    recs = data.get("recommendations")
    if not isinstance(recs, list):
        return base_recommendations

    out: list[dict[str, str]] = []
    for item in recs:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        suggestion = str(item.get("suggestion", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if not title or not suggestion:
            continue
        out.append(
            {
                "title": title,
                "suggestion": suggestion,
                "explanation": explanation,
                "disclaimer": DISCLAIMER_TEXT,
            }
        )
        if len(out) >= _MAX_ITEMS:
            break

    if len(out) < 1:
        return base_recommendations

    return out

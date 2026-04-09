"""
Recommendation service for Calm AI.

Goal:
- Given a user's latest daily log plus a model-predicted risk level,
  return supportive, non-clinical suggestions.

Design:
- Purely rule-based, easy to extend.
- No medical advice; everything is framed as gentle suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


RiskLevel = Literal["low", "medium", "high"]

DISCLAIMER_TEXT = (
    "These suggestions are for general wellbeing support only and are "
    "not medical or clinical advice. If you are in crisis or worried "
    "about your safety, please contact local emergency services or a "
    "qualified professional."
)


@dataclass
class DailyLogSnapshot:
    """Minimal view of a daily log used for recommendations."""

    mood: str | None
    stress: float | None
    craving: float | None
    sleep_hours: float | None
    exercise_minutes: float | None
    social_interaction: float | None
    trigger_boredom: int | None
    trigger_loneliness: int | None
    trigger_conflict: int | None
    days_since_last_relapse: int | None


def _coerce_snapshot(data: dict[str, Any]) -> DailyLogSnapshot:
    """
    Map an arbitrary dict (e.g. from DB row or request body)
    into a typed DailyLogSnapshot.
    """

    def _get_float(key: str) -> float | None:
        value = data.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _get_int(key: str) -> int | None:
        value = data.get(key)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return DailyLogSnapshot(
        mood=(str(data.get("mood")).lower() if data.get("mood") is not None else None),
        stress=_get_float("stress"),
        craving=_get_float("craving"),
        sleep_hours=_get_float("sleep_hours"),
        exercise_minutes=_get_float("exercise_minutes"),
        social_interaction=_get_float("social_interaction"),
        trigger_boredom=_get_int("trigger_boredom"),
        trigger_loneliness=_get_int("trigger_loneliness"),
        trigger_conflict=_get_int("trigger_conflict"),
        days_since_last_relapse=_get_int("days_since_last_relapse"),
    )


def _base_disclaimer() -> str:
    # Backwards-friendly helper for rules that attach disclaimers per-item.
    return DISCLAIMER_TEXT


def _add_sleep_craving_rules(s: DailyLogSnapshot, risk: RiskLevel, out: list[dict[str, str]]) -> None:
    if s.sleep_hours is None or s.craving is None:
        return

    very_tired = s.sleep_hours < 6
    high_craving = s.craving >= 7

    if very_tired and high_craving:
        out.append(
            {
                "title": "Focus on rest before urges",
                "suggestion": (
                    "Tonight, try to prioritize winding down early and giving "
                    "yourself space to rest before engaging with triggers "
                    "like screens, social media, or stressful conversations."
                ),
                "explanation": (
                    "Low sleep can make cravings feel louder. Supporting your "
                    "body with rest can make it a bit easier to ride out urges."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )
    elif very_tired:
        out.append(
            {
                "title": "Gentle evening routine",
                "suggestion": (
                    "Consider a simple, calming wind-down tonight—like a warm "
                    "shower, light reading, or breathing exercises—before bed."
                ),
                "explanation": (
                    "Improving sleep even slightly over time can support mood, "
                    "stress, and craving levels."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )
    elif high_craving and risk in {"medium", "high"}:
        out.append(
            {
                "title": "Surf the urge with a safe activity",
                "suggestion": (
                    "Pick one short, safe activity you can use when cravings "
                    "spike—like a five-minute walk, a glass of water, or "
                    "texting a trusted friend."
                ),
                "explanation": (
                    "Having a small, specific plan ready can make it easier to "
                    "get through intense craving waves without acting on them."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )


def _add_stress_loneliness_rules(s: DailyLogSnapshot, risk: RiskLevel, out: list[dict[str, str]]) -> None:
    lonely = bool(s.trigger_loneliness)
    high_stress = (s.stress is not None) and s.stress >= 7

    if lonely and high_stress:
        out.append(
            {
                "title": "Reach out in a low-pressure way",
                "suggestion": (
                    "Consider sending a short check-in message to someone you "
                    "trust, or joining a low-effort online community or group "
                    "that feels safe."
                ),
                "explanation": (
                    "Stress plus feeling alone can feel heavy. Even a small "
                    "moment of connection can help lighten that load."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )
    elif lonely:
        out.append(
            {
                "title": "Plan a small connection moment",
                "suggestion": (
                    "Think of one small, realistic social interaction you can "
                    "aim for in the next day—like a brief chat, call, or "
                    "message."
                ),
                "explanation": (
                    "Regular, gentle connection can help ease loneliness over "
                    "time without needing big social events."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )
    elif high_stress and risk in {"medium", "high"}:
        out.append(
            {
                "title": "Create a tiny stress buffer",
                "suggestion": (
                    "Choose one short grounding practice you can do today, "
                    "such as slow breathing, naming five things you see, or "
                    "taking a brief walk."
                ),
                "explanation": (
                    "Short, consistent practices can create small buffers "
                    "against stress without needing a lot of time or energy."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )


def _add_conflict_instability_rules(s: DailyLogSnapshot, risk: RiskLevel, out: list[dict[str, str]]) -> None:
    conflict = bool(s.trigger_conflict)
    boredom = bool(s.trigger_boredom)

    if conflict and risk in {"medium", "high"}:
        out.append(
            {
                "title": "Give yourself space after conflict",
                "suggestion": (
                    "If you can, take a short break away from the situation—"
                    "step outside, listen to calming music, or write down "
                    "what you’re feeling without judging it."
                ),
                "explanation": (
                    "Conflict can leave your system stirred up. A short pause "
                    "can help you respond more intentionally instead of from "
                    "pure reaction."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )

    if boredom and risk in {"medium", "high"}:
        out.append(
            {
                "title": "Plan a safe distraction",
                "suggestion": (
                    "Pick one low-effort, safe activity you can turn to when "
                    "you feel bored—like a short walk, a puzzle game, or a "
                    "simple creative hobby."
                ),
                "explanation": (
                    "Boredom can sometimes pull you toward risky habits. "
                    "Having a few safe options ready can make those moments "
                    "easier to handle."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )


def _add_protective_factor_rules(s: DailyLogSnapshot, risk: RiskLevel, out: list[dict[str, str]]) -> None:
    # Highlight protective patterns when they appear, regardless of risk.
    good_sleep = (s.sleep_hours is not None) and s.sleep_hours >= 7
    some_exercise = (s.exercise_minutes is not None) and s.exercise_minutes >= 20
    some_social = (s.social_interaction is not None) and s.social_interaction >= 30

    if good_sleep or some_exercise or some_social:
        out.append(
            {
                "title": "Notice what’s already helping",
                "suggestion": (
                    "Take a moment to notice any habits that might already be "
                    "supporting you—like sleep, movement, or connection—and "
                    "consider gently protecting them in your schedule."
                ),
                "explanation": (
                    "Recognizing your existing strengths can make it easier to "
                    "build on them, especially when things feel difficult."
                ),
                "disclaimer": _base_disclaimer(),
            }
        )


def _add_general_grounding_rule(out: list[dict[str, str]]) -> None:
    # General, always-safe grounding suggestion to ensure we have something
    # even when data is sparse.
    out.append(
        {
            "title": "Simple check-in with yourself",
            "suggestion": (
                "If it feels okay, pause for a moment to notice how your body "
                "feels (tension, temperature, breathing) and name one small, "
                "kind thing you could offer yourself today."
            ),
            "explanation": (
                "Gentle self-check-ins can increase awareness of your needs "
                "without pressuring you to change everything at once."
            ),
            "disclaimer": _base_disclaimer(),
        }
    )


def generate_recommendations(
    latest_log: dict[str, Any],
    *,
    risk_level: RiskLevel,
    max_suggestions: int = 5,
) -> list[dict[str, str]]:
    """
    Produce 3–5 supportive, non-clinical recommendations.

    Parameters
    ----------
    latest_log:
        A mapping with keys like `mood`, `stress`, `craving`, `sleep_hours`,
        `exercise_minutes`, `social_interaction`, and trigger flags.
    risk_level:
        Model-derived risk level: "low", "medium", or "high".
    max_suggestions:
        Upper bound on the number of suggestions to return (default: 5).
    """

    snapshot = _coerce_snapshot(latest_log)
    suggestions: list[dict[str, str]] = []

    # Rule groups are kept separate so they are easy to extend.
    _add_sleep_craving_rules(snapshot, risk_level, suggestions)
    _add_stress_loneliness_rules(snapshot, risk_level, suggestions)
    _add_conflict_instability_rules(snapshot, risk_level, suggestions)
    _add_protective_factor_rules(snapshot, risk_level, suggestions)

    if not suggestions:
        _add_general_grounding_rule(suggestions)
    else:
        # Ensure we always return at least one general grounding suggestion,
        # while still respecting max_suggestions.
        if len(suggestions) < max_suggestions:
            _add_general_grounding_rule(suggestions)

    # Deduplicate by title to avoid repeated suggestions as the rules grow.
    seen_titles: set[str] = set()
    unique: list[dict[str, str]] = []
    for s in suggestions:
        title = s.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            unique.append(s)

    # Respect the requested range: 3–5 suggestions where possible.
    # - If we have fewer than 3 unique suggestions, return what we have.
    # - If we have many, trim to max_suggestions.
    if len(unique) > max_suggestions:
        unique = unique[:max_suggestions]

    return unique


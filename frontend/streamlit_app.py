"""
Calm AI Streamlit frontend.

What this app does:
- Collect a daily log via a simple form.
- Send it to the FastAPI backend for risk prediction + recommendations.
- Display the latest predicted risk and supportive recommendations.

Notes:
- The backend URL is configurable in the sidebar.
- This file is intentionally small and easy to modify.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests
from requests import exceptions as requests_exceptions
import streamlit as st


# -----------------------------
# Configuration / helpers
# -----------------------------

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 15.0) -> dict[str, Any]:
    """POST JSON and return parsed JSON or raise a readable exception."""
    resp = requests.post(url, json=payload, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface backend error bodies if present (helpful while iterating).
        msg = f"{e} (status={resp.status_code})"
        try:
            msg += f" body={resp.text}"
        except Exception:
            pass
        raise RuntimeError(msg) from e
    return resp.json()


def _normalize_backend_url(raw: str) -> str:
    return raw.rstrip("/")


def _format_risk_label(raw: Any) -> str:
    # Display friendly labels like "Low", "Medium", "High".
    return str(raw).strip().title()


def _format_probabilities_percent(raw_probs: dict[str, Any]) -> dict[str, str]:
    """
    Convert model probabilities from decimals to percentage strings.
    Example: 0.8231 -> "82.31%"
    """
    formatted: dict[str, str] = {}
    for label, value in raw_probs.items():
        try:
            pct = float(value) * 100.0
            formatted[str(label).title()] = f"{pct:.2f}%"
        except (TypeError, ValueError):
            formatted[str(label).title()] = "N/A"
    return formatted


def _normalize_probabilities(raw_probs: dict[str, Any]) -> list[tuple[str, float]]:
    """
    Convert backend probabilities into sorted (Label, value_0_to_1) rows.
    """
    rows: list[tuple[str, float]] = []
    for label, value in raw_probs.items():
        try:
            prob = float(value)
        except (TypeError, ValueError):
            continue
        # Clamp defensively to valid probability range.
        prob = max(0.0, min(1.0, prob))
        rows.append((str(label).title(), prob))

    # Show highest probability first.
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Calm AI", layout="centered")

st.title("Calm AI")
st.caption("A minimal UI for daily logs, risk prediction, and supportive suggestions.")

with st.sidebar:
    st.header("Settings")
    backend_url = _normalize_backend_url(
        st.text_input("Backend URL", value=DEFAULT_BACKEND_URL, help="Where your FastAPI server is running.")
    )
    st.write("Tip: start the API with `uvicorn app.main:app --reload`.")


st.header("Daily log")

# Keep inputs explicit and easy to expand later.
with st.form("daily_log_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        log_date = st.date_input("Log date", value=date.today())
        mood = st.selectbox("Mood", options=["Great", "Good", "Okay", "Down", "Bad"], index=2)
        sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=16.0, value=7.0, step=0.25)
        exercise_minutes = st.number_input("Exercise (minutes)", min_value=0, max_value=300, value=20, step=5)

    with col2:
        stress = st.slider("Stress (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        craving = st.slider("Craving (0-10)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        social_interaction = st.number_input(
            "Social interaction (minutes)",
            min_value=0,
            max_value=600,
            value=30,
            step=5,
        )
        days_since_last_relapse = st.number_input(
            "Days since last relapse",
            min_value=0,
            max_value=3650,
            value=7,
            step=1,
        )

    st.subheader("Triggers")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        trigger_boredom = st.checkbox("Boredom", value=False)
    with tcol2:
        trigger_loneliness = st.checkbox("Loneliness", value=False)
    with tcol3:
        trigger_conflict = st.checkbox("Conflict", value=False)

    submitted = st.form_submit_button("Submit log")


st.header("Results")

if submitted:
    # Payload shape mirrors the features used by the model/inference module.
    # If your backend stores logs, it can ignore extra fields or persist them.
    daily_log_payload: dict[str, Any] = {
        "log_date": log_date.isoformat(),
        "mood": mood,
        "stress": float(stress),
        "craving": float(craving),
        "sleep_hours": float(sleep_hours),
        "exercise_minutes": int(exercise_minutes),
        "social_interaction": int(social_interaction),
        "trigger_boredom": int(trigger_boredom),
        "trigger_loneliness": int(trigger_loneliness),
        "trigger_conflict": int(trigger_conflict),
        "days_since_last_relapse": int(days_since_last_relapse),
    }

    # These endpoints are intentionally simple conventions:
    # - POST /ml/predict returns {"risk_class": "...", "risk_probabilities": {...}}
    # - POST /recommendations returns {"risk_class": "...", "recommendations": [...]}
    #
    # If you name your routes differently, just update the two URL paths below.
    predict_url = f"{backend_url}/ml/predict"
    recs_url = f"{backend_url}/recommendations"

    try:
        pred = _post_json(predict_url, daily_log_payload)
        risk_class = str(pred.get("risk_class", "unknown"))
        display_risk_class = _format_risk_label(risk_class)

        st.subheader("Predicted risk")
        st.write(f"**{display_risk_class}**")

        probs = pred.get("risk_probabilities")
        if isinstance(probs, dict):
            st.caption("Probabilities")
            prob_rows = _normalize_probabilities(probs)
            if prob_rows:
                for label, prob in prob_rows:
                    left, right = st.columns([3, 1])
                    with left:
                        st.write(f"**{label}**")
                        st.progress(prob)
                    with right:
                        st.write(f"{prob * 100:.2f}%")
            else:
                st.write("No probability values available.")

        # Fetch recommendations using the predicted class (if the endpoint supports it).
        recs_payload = {
            "latest_log": daily_log_payload,
            "risk_level": risk_class,
        }
        recs_resp = _post_json(recs_url, recs_payload)

        st.subheader("Recommendations")
        top_disclaimer = recs_resp.get("disclaimer")
        if isinstance(top_disclaimer, str) and top_disclaimer.strip():
            st.caption(top_disclaimer)

        recs = recs_resp.get("recommendations")
        if isinstance(recs, list) and recs:
            for i, item in enumerate(recs, start=1):
                title = str(item.get("title", f"Suggestion {i}"))
                suggestion = str(item.get("suggestion", ""))
                explanation = str(item.get("explanation", ""))
                disclaimer = str(item.get("disclaimer", ""))

                st.markdown(f"**{i}. {title}**")
                if suggestion:
                    st.write(suggestion)
                if explanation:
                    st.caption(explanation)
                # Backwards compatibility: older backends may include per-item disclaimers.
                if (not top_disclaimer) and disclaimer:
                    st.caption(disclaimer)
                st.divider()
        else:
            st.info("No recommendations returned yet.")

    except (requests_exceptions.ConnectionError, requests_exceptions.Timeout) as e:
        st.error(
            "Could not connect to the backend.\n\n"
            "Start the API from the project root (with your venv activated):\n"
            "`uvicorn app.main:app --reload`\n\n"
            "Then confirm **Backend URL** in the sidebar matches the server (default `http://127.0.0.1:8000`)."
        )
        st.code(str(e))
    except RuntimeError as e:
        err_text = str(e)
        if "status=503" in err_text or "Missing model artifact" in err_text:
            st.error(
                "The API is running, but the trained model file is missing.\n\n"
                "From the project root, run:\n"
                "`python scripts/generate_synthetic_data.py --users 500 --days 60 --seed 42`\n"
                "`python scripts/train_model.py`\n\n"
                "Restart `uvicorn` after `train_model.py` writes `app/ml/artifacts/risk_model.joblib`."
            )
        else:
            st.error(
                "Couldn’t get results from the backend.\n\n"
                "Common causes:\n"
                "- The API isn’t running\n"
                "- The backend URL is incorrect\n"
                "- The request failed (see details below)\n"
            )
        st.code(err_text)
    except Exception as e:
        st.error(
            "Couldn’t get results from the backend.\n\n"
            "Common causes:\n"
            "- The API isn’t running\n"
            "- The backend routes aren’t implemented yet (`POST /ml/predict`, `POST /recommendations`)\n"
            "- The backend URL is incorrect\n"
        )
        st.code(str(e))
else:
    st.info("Submit a daily log to see predicted risk and recommendations.")


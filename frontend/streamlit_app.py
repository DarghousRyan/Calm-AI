"""
Calm AI Streamlit frontend.

Features:
- Daily check-in submission + risk prediction + recommendations.
- Side chat assistant powered by backend `/chat`.
- Past check-in history loaded from backend `/checkins`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import requests
from requests import exceptions as requests_exceptions
import streamlit as st


#DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_BACKEND_URL = "https://calm-ai.onrender.com"

def _format_timestamp_hhmm(ts: str) -> str:
    """Format timestamp as HH:MM."""
    if not ts:
        return "n/a"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y/%d/%m %-I:%M %p")
    except Exception:
        return ts

def _normalize_backend_url(raw: str) -> str:
    return raw.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 15.0) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        msg = f"{e} (status={resp.status_code})"
        try:
            msg += f" body={resp.text}"
        except Exception:
            pass
        raise RuntimeError(msg) from e
    return resp.json()


def _get_json(url: str, *, timeout_s: float = 15.0) -> dict[str, Any]:
    resp = requests.get(url, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        msg = f"{e} (status={resp.status_code})"
        try:
            msg += f" body={resp.text}"
        except Exception:
            pass
        raise RuntimeError(msg) from e
    return resp.json()


def _format_risk_label(raw: Any) -> str:
    return str(raw).strip().title()


def _normalize_probabilities(raw_probs: dict[str, Any]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for label, value in raw_probs.items():
        try:
            prob = float(value)
        except (TypeError, ValueError):
            continue
        prob = max(0.0, min(1.0, prob))
        rows.append((str(label).title(), prob))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_journey_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        raw_day = str(row.get("log_date", "")).strip()
        if not raw_day:
            continue
        try:
            day_obj = date.fromisoformat(raw_day)
        except ValueError:
            continue
        prepared.append(
            {
                "log_date": day_obj,
                "stress": _to_float(row.get("stress")),
                "craving": _to_float(row.get("craving")),
                "sleep_hours": _to_float(row.get("sleep_hours")),
                "exercise_minutes": _to_float(row.get("exercise_minutes")),
                "social_interaction": _to_float(row.get("social_interaction")),
                "days_since_last_relapse": _to_float(row.get("days_since_last_relapse")),
            }
        )
    prepared.sort(key=lambda x: x["log_date"])
    return prepared


def _plot_journey_chart(rows: list[dict[str, Any]], metric: str, label: str, color: str, y_label: str) -> None:
    x_vals: list[date] = []
    y_vals: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        x_vals.append(row["log_date"])
        y_vals.append(float(value))

    if not x_vals:
        st.info(f"No data available yet for {label}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5    ))
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, color=color)
    ax.set_title(label)
    ax.set_xlabel("Log date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.set_page_config(page_title="Calm AI", layout="centered")
st.title("Calm AI")
st.caption("Daily check-ins, supportive recommendations, and a side chatbot.")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7FAFC;
        color: #1F2937;
    }
    [data-testid="stSidebar"] {
        background-color: #EAF1FF;
        border-right: 1px solid #D5E4FF;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #2F6FED !important;
        border-bottom-color: #2F6FED !important;
        font-weight: 600;
    }
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE9FF;
        border-radius: 12px;
    }
    div.stButton > button {
        background-color: #2F6FED;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #2F6FED;
    }
    div.stButton > button:hover {
        background-color: #2457BA;
        border-color: #2457BA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi, I am here to listen. What is on your mind today?"}
    ]

with st.sidebar:
    st.header("Settings")
    backend_url = _normalize_backend_url(
        st.text_input("Backend URL", value=DEFAULT_BACKEND_URL, help="Where your FastAPI server is running.")
    )
    st.write("Tip: start the API with `uvicorn app.main:app --reload`.")

tab_checkin, tab_chat, tab_history, tab_journey = st.tabs(
    ["Check-in", "Chat", "Past check-ins", "Journey"]
)


with tab_checkin:
    st.header("Daily log")
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

        predict_url = f"{backend_url}/ml/predict"
        recs_url = f"{backend_url}/recommendations"
        checkins_url = f"{backend_url}/checkins"

        try:
            try:
                _post_json(checkins_url, daily_log_payload)
            except Exception as save_error:
                st.warning(f"Check-in was not saved to history: {save_error}")

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
            st.error("Could not get results from the backend.")
            st.code(str(e))
        except Exception as e:
            st.error("Could not get results from the backend.")
            st.code(str(e))
    else:
        st.info("Submit a daily log to see predicted risk and recommendations.")


with tab_chat:
    st.header("Chat assistant")
    st.caption("Talk with a supportive assistant powered by your configured backend provider.")

    messages_container = st.container()

    prompt = st.chat_input("Type your message...")
    with messages_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

       
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            chat_url = f"{backend_url}/chat"
            try:
                history_payload = [
                    {
                        "role": str(msg.get("role", "")),
                        "content": str(msg.get("content", "")),
                    }
                    for msg in st.session_state.chat_messages[-20:]
                ]
                chat_resp = _post_json(
                    chat_url,
                    {
                        "message": prompt,
                        "history": history_payload,
                    },
                    timeout_s=45.0,
                )
                reply = str(chat_resp.get("reply", "")).strip() or "I could not generate a response right now."
            except Exception as e:
                reply = f"Chat is unavailable right now: {e}"

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)


with tab_history:
    st.header("Past check-ins")
    st.caption("Most recent check-ins.")
    if st.button("Refresh history"):
        pass

    try:
        history_url = f"{backend_url}/checkins?limit=100"
        data = _get_json(history_url)
        rows = data.get("checkins", [])
        if not isinstance(rows, list) or not rows:
            st.info("No saved check-ins yet. Submit a daily log first.")
        else:
            for row in rows:
                mood = row.get("mood", "unknown")
                log_day = row.get("log_date", "")
                created_at = _format_timestamp_hhmm(str(row.get("created_at", "")))
                st.markdown(f"**{log_day} - {mood}**")
                c1, c2, c3 = st.columns(3)
                c1.write(f"Stress: {row.get('stress', 'n/a')}")
                c2.write(f"Craving: {row.get('craving', 'n/a')}")
                c3.write(f"Sleep: {row.get('sleep_hours', 'n/a')} hrs")
                st.caption(
                    "Exercise: "
                    f"{row.get('exercise_minutes', 'n/a')} min | "
                    "Social: "
                    f"{row.get('social_interaction', 'n/a')} min | "
                    "Days since relapse: "
                    f"{row.get('days_since_last_relapse', 'n/a')} | "
                    "Saved: "
                    f"{created_at}"
                )
                st.divider()
    except Exception as e:
        st.error("Could not load history from backend.")
        st.code(str(e))


with tab_journey:
    st.header("Journey trends")
    st.caption("Visualize how your wellness signals change over time.")
    if st.button("Refresh journey charts"):
        pass

    try:
        history_url = f"{backend_url}/checkins?limit=365"
        data = _get_json(history_url)
        rows = data.get("checkins", [])
        if not isinstance(rows, list) or not rows:
            st.info("No trend data yet. Submit check-ins to build your journey chart.")
        else:
            prepared_rows = _prepare_journey_rows(rows)
            if not prepared_rows:
                st.info("Trend data is present but missing valid log dates.")
            else:
                st.subheader("Key metrics over time")


        col1, col2 = st.columns(2)
        with col1:
            _plot_journey_chart(prepared_rows, "stress", "Stress trend", "#E76F51", "Stress (0-10)")
        with col2:
            _plot_journey_chart(prepared_rows, "craving", "Craving trend", "#F4A261", "Craving (0-10)")
        col3, col4 = st.columns(2)
        with col3:
            _plot_journey_chart(prepared_rows, "sleep_hours", "Sleep trend", "#2A9D8F", "Sleep (hours)")
        with col4:
            _plot_journey_chart(prepared_rows, "exercise_minutes", "Exercise trend", "#457B9D", "Exercise (minutes)")
        
        col5, col6 = st.columns(2)
        with col5:
            _plot_journey_chart(prepared_rows, "social_interaction", "Social interaction trend", "#7B2CBF", "Social interaction (minutes)")
        with col6:
            _plot_journey_chart(prepared_rows, "days_since_last_relapse", "Days since relapse trend", "#2B9348", "Days since last relapse")
    except Exception as e:
        st.error("Could not load journey visualization data.")
        st.code(str(e))

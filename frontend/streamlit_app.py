"""
Calm AI Streamlit frontend.

Features:
- Daily check-in submission + risk prediction + recommendations.
- Side chat assistant powered by backend `/chat`.
- Past check-in history loaded from backend `/checkins`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import requests
from requests import exceptions as requests_exceptions
from dotenv import load_dotenv
import streamlit as st
from streamlit_cookies_controller import CookieController, RemoveEmptyElementContainer


DEFAULT_BACKEND_URL = "https://calm-ai.onrender.com"
load_dotenv()

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


def _friendly_response_error(resp: requests.Response) -> str:
    """Turn backend/API errors into messages that make sense to users."""
    try:
        body = resp.json()
    except ValueError:
        body = None

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        detail = "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )
    detail = str(detail).strip() if detail else ""
    normalized = detail.lower()

    if resp.status_code == 400 and "invalid login credentials" in normalized:
        return "The email or password is incorrect. If you just created your account, confirm your email first."
    if resp.status_code == 429 or "rate limit" in normalized:
        return "Too many requests right now. Please wait a little while and try again."
    if resp.status_code == 401:
        return "Your session has expired. Please log in again."
    if resp.status_code == 403:
        return "You do not have permission to do that."
    if resp.status_code == 404:
        return "Calm AI could not find that service. Please try again later."
    if resp.status_code >= 500:
        return "Calm AI is having trouble right now. Please try again in a moment."
    if detail:
        return detail
    return "Calm AI could not complete that request. Please try again."


def _post_json(url: str, payload: dict[str, Any], *, token: str | None = None, timeout_s: float = 15.0) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(_friendly_response_error(resp)) from e
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError("Calm AI returned an unexpected response. Please try again.") from e


def _get_json(url: str, *, token: str | None = None, timeout_s: float = 15.0) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(_friendly_response_error(resp)) from e
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError("Calm AI returned an unexpected response. Please try again.") from e


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


def _render_checkin_result(result: dict[str, Any]) -> None:
    """Render the latest prediction and recommendations on every rerun."""
    pred = result.get("prediction") or {}
    recs_resp = result.get("recommendations") or {}
    risk_class = str(pred.get("risk_class", "unknown"))
    display_risk_class = _format_risk_label(risk_class)

    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(display_risk_class, "blue")
    with st.container(border=True):
        st.subheader("Estimated likelihood of returning to the habit", anchor=False, icon=":material/monitor_heart:")
        st.caption("A model estimate based on today’s check-in—not a diagnosis or certainty.")
        risk_col, context_col = st.columns([1, 2])
        with risk_col:
            st.badge(display_risk_class, icon=":material/insights:", color=risk_color)
        with context_col:
            st.write("This is a signal to help you choose a supportive next step, not a label for you.")

    probs = pred.get("risk_probabilities")
    if isinstance(probs, dict):
        with st.expander("See the model estimate details", icon=":material/bar_chart:"):
            st.caption("These percentages reflect the model’s estimated class probabilities.")
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

    st.subheader("A few supportive next steps", anchor=False, icon=":material/lightbulb:")
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


def _render_app_guide() -> None:
    """Explain the app's purpose and the first steps for new users."""
    if not st.session_state.get("show_app_guide", True):
        return

    with st.container(border=True):
        st.subheader("How Calm AI works", anchor=False, icon=":material/help_outline:")
        st.write(
            "Calm AI gives you a private place to reflect on daily habits, notice patterns, "
            "and choose a supportive next step."
        )

        step_one, step_two, step_three = st.columns(3)
        with step_one:
            st.markdown("**1. Check in**")
            st.caption("Record how you feel today, including stress, cravings, sleep, and possible triggers.")
        with step_two:
            st.markdown("**2. Review your reflection**")
            st.caption("Calm AI estimates how likely today’s pattern may be to lead back to the habit.")
        with step_three:
            st.markdown("**3. Keep building awareness**")
            st.caption("Use recommendations, past check-ins, journey trends, and chat to support your progress.")

        st.info(
            "The estimate is a guidance tool—not a diagnosis, a judgment, or a guarantee. "
            "You are always in control of what you share and what you do next.",
            icon=":material/health_and_safety:",
        )
        if st.button("Got it", icon=":material/check:"):
            st.session_state.show_app_guide = False
            st.rerun()


def _render_public_overview() -> None:
    """Explain Calm AI to visitors before asking them to create an account."""
    st.subheader("A private space to understand your patterns", anchor=False, icon=":material/self_improvement:")
    st.write(
        "Calm AI helps you reflect on daily habits, notice what may affect your progress, "
        "and choose a supportive next step. It is designed for ongoing self-awareness—not judgment."
    )

    reflect_col, notice_col, support_col = st.columns(3)
    with reflect_col:
        st.markdown("**Reflect**")
        st.caption("Complete a short daily check-in about your mood, sleep, stress, cravings, and triggers.")
    with notice_col:
        st.markdown("**Notice patterns**")
        st.caption("Review your past check-ins and journey trends to see how your experiences change over time.")
    with support_col:
        st.markdown("**Find support**")
        st.caption("Receive a plain-language estimate and practical suggestions for your next step.")

    st.info(
        "Calm AI is a reflection and habit-support tool. It is not a doctor, therapist, diagnosis, "
        "or guarantee about what will happen.",
        icon=":material/health_and_safety:",
    )


def _render_history_checkin(row: dict[str, Any]) -> None:
    """Render one saved check-in, including its recorded triggers."""
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
        f"{row.get('days_since_last_relapse', 'n/a')}"
    )

    trigger_labels = [
        label
        for label, key in (
            ("Boredom", "trigger_boredom"),
            ("Loneliness", "trigger_loneliness"),
            ("Conflict", "trigger_conflict"),
        )
        if row.get(key)
    ]
    custom_trigger = str(row.get("custom_trigger") or "").strip()
    if custom_trigger:
        trigger_labels.append(f"Other: {custom_trigger}")

    if trigger_labels:
        st.caption("Triggers: " + " · ".join(trigger_labels))
    else:
        st.caption("Triggers: None recorded")

    created_at = _format_timestamp_hhmm(str(row.get("created_at", "")))
    st.caption(f"Saved at {created_at}")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_journey_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare one chart point per date using that date's newest check-in."""
    latest_by_day: dict[str, dict[str, Any]] = {}
    # The API returns rows newest-first by date and creation time. Keeping the
    # first row for each date makes the chart represent the latest daily state.
    for row in rows:
        raw_day = str(row.get("log_date", "")).strip()
        if raw_day and raw_day not in latest_by_day:
            latest_by_day[raw_day] = row

    prepared: list[dict[str, Any]] = []
    for row in latest_by_day.values():
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
    st.pyplot(fig, width="stretch")
    plt.close(fig)


st.set_page_config(
    page_title="Calm AI",
    page_icon=":material/self_improvement:",
    layout="wide",
)
cookie_controller = CookieController(key="calm_ai_auth")
RemoveEmptyElementContainer()
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #F5F8FF 0%, #F8FBFA 55%, #FFF9F3 100%);
        color: #1F2937;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF1FF 0%, #F3F6FF 100%);
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
        box-shadow: 0 4px 16px rgba(57, 86, 128, 0.06);
    }
    div[data-testid="stForm"] {
        border-color: #DCE9FF;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(57, 86, 128, 0.06);
    }
    .calm-hero {
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.5rem;
        border: 1px solid #DCE9FF;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(236,244,255,0.85));
        box-shadow: 0 12px 32px rgba(57, 86, 128, 0.08);
    }
    .calm-kicker {
        color: #2F6FED;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .calm-hero h1 {
        color: #17233B;
        font-size: 2.6rem;
        margin: 0 0 0.5rem 0;
    }
    .calm-hero p {
        color: #63708A;
        font-size: 1.05rem;
        margin: 0;
    }
    .calm-section-note {
        color: #68758C;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
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

backend_url = _normalize_backend_url(os.environ.get("CALM_AI_BACKEND_URL", DEFAULT_BACKEND_URL))
st.session_state.setdefault("latest_checkin_result", None)
st.session_state.setdefault("checkin_submitting", False)

# Cookie components communicate with the browser asynchronously. Give the
# component a moment to return existing cookies after a full page refresh.
auth_cookies = cookie_controller.getAll() or {}
if not auth_cookies and not st.session_state.get("auth_cookie_checked"):
    time.sleep(0.5)
    auth_cookies = cookie_controller.getAll() or {}
st.session_state.auth_cookie_checked = True

if not st.session_state.get("auth_token"):
    stored_access_token = auth_cookies.get("calm_ai_access_token")
    stored_refresh_token = auth_cookies.get("calm_ai_refresh_token")
    stored_email = auth_cookies.get("calm_ai_user_email")

    if stored_access_token:
        try:
            me = _get_json(f"{backend_url}/auth/me", token=str(stored_access_token), timeout_s=10.0)
            st.session_state.auth_token = str(stored_access_token)
            st.session_state.user_email = str(me.get("email", stored_email or "user"))
        except (RuntimeError, requests_exceptions.RequestException):
            if stored_refresh_token:
                try:
                    refreshed = _post_json(
                        f"{backend_url}/auth/refresh",
                        {"refresh_token": str(stored_refresh_token)},
                        timeout_s=15.0,
                    )
                    refreshed_access = refreshed.get("access_token")
                    if refreshed_access:
                        st.session_state.auth_token = str(refreshed_access)
                        st.session_state.user_email = str(refreshed.get("email", stored_email or "user"))
                        cookie_controller.set("calm_ai_access_token", str(refreshed_access), max_age=7 * 24 * 60 * 60)
                        if refreshed.get("refresh_token"):
                            cookie_controller.set(
                                "calm_ai_refresh_token",
                                str(refreshed["refresh_token"]),
                                max_age=30 * 24 * 60 * 60,
                            )
                except (RuntimeError, requests_exceptions.RequestException):
                    cookie_controller.remove("calm_ai_access_token")
                    cookie_controller.remove("calm_ai_refresh_token")
                    cookie_controller.remove("calm_ai_user_email")

auth_token = st.session_state.get("auth_token")

if not auth_token:
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        st.markdown(
            """
            <div class="calm-hero">
                <div class="calm-kicker">A calmer place to begin</div>
                <h1>Welcome to Calm AI</h1>
                <p>A private space for daily check-ins, reflection, and supportive guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        overview_tab, account_tab = st.tabs(["About Calm AI", "Log in / Sign up"])

        with overview_tab:
            with st.container(border=True):
                _render_public_overview()

        with account_tab:
            with st.container(border=True):
                st.subheader("Your private wellness space", anchor=False, icon=":material/lock:")
                st.write("Create an account or log in to save your check-ins, journey, and conversations.")
                auth_mode = st.radio("Account action", ["Log in", "Create account"], horizontal=True)
                with st.form("auth_form", enter_to_submit=True):
                    email = st.text_input("Email address", autocomplete="email")
                    password = st.text_input(
                        "Password",
                        type="password",
                        help="Use at least 8 characters.",
                        autocomplete="current-password" if auth_mode == "Log in" else "new-password",
                    )
                    password_confirmation = ""
                    if auth_mode == "Create account":
                        password_confirmation = st.text_input(
                            "Re-type password",
                            type="password",
                            help="Re-enter the same password to confirm it.",
                            autocomplete="new-password",
                        )
                    submitted_auth = st.form_submit_button(
                        "Log in" if auth_mode == "Log in" else "Create account",
                        type="primary",
                        width="stretch",
                    )
                    if submitted_auth:
                        if auth_mode == "Create account" and password != password_confirmation:
                            st.error("The passwords do not match. Please re-type the same password in both fields.")
                        else:
                            endpoint = "/auth/login" if auth_mode == "Log in" else "/auth/register"
                            try:
                                with st.spinner("Creating your account..." if auth_mode == "Create account" else "Signing you in..."):
                                    auth = _post_json(
                                        f"{backend_url}{endpoint}",
                                        {"email": email, "password": password},
                                        timeout_s=60.0,
                                    )
                                access_token = auth.get("access_token")
                                if not access_token:
                                    st.success(str(auth.get("message", "Account created. Check your email, then log in.")))
                                else:
                                    st.session_state.auth_token = str(access_token)
                                    st.session_state.user_email = str(auth.get("email", email))
                                    cookie_controller.set(
                                        "calm_ai_access_token",
                                        str(access_token),
                                        max_age=7 * 24 * 60 * 60,
                                    )
                                    if auth.get("refresh_token"):
                                        cookie_controller.set(
                                            "calm_ai_refresh_token",
                                            str(auth["refresh_token"]),
                                            max_age=30 * 24 * 60 * 60,
                                        )
                                    cookie_controller.set(
                                        "calm_ai_user_email",
                                        str(auth.get("email", email)),
                                        max_age=30 * 24 * 60 * 60,
                                    )
                                    # Allow the browser component to finish writing the
                                    # cookies before Streamlit starts a fresh run.
                                    time.sleep(1.0)
                                    st.session_state.pop("chat_loaded", None)
                                    st.rerun()
                            except (requests_exceptions.ConnectionError, requests_exceptions.Timeout):
                                st.error("Calm AI is taking too long to respond. Please try again in a moment.")
                            except RuntimeError as e:
                                st.error(str(e))
                st.caption("Your check-ins, journey graphs, and conversations are saved privately to your account.")
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.get('user_email', 'user')}**")
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        for key in ("auth_token", "user_email", "chat_messages", "chat_loaded", "latest_checkin_result"):
            st.session_state.pop(key, None)
        cookie_controller.remove("calm_ai_access_token")
        cookie_controller.remove("calm_ai_refresh_token")
        cookie_controller.remove("calm_ai_user_email")
        st.rerun()

token = auth_token
st.markdown(
    """
    <div class="calm-hero">
        <div class="calm-kicker">Your private wellness space</div>
        <h1>Small check-ins. Steadier progress.</h1>
        <p>Notice what is affecting you today, then turn that insight into one supportive next step.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_app_guide()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if not st.session_state.get("chat_loaded"):
    try:
        saved_chat = _get_json(f"{backend_url}/chat/history", token=token)
        st.session_state.chat_messages = saved_chat.get("messages", []) or [
            {"role": "assistant", "content": "Hi, I am here to listen. What is on your mind today?"}
        ]
        st.session_state.chat_loaded = True
    except Exception as e:
        st.error(f"Could not load your chat history: {e}")

tab_checkin, tab_chat, tab_history, tab_journey = st.tabs(
    ["Check-in", "Chat", "Past check-ins", "Journey"]
)


with tab_checkin:
    st.header("Daily check-in", icon=":material/edit_note:")
    st.markdown(
        '<div class="calm-section-note">A quick snapshot of how today feels. There are no perfect answers.</div>',
        unsafe_allow_html=True,
    )
    with st.form("daily_log_form", clear_on_submit=False, border=True):
        col1, col2 = st.columns(2)

        with col1:
            log_date = st.date_input("Log date", value=date.today())
            mood = st.selectbox(
                "Mood",
                options=["Great", "Good", "Okay", "Down", "Bad"],
                index=2,
                help="Choose the option that best describes how you feel today.",
            )
            sleep_hours = st.number_input(
                "Sleep (hours)",
                min_value=0.0,
                max_value=16.0,
                value=7.0,
                step=0.25,
                help="Enter approximately how many hours you slept last night.",
            )
            exercise_minutes = st.number_input(
                "Exercise (minutes)",
                min_value=0,
                max_value=300,
                value=20,
                step=5,
                help="Include intentional movement or exercise from today.",
            )

        with col2:
            stress = st.slider(
                "Stress (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="0 means no stress; 10 means the highest stress you are experiencing today.",
            )
            craving = st.slider(
                "Craving (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.1,
                help="0 means no craving; 10 means the strongest craving you are experiencing today.",
            )
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
                help="How many days have passed since the last time you returned to the habit you are tracking.",
            )

        st.subheader("Triggers")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            trigger_boredom = st.checkbox("Boredom", value=False)
        with tcol2:
            trigger_loneliness = st.checkbox("Loneliness", value=False)
        with tcol3:
            trigger_conflict = st.checkbox("Conflict", value=False)
        custom_trigger = st.text_input(
            "Other trigger (optional)",
            max_chars=500,
            help="Describe another situation, feeling, or event that affected you today. This is saved with your check-in but does not change the estimate yet.",
        )

        submitted = st.form_submit_button(
            "Saving..." if st.session_state.checkin_submitting else "Submit log",
            disabled=st.session_state.checkin_submitting,
            icon=":material/arrow_forward:" if not st.session_state.checkin_submitting else ":material/hourglass_top:",
        )

    st.header("Your reflection", icon=":material/insights:")
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
            "custom_trigger": custom_trigger.strip() or None,
            "days_since_last_relapse": int(days_since_last_relapse),
        }

        predict_url = f"{backend_url}/ml/predict"
        recs_url = f"{backend_url}/recommendations"
        checkins_url = f"{backend_url}/checkins"

        st.session_state.checkin_submitting = True
        try:
            with st.spinner("Saving your check-in and preparing recommendations..."):
                try:
                    _post_json(checkins_url, daily_log_payload, token=token)
                except Exception as save_error:
                    st.warning(f"Your results are ready, but the check-in was not saved: {save_error}")

                pred = _post_json(predict_url, daily_log_payload, token=token)
                risk_class = str(pred.get("risk_class", "unknown"))

            recs_payload = {
                "latest_log": daily_log_payload,
                "risk_level": risk_class,
            }
            recs_resp = _post_json(recs_url, recs_payload, token=token)
            st.session_state.latest_checkin_result = {
                "prediction": pred,
                "recommendations": recs_resp,
            }

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
        finally:
            st.session_state.checkin_submitting = False

    if st.session_state.get("latest_checkin_result"):
        _render_checkin_result(st.session_state.latest_checkin_result)
    elif not submitted:
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
                    token=token,
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
        data = _get_json(history_url, token=token)
        rows = data.get("checkins", [])
        if not isinstance(rows, list) or not rows:
            st.info("No saved check-ins yet. Submit a daily log first.")
        else:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                log_day = str(row.get("log_date", "Unknown date"))
                grouped.setdefault(log_day, []).append(row)

            for log_day, day_rows in grouped.items():
                latest = day_rows[0]
                latest_mood = latest.get("mood", "unknown")
                with st.container(border=True):
                    st.markdown(f"**{log_day} · Latest check-in ({latest_mood})**")
                    _render_history_checkin(latest)

                    previous_rows = day_rows[1:]
                    if previous_rows:
                        with st.expander(
                            f"View {len(previous_rows)} earlier check-in{'s' if len(previous_rows) != 1 else ''} for {log_day}",
                            icon=":material/history:",
                        ):
                            for previous in previous_rows:
                                st.markdown(f"**Earlier check-in ({previous.get('mood', 'unknown')})**")
                                _render_history_checkin(previous)
                                st.divider()
    except Exception as e:
        st.error("Could not load history from backend.")
        st.code(str(e))


with tab_journey:
    st.header("Journey trends")
    st.caption("Visualize how your wellness signals change over time. Charts use your latest check-in for each day.")
    if st.button("Refresh journey charts"):
        pass

    try:
        history_url = f"{backend_url}/checkins?limit=365"
        data = _get_json(history_url, token=token)
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

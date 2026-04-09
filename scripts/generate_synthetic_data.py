"""
Generate semi-realistic synthetic daily wellness/risk logs.

Output:
  data/synthetic/daily_logs.csv

Design goals (not purely random):
- Low sleep increases stress and craving.
- Loneliness/conflict increases the probability of risk relapses.
- Risk relapses reset `days_since_last_relapse`.

The dataset is useful for local development and testing ML/business logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _sigmoid(x: float) -> float:
    # Numerically-stable sigmoid for typical score ranges.
    if x >= 0:
        z = np.exp(-x)
        return float(1 / (1 + z))
    z = np.exp(x)
    return float(z / (1 + z))


def _mood_from_score(score_0_10: float) -> str:
    # Map a 0..10 mood score to a small set of categories.
    if score_0_10 >= 7:
        return "great"
    if score_0_10 >= 5:
        return "good"
    if score_0_10 >= 3:
        return "okay"
    if score_0_10 >= 1:
        return "down"
    return "bad"


@dataclass(frozen=True)
class UserTraits:
    # Baselines per user (slowly changing / stable).
    baseline_stress: float  # 0..10
    sleep_pref_hours: float  # typical sleep duration
    social_baseline: float  # 0..100 "social energy"
    loneliness_tendency: float  # 0..1
    conflict_tendency: float  # 0..1
    relapse_tendency: float  # 0..1 (how relapse-prone this user is)

    # For realism we give each user a different initial relapse timer.
    initial_days_since_relapse: int


def _sample_user_traits(rng: np.random.Generator) -> UserTraits:
    baseline_stress = _clamp(rng.normal(5.0, 1.2), 1.0, 9.0)
    sleep_pref_hours = _clamp(rng.normal(7.2, 0.7), 5.0, 9.5)
    social_baseline = _clamp(rng.normal(55.0, 18.0), 0.0, 100.0)

    # Tendencies are bounded probabilities.
    loneliness_tendency = float(rng.beta(2.0, 5.0))
    conflict_tendency = float(rng.beta(2.3, 4.7))

    # Users differ in relapse propensity.
    relapse_tendency = float(rng.beta(1.6, 3.4))

    # Start "mid-stream" for variety.
    initial_days_since_relapse = int(rng.integers(0, 15))

    return UserTraits(
        baseline_stress=baseline_stress,
        sleep_pref_hours=sleep_pref_hours,
        social_baseline=social_baseline,
        loneliness_tendency=loneliness_tendency,
        conflict_tendency=conflict_tendency,
        relapse_tendency=relapse_tendency,
        initial_days_since_relapse=initial_days_since_relapse,
    )


def generate_daily_logs(
    num_users: int,
    num_days: int,
    *,
    start_date: date,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Pre-sample stable per-user traits for realism.
    users = [(_sample_user_traits(rng), uid) for uid in range(1, num_users + 1)]

    rows: list[dict[str, object]] = []

    for traits, user_id in users:
        days_since_last_relapse = traits.initial_days_since_relapse

        for day_idx in range(num_days):
            current_date = start_date + timedelta(days=day_idx)
            dow = current_date.weekday()  # 0=Mon .. 6=Sun
            weekend = 1.0 if dow >= 5 else 0.0

            # --- Sleep ---
            # Weekend sleep drift + random day-to-day noise.
            sleep_hours = traits.sleep_pref_hours + weekend * 0.4 + rng.normal(0, 0.6)

            # Occasional "bad sleep" events (more likely for stressed users).
            bad_sleep_event_p = 0.03 + (traits.baseline_stress / 10) * 0.03
            if rng.random() < bad_sleep_event_p:
                sleep_hours -= rng.uniform(0.8, 1.8)

            sleep_hours = _clamp(sleep_hours, 3.5, 10.5)

            # --- Stress + Craving ---
            # Core correlation: lower sleep => higher stress/craving.
            sleep_deficit = max(0.0, traits.sleep_pref_hours - sleep_hours)

            stress = (
                traits.baseline_stress
                + sleep_deficit * 1.15
                + rng.normal(0.0, 0.75)
            )

            # Stress can be aggravated by conflict trigger (determined later)
            # but we keep it simple by adding a small "latent" effect based on
            # conflict tendency.
            stress += traits.conflict_tendency * 0.4
            stress = _clamp(stress, 0.0, 10.0)

            craving = (
                0.9
                + stress * 0.55
                + sleep_deficit * 0.95
                + rng.normal(0.0, 0.7)
            )
            craving = _clamp(craving, 0.0, 10.0)

            # --- Social interaction (minutes) ---
            # Social interaction is weaker when loneliness triggers occur.
            # We will decide loneliness trigger using a probability model below.

            base_social = traits.social_baseline + rng.normal(0.0, 14.0) + (sleep_hours - 7.0) * 6.0
            social_interaction = _clamp(base_social, 0.0, 180.0)

            # --- Triggers (0/1) ---
            # These are used downstream to drive risk.
            p_lonely = 0.06 + traits.loneliness_tendency * 0.25 + (social_interaction < 35) * 0.06 + (sleep_hours < 6.2) * 0.05
            p_conflict = 0.05 + traits.conflict_tendency * 0.22 + (stress > 6.3) * 0.06 + (sleep_hours < 6.2) * 0.04
            p_boredom = 0.07 + (social_interaction < 35) * 0.08 + (sleep_hours < 6.0) * 0.05

            trigger_loneliness = int(rng.random() < _clamp(p_lonely, 0.0, 0.6))
            trigger_conflict = int(rng.random() < _clamp(p_conflict, 0.0, 0.6))
            trigger_boredom = int(rng.random() < _clamp(p_boredom, 0.0, 0.6))

            # Apply trigger effects after they're sampled.
            social_interaction -= trigger_loneliness * 25.0
            social_interaction -= trigger_conflict * 8.0
            social_interaction = _clamp(social_interaction, 0.0, 180.0)

            # --- Exercise ---
            # Exercise is lower when stress/craving is high and sleep is low.
            exercise_minutes = (
                95.0
                - stress * 8.0
                - craving * 2.2
                + (sleep_hours - 7.0) * 12.0
                + (social_interaction / 180.0) * 20.0
            )
            exercise_minutes -= trigger_boredom * 25.0
            exercise_minutes -= trigger_conflict * 10.0
            exercise_minutes += rng.normal(0.0, 12.0)
            exercise_minutes = int(_clamp(exercise_minutes, 0.0, 140.0))

            # --- Mood ---
            mood_score = (
                10.0
                - stress * 0.65
                - craving * 0.30
                + (exercise_minutes / 60.0) * 1.6
                + (social_interaction / 90.0) * 1.2
                + rng.normal(0.0, 0.35)
            )
            mood_score = _clamp(mood_score, 0.0, 10.0)
            mood = _mood_from_score(mood_score)

            # --- Risk label (relapse event) ---
            # Core correlation: loneliness/conflict increases risk.
            # We also include low-exercise/low-social as amplifiers.
            exercise_factor = exercise_minutes / 60.0  # 0..~2.3
            social_factor = social_interaction / 90.0  # 0..~2

            relapse_pressure = (
                0.26 * stress
                + 0.26 * craving
                + 1.15 * trigger_loneliness
                + 1.00 * trigger_conflict
                + 0.55 * trigger_boredom
                - 0.35 * exercise_factor
                - 0.25 * social_factor
            )

            # "Days since last relapse" lowers risk while it's small, then it saturates.
            # Users with higher relapse_tendency are more affected.
            decay = 1.0 / (1.0 + days_since_last_relapse / 3.0)
            relapse_pressure += traits.relapse_tendency * 1.6 * decay

            # Turn pressure into a probability via logistic mapping.
            # The constants are tuned for a moderate relapse frequency.
            risk_prob = _sigmoid(-4.0 + relapse_pressure * 0.85)
            risk_label = int(rng.random() < risk_prob)

            if risk_label == 1:
                days_since_last_relapse = 0
            else:
                days_since_last_relapse += 1

            rows.append(
                {
                    "user_id": int(user_id),
                    "log_date": current_date.isoformat(),
                    "mood": mood,
                    "stress": round(float(stress), 3),
                    "craving": round(float(craving), 3),
                    "sleep_hours": round(float(sleep_hours), 2),
                    "exercise_minutes": int(exercise_minutes),
                    "social_interaction": int(round(float(social_interaction))),
                    "trigger_boredom": int(trigger_boredom),
                    "trigger_loneliness": int(trigger_loneliness),
                    "trigger_conflict": int(trigger_conflict),
                    "days_since_last_relapse": int(days_since_last_relapse),
                    "risk_label": int(risk_label),
                }
            )

    df = pd.DataFrame(rows)

    # Keep column order stable for downstream consumers.
    df = df[
        [
            "user_id",
            "log_date",
            "mood",
            "stress",
            "craving",
            "sleep_hours",
            "exercise_minutes",
            "social_interaction",
            "trigger_boredom",
            "trigger_loneliness",
            "trigger_conflict",
            "days_since_last_relapse",
            "risk_label",
        ]
    ]

    return df.sort_values(["user_id", "log_date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD). Defaults to today - (days-1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/daily_logs.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.start_date is None:
        start = date.today() - timedelta(days=args.days - 1)
    else:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_daily_logs(
        num_users=args.users,
        num_days=args.days,
        start_date=start,
        seed=args.seed,
    )

    df.to_csv(output_path, index=False)
    print(
        f"Wrote {len(df):,} rows to {output_path} "
        f"(users={args.users}, days={args.days}, start={start.isoformat()})"
    )


if __name__ == "__main__":
    main()


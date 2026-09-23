#!/usr/bin/env python3
"""
IronSense - operator skill profiles + baselines (offline build).

Not a trained model: rolling statistics per operator, which is exactly why it can be
explained to an operator line by line. Two outputs:

  operator_profiles.json  - weekly skill score per operator (efficiency / compliance /
                            safety / smoothness), current level and trend
  operator_baselines.json - each operator's normal idle ratio, fuel per cycle, cycles per
                            hour and compliance; the anomaly detector's explanations compare
                            against these ("idle 55 min vs your usual 22")

Every component is scored 0-100 by percentile against *all* operators over the same period,
so the scale stays stable as the fleet changes.

Run:  .venv/bin/python backend/training/build_profiles.py
Out:  backend/artifacts/operator_profiles.json, backend/artifacts/operator_baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.profile import COMPONENT_WEIGHTS, COMPONENTS, level_for  # noqa: E402

DS = ROOT / "data" / "datasets"
ART = ROOT / "backend" / "artifacts"
MIN_WEEKS_FOR_TREND = 3
SMOOTH_WEEKS = 4        # rolling window before measuring a trend


def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 percentile rank; a flat column scores everyone 50 rather than dividing by zero."""
    if s.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=s.index)
    r = s.rank(pct=True, na_option="keep")
    return (r if higher_is_better else 1 - r) * 100


def weekly_frames() -> pd.DataFrame:
    """One row per operator-week with the four raw component measures."""
    tasks = pd.read_csv(DS / "task_records.csv", parse_dates=["start_ts"])
    tasks = tasks[tasks.source == "synthetic"].copy()
    tasks["week"] = tasks.start_ts.dt.tz_localize(None).dt.to_period("W").dt.start_time
    eff = tasks.groupby(["operator_id", "week"]).apply(
        lambda g: np.log(g.actual_time_min / g.estimated_time_min).median(), include_groups=False
    ).rename("overrun_log").reset_index()

    hourly = pd.read_csv(DS / "hourly_summaries.csv", parse_dates=["timestamp"])
    hourly["week"] = hourly.timestamp.dt.to_period("W").dt.start_time
    comp = hourly.groupby(["operator_id", "week"]).agg(
        seatbelt_compliance_pct=("seatbelt_compliance_pct", "mean"),
        idle_ratio=("idle_ratio", "median"),
        alerts_per_hour=("safety_alerts", "mean"),
        engine_hours=("engine_on_min", lambda s: s.sum() / 60),
    ).reset_index()

    win = pd.read_csv(DS / "anomaly_windows_5min.csv",
                      usecols=["operator_id", "timestamp", "engine_on_min", "harsh_brake_count",
                               "fast_swing_count", "over_rev_min", "overload_count"],
                      parse_dates=["timestamp"])
    win["week"] = win.timestamp.dt.to_period("W").dt.start_time
    win["rough"] = win.harsh_brake_count + win.fast_swing_count + win.over_rev_min + win.overload_count
    smooth = win.groupby(["operator_id", "week"]).apply(
        lambda g: g.rough.sum() / max(g.engine_on_min.sum() / 60, 0.1), include_groups=False
    ).rename("rough_per_hour").reset_index()

    inc = pd.read_csv(DS / "incidents.csv", parse_dates=["timestamp"])
    inc["week"] = inc.timestamp.dt.to_period("W").dt.start_time
    inc["weight"] = inc.severity.map({"critical": 4.0, "high": 2.0, "warning": 1.0}).fillna(1.0)
    incw = inc.groupby(["operator_id", "week"]).weight.sum().rename("incident_weight").reset_index()

    df = (eff.merge(comp, on=["operator_id", "week"], how="outer")
             .merge(smooth, on=["operator_id", "week"], how="outer")
             .merge(incw, on=["operator_id", "week"], how="left"))
    df["incident_weight"] = df.incident_weight.fillna(0.0)
    df["engine_hours"] = df.engine_hours.fillna(0.0)
    df["incidents_per_100h"] = 100 * df.incident_weight / df.engine_hours.clip(lower=1.0)
    return df[df.engine_hours >= 4].copy()          # ignore near-empty weeks


def score_components(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each raw measure into a 0-100 component score."""
    df = df.copy()
    df["efficiency"] = pct_rank(df.overrun_log, higher_is_better=False)
    df["compliance"] = 0.7 * pct_rank(df.seatbelt_compliance_pct) + 0.3 * pct_rank(df.idle_ratio, False)
    df["safety"] = 0.6 * pct_rank(df.incidents_per_100h, False) + 0.4 * pct_rank(df.alerts_per_hour, False)
    df["smoothness"] = pct_rank(df.rough_per_hour, higher_is_better=False)
    for c in COMPONENTS:
        df[c] = df[c].fillna(50.0)
    df["skill_score"] = sum(df[c] * w for c, w in COMPONENT_WEIGHTS.items())
    return df


def build() -> tuple[dict, dict]:
    weekly = score_components(weekly_frames()).sort_values(["operator_id", "week"])
    profiles, series = {}, {}

    for op, g in weekly.groupby("operator_id"):
        g = g.sort_values("week")
        # Recent weeks carry more weight than the whole history.
        recent = g.tail(4)
        comp = {c: round(float(np.average(recent[c], weights=np.linspace(1, 2, len(recent)))), 1)
                for c in COMPONENTS}
        score = round(float(sum(comp[c] * w for c, w in COMPONENT_WEIGHTS.items())), 1)

        # Trend = slope of the weekly series in points per week. A single week holds only a
        # handful of tasks, so the raw series swings 20+ points week to week and buries any
        # real trend; a 4-week rolling mean first is what makes the slope mean anything.
        # Applied identically to every operator and every component.
        trend, slope = "new", 0.0
        comp_slope = {}
        if len(g) >= MIN_WEEKS_FOR_TREND:
            smooth = g.skill_score.rolling(SMOOTH_WEEKS, min_periods=2).mean().dropna()
            x = np.arange(len(smooth), dtype=float)
            slope = float(np.polyfit(x, smooth.values, 1)[0])
            trend = "improving" if slope > 0.15 else "declining" if slope < -0.15 else "steady"
            # Component trends: someone can be getting faster while their compliance slips,
            # and the training hub needs to know which.
            for c in COMPONENTS:
                sm = g[c].rolling(SMOOTH_WEEKS, min_periods=2).mean().dropna()
                comp_slope[c] = round(float(np.polyfit(np.arange(len(sm), dtype=float), sm.values, 1)[0]), 3)

        profiles[op] = {
            "skill_score": score, "level": level_for(score), "trend": trend,
            "components": comp, "weeks_of_history": int(len(g)),
            "as_of": str(g.week.max().date()),
            "slope_per_week": round(slope, 3),
            "component_trend": comp_slope,
            "delta_vs_first_weeks": round(float(g.skill_score.tail(3).mean()
                                                - g.skill_score.head(3).mean()), 1),
        }
        series[op] = [{"week": str(w.date()), "skill_score": round(float(s), 1)}
                      for w, s in zip(g.week, g.skill_score)]

    hourly = pd.read_csv(DS / "hourly_summaries.csv")
    base = hourly.groupby("operator_id").agg(
        idle_ratio=("idle_ratio", "median"), idle_min_per_hour=("idling_time_min", "median"),
        fuel_per_cycle_l=("fuel_per_cycle_l", "median"),
        cycles_per_engine_hour=("cycles_per_engine_hour", "median"),
        seatbelt_compliance_pct=("seatbelt_compliance_pct", "median"),
    ).round(3)
    fleet = base.median().round(3).to_dict()
    baselines = {"fleet": fleet, "operators": base.to_dict("index"),
                 "note": "medians over the full history; used for anomaly explanations"}
    return {"operators": profiles, "weekly": series,
            "built_at": pd.Timestamp.now("UTC").isoformat()}, baselines


def main():
    ART.mkdir(parents=True, exist_ok=True)
    profiles, baselines = build()
    (ART / "operator_profiles.json").write_text(json.dumps(profiles, indent=2))
    (ART / "operator_baselines.json").write_text(json.dumps(baselines, indent=2))

    rows = [{"operator_id": k, **{kk: v[kk] for kk in ("skill_score", "level", "trend", "weeks_of_history",
                                                      "slope_per_week", "delta_vs_first_weeks")},
             "efficiency_slope": v["component_trend"].get("efficiency", 0.0)}
            for k, v in profiles["operators"].items()]
    df = pd.DataFrame(rows).sort_values("skill_score", ascending=False)
    print(df.to_string(index=False))

    truth = pd.read_csv(DS / "operators_ground_truth.csv")[["operator_id", "skill_level", "improvement_log"]]
    chk = df.merge(truth, on="operator_id")
    print("\nvalidation against the generator's hidden traits (never used as input):")
    print(chk.groupby("skill_level").skill_score.mean().round(1).to_string())
    print("\nlevel distribution:", df.level.value_counts().to_dict())
    improving = chk[chk.improvement_log < 0]
    print("\noperators the generator made improve (expect trend=improving, positive slopes):")
    print(improving[["operator_id", "trend", "slope_per_week", "efficiency_slope"]].to_string(index=False))


if __name__ == "__main__":
    main()

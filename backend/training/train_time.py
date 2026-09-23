#!/usr/bin/env python3
"""
IronSense - task time estimator training.

Predicts log(actual_time_min / estimated_time_min), i.e. a correction multiplier on the
planner's estimate, rather than raw minutes. That works with little data, generalises
across task types and stays explainable.

Two models, both saved:
  * linear  : Ridge on one-hot categoricals -> the `factors` list shown in the cab
              ("rain +10%", "intermediate operator +5%").
  * lgbm    : three LightGBM quantile models (alpha 0.1 / 0.5 / 0.9) -> p10/p50/p90.

Split is by time (train Nov-Mar, test April), never random: two beginners improve over
the six months, so a random split would overstate accuracy.

Run:  .venv/bin/python backend/training/train_time.py
Out:  backend/artifacts/task_time_model.joblib, backend/artifacts/task_time_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets" / "task_records.csv"
ART = ROOT / "backend" / "artifacts"

TEST_FROM = "2025-04-01"          # April is the held-out month
CAL_FROM  = "2025-03-01"          # March calibrates the p10-p90 band
ORGANIZER_WEIGHT = 20.0           # the 5 real rows count for more than one synthetic task

CAT_FEATURES = ["task_type", "weather", "operator_skill", "light", "ground_condition"]
NUM_FEATURES = ["estimated_time_min", "machine_age_yrs", "ambient_temp_c", "rain_mm_h",
                "wind_speed_kmh", "visibility_m", "haul_trucks_assigned",
                "operator_experience_yrs", "time_of_day_hour", "target_volume_m3",
                "spotter_present", "night_shift"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# Fallbacks for rows that only carry the organizers' six columns.
DEFAULTS = {
    "light": "day", "ground_condition": "dry", "ambient_temp_c": 28.0, "rain_mm_h": 0.0,
    "wind_speed_kmh": 12.0, "visibility_m": 8000.0, "haul_trucks_assigned": 2.0,
    "operator_experience_yrs": {"Beginner": 1.0, "Intermediate": 5.0, "Expert": 12.0},
    "time_of_day_hour": 10.0, "target_volume_m3": np.nan, "spotter_present": 1.0, "night_shift": 0.0,
}
WEATHER_GROUND = {"Rainy": "wet", "Storm": "muddy", "Dust": "loose"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the columns the organizers' sample rows don't carry, and coerce types."""
    df = df.copy()
    for c in ("spotter_present", "night_shift"):
        df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    df["ground_condition"] = df.ground_condition.fillna(df.weather.map(WEATHER_GROUND)).fillna(DEFAULTS["ground_condition"])
    df["operator_experience_yrs"] = df.operator_experience_yrs.fillna(
        df.operator_skill.map(DEFAULTS["operator_experience_yrs"]))
    for c, v in DEFAULTS.items():
        if c in ("operator_experience_yrs", "ground_condition") or isinstance(v, dict):
            continue
        df[c] = df[c].fillna(v)
    # target_volume_m3 is unknown for the organizer rows: approximate from the plan
    rate = df.groupby("task_type").apply(
        lambda g: (g.target_volume_m3 / g.estimated_time_min).median(), include_groups=False)
    df["target_volume_m3"] = df.target_volume_m3.fillna(df.task_type.map(rate) * df.estimated_time_min)
    df["y"] = np.log(df.actual_time_min / df.estimated_time_min)
    df["date"] = df.date.fillna("2025-04-15")  # organizer rows: keep them in training
    return df


def build_linear() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ])),
        ("reg", Ridge(alpha=1.0)),
    ])


def factor_table(pipe: Pipeline, train_df: pd.DataFrame) -> dict:
    """
    Effect of each categorical level as a percentage on the planned time, keyed by the real
    column name ("weather" -> {"Rainy": 7.5, ...}).

    OneHotEncoder(drop="first") measures every level against whichever level sorted first
    (Beginner, Cloudy, ...), which would read oddly in the cab. So each column is re-centred
    on its average level, weighted by how often each level actually occurs in training: a
    positive number means "slower than a typical job", which is what the operator wants to
    know. Frequency weighting matters -- a plain mean would let rare Storm rows drag the
    centre up until ordinary rain looked neutral.
    """
    ct: ColumnTransformer = pipe.named_steps["prep"]
    enc: OneHotEncoder = ct.named_transformers_["cat"]
    coefs = pipe.named_steps["reg"].coef_[:len(ct.get_feature_names_out()) - len(NUM_FEATURES)]
    out, i = {}, 0
    for col, cats in zip(CAT_FEATURES, enc.categories_):
        levels = {str(cats[0]): 0.0}  # the dropped reference level
        for level in cats[1:]:
            levels[str(level)] = float(coefs[i])
            i += 1
        freq = train_df[col].value_counts(normalize=True)
        centre = float(sum(v * freq.get(lv, 0.0) for lv, v in levels.items()))
        out[col] = {lv: round(float(np.exp(v - centre) - 1) * 100, 1) for lv, v in levels.items()}
    return out


def quantile_models(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, cats: list[str]) -> dict:
    models = {}
    for tag, alpha in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05,
                              num_leaves=31, min_child_samples=30, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, random_state=42, verbose=-1)
        m.fit(X, y, sample_weight=w, categorical_feature=cats)
        models[tag] = m
    return models


def calibrate(models: dict, X: pd.DataFrame, y: np.ndarray, target: float = 0.80) -> dict:
    """Shift p10 down and p90 up until the band covers `target` of held-out tasks."""
    lo_r = y - models["p10"].predict(X)   # how far below p10 the truth actually falls
    hi_r = y - models["p90"].predict(X)
    half = (1 - target) / 2
    lo_off = min(float(np.quantile(lo_r, half)), 0.0)
    hi_off = max(float(np.quantile(hi_r, 1 - half)), 0.0)
    return {"p10": lo_off, "p50": 0.0, "p90": hi_off}


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_csv(DATA))
    X_all = df[FEATURES].copy()
    for c in CAT_FEATURES:
        X_all[c] = X_all[c].astype("category")

    is_test = (df.date >= TEST_FROM) & (df.source == "synthetic")
    tr, te = ~is_test, is_test
    w = np.where(df.source == "organizer_sample", ORGANIZER_WEIGHT, 1.0)
    print(f"train {tr.sum():,} tasks (to {TEST_FROM})   test {te.sum():,} tasks (April)")

    # ---- linear (explanations)
    lin = build_linear()
    lin.fit(df.loc[tr, FEATURES], df.loc[tr, "y"], reg__sample_weight=w[tr.values])
    factors = factor_table(lin, df.loc[tr])

    # ---- lightgbm quantiles
    # Quantile models fit on their own training data come out over-confident, so the band is
    # calibrated on a held-out month (March) and the offsets are applied at predict time.
    cal = tr & (df.date >= CAL_FROM) & (df.date < TEST_FROM) & (df.source == "synthetic")
    fit_only = tr & ~cal
    cal_models = quantile_models(X_all[fit_only.values], df.loc[fit_only, "y"].values,
                                 w[fit_only.values], CAT_FEATURES)
    offsets = calibrate(cal_models, X_all[cal.values], df.loc[cal, "y"].values, target=0.80)
    print(f"calibration on {cal.sum():,} March tasks -> log offsets "
          f"p10 {offsets['p10']:+.3f}, p90 {offsets['p90']:+.3f}")

    models = quantile_models(X_all[tr.values], df.loc[tr, "y"].values, w[tr.values], CAT_FEATURES)

    # ---- evaluate on April
    Xte, dte = X_all[te.values], df[te]
    planned = dte.estimated_time_min.values
    actual = dte.actual_time_min.values
    pred = {k: planned * np.exp(m.predict(Xte) + offsets.get(k, 0.0)) for k, m in models.items()}
    lin_pred = planned * np.exp(lin.predict(dte[FEATURES]))

    def mae(p):
        return float(np.mean(np.abs(p - actual)))

    def mape(p):
        return float(np.mean(np.abs(p - actual) / actual) * 100)

    lo, hi = np.minimum(pred["p10"], pred["p90"]), np.maximum(pred["p10"], pred["p90"])
    metrics = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "test_from": TEST_FROM,
        "mae_min": {"planner_estimate": round(mae(planned), 2), "linear": round(mae(lin_pred), 2),
                    "lgbm_p50": round(mae(pred["p50"]), 2)},
        "mape_pct": {"planner_estimate": round(mape(planned), 2), "linear": round(mape(lin_pred), 2),
                     "lgbm_p50": round(mape(pred["p50"]), 2)},
        "p10_p90_coverage_pct": round(float(np.mean((actual >= lo) & (actual <= hi)) * 100), 1),
        "mean_interval_width_min": round(float(np.mean(hi - lo)), 1),
        "mae_by_skill": {}, "mae_by_weather": {},
    }
    for k, g in dte.groupby("operator_skill"):
        i = dte.operator_skill.values == k
        metrics["mae_by_skill"][k] = {"n": int(i.sum()), "planner": round(mae_sub(planned, actual, i), 2),
                                      "model": round(mae_sub(pred["p50"], actual, i), 2)}
    for k, g in dte.groupby("weather"):
        i = dte.weather.values == k
        metrics["mae_by_weather"][k] = {"n": int(i.sum()), "planner": round(mae_sub(planned, actual, i), 2),
                                        "model": round(mae_sub(pred["p50"], actual, i), 2)}

    joblib.dump({"lgbm": models, "linear": lin, "factors": factors, "features": FEATURES,
                 "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES, "defaults": DEFAULTS,
                 "weather_ground": WEATHER_GROUND, "offsets": offsets, "trained_at": pd.Timestamp.now('UTC').isoformat(),
                 "metrics": metrics}, ART / "task_time_model.joblib")
    (ART / "task_time_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: metrics[k] for k in ("mae_min", "mape_pct", "p10_p90_coverage_pct",
                                              "mean_interval_width_min")}, indent=2))
    print("\nlinear factors (effect on planned time, %):")
    print(json.dumps(factors, indent=2))
    print("\nMAE by skill (planner vs model):")
    for k, v in metrics["mae_by_skill"].items():
        print(f"  {k:<13} n={v['n']:<5} planner {v['planner']:>6}  model {v['model']:>6}")
    print("MAE by weather (planner vs model):")
    for k, v in metrics["mae_by_weather"].items():
        print(f"  {k:<13} n={v['n']:<5} planner {v['planner']:>6}  model {v['model']:>6}")


def mae_sub(pred, actual, mask):
    return float(np.mean(np.abs(pred[mask] - actual[mask])))


if __name__ == "__main__":
    main()

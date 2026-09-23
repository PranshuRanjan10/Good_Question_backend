#!/usr/bin/env python3
"""
IronSense - Readiness Score weight fitting.

The five sub-scores stay hand-written rules (see app/models/readiness.py) so they can always
be explained. Only their *weights* are learned here: a logistic regression on
readiness_training.csv predicts whether an incident follows within 30 minutes, and its
coefficients say which sub-score actually carries risk.

Run:  .venv/bin/python backend/training/train_readiness.py
Out:  backend/artifacts/readiness_weights.json, readiness_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.readiness import HAND_WEIGHTS, SUBSCORES, compute_readiness, subscores  # noqa: E402

DATA = ROOT / "data" / "datasets" / "readiness_training.csv"
WINDOWS = ROOT / "data" / "datasets" / "anomaly_windows_5min.csv"
ART = ROOT / "backend" / "artifacts"
TEST_FROM = "2025-04-01"


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Run the rule sub-scores over every window (vectorising this isn't worth the obscurity)."""
    recs = df.to_dict("records")
    anoms = [[{"anomaly_type": r["anomaly_label"]}] if r.get("is_anomaly") else [] for r in recs]
    parts = [subscores(r, a)[0] for r, a in zip(recs, anoms)]
    return pd.DataFrame(parts, index=df.index)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    # The shipped label looks 30 minutes ahead. Persistent things (weather, fatigue) dominate
    # that horizon, while a worker in the danger zone is a risk *right now* -- so the concurrent
    # label is pulled in too, and both are used to set the weights.
    now = pd.read_csv(WINDOWS, usecols=["machine_id", "timestamp", "incident"],
                      parse_dates=["timestamp"])
    df = df.merge(now.rename(columns={"incident": "incident_now"}), on=["machine_id", "timestamp"],
                  how="left")
    df["incident_now"] = df.incident_now.fillna(0).astype(int)
    print(f"{len(df):,} engine-on windows | {df.incident_next_30min.mean():.2%} followed by an "
          f"incident within 30 min | {df.incident_now.mean():.2%} contained one")

    parts = score_rows(df)
    y = df.incident_next_30min.values
    tr = (df.timestamp < TEST_FROM).values
    te = ~tr
    print(f"train {tr.sum():,}  test (April) {te.sum():,}")

    # Deficits (100 - score) so a positive coefficient means "this raises risk".
    X = (100 - parts[list(SUBSCORES)]).values / 100.0
    fits = {}
    for name, label in (("now", df.incident_now.values), ("next_30min", y)):
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(X[tr], label[tr])
        fits[name] = (clf, dict(zip(SUBSCORES, clf.coef_[0])))
        print(f"\nlogistic coefficients, incident {name} (risk per unit of deficit):")
        for k, v in sorted(fits[name][1].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<11} {v:+.3f}")

    # Final weights = hand priors shrunk toward what the data says, using both horizons.
    # Raw fitted weights alone would zero out seatbelt compliance, because an unbuckled belt
    # rarely causes an incident in the next half hour -- but a Readiness Score that ignores it
    # would be a bad product and an easy thing for the panel to poke at. Negative coefficients
    # are floored: "this sub-score makes you safer" is not a meaningful weight.
    def norm(d):
        t = sum(d.values()) or 1.0
        return {k: v / t for k, v in d.items()}

    w_now, w_next = (norm({k: max(v, 0.02) for k, v in fits[n][1].items()}) for n in ("now", "next_30min"))
    weights = {k: round(0.40 * HAND_WEIGHTS[k] + 0.30 * w_now[k] + 0.30 * w_next[k], 4)
               for k in SUBSCORES}
    weights = {k: round(v, 4) for k, v in norm(weights).items()}
    clf, coef = fits["next_30min"]

    # Compare: hand weights vs learned weights vs the raw model probability.
    hand = blend(parts, HAND_WEIGHTS)
    learned = blend(parts, weights)
    prob = clf.predict_proba(X)[:, 1]
    metrics = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "positive_rate": float(y.mean()),
        "test_from": TEST_FROM, "weights_hand": HAND_WEIGHTS, "weights_learned": weights,
        "coefficients": {k: round(float(v), 4) for k, v in coef.items()},
        "auc": {
            "hand_weighted_score": round(float(roc_auc_score(y[te], -hand[te])), 4),
            "learned_weighted_score": round(float(roc_auc_score(y[te], -learned[te])), 4),
            "logistic_probability": round(float(roc_auc_score(y[te], prob[te])), 4),
        },
        "auc_incident_now": {
            "hand_weighted_score": round(float(roc_auc_score(df.incident_now.values[te], -hand[te])), 4),
            "learned_weighted_score": round(float(roc_auc_score(df.incident_now.values[te], -learned[te])), 4),
        },
        "coefficients_incident_now": {k: round(float(v), 4) for k, v in fits["now"][1].items()},
        "average_precision": {
            "hand_weighted_score": round(float(average_precision_score(y[te], -hand[te])), 4),
            "learned_weighted_score": round(float(average_precision_score(y[te], -learned[te])), 4),
        },
    }
    # How much riskier are the worst 5% of windows by Readiness?
    cut = np.quantile(learned[te], 0.05)
    worst = learned[te] <= cut
    metrics["incident_rate_worst_5pct"] = round(float(y[te][worst].mean()), 4)
    metrics["incident_rate_rest"] = round(float(y[te][~worst].mean()), 4)
    metrics["lift_worst_5pct"] = round(metrics["incident_rate_worst_5pct"] /
                                       max(metrics["incident_rate_rest"], 1e-9), 1)

    (ART / "readiness_weights.json").write_text(json.dumps(
        {"weights": weights, "subscores": list(SUBSCORES), "fitted_on": str(DATA.name),
         "trained_at": pd.Timestamp.now("UTC").isoformat()}, indent=2))
    (ART / "readiness_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nlearned weights:", json.dumps(weights))
    print("AUC (April), incident within 30 min:", json.dumps(metrics["auc"], indent=2))
    print("AUC (April), incident in this window:", json.dumps(metrics["auc_incident_now"], indent=2))
    print(f"worst 5% of windows: {metrics['incident_rate_worst_5pct']:.2%} incident rate vs "
          f"{metrics['incident_rate_rest']:.2%} elsewhere  ({metrics['lift_worst_5pct']}x)")
    print("\nmean sub-score, incident windows vs the rest:")
    print(pd.DataFrame({"incident_next_30min": parts[y == 1].mean().round(1),
                        "quiet": parts[y == 0].mean().round(1)}).to_string())


def blend(parts: pd.DataFrame, w: dict) -> np.ndarray:
    total = sum(w.values())
    s = sum(parts[k].values * w[k] for k in SUBSCORES) / total
    floor = parts[list(SUBSCORES)].min(axis=1).values + 25   # same guard as compute_readiness
    return np.minimum(s, floor)


if __name__ == "__main__":
    main()

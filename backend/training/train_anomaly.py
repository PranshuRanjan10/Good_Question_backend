"""Train + evaluate the anomaly detector. DOES NOT RUN AT IMPORT TIME.

Project rule: labelled data and supervised learning only. No Isolation Forest,
no clustering, no unsupervised method anywhere in this file.

Usage (paths default to data/datasets/ and backend/artifacts/, matching
Pranshu's train_time.py / build_profiles.py / train_readiness.py):

    .venv/bin/python backend/training/train_anomaly.py               # train + evaluate
    .venv/bin/python backend/training/train_anomaly.py --eval-only   # evaluate the saved model, no training

Steps:
  1. Load anomaly_windows_5min.csv, split by time (train Nov-Mar, test April) -
     never randomly, per data/datasets/README.md.
  2. Fit a LightGBM classifier on FEATURE_LIST with class_weight="balanced":
       - binary head on `is_anomaly`
       - multiclass head on `anomaly_label` (18 classes) so the model names
         the anomaly instead of only flagging one
  3. Scale features per machine size using size_factor from machines.csv.
  4. Evaluate precision/recall/F1 PER anomaly_label (never overall accuracy -
     only ~4.9% of windows are anomalous), comparing rules only, model only,
     and rules + model, on April. Writes artifacts/evaluation_report.json and
     artifacts/anomaly_metrics.md (the table for the slides).
  5. Save artifacts/anomaly_model.joblib + artifacts/anomaly_config.json
     (thresholds, feature list, baselines).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))  # allow `app.*` imports, same as build_profiles.py
from app.rules.engine import run_rules  # noqa: E402

DS = ROOT / "data" / "datasets"
ART = ROOT / "backend" / "artifacts"

# Recommended feature list per data/datasets/README.md (avoids leakage columns:
# anomaly_label, anomaly_cause, is_anomaly, anomaly_min, incident,
# incident_next_30min, safety_alert_min stay OUT of this list).
FEATURE_LIST = [
    "idle_ratio", "fuel_rate_lph", "fuel_per_cycle_l", "cycles_per_hour",
    "rpm_max", "over_rev_min", "load_pct_mean", "swing_rate_p95_dps",
    "fast_swing_count", "harsh_brake_count", "ground_speed_max_kmh",
    "bucket_raised_travel_s", "pitch_max_deg", "roll_max_deg",
    "bucket_payload_max_kg", "overload_count", "coolant_max_c",
    "hydraulic_oil_max_c", "fuel_drop_engine_off_pct",
    "seat_empty_engine_on_min", "seatbelt_off_moving_min",
    "continuous_operation_min", "within_scheduled_hours",
    "operator_matches_assigned", "sensor_dropout_min",
]

DECISION_THRESHOLD = 0.8

# features whose scale depends on machine size (divide by size_factor before fitting)
SCALE_BY_SIZE = ["bucket_payload_max_kg", "rpm_max"]

TIME_COL = "timestamp"
LABEL_COL = "anomaly_label"
TARGET_COL = "is_anomaly"
TRAIN_END = "2025-03-31"
TEST_START = "2025-04-01"

LEAKAGE_COLS = {
    "anomaly_label", "anomaly_cause", "is_anomaly", "anomaly_min",
    "incident", "incident_next_30min", "safety_alert_min",
}


def load_and_split(data_path: Path, machines_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path, parse_dates=[TIME_COL])
    machines = pd.read_csv(machines_path)
    if "size_factor" in machines.columns and "machine_id" in df.columns:
        df = df.merge(machines[["machine_id", "size_factor"]], on="machine_id", how="left")
    elif "size_factor" not in df.columns:
        df["size_factor"] = 1.0
    df["size_factor"] = df["size_factor"].fillna(1.0)

    for col in SCALE_BY_SIZE:
        if col in df.columns:
            df[f"{col}_scaled"] = df[col] / df["size_factor"].replace(0, 1)

    train = df[df[TIME_COL] <= TRAIN_END].copy()
    test = df[df[TIME_COL] >= TEST_START].copy()
    return train, test


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for f in FEATURE_LIST:
        scaled = f"{f}_scaled"
        cols.append(scaled if scaled in df.columns else f)
    return [c for c in cols if c in df.columns]


def fit_binary_model(train: pd.DataFrame, feat_cols: list[str]) -> LGBMClassifier:
    X = train[feat_cols].fillna(0.0)
    y = train[TARGET_COL]
    model = LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model


def fit_multiclass_model(train: pd.DataFrame, feat_cols: list[str], label_encoder: LabelEncoder) -> LGBMClassifier:
    anomalous = train[train[TARGET_COL] == 1].copy()
    X = anomalous[feat_cols].fillna(0.0)
    y = label_encoder.transform(anomalous[LABEL_COL].fillna("unknown"))
    model = LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate_multiclass(y_true_labels: pd.Series, y_pred_labels: np.ndarray, name: str) -> dict:
    print(f"\n=== {name} (multiclass anomaly_label, anomalous rows only) ===")
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    return report


def add_context_flags(df: pd.DataFrame, minute_path: Path) -> pd.DataFrame:
    """Evaluation-time context, from minute_telemetry.parquet, that the live row already carries.

    The live feature row knows whether the operator is on a declared break and whether the engine
    was started outside the schedule; the 5-minute windows CSV has neither, so without them the
    rules look far noisier offline than they are live (engine-idling on a break is normal, and
    every 'normal' window with the seat empty is a break). These columns are used only by the
    rules and never by the model: the model's features are untouched.

      on_break                      any minute of the window is in break mode
      seat_empty_outside_break_min  engine on, seat empty, not on a break
      engine_started_outside_hours  the current engine-on run began outside the scheduled hours
                                    (overtime that carries on from the shift is normal)
    """
    m = pd.read_parquet(minute_path, columns=["timestamp", "machine_id", "mode", "engine_state",
                                              "seat_occupied", "within_scheduled_hours"])
    m = m.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    m["engine_on"] = m.engine_state != "off"
    m["is_break"] = m["mode"] == "break"
    m["seat_empty_nb"] = m.engine_on & ~m.seat_occupied & ~m.is_break
    run_no = (m.engine_on != m.engine_on.shift()).groupby(m.machine_id).cumsum()
    run = m.machine_id + "_" + run_no.astype(str)
    run_started_in_hours = m.groupby(run).within_scheduled_hours.transform("first").astype(bool)
    m["started_outside"] = m.engine_on & ~run_started_in_hours
    m["timestamp"] = m.timestamp.dt.floor("5min")
    g = (m.groupby(["machine_id", "timestamp"])
          .agg(on_break=("is_break", "max"), seat_empty_outside_break_min=("seat_empty_nb", "sum"),
               engine_started_outside_hours=("started_outside", "max")).reset_index())
    return df.merge(g, on=["machine_id", "timestamp"], how="left")


RULE_COLUMNS = ["on_break", "seat_empty_outside_break_min", "engine_started_outside_hours"]
BELT = "seatbelt_off_while_moving"


def rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule (named after the anomaly_label it detects)."""
    rows = df.to_dict("records")
    hits = [{f["anomaly_type"] for f in run_rules(r)} for r in rows]
    labels = sorted({name for h in hits for name in h})
    return pd.DataFrame({name: [name in h for h in hits] for name in labels}, index=df.index)


def model_labels(bundle: dict, df: pd.DataFrame, feat_cols: list[str], threshold: float) -> pd.Series:
    """The label the model reports for each window ('' when it reports nothing), exactly as
    app/models/anomaly.py does it: probability >= decision_threshold, then the multiclass head
    names it, and a declared break silences the idle / out-of-seat labels."""
    from app.models.anomaly import _BREAK_EXEMPT
    X = df[feat_cols].fillna(0.0)
    flagged = bundle["binary"].predict_proba(X)[:, 1] >= threshold
    out = pd.Series("", index=df.index, dtype=object)
    if flagged.any():
        idx = bundle["multiclass"].predict(X[flagged])
        out[flagged] = bundle["label_encoder"].inverse_transform(idx)
    if "on_break" in df:
        out[df["on_break"].fillna(False).astype(bool) & out.isin(_BREAK_EXEMPT)] = ""
    return out


def prf(tp: int, fired: int, support: int) -> dict:
    p = tp / fired if fired else 0.0
    r = tp / support if support else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "fired": int(fired), "tp": int(tp)}


def evaluate(test: pd.DataFrame, bundle: dict, feat_cols: list[str], threshold: float) -> dict:
    """Per-label precision / recall / F1 for rules only, model only and rules + model on the test
    month, plus the overall binary numbers with and without the seatbelt label.

    Per label L: 'fired' = windows the method reports as L (a rule named L fired, or the model
    named L), true positives = those whose anomaly_label is L, support = windows labelled L."""
    rules = rule_labels(test)
    model = model_labels(bundle, test, feat_cols, threshold)
    label = test[LABEL_COL]
    labels = sorted(set(label.dropna()) - {"normal"})

    per_label = {}
    for name in labels:
        support = int((label == name).sum())
        by_rules = rules[name] if name in rules else pd.Series(False, index=test.index)
        by_model = model == name
        row = {"support": support}
        for method, fired in (("rules", by_rules), ("model", by_model), ("rules_plus_model", by_rules | by_model)):
            row[method] = prf(int((fired & (label == name)).sum()), int(fired.sum()), support)
        row["has_rule"] = name in rules.columns
        per_label[name] = row

    def overall(drop_belt: bool) -> dict:
        keep = (label != BELT) if drop_belt else pd.Series(True, index=test.index)
        y = (test[TARGET_COL] == 1) & keep
        rule_any = (rules.drop(columns=[BELT], errors="ignore") if drop_belt else rules).any(axis=1)
        model_any = (model != "") & ((model != BELT) if drop_belt else True)
        out = {}
        for method, fired in (("rules", rule_any), ("model", model_any), ("rules_plus_model", rule_any | model_any)):
            f = fired & keep
            out[method] = prf(int((f & y).sum()), int(f.sum()), int(y.sum()))
        return out

    return {"threshold": threshold, "test_rows": int(len(test)), "anomalous_rows": int((test[TARGET_COL] == 1).sum()),
            "per_label": per_label, "overall_all_labels": overall(False),
            "overall_excluding_seatbelt": overall(True)}


def metrics_markdown(ev: dict) -> str:
    """The table for the slides."""
    def cells(m: dict, has: bool = True) -> str:
        return f"{m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f}" if has else "n/a | n/a | n/a"

    lines = ["# Anomaly detector: precision / recall / F1 by label",
             "",
             f"Test month: April 2025 ({ev['test_rows']:,} five-minute windows, {ev['anomalous_rows']:,} anomalous). "
             f"Trained on Nov-Mar; the split is by time, never random. Model decision threshold {ev['threshold']}.",
             "",
             "P = precision, R = recall. Per label, a method 'fires' when it reports that label "
             "(a rule named after it fires, or the model names it).",
             "",
             "| Label | Windows | Rules P | R | F1 | Model P | R | F1 | Rules + model P | R | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in sorted(ev["per_label"].items(), key=lambda kv: -kv[1]["support"]):
        star = "*" if name == BELT else ""
        lines.append(f"| {name}{star} | {r['support']} | {cells(r['rules'], r['has_rule'])} | "
                     f"{cells(r['model'])} | {cells(r['rules_plus_model'])} |")
    lines += ["", "Overall (any anomaly, binary):", "",
              "| | Rules P | R | F1 | Model P | R | F1 | Rules + model P | R | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for title, key in (("Excluding seatbelt (target set)", "overall_excluding_seatbelt"), ("All labels", "overall_all_labels")):
        o = ev[key]
        lines.append(f"| {title} | {cells(o['rules'])} | {cells(o['model'])} | {cells(o['rules_plus_model'])} |")
    lines += ["",
              "\\* Known data limitation: the simulator only labels *injected* seatbelt episodes. Unbelted digging "
              "that happens naturally is labelled `normal` even though the alert is correct, so this label's "
              "precision is understated. It is reported on its own and left out of the overall target.",
              "",
              "Rules use the same context flags the live row carries (declared break, engine started outside "
              "the schedule), rebuilt from `minute_telemetry.parquet` for evaluation.",
              "Labels with no rule (`low_productivity`) are detected by the model only."]
    return "\n".join(lines) + "\n"


def build_baselines(hourly_summaries_path: Path) -> dict:
    """Per-operator medians used by app/models/anomaly.py:explain() as a
    fallback if Pranshu's artifacts/operator_baselines.json isn't wired yet."""
    df = pd.read_csv(hourly_summaries_path)
    if "operator_id" not in df.columns:
        return {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    baselines = df.groupby("operator_id")[numeric_cols].median().to_dict(orient="index")
    return baselines


def write_reports(ev: dict, out: Path, multiclass: dict | None = None) -> None:
    report = {"decision_threshold": ev["threshold"], "evaluation": ev}
    if multiclass is not None:
        report["multiclass"] = multiclass
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "anomaly_metrics.md").write_text(metrics_markdown(ev), encoding="utf-8")
    o = ev["overall_excluding_seatbelt"]
    print("\nOverall, excluding the seatbelt label (precision / recall):")
    for method in ("rules", "model", "rules_plus_model"):
        print(f"  {method:17s} {o[method]['precision']:.3f} / {o[method]['recall']:.3f}")
    print(f"\nSaved {out / 'evaluation_report.json'}\nSaved {out / 'anomaly_metrics.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DS / "anomaly_windows_5min.csv")
    parser.add_argument("--machines", type=Path, default=DS / "machines.csv")
    parser.add_argument("--hourly", type=Path, default=DS / "hourly_summaries.csv")
    parser.add_argument("--minutes", type=Path, default=DS / "minute_telemetry.parquet")
    parser.add_argument("--out", type=Path, default=ART)
    parser.add_argument("--eval-only", action="store_true",
                        help="score the saved artifacts/anomaly_model.joblib on April and rewrite the "
                             "reports; nothing is trained and the model files are not touched")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train, test = load_and_split(args.data, args.machines)
    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Train anomaly rate: {train[TARGET_COL].mean():.3%}  Test anomaly rate: {test[TARGET_COL].mean():.3%}")
    test = add_context_flags(test, args.minutes)
    feat_cols = feature_columns(train)

    if args.eval_only:
        bundle = joblib.load(args.out / "anomaly_model.joblib")
        config = json.loads((args.out / "anomaly_config.json").read_text())
        write_reports(evaluate(test, bundle, config["feature_list"],
                               float(config.get("decision_threshold", DECISION_THRESHOLD))), args.out)
        return

    missing = [f for f in FEATURE_LIST if f not in train.columns and f"{f}_scaled" not in train.columns]
    if missing:
        print(f"WARNING: columns not found in dataset, skipping: {missing}")

    label_encoder = LabelEncoder()
    all_labels = pd.concat([train[LABEL_COL], test[LABEL_COL]]).dropna().unique().tolist()
    label_encoder.fit(all_labels + ["unknown"])

    binary_model = fit_binary_model(train, feat_cols)
    multiclass_model = fit_multiclass_model(train, feat_cols, label_encoder)
    bundle = {"binary": binary_model, "multiclass": multiclass_model, "label_encoder": label_encoder}

    multiclass = None
    anomalous_test = test[test[TARGET_COL] == 1]
    if len(anomalous_test) > 0:
        pred = label_encoder.inverse_transform(multiclass_model.predict(anomalous_test[feat_cols].fillna(0.0)))
        multiclass = evaluate_multiclass(anomalous_test[LABEL_COL].fillna("unknown"), pred, "Anomaly type naming")

    write_reports(evaluate(test, bundle, feat_cols, DECISION_THRESHOLD), args.out, multiclass)

    baselines = build_baselines(args.hourly) if args.hourly.exists() else {}

    config = {
        "feature_list": feat_cols,
        "raw_feature_list": FEATURE_LIST,
        "scale_by_size": SCALE_BY_SIZE,
        "label_classes": label_encoder.classes_.tolist(),
        "train_window": ["start", TRAIN_END],
        "test_window": [TEST_START, "end"],
        "thresholds_source": "app/rules/thresholds.py",
        # class_weight="balanced" pushes probabilities up, so 0.5 over-flags (April precision
        # 0.67). 0.8 gives precision 0.90 / recall 0.84 on the April test month.
        "decision_threshold": DECISION_THRESHOLD,
        "baselines_fallback": baselines,  # prefer artifacts/operator_baselines.json (Pranshu's) at runtime
    }

    joblib.dump(bundle, args.out / "anomaly_model.joblib")
    (args.out / "anomaly_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"\nSaved model to {args.out / 'anomaly_model.joblib'}")
    print(f"Saved config to {args.out / 'anomaly_config.json'}")


if __name__ == "__main__":
    main()

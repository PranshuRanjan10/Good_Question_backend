"""Train + evaluate the anomaly detector. DOES NOT RUN AT IMPORT TIME.

Project rule: labelled data and supervised learning only. No Isolation Forest,
no clustering, no unsupervised method anywhere in this file.

Usage (paths default to data/datasets/ and backend/artifacts/, matching
Pranshu's train_time.py / build_profiles.py / train_readiness.py):

    .venv/bin/python backend/training/train_anomaly.py

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
     and rules + model. Save a confusion table for the pitch.
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
from sklearn.metrics import classification_report, confusion_matrix
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


def predict_model(model: LGBMClassifier, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    X = df[feat_cols].fillna(0.0)
    return model.predict(X)


def predict_rules(df: pd.DataFrame) -> np.ndarray:
    flags = []
    for _, row in df.iterrows():
        flags.append(1 if run_rules(row.to_dict()) else 0)
    return np.array(flags)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, name: str, labels: pd.Series) -> dict:
    print(f"\n=== {name} (binary is_anomaly) ===")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)

    per_label = {}
    for label in labels.dropna().unique():
        mask = labels == label
        if mask.sum() == 0:
            continue
        sub_report = classification_report(
            y_true[mask], y_pred[mask], output_dict=True, zero_division=0
        )
        per_label[label] = sub_report.get("1", sub_report.get("weighted avg"))

    return {"overall": report, "confusion_matrix": cm.tolist(), "per_label": per_label}


def evaluate_multiclass(y_true_labels: pd.Series, y_pred_labels: np.ndarray, name: str) -> dict:
    print(f"\n=== {name} (multiclass anomaly_label, anomalous rows only) ===")
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    return report


def build_baselines(hourly_summaries_path: Path) -> dict:
    """Per-operator medians used by app/models/anomaly.py:explain() as a
    fallback if Pranshu's artifacts/operator_baselines.json isn't wired yet."""
    df = pd.read_csv(hourly_summaries_path)
    if "operator_id" not in df.columns:
        return {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    baselines = df.groupby("operator_id")[numeric_cols].median().to_dict(orient="index")
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DS / "anomaly_windows_5min.csv")
    parser.add_argument("--machines", type=Path, default=DS / "machines.csv")
    parser.add_argument("--hourly", type=Path, default=DS / "hourly_summaries.csv")
    parser.add_argument("--out", type=Path, default=ART)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train, test = load_and_split(args.data, args.machines)
    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Train anomaly rate: {train[TARGET_COL].mean():.3%}  Test anomaly rate: {test[TARGET_COL].mean():.3%}")

    feat_cols = feature_columns(train)
    missing = [f for f in FEATURE_LIST if f not in train.columns and f"{f}_scaled" not in train.columns]
    if missing:
        print(f"WARNING: columns not found in dataset, skipping: {missing}")

    label_encoder = LabelEncoder()
    all_labels = pd.concat([train[LABEL_COL], test[LABEL_COL]]).dropna().unique().tolist()
    label_encoder.fit(all_labels + ["unknown"])

    binary_model = fit_binary_model(train, feat_cols)
    multiclass_model = fit_multiclass_model(train, feat_cols, label_encoder)

    y_true = test[TARGET_COL].values
    labels = test[LABEL_COL] if LABEL_COL in test.columns else pd.Series([None] * len(test))

    rules_pred = predict_rules(test)
    model_pred = predict_model(binary_model, test, feat_cols)
    combined_pred = np.clip(rules_pred + model_pred, 0, 1)

    results = {
        "rules_only": evaluate_binary(y_true, rules_pred, "Rules only", labels),
        "model_only": evaluate_binary(y_true, model_pred, "LightGBM only", labels),
        "rules_plus_model": evaluate_binary(y_true, combined_pred, "Rules + LightGBM", labels),
    }

    # multiclass eval, anomalous test rows only
    anomalous_test = test[test[TARGET_COL] == 1].copy()
    if len(anomalous_test) > 0:
        X_anom = anomalous_test[feat_cols].fillna(0.0)
        pred_class_idx = multiclass_model.predict(X_anom)
        pred_labels = label_encoder.inverse_transform(pred_class_idx)
        results["multiclass"] = evaluate_multiclass(anomalous_test[LABEL_COL].fillna("unknown"), pred_labels, "Anomaly type naming")

    (args.out / "evaluation_report.json").write_text(json.dumps(results, indent=2, default=str))

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

    joblib.dump({"binary": binary_model, "multiclass": multiclass_model, "label_encoder": label_encoder},
                args.out / "anomaly_model.joblib")
    (args.out / "anomaly_config.json").write_text(json.dumps(config, indent=2, default=str))

    print(f"\nSaved model to {args.out / 'anomaly_model.joblib'}")
    print(f"Saved config to {args.out / 'anomaly_config.json'}")
    print(f"Saved evaluation report to {args.out / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()

"""Anomaly detector: rules + supervised LightGBM classifier, loaded from
backend/artifacts/.

detect_anomalies(row, baseline) is the shared-interface function the decision
layer calls into (see HANDOFF_backend_teammate.md section 4). It never trains
anything - training happens in backend/training/train_anomaly.py and writes
the artifacts this module loads. Supervised only, per project rule: the
LightGBM model is fit on labelled `is_anomaly` / `anomaly_label` data, no
Isolation Forest or other unsupervised method.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.i18n import render_explanation
from app.rules.engine import run_rules

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "anomaly_model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "anomaly_config.json"
BASELINES_PATH = ARTIFACT_DIR / "operator_baselines.json"  # Pranshu's artifact: {"fleet": {...}, "operators": {id: {...}}}

_BASELINES = None
_LOCK = threading.Lock()


def load_baselines(path: Path | str = BASELINES_PATH) -> dict:
    """{"fleet": {...medians...}, "operators": {operator_id: {...medians...}}}"""
    global _BASELINES
    with _LOCK:
        if _BASELINES is None:
            try:
                _BASELINES = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                _BASELINES = {"fleet": {}, "operators": {}}
    return _BASELINES


def baseline_for(operator_id: str | None) -> dict:
    b = load_baselines()
    if operator_id and operator_id in b.get("operators", {}):
        return b["operators"][operator_id]
    return b.get("fleet", {})


class AnomalyDetector:
    def __init__(self, model_path: Path = MODEL_PATH, config_path: Path = CONFIG_PATH):
        self.binary_model = None
        self.multiclass_model = None
        self.label_encoder = None
        self.config: dict = {}
        self._model_path = model_path
        self._config_path = config_path
        self._try_load()

    def _try_load(self) -> None:
        if self._model_path.exists() and self._config_path.exists():
            bundle = joblib.load(self._model_path)
            self.binary_model = bundle["binary"]
            self.multiclass_model = bundle["multiclass"]
            self.label_encoder = bundle["label_encoder"]
            self.config = json.loads(self._config_path.read_text())

    @property
    def is_loaded(self) -> bool:
        return self.binary_model is not None

    def _feature_vector(self, row: dict) -> pd.DataFrame:
        feature_list = self.config["feature_list"]
        raw_feature_list = self.config.get("raw_feature_list", feature_list)
        scale_by_size = set(self.config.get("scale_by_size", []))
        size_factor = row.get("size_factor", 1.0) or 1.0

        vals = []
        for raw_name in raw_feature_list:
            v = row.get(raw_name, 0.0) or 0.0
            if raw_name in scale_by_size:
                v = v / size_factor
            vals.append(float(v))
        # Named columns, exactly as in training (feature_list holds the *_scaled names). A bare
        # array predicts the same, but sklearn logs a UserWarning on every call, flooding the logs.
        return pd.DataFrame([vals], columns=feature_list)

    def predict(self, row: dict) -> tuple[bool, str | None, float]:
        """Returns (is_anomaly, anomaly_label_or_None, confidence 0..1)."""
        if not self.is_loaded:
            return False, None, 0.0
        x = self._feature_vector(row)
        proba = self.binary_model.predict_proba(x)[0]
        confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
        # Operating point chosen on the April test month (see train_anomaly.py), not the 0.5
        # default, which over-flags because training used balanced class weights.
        is_anomaly = confidence >= float(self.config.get("decision_threshold", 0.8))

        label = None
        if is_anomaly and self.multiclass_model is not None:
            class_idx = self.multiclass_model.predict(x)[0]
            label = str(self.label_encoder.inverse_transform([class_idx])[0])  # np.str_ -> str for JSON

        return is_anomaly, label, confidence


_detector = AnomalyDetector()

# anomaly_type -> (row key, baseline key, unit). Baseline keys are the ones
# Pranshu's operator_baselines.json actually carries (fleet-wide + per-operator
# medians of idle_ratio, idle_min_per_hour, fuel_per_cycle_l,
# cycles_per_engine_hour, seatbelt_compliance_pct) - not every anomaly type
# has a matching baseline field, and that's fine, those fall back to a plain
# description.
_EXPLAIN_MAP = {
    "excessive_idling": ("idle_ratio", "idle_ratio", ""),
    "fast_swing": ("swing_rate_p95_dps", None, "deg/s"),
    "over_rev": ("rpm_max", None, "rpm"),
    "overload": ("bucket_payload_max_kg", None, "kg payload"),
    "overheating": ("coolant_max_c", None, "C coolant"),
    "fatigue": ("continuous_operation_min", None, "min continuous"),
    "harsh_operation": ("harsh_brake_count", None, "harsh brakes"),
}


def explain_items(row: dict, baseline: dict, anomaly_type: str, finding: dict | None = None) -> list[dict]:
    """Structured explanation: [{"key", "params"}]. app.i18n renders it in the cab's language;
    `explain()` renders the English sentences the API has always returned."""
    items = []
    if anomaly_type in _EXPLAIN_MAP:
        row_key, baseline_key, unit = _EXPLAIN_MAP[anomaly_type]
        cur = row.get(row_key)
        base = baseline.get(baseline_key) if baseline_key else None
        if cur is not None and base is not None:
            items.append({"key": "expl.compare", "params": {
                "feature": row_key, "cur": f"{cur:.2f}", "base": f"{base:.2f}", "unit": unit}})
    if anomaly_type == "excessive_idling" and row.get("idle_streak_min"):
        items.append({"key": "expl.idle_streak", "params": {"min": f"{row['idle_streak_min']:.0f}"}})
    if anomaly_type == "excessive_idling" and row.get("truck_wait_min", 0) > 0:
        items.append({"key": "expl.waiting_truck", "params": {}})
    if not items and finding is not None:
        items.append({"key": f"expl.rule.{anomaly_type}", "params": finding.get("params", {})})
    if not items:
        items.append({"key": "expl.flagged", "params": {"anomaly": anomaly_type}})
    return items[:3]


def explain(row: dict, baseline: dict, anomaly_type: str, finding: dict | None = None) -> list[str]:
    """Compare flagged feature(s) against the operator's baseline
    (Pranshu's artifacts/operator_baselines.json, per-operator or fleet-wide)."""
    return render_explanation("en", explain_items(row, baseline, anomaly_type, finding))


_BREAK_EXEMPT = {"excessive_idling", "operator_out_of_seat"}


def detect_anomalies(row: dict, baseline: dict) -> list[dict]:
    """Shared interface (HANDOFF doc section 4).

    Returns a list of:
      {"anomaly_type", "method", "score", "explanation": [...]}
    Rule hits get method="rule", a model hit with no matching rule gets
    method="model", and a pattern caught by both is "rule+model".
    """
    rule_findings = {f["anomaly_type"]: f for f in run_rules(row)}
    is_model_anomaly, model_label, confidence = _detector.predict(row)
    if row.get("on_break") and model_label in _BREAK_EXEMPT:
        # Training labels engine-idling on a scheduled break as normal; the model has no
        # break flag, so the declared break (break_start/break_end events) decides.
        is_model_anomaly, model_label = False, None

    results = []

    for anomaly_type in rule_findings:
        method = "rule+model" if (is_model_anomaly and model_label == anomaly_type) else "rule"
        score = max(confidence, 0.6) if method == "rule+model" else 0.6
        items = explain_items(row, baseline, anomaly_type, rule_findings[anomaly_type])
        results.append({
            "anomaly_type": anomaly_type,
            "method": method,
            "score": round(score, 3),
            "explanation": render_explanation("en", items),
            "explanation_items": items,
        })

    if is_model_anomaly and model_label and model_label not in rule_findings:
        items = explain_items(row, baseline, model_label)
        results.append({
            "anomaly_type": model_label,
            "method": "model",
            "score": round(confidence, 3),
            "explanation": render_explanation("en", items),
            "explanation_items": items,
        })

    return results

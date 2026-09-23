"""Rule thresholds, from frontend_handoff_spec_v1.md section 7 (realistic
value ranges table) and datasets/README.md.

Zone widening (danger/caution radius) lives in app/models/readiness.py:
danger_radius_m() instead of here, so the Readiness Score and the cab
alerts in app/rules/safety.py always agree on where "danger" is.
"""

THRESHOLDS = {
    "idle_min": 20,
    "idle_ratio": 0.6,
    "rpm_max": 2100,
    "swing_deg_s_max": 55,
    "speed_kmh_max": 6,
    "pitch_roll_deg_max": 15,
    "coolant_c_max": 105,
    "continuous_op_min_max": 240,
    # Fuel gauges read +/-0.15%: normal engine-off windows show up to ~0.9% of noise, theft
    # windows start at ~1.2%. Anything over 0 fired on noise.
    "fuel_drop_off_pct": 1.0,
    # Brief raises while repositioning are normal (training: median 8 s, p95 28 s per window);
    # the violation is sustained travel with the bucket up (25-235 s).
    "bucket_raised_travel_s": 30,
}

PAYLOAD_KG_PER_SIZE_FACTOR = 2000  # payload > 2000 kg * size_factor

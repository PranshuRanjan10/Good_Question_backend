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
    # Brief raises while repositioning are normal (training: median 8 s, p95 28 s per window);
    # the violation is sustained travel with the bucket up (25-235 s).
    "bucket_raised_travel_s": 30,
    # Normal windows briefly touch 2100-2199 rpm (sensor noise): >2100 fired on 8,000 of them.
    # A real over-rev is a sustained one: >=2 min above 2100 with the peak past 2200
    # (Nov-Mar: precision 0.97 / recall 0.99; April 0.89 / 1.00).
    "over_rev_peak_rpm": 2200,
    "over_rev_min": 2,
    # Gauge noise is ~1.5% at the low end; theft windows above 1.5% (Nov-Mar 0.97 / 0.94, April 1.00 / 1.00).
    "fuel_drop_off_pct": 1.5,
    # Ordinary braking gives 1-4 harsh brakes in a window; harsh operation shows 5+
    # (Nov-Mar and April: precision ~0.96, recall ~0.99). Needing a fast swing too caught none.
    "harsh_brake_count": 5,
}

PAYLOAD_KG_PER_SIZE_FACTOR = 2000  # payload > 2000 kg * size_factor

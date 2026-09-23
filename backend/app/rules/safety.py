"""Fast-path safety rules -> `alerts` in the `assessment` message (spec section 10).
Runs off the live feature row + machine state, separate from the offline-trained
anomaly model. Zone widening mirrors app/models/readiness.py:danger_radius_m so
the Readiness Score and the cab alerts always agree on where "danger" is.
"""

from __future__ import annotations
from app.models.readiness import danger_radius_m

CAUTION_MARGIN_M = 7.0  # spec sample: rain -> danger 8, caution 15
BLIND_SPOT_BEARING = (135.0, 225.0)


def zone_radii(row: dict) -> dict:
    danger = danger_radius_m(row)
    return {"danger_radius_m": danger, "caution_radius_m": danger + CAUTION_MARGIN_M,
            "reason": _zone_reason(row)}


def _zone_reason(row: dict) -> str | None:
    reasons = []
    weather = row.get("weather")
    if weather in ("Rainy", "Fog", "Storm", "Dust"):
        reasons.append(weather.lower())
    if row.get("light") == "night":
        reasons.append("low visibility")
    return ", ".join(reasons) or None


def proximity_view(machine_state) -> list[dict]:
    """The `proximity_view` block of `assessment`: every currently-tracked object
    with the zone/blind-spot flags the backend (not the sim) computes."""
    if not machine_state.latest_proximity:
        return []
    row = machine_state.to_feature_row()
    danger = danger_radius_m(row)
    caution = danger + CAUTION_MARGIN_M
    out = []
    for obj in machine_state.latest_proximity.objects:
        if obj.distance_m < danger:
            zone = "red"
        elif obj.distance_m < caution:
            zone = "amber"
        else:
            zone = "green"
        in_blind_spot = BLIND_SPOT_BEARING[0] <= obj.bearing_deg <= BLIND_SPOT_BEARING[1]
        out.append({
            "object_id": obj.object_id, "object_type": obj.object_type,
            "distance_m": obj.distance_m, "bearing_deg": obj.bearing_deg,
            "zone": zone, "in_blind_spot": in_blind_spot,
        })
    return out


def check_proximity_alerts(machine_state) -> list[dict]:
    if not machine_state.latest_proximity:
        return []
    row = machine_state.to_feature_row()
    danger = danger_radius_m(row)
    findings = []
    for obj in machine_state.latest_proximity.objects:
        if obj.object_type != "person" or obj.distance_m >= danger:
            continue
        blind = BLIND_SPOT_BEARING[0] <= obj.bearing_deg <= BLIND_SPOT_BEARING[1]
        findings.append({
            "alert_type": "proximity_person_blind_spot" if blind else "proximity_person_danger",
            "severity": "critical",
            "message": f"Worker {obj.distance_m:.1f} m away" + (", in blind spot" if blind else ""),
            "recommended_action": "Stop swing and sound horn" if blind else "Stop and check surroundings",
            "adjusted_threshold_note": _zone_reason(row) and f"Danger zone widened to {danger:.0f} m ({_zone_reason(row)})",
        })
    return findings


def check_seatbelt(machine_state) -> dict | None:
    """Escalation ladder: warning (0-20s unbelted) -> alarm (20-60s) -> logged_violation (>60s)."""
    op = machine_state.latest_operation
    if op is None or op.cab.seatbelt_fastened is not False:
        return None
    if (op.motion.ground_speed_kmh or 0) <= 0.5:
        return None
    elapsed_min = 0.0
    for ts, tick in reversed(machine_state.operation_ticks):
        if tick.cab.seatbelt_fastened is False:
            elapsed_min = (op.timestamp - ts).total_seconds() / 60.0
        else:
            break
    if elapsed_min > 1.0:
        return {"alert_type": "seatbelt_off_moving", "severity": "critical",
                "message": "Seatbelt off while moving - logged as a violation",
                "recommended_action": "Stop and fasten seatbelt", "adjusted_threshold_note": None}
    if elapsed_min > 20.0 / 60.0:
        return {"alert_type": "seatbelt_off_moving", "severity": "high",
                "message": "Seatbelt still off while moving",
                "recommended_action": "Fasten seatbelt now", "adjusted_threshold_note": None}
    return {"alert_type": "seatbelt_off_moving", "severity": "warning",
            "message": "Seatbelt off while moving",
            "recommended_action": "Fasten seatbelt", "adjusted_threshold_note": None}


def check_tilt(row: dict) -> dict | None:
    pitch, roll = abs(row.get("pitch_max_deg", 0) or 0), abs(row.get("roll_max_deg", 0) or 0)
    if pitch > 15 or roll > 15:
        return {"alert_type": "tilt_warning", "severity": "critical",
                "message": f"Pitch/roll {max(pitch, roll):.0f} deg exceeds 15 deg slope limit",
                "recommended_action": "Reposition to level ground", "adjusted_threshold_note": None}
    return None


def check_lightning(row: dict) -> dict | None:
    dist = row.get("lightning_distance_km")
    if dist is not None and 0 < dist < 10:
        return {"alert_type": "lightning_nearby", "severity": "critical",
                "message": f"Lightning {dist:.0f} km away: stop work",
                "recommended_action": "Stop work and move to shelter", "adjusted_threshold_note": None}
    return None


def check_refuel_engine_on(machine_state) -> dict | None:
    row = machine_state.to_feature_row()
    if row.get("engine_on") and row.get("fuel_rising"):
        return {"alert_type": "refuel_engine_on", "severity": "critical",
                "message": "Fuel level rising while engine is running",
                "recommended_action": "Stop engine before refuelling", "adjusted_threshold_note": None}
    return None


def check_geofence(machine_state) -> dict | None:
    """Looks for an unresolved geofence_enter into a no_go_zone with no matching exit."""
    if not machine_state.shift_context:
        return None
    no_go_ids = {z.zone_id for z in machine_state.shift_context.zones if z.zone_type == "no_go_zone"}
    entered, exited = None, None
    for ts, e in machine_state.events:
        zone_id = e.details.get("zone_id")
        if zone_id not in no_go_ids:
            continue
        if e.event_type == "geofence_enter":
            entered = ts
        elif e.event_type == "geofence_exit":
            exited = ts
    if entered and (exited is None or exited < entered):
        return {"alert_type": "geofence_violation", "severity": "high",
                "message": "Machine inside a no-go zone",
                "recommended_action": "Move out of the restricted zone", "adjusted_threshold_note": None}
    return None


def run_safety_checks(machine_state) -> list[dict]:
    row = machine_state.to_feature_row()
    findings = check_proximity_alerts(machine_state)
    for check in (check_seatbelt(machine_state), check_tilt(row), check_lightning(row),
                  check_refuel_engine_on(machine_state), check_geofence(machine_state)):
        if check:
            findings.append(check)
    return findings

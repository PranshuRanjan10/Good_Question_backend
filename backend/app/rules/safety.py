"""Fast-path safety rules -> `alerts` in the `assessment` message (spec section 10).

These read the *latest* raw readings (current proximity scan, last few seconds of motion),
not the 5-minute feature row: a worker behind the machine or a tilt past 15 degrees has to
show now and clear when it ends. Zone widening mirrors app/models/readiness.py:danger_radius_m
so the Readiness Score and the cab alerts always agree on where "danger" is.

Every finding carries a private `_key` (alert type + object) that the decision layer uses to
keep one stable alert_id while the condition lasts. It is stripped before sending.
"""

from __future__ import annotations
from app.i18n import t
from app.models.readiness import danger_radius_m

CAUTION_MARGIN_M = 7.0  # spec sample: rain -> danger 8, caution 15
BLIND_SPOT_BEARING = (135.0, 225.0)
TILT_LIMIT_DEG = 15.0
LIGHTNING_STOP_KM = 10.0


def _alert(alert_type: str, severity: str, params: dict | None = None, variant: str | None = None,
           note: dict | None = None, key: str | None = None) -> dict:
    """English text comes from the catalogue; `params` (+ variant / note) ride along so the cab
    hub can re-render the alert in the operator's language (app/i18n)."""
    params = dict(params or {})
    if variant:
        params["variant"] = variant
    if note:
        params["note"] = note
    cat = f"{alert_type}.{variant}" if variant else alert_type
    fmt = {k: v for k, v in params.items() if k not in ("variant", "note")}
    return {"alert_type": alert_type, "severity": severity,
            "message": t("en", f"{cat}.msg", **fmt),
            "recommended_action": t("en", f"{cat}.act", **fmt),
            "adjusted_threshold_note": (t("en", "note.zone_widened", radius=note["radius"],
                                          reason=", ".join(t("en", f"reason.{r}") for r in note["reasons"]))
                                        if note else None),
            "params": params, "_key": key or alert_type}


def zone_radii(row: dict) -> dict:
    danger = danger_radius_m(row)
    return {"danger_radius_m": danger, "caution_radius_m": danger + CAUTION_MARGIN_M,
            "reason": _zone_reason(row)}


def _zone_reasons(row: dict) -> list[str]:
    reasons = []
    weather = row.get("weather")
    if weather in ("Rainy", "Fog", "Storm", "Dust"):
        reasons.append(weather.lower())
    if row.get("light") == "night":
        reasons.append("low visibility")
    return reasons


def _zone_reason(row: dict) -> str | None:
    return ", ".join(_zone_reasons(row)) or None


def _in_blind_spot(bearing: float) -> bool:
    return BLIND_SPOT_BEARING[0] <= bearing <= BLIND_SPOT_BEARING[1]


def proximity_view(machine_state, row: dict | None = None) -> list[dict]:
    """The `proximity_view` block of `assessment`: every currently-tracked object
    with the zone/blind-spot flags the backend (not the sim) computes."""
    scan = machine_state.fresh_proximity()
    if not scan:
        return []
    row = row or machine_state.to_feature_row()
    danger = danger_radius_m(row)
    caution = danger + CAUTION_MARGIN_M
    out = []
    for obj in scan.objects:
        zone = "red" if obj.distance_m < danger else "amber" if obj.distance_m < caution else "green"
        out.append({
            "object_id": obj.object_id, "object_type": obj.object_type,
            "distance_m": obj.distance_m, "bearing_deg": obj.bearing_deg,
            "zone": zone, "in_blind_spot": _in_blind_spot(obj.bearing_deg),
        })
    return out


def check_proximity_alerts(machine_state, row: dict) -> list[dict]:
    scan = machine_state.fresh_proximity()
    if not scan:
        return []
    danger = danger_radius_m(row)
    reasons = _zone_reasons(row)
    note = {"radius": f"{danger:.0f}", "reasons": reasons} if reasons else None
    findings = []
    for obj in scan.objects:
        if obj.object_type != "person" or obj.distance_m >= danger:
            continue
        blind = _in_blind_spot(obj.bearing_deg)
        alert_type = "proximity_person_blind_spot" if blind else "proximity_person_danger"
        findings.append(_alert(alert_type, "critical", {"distance": f"{obj.distance_m:.1f}"},
                               note=note, key=f"proximity:{obj.object_id}"))
    return findings


def check_seatbelt(machine_state) -> dict | None:
    """Escalation ladder while working or travelling unbelted:
    warning (0-20 s) -> high alarm (20-60 s) -> critical logged violation (> 60 s).
    'Working' includes digging in place, not only driving."""
    op = machine_state.latest_operation
    if op is None or op.cab.seatbelt_fastened is not False or not op.cab.seat_occupied:
        return None
    if not machine_state.is_moving():
        return None
    since = op.timestamp
    for ts, tick in reversed(machine_state.operation_ticks):
        if tick.cab.seatbelt_fastened is False:
            since = ts
        else:
            break
    elapsed_s = (op.timestamp - since).total_seconds()
    if elapsed_s > 60:
        return _alert("seatbelt_off_moving", "critical", variant="critical")
    if elapsed_s > 20:
        return _alert("seatbelt_off_moving", "high", variant="high")
    return _alert("seatbelt_off_moving", "warning", variant="warning")


def check_out_of_cab(machine_state) -> dict | None:
    """Scenario 3: operator left the seat with the engine running (not on a declared break)."""
    op = machine_state.latest_operation
    if op is None or not machine_state.engine_running() or op.cab.seat_occupied is not False:
        return None
    if machine_state.on_break:
        return None
    return _alert("operator_out_of_cab", "high")


def check_tilt(machine_state) -> dict | None:
    recent = machine_state.recent_motion(15)
    if not recent:
        return None
    pitch = max(abs(r.get("pitch_deg") or 0) for r in recent)
    roll = max(abs(r.get("roll_deg") or 0) for r in recent)
    if max(pitch, roll) > TILT_LIMIT_DEG:
        return _alert("tilt_warning", "critical",
                      {"deg": f"{max(pitch, roll):.0f}", "limit": f"{TILT_LIMIT_DEG:.0f}"})
    return None


def check_lightning(row: dict, machine_state=None) -> dict | None:
    """From the environment reading, or a lightning_nearby event in the last 10 sim minutes
    (the event arrives immediately; the environment message may be up to 15 min away)."""
    dist = row.get("lightning_distance_km")
    if machine_state is not None:
        now = machine_state.sim_now()
        for ts, e in reversed(machine_state.events):
            if e.event_type == "lightning_nearby" and (now - ts).total_seconds() <= 600:
                ev = (e.details or {}).get("distance_km")
                if ev is not None and (dist is None or ev < dist):
                    dist = float(ev)
                break
    if dist is not None and 0 < dist < LIGHTNING_STOP_KM:
        return _alert("lightning_nearby", "critical", {"km": f"{dist:.0f}"})
    return None


def check_refuel_engine_on(row: dict) -> dict | None:
    if row.get("engine_on") and row.get("fuel_rising"):
        return _alert("refuel_engine_on", "critical")
    return None


def check_geofence(machine_state) -> dict | None:
    """Looks for an unresolved geofence_enter into a no_go_zone with no matching exit."""
    if not machine_state.shift_context:
        return None
    no_go_ids = {z.zone_id for z in machine_state.shift_context.zones if z.zone_type == "no_go_zone"}
    entered, exited, zone = None, None, None
    for ts, e in machine_state.events:
        zone_id = (e.details or {}).get("zone_id")
        if zone_id not in no_go_ids:
            continue
        if e.event_type == "geofence_enter":
            entered, zone = ts, zone_id
        elif e.event_type == "geofence_exit":
            exited = ts
    if entered and (exited is None or exited < entered):
        return _alert("geofence_violation", "high", {"zone": zone}, key=f"geofence:{zone}")
    return None


def run_safety_checks(machine_state, row: dict | None = None) -> list[dict]:
    row = row or machine_state.to_feature_row()
    findings = check_proximity_alerts(machine_state, row)
    for check in (check_seatbelt(machine_state), check_out_of_cab(machine_state), check_tilt(machine_state),
                  check_lightning(row, machine_state), check_refuel_engine_on(row), check_geofence(machine_state)):
        if check:
            findings.append(check)
    return findings

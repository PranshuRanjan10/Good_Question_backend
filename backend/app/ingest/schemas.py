"""Pydantic models for the 7 inbound message types, per
docs/frontend_handoff_spec_v1.md section 5 (the spec wins over
telemetry_contract_v1.json where they differ - the .md uses split
per-purpose messages, not one flat "telemetry" tick).
"""

from __future__ import annotations
from typing import Literal, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field

# ---- shared enums (telemetry_contract_v1.json "enums") ----
MachineType = Literal["excavator", "wheel_loader", "backhoe_loader", "dozer", "dump_truck", "roller"]
SkillLevel = Literal["Beginner", "Intermediate", "Expert"]
TaskType = Literal["Earth Excavation", "Trenching", "Material Loading", "Grading", "Demolition"]
EngineState = Literal["off", "idle", "running"]
TravelDirection = Literal["forward", "reverse", "stationary"]
WorkMode = Literal["idle", "dig", "swing_loaded", "dump", "swing_empty", "travel", "grade", "break"]
Weather = Literal["Sunny", "Cloudy", "Rainy", "Windy", "Storm", "Fog", "Extreme Heat", "Dust"]
Light = Literal["day", "dusk", "night"]
GroundCondition = Literal["dry", "wet", "muddy", "loose"]
ObjectType = Literal["person", "vehicle", "machine", "static_obstacle"]
PersonRole = Literal["labourer", "spotter", "surveyor", "supervisor", "visitor", "public"]
SensorType = Literal["radar", "camera", "gps_tag"]
EventType = Literal[
    "engine_start", "engine_stop", "seatbelt_fastened", "seatbelt_unfastened",
    "operator_left_seat", "operator_returned", "geofence_enter", "geofence_exit",
    "harsh_brake", "collision", "tilt_warning", "fault_code", "refuel_start",
    "refuel_end", "task_start", "task_complete", "break_start", "break_end",
    "manual_incident", "manual_near_miss", "walkaround_completed", "weather_change",
    "lightning_nearby",  # spec section 6 (scenario 6)
]
EventSource = Literal["sensor", "operator", "director_console"]


class _Base(BaseModel):
    msg_type: str
    schema_version: str = "1.0"
    timestamp: datetime
    sim_tick: int
    machine_id: str


# ---- 5.1 shift_context ----

class MachineInfo(BaseModel):
    machine_id: str
    model: str
    machine_type: MachineType
    machine_age_yrs: float


class OperatorInfo(BaseModel):
    operator_id: str
    skill_level: SkillLevel
    shift_start: datetime


class DailyTask(BaseModel):
    task_id: str
    task_type: TaskType
    zone_id: Optional[str] = None
    planned_estimate_min: float
    target_volume_m3: Optional[float] = None
    scheduled_start: Optional[datetime] = None


class Zone(BaseModel):
    zone_id: str
    zone_type: str
    polygon: list[list[float]]


class Hazard(BaseModel):
    hazard_id: str
    hazard_type: str
    line: list[list[float]]
    clearance_height_m: Optional[float] = None


class WalkaroundResult(BaseModel):
    completed: bool
    issues: list[str] = Field(default_factory=list)


class ShiftContextMessage(_Base):
    msg_type: Literal["shift_context"] = "shift_context"
    time_scale: int = 1
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    site_id: Optional[str] = None
    machine: MachineInfo
    operator: OperatorInfo
    daily_tasks: list[DailyTask] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    hazards: list[Hazard] = Field(default_factory=list)
    walkaround_checklist_result: Optional[WalkaroundResult] = None


# ---- 5.2 operation ----

class EngineOp(BaseModel):
    state: EngineState
    rpm: Optional[float] = None
    load_pct: Optional[float] = None
    fuel_rate_lph: Optional[float] = None


class Position(BaseModel):
    x_m: float
    y_m: float


class MotionOp(BaseModel):
    position: Position
    heading_deg: Optional[float] = None
    ground_speed_kmh: Optional[float] = None
    travel_direction: Optional[TravelDirection] = None
    parking_brake: Optional[bool] = None
    hydraulic_lockout: Optional[bool] = None
    travel_alarm_active: Optional[bool] = None


class ImplementOp(BaseModel):
    work_mode: Optional[WorkMode] = None
    hydraulic_pressure_bar: Optional[float] = None


class CabOp(BaseModel):
    seat_occupied: Optional[bool] = None
    seatbelt_fastened: Optional[bool] = None
    door_open: Optional[bool] = None
    controls_active: Optional[bool] = None


class OperationMessage(_Base):
    msg_type: Literal["operation"] = "operation"
    operator_id: Optional[str] = None
    task_id: Optional[str] = None
    engine: EngineOp
    motion: MotionOp
    implement: ImplementOp
    cab: CabOp


# ---- 5.3 motion_batch ----

class MotionBatchMessage(_Base):
    msg_type: Literal["motion_batch"] = "motion_batch"
    start_timestamp: datetime
    sample_interval_s: float
    fields: list[str]
    samples: list[list[Optional[float]]]

    def as_records(self) -> list[dict]:
        return [dict(zip(self.fields, row)) for row in self.samples]


# ---- 5.4 proximity ----

class ProximityObject(BaseModel):
    object_id: str
    object_type: ObjectType
    role: Optional[str] = None
    distance_m: float
    bearing_deg: float
    relative_speed_ms: Optional[float] = None
    sensor: Optional[SensorType] = None
    detection_confidence: Optional[float] = None


class ProximityMessage(_Base):
    msg_type: Literal["proximity"] = "proximity"
    objects: list[ProximityObject] = Field(default_factory=list)


# ---- 5.5 environment ----

class EnvironmentData(BaseModel):
    weather: Weather
    ambient_temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rain_mm_h: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    visibility_m: Optional[float] = None
    light: Light
    dust_index: Optional[float] = None
    ground_condition: Optional[GroundCondition] = None
    lightning_distance_km: Optional[float] = None


class EnvironmentMessage(_Base):
    msg_type: Literal["environment"] = "environment"
    environment: EnvironmentData


# ---- 5.6 status ----

class EngineStatus(BaseModel):
    coolant_temp_c: Optional[float] = None
    fuel_level_pct: Optional[float] = None
    cumulative_hours: Optional[float] = None
    cumulative_idle_hours: Optional[float] = None
    cumulative_fuel_used_l: Optional[float] = None
    def_level_pct: Optional[float] = None
    battery_voltage_v: Optional[float] = None


class ImplementStatus(BaseModel):
    hydraulic_oil_temp_c: Optional[float] = None
    cumulative_load_cycles: Optional[int] = None


class CabStatus(BaseModel):
    cab_temp_c: Optional[float] = None


class TaskStatus(BaseModel):
    task_id: Optional[str] = None
    progress_pct: Optional[float] = None
    volume_moved_m3: Optional[float] = None


class Diagnostics(BaseModel):
    active_fault_codes: list[str] = Field(default_factory=list)


class StatusMessage(_Base):
    msg_type: Literal["status"] = "status"
    engine: EngineStatus
    implement: ImplementStatus
    cab: CabStatus
    task: Optional[TaskStatus] = None
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)


# ---- 5.7 event ----

class EventMessage(_Base):
    msg_type: Literal["event"] = "event"
    operator_id: Optional[str] = None
    event_id: str
    event_type: EventType
    source: EventSource
    details: dict[str, Any] = Field(default_factory=dict)


MESSAGE_TYPES: dict[str, type[BaseModel]] = {
    "shift_context": ShiftContextMessage,
    "operation": OperationMessage,
    "motion_batch": MotionBatchMessage,
    "proximity": ProximityMessage,
    "environment": EnvironmentMessage,
    "status": StatusMessage,
    "event": EventMessage,
}


class ErrorMessage(BaseModel):
    msg_type: Literal["error"] = "error"
    ref_msg_type: Optional[str] = None
    sim_tick: Optional[int] = None
    detail: str


def parse_message(raw: dict) -> BaseModel:
    """Raises KeyError if msg_type is missing/unknown, ValidationError otherwise."""
    msg_type = raw.get("msg_type")
    model = MESSAGE_TYPES[msg_type]  # KeyError -> caller turns into an error reply
    return model.model_validate(raw)

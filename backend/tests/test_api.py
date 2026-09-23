"""REST API tests against a temporary database (see conftest.py). Run from backend/:

    ../.venv/bin/python -m pytest -q tests/test_api.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.db import writes
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- training hub (B2-3)

def test_module_detail_has_content_and_hides_answers(client):
    r = client.get("/api/training/modules/TRN-IDLE-01")
    assert r.status_code == 200
    body = r.json()
    assert body["module_id"] == "TRN-IDLE-01" and body["title"] and body["video_url"]
    assert len(body["quiz"]) == 3
    for q in body["quiz"]:
        assert q["question"] and 3 <= len(q["options"]) <= 4
        assert "correct" not in q and "explanation" not in q


def test_every_catalogue_module_has_three_questions(client):
    for m in client.get("/api/training/recommendations/OP1001").json()["modules"]:
        r = client.get(f"/api/training/modules/{m['module_id']}")
        assert r.status_code == 200 and len(r.json()["quiz"]) == 3, m["module_id"]


def test_unknown_module_is_404(client):
    assert client.get("/api/training/modules/NOPE").status_code == 404
    assert client.post("/api/training/complete", json={"operator_id": "OP1001", "module_id": "NOPE"}).status_code == 404


def test_complete_grades_answers_server_side(client):
    from app.api.rest import _content
    correct = [q["correct"] for q in _content()["TRN-BELT-01"]["quiz"]]
    r = client.post("/api/training/complete",
                    json={"operator_id": "OP1002", "module_id": "TRN-BELT-01", "answers": correct})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "recorded"
    assert body["score"] == 100.0 and body["correct"] == body["total"] == 3 and body["passed"] is True

    wrong = [(c + 1) % 3 for c in correct]
    body = client.post("/api/training/complete",
                       json={"operator_id": "OP1002", "module_id": "TRN-BELT-01", "answers": wrong}).json()
    assert body["score"] == 0.0 and body["passed"] is False
    assert all(not x["correct"] and x["explanation"] for x in body["results"])


def test_complete_rejects_wrong_answer_count(client):
    r = client.post("/api/training/complete",
                    json={"operator_id": "OP1002", "module_id": "TRN-BELT-01", "answers": [0]})
    assert r.status_code == 422


def test_complete_still_accepts_a_plain_quiz_score(client):
    r = client.post("/api/training/complete",
                    json={"operator_id": "OP1003", "module_id": "TRN-IDLE-01", "quiz_score": 80})
    assert r.status_code == 200 and r.json()["score"] == 80


def test_instructor_booking_confirms_and_avoids_double_booking(client):
    slot = "2030-05-02T10:00:00"
    first = client.post("/api/training/book", json={"operator_id": "OP1004", "module_id": "TRN-INSTR-01",
                                                    "preferred_slot": slot})
    assert first.status_code == 200
    a = first.json()
    assert a["booking_id"].startswith("BK-") and a["confirmed_slot"] == slot and a["status"] == "confirmed"

    second = client.post("/api/training/book", json={"operator_id": "OP1005", "module_id": "TRN-INSTR-01",
                                                     "preferred_slot": slot}).json()
    assert second["preferred_slot"] == slot and second["confirmed_slot"] == "2030-05-02T11:00:00"

    listed = client.get("/api/training/bookings/OP1004").json()
    assert listed["operator_id"] == "OP1004"
    assert [b["booking_id"] for b in listed["bookings"]] == [a["booking_id"]]
    assert client.get("/api/training/bookings/OP9999").json()["bookings"] == []


def test_booking_validates_input(client):
    assert client.post("/api/training/book", json={"operator_id": "OP1", "module_id": "TRN-INSTR-01",
                                                   "preferred_slot": "tomorrow morning"}).status_code == 422
    assert client.post("/api/training/book", json={"operator_id": "OP1", "module_id": "NOPE",
                                                   "preferred_slot": "2030-05-02T10:00:00"}).status_code == 404


# ---------------------------------------------------------------- incidents and alerts (B2-4)

T0 = datetime(2031, 1, 1, 8, 0, tzinfo=timezone.utc)


def _alert(n: int) -> dict:
    return {"alert_id": f"AL-T{n:03d}", "alert_type": "tilt_warning", "severity": "critical",
            "message": f"Pitch/roll {n} deg exceeds 15 deg slope limit"}


def test_incident_detail_includes_telemetry_window(client):
    before = [{"msg_type": "operation", "sim_tick": 1}, {"msg_type": "proximity", "sim_tick": 2}]
    inc_id = writes.save_auto_incident("TESTM1", "OP7001", _alert(1), T0, before)
    writes.attach_incident_after(inc_id, [{"msg_type": "operation", "sim_tick": 40}])

    r = client.get(f"/api/incidents/{inc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == inc_id and body["machine_id"] == "TESTM1" and body["operator_id"] == "OP7001"
    assert body["incident_type"] == "tip_over_risk" and body["severity"] == "critical" and body["source"] == "auto"
    assert body["telemetry_window"]["before"] == before
    assert body["telemetry_window"]["after"] == [{"msg_type": "operation", "sim_tick": 40}]
    assert body["telemetry_window"]["alert_id"] == "AL-T001"


def test_incident_detail_matches_the_list_entry_and_404s(client):
    listed = client.get("/api/incidents", params={"operator_id": "OP7001"}).json()
    assert listed
    detail = client.get(f"/api/incidents/{listed[0]['id']}").json()
    assert {k: detail[k] for k in listed[0]} == listed[0]
    assert client.get("/api/incidents/99999999").status_code == 404
    assert client.get("/api/incidents/abc").status_code == 422


def test_manual_incident_has_an_empty_window(client):
    posted = client.post("/api/incidents", json={"machine_id": "TESTM1", "operator_id": "OP7002",
                                                 "note": "Worker walked behind me"}).json()
    detail = client.get(f"/api/incidents/{posted['id']}").json()
    assert detail["telemetry_window"] == {} and detail["source"] == "operator"
    assert detail["cause"] == "Worker walked behind me"


def test_alerts_are_newest_first_filtered_and_limited(client):
    for n in range(2, 7):
        writes.save_alert("TESTM2", "OP7003", _alert(n), T0 + timedelta(minutes=n))
    writes.save_alert("TESTM3", "OP7003", _alert(9), T0 + timedelta(minutes=99))

    rows = client.get("/api/alerts", params={"machine_id": "TESTM2"}).json()
    assert [a["alert_id"] for a in rows] == [f"AL-T{n:03d}" for n in (6, 5, 4, 3, 2)]
    assert rows[0]["message"] and rows[0]["acknowledged"] is False and rows[0]["severity"] == "critical"
    assert {"id", "alert_id", "machine_id", "operator_id", "alert_type", "timestamp"} <= set(rows[0])

    assert len(client.get("/api/alerts", params={"machine_id": "TESTM2", "limit": 2}).json()) == 2
    everything = client.get("/api/alerts", params={"limit": 500}).json()
    assert everything[0]["alert_id"] == "AL-T009"                       # newest overall
    assert client.get("/api/alerts", params={"limit": 0}).status_code == 422
    assert client.get("/api/alerts", params={"limit": 501}).status_code == 422
    assert client.get("/api/alerts", params={"machine_id": "NOBODY"}).json() == []


def test_ack_over_rest_matches_the_cab_socket(client):
    writes.save_alert("TESTM4", "OP7004", _alert(20), T0)
    writes.save_alert("TESTM4", "OP7004", _alert(21), T0 + timedelta(minutes=1))

    r = client.post("/api/alerts/AL-T020/ack")
    assert r.status_code == 200 and r.json() == {"alert_id": "AL-T020", "acknowledged": True}
    state = {a["alert_id"]: a["acknowledged"] for a in client.get("/api/alerts", params={"machine_id": "TESTM4"}).json()}
    assert state == {"AL-T020": True, "AL-T021": False}

    assert writes.acknowledge_alert("AL-T021") is True                  # the socket's code path
    assert client.post("/api/alerts/AL-NOPE/ack").status_code == 404
    assert writes.acknowledge_alert("AL-NOPE") is False

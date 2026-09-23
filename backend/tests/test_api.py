"""REST API tests against a temporary database (see conftest.py). Run from backend/:

    ../.venv/bin/python -m pytest -q tests/test_api.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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

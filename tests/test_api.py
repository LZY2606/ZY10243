import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PHOTOBLEACH_DB"] = str(ROOT / "data" / "test_api.sqlite3")

from app import app  # noqa: E402


client = TestClient(app)


@pytest.fixture()
def client_isolated_db(monkeypatch):
    path = ROOT / "data" / "test_api_isolated.sqlite3"
    path.unlink(missing_ok=True)
    monkeypatch.setenv("PHOTOBLEACH_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_home_title(client_isolated_db):
    response = client.get("/")
    assert response.status_code == 200
    assert "光衰校正室" in response.text


def test_replay_compare_export_and_clear_cycle(client_isolated_db):
    replay = client.post("/api/replay")
    assert replay.status_code == 200
    run_ids = replay.json()["run_ids"]
    assert len(run_ids) == 4

    comparison = client.get("/api/compare?segment_id=S1")
    assert comparison.status_code == 200
    assert len(comparison.json()) == 3
    scores = [item["metrics"]["stability_score"] for item in comparison.json()]
    assert scores == sorted(scores)

    run = client.get(f"/api/runs/{run_ids[0]}")
    assert run.status_code == 200
    assert len(run.json()["frames"]) == 24

    exported = client.get("/api/export")
    assert exported.status_code == 200
    assert len(exported.json()["runs"]) == 4

    assert client.delete("/api/state").status_code == 200
    state = client.get("/api/state").json()
    assert state["frames"] == []
    assert state["runs"] == []

    assert client.post("/api/import").status_code == 200
    state = client.get("/api/state").json()
    assert len(state["frames"]) == 24


def test_api_blocks_cross_break_fit(client_isolated_db):
    client.post("/api/import")
    response = client.post("/api/runs", json={
        "name": "bad-window",
        "model": "single",
        "segment_id": "S1",
        "fit_start": 7,
        "fit_end": 14,
        "factor_floor": 0.2,
    })
    assert response.status_code == 400
    assert "跨越" in response.json()["detail"]

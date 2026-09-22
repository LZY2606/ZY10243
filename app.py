from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import (
    break_records,
    connect,
    create_run,
    event_records,
    frame_records,
    get_run,
    import_fixtures,
    init_db,
    list_runs,
    replay_demo,
)


app = FastAPI(title="光衰校正室", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


def database():
    connection = connect()
    try:
        init_db(connection)
        yield connection
    finally:
        connection.close()


class RunRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    model: Literal["single", "double", "monotone"]
    segment_id: str = Field(min_length=1)
    fit_start: int = Field(ge=0)
    fit_end: int = Field(ge=0)
    factor_floor: float = Field(default=0.2, gt=0, le=1)


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/api/state")
def state(connection=Depends(database)):
    return {
        "frames": frame_records(connection),
        "events": event_records(connection),
        "breaks": break_records(connection),
        "runs": sorted(list_runs(connection), key=lambda item: (item["segment_id"], item["metrics"]["stability_score"])),
    }


@app.post("/api/import")
def import_data(connection=Depends(database)):
    return import_fixtures(connection)


@app.post("/api/replay")
def replay(connection=Depends(database)):
    runs = replay_demo(connection)
    return {"imported": len({frame["frame_index"] for frame in frame_records(connection)}), "run_ids": [run["run_id"] for run in runs]}


@app.delete("/api/state")
def clear(connection=Depends(database)):
    connection.executescript("DELETE FROM run_frames; DELETE FROM runs; DELETE FROM events; DELETE FROM breaks; DELETE FROM frames;")
    connection.commit()
    return {"cleared": True}


@app.post("/api/runs", status_code=201)
def add_run(request: RunRequest, connection=Depends(database)):
    try:
        return create_run(
            connection,
            name=request.name,
            model=request.model,
            segment_id=request.segment_id,
            fit_start=request.fit_start,
            fit_end=request.fit_end,
            factor_floor=request.factor_floor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="分支名称已存在") from exc
        raise


@app.get("/api/runs/{run_id}")
def read_run(run_id: int, connection=Depends(database)):
    try:
        return get_run(connection, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在") from exc


@app.get("/api/compare")
def compare(segment_id: str, connection=Depends(database)):
    runs = [run for run in list_runs(connection) if run["segment_id"] == segment_id]
    return sorted(runs, key=lambda item: item["metrics"]["stability_score"])


@app.get("/api/export")
def export_data(connection=Depends(database)):
    payload = {
        "frames": frame_records(connection),
        "events": event_records(connection),
        "breaks": break_records(connection),
        "runs": [get_run(connection, run["run_id"]) for run in list_runs(connection)],
    }
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": "attachment; filename=photobleach-room-export.json"},
    )

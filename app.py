"""光衰校正室 - FastAPI 服务。

演示：
    .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5583
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db
from core import (Breakpoint, FitWindowError, Frame, ModelSpec,
                  run_pipeline)
from fixtures import fixture_payload

DB_PATH = os.environ.get("BLEACH_DB", str(Path(__file__).parent / "bleach.db"))

app = FastAPI(title="光衰校正室")
STATIC = Path(__file__).parent / "static"


def conn():
    return db.connect(DB_PATH)


class ModelSpecIn(BaseModel):
    model_type: str = Field(..., description="single_exp | double_exp | monotone")
    fit_start: int
    fit_end: int
    exclude_events: list[tuple[int, int]] = []
    noise_cap: float = 0.2


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/import-fixture")
def import_fixture():
    payload = fixture_payload()
    c = conn()
    try:
        ds_id = db.insert_dataset(c, payload["name"], payload["meta"],
                                  payload["frames"], payload["breakpoints"])
        return {"dataset_id": ds_id, "name": payload["name"],
                "n_frames": len(payload["frames"]),
                "n_breakpoints": len(payload["breakpoints"])}
    finally:
        c.close()


@app.post("/api/reset")
def reset():
    c = conn()
    try:
        db.reset(c)
        return {"ok": True}
    finally:
        c.close()


@app.get("/api/datasets")
def datasets():
    c = conn()
    try:
        return db.list_datasets(c)
    finally:
        c.close()


@app.get("/api/datasets/{ds_id}")
def dataset(ds_id: int):
    c = conn()
    try:
        d = db.get_dataset(c, ds_id)
        if not d:
            raise HTTPException(404, "数据集不存在")
        return d
    finally:
        c.close()


@app.post("/api/datasets/{ds_id}/models")
def create_model(ds_id: int, spec: ModelSpecIn):
    c = conn()
    try:
        d = db.get_dataset(c, ds_id)
        if not d:
            raise HTTPException(404, "数据集不存在")
        frames = [Frame(t=f["t"], roi=f["roi"], background=f["background"],
                        exposure=f["exposure"], event=f["event"])
                  for f in d["frames"]]
        bps = [Breakpoint(t=b["t"], type=b["type"], note=b["note"])
               for b in d["breakpoints"]]
        ms = ModelSpec(model_type=spec.model_type, fit_start=spec.fit_start,
                       fit_end=spec.fit_end,
                       exclude_events=[tuple(e) for e in spec.exclude_events],
                       noise_cap=spec.noise_cap)
        try:
            result = run_pipeline(frames, bps, ms)
        except FitWindowError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        mid = db.save_model(c, ds_id, spec.model_dump(), result)
        return {"model_id": mid, "segment": result["segment"],
                "params": result["params"], "stability": result["stability"],
                "diagnostics": result["diagnostics"]}
    finally:
        c.close()


@app.get("/api/models/{mid}")
def model_detail(mid: int):
    c = conn()
    try:
        m = db.get_model(c, mid)
        if not m:
            raise HTTPException(404, "模型不存在")
        return m
    finally:
        c.close()


@app.get("/api/datasets/{ds_id}/compare")
def compare(ds_id: int):
    """同一数据的多个模型分支，按外推稳定性排序。"""
    c = conn()
    try:
        models = db.list_models(c, ds_id)
        if not models:
            raise HTTPException(404, "该数据集还没有模型分支")
        branches = []
        for m in models:
            rel = m["stability"].get("relative_diff")
            branches.append({
                "model_id": m["id"], "model_type": m["model_type"],
                "fit_window": [m["fit_start"], m["fit_end"]],
                "relative_diff": rel,
                "stability": m["stability"],
                "noise_affected": len(
                    m["diagnostics"].get("noise_amplification", {})
                    .get("affected_frames", [])),
            })
        ranked = sorted(
            branches,
            key=lambda b: (b["relative_diff"] is None,
                           b["relative_diff"] if b["relative_diff"] is not None else 0))
        for i, b in enumerate(ranked):
            b["rank"] = i + 1
        return {"dataset_id": ds_id, "branches": ranked}
    finally:
        c.close()


@app.get("/api/runs/export")
def export_runs():
    """导出运行记录（导入/拟合/重置），用于复核与重放。"""
    c = conn()
    try:
        return {"runs": db.list_runs(c)}
    finally:
        c.close()

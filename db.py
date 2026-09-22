"""SQLite 数据层：数据集、帧、断点、模型分支、逐帧结果、运行记录。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS frames(
    dataset_id INTEGER NOT NULL,
    t INTEGER NOT NULL,
    roi REAL NOT NULL,
    background REAL NOT NULL,
    exposure REAL,
    event TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(dataset_id, t));
CREATE TABLE IF NOT EXISTS breakpoints(
    dataset_id INTEGER NOT NULL,
    t INTEGER NOT NULL,
    type TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(dataset_id, t));
CREATE TABLE IF NOT EXISTS models(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    model_type TEXT NOT NULL,
    fit_start INTEGER NOT NULL,
    fit_end INTEGER NOT NULL,
    exclude_events TEXT NOT NULL DEFAULT '[]',
    noise_cap REAL NOT NULL DEFAULT 0.2,
    params TEXT NOT NULL,
    stability TEXT NOT NULL DEFAULT '{}',
    diagnostics TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results(
    model_id INTEGER NOT NULL,
    t INTEGER NOT NULL,
    segment INTEGER NOT NULL,
    raw REAL NOT NULL,
    background REAL NOT NULL,
    bg_corrected REAL NOT NULL,
    exposure REAL,
    exposure_valid INTEGER NOT NULL,
    exp_norm REAL,
    bleach_factor REAL,
    corrected REAL,
    residual REAL,
    flags TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(model_id, t));
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_run(conn, action: str, detail: dict) -> None:
    conn.execute("INSERT INTO runs(action, detail, created_at) VALUES(?,?,?)",
                 (action, json.dumps(detail, ensure_ascii=False), now()))


def insert_dataset(conn, name: str, meta: dict, frames: list[dict],
                   breakpoints: list[dict]) -> int:
    cur = conn.execute(
        "INSERT INTO datasets(name, meta, created_at) VALUES(?,?,?)",
        (name, json.dumps(meta, ensure_ascii=False), now()))
    ds_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO frames(dataset_id,t,roi,background,exposure,event)"
        " VALUES(?,?,?,?,?,?)",
        [(ds_id, f["t"], f["roi"], f["background"], f.get("exposure"),
          f.get("event", "")) for f in frames])
    conn.executemany(
        "INSERT INTO breakpoints(dataset_id,t,type,note) VALUES(?,?,?,?)",
        [(ds_id, b["t"], b["type"], b.get("note", "")) for b in breakpoints])
    log_run(conn, "import_dataset",
            {"dataset_id": ds_id, "name": name,
             "n_frames": len(frames), "n_breakpoints": len(breakpoints)})
    conn.commit()
    return ds_id


def get_dataset(conn, ds_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
    if not row:
        return None
    frames = [dict(r) for r in conn.execute(
        "SELECT t,roi,background,exposure,event FROM frames"
        " WHERE dataset_id=? ORDER BY t", (ds_id,))]
    bps = [dict(r) for r in conn.execute(
        "SELECT t,type,note FROM breakpoints WHERE dataset_id=? ORDER BY t",
        (ds_id,))]
    return {"id": row["id"], "name": row["name"], "meta": json.loads(row["meta"]),
            "created_at": row["created_at"], "frames": frames, "breakpoints": bps}


def list_datasets(conn) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM datasets ORDER BY id"):
        n_frames = conn.execute(
            "SELECT COUNT(*) c FROM frames WHERE dataset_id=?",
            (row["id"],)).fetchone()["c"]
        n_bp = conn.execute(
            "SELECT COUNT(*) c FROM breakpoints WHERE dataset_id=?",
            (row["id"],)).fetchone()["c"]
        n_models = conn.execute(
            "SELECT COUNT(*) c FROM models WHERE dataset_id=?",
            (row["id"],)).fetchone()["c"]
        out.append({"id": row["id"], "name": row["name"],
                    "meta": json.loads(row["meta"]),
                    "created_at": row["created_at"], "n_frames": n_frames,
                    "n_breakpoints": n_bp, "n_models": n_models})
    return out


def save_model(conn, ds_id: int, spec: dict, result: dict) -> int:
    cur = conn.execute(
        "INSERT INTO models(dataset_id,model_type,fit_start,fit_end,"
        "exclude_events,noise_cap,params,stability,diagnostics,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ds_id, spec["model_type"], spec["fit_start"], spec["fit_end"],
         json.dumps(spec.get("exclude_events", [])),
         spec.get("noise_cap", 0.2),
         json.dumps(result["params"], ensure_ascii=False),
         json.dumps(result["stability"], ensure_ascii=False),
         json.dumps(result["diagnostics"], ensure_ascii=False), now()))
    mid = cur.lastrowid
    conn.executemany(
        "INSERT INTO results(model_id,t,segment,raw,background,bg_corrected,"
        "exposure,exposure_valid,exp_norm,bleach_factor,corrected,residual,flags)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(mid, r["t"], r["segment"], r["raw"], r["background"],
          r["bg_corrected"], r["exposure"], int(r["exposure_valid"]),
          r["exp_norm"], r["bleach_factor"], r["corrected"], r["residual"],
          json.dumps(r["flags"])) for r in result["frames"]])
    log_run(conn, "fit_model",
            {"model_id": mid, "dataset_id": ds_id,
             "model_type": spec["model_type"],
             "fit_window": [spec["fit_start"], spec["fit_end"]],
             "stability": result["stability"]})
    conn.commit()
    return mid


def get_model(conn, mid: int) -> dict | None:
    row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        return None
    frames = []
    for r in conn.execute(
            "SELECT * FROM results WHERE model_id=? ORDER BY t", (mid,)):
        d = dict(r)
        d["flags"] = json.loads(d["flags"])
        d["exposure_valid"] = bool(d["exposure_valid"])
        frames.append(d)
    return {"id": row["id"], "dataset_id": row["dataset_id"],
            "model_type": row["model_type"],
            "fit_start": row["fit_start"], "fit_end": row["fit_end"],
            "exclude_events": json.loads(row["exclude_events"]),
            "noise_cap": row["noise_cap"],
            "params": json.loads(row["params"]),
            "stability": json.loads(row["stability"]),
            "diagnostics": json.loads(row["diagnostics"]),
            "created_at": row["created_at"], "frames": frames}


def list_models(conn, ds_id: int) -> list[dict]:
    out = []
    for row in conn.execute(
            "SELECT * FROM models WHERE dataset_id=? ORDER BY id", (ds_id,)):
        out.append({"id": row["id"], "dataset_id": row["dataset_id"],
                    "model_type": row["model_type"],
                    "fit_start": row["fit_start"], "fit_end": row["fit_end"],
                    "exclude_events": json.loads(row["exclude_events"]),
                    "noise_cap": row["noise_cap"],
                    "stability": json.loads(row["stability"]),
                    "diagnostics": json.loads(row["diagnostics"]),
                    "created_at": row["created_at"]})
    return out


def list_runs(conn) -> list[dict]:
    return [{"id": r["id"], "action": r["action"],
             "detail": json.loads(r["detail"]), "created_at": r["created_at"]}
            for r in conn.execute("SELECT * FROM runs ORDER BY id")]


def reset(conn) -> None:
    for tbl in ("results", "models", "frames", "breakpoints", "datasets"):
        conn.execute(f"DELETE FROM {tbl}")
    log_run(conn, "reset_database", {})
    conn.commit()

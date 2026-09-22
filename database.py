import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from correction import analyze_run


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "photobleach_room.sqlite3"


def db_path() -> Path:
    return Path(os.environ.get("PHOTOBLEACH_DB", DEFAULT_DB))


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS frames (
            frame_index INTEGER PRIMARY KEY,
            time_seconds REAL NOT NULL,
            segment_id TEXT NOT NULL,
            roi_id TEXT NOT NULL,
            intensity REAL NOT NULL,
            background REAL NOT NULL,
            exposure_ms REAL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            segment_id TEXT NOT NULL,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS breaks (
            break_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            after_frame INTEGER NOT NULL,
            from_segment TEXT,
            to_segment TEXT,
            from_roi TEXT,
            to_roi TEXT
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            model TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            fit_start INTEGER NOT NULL,
            fit_end INTEGER NOT NULL,
            factor_floor REAL NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS run_frames (
            run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            frame_index INTEGER NOT NULL,
            segment_id TEXT NOT NULL,
            roi_id TEXT NOT NULL,
            raw_intensity REAL NOT NULL,
            background REAL NOT NULL,
            exposure_ms REAL,
            background_corrected REAL NOT NULL,
            exposure_normalized REAL,
            bleach_factor REAL,
            noise_gain REAL,
            corrected_intensity REAL,
            residual REAL,
            in_fit_window INTEGER NOT NULL,
            used_for_fit INTEGER NOT NULL,
            is_extrapolation INTEGER NOT NULL,
            diagnostics_json TEXT NOT NULL,
            PRIMARY KEY (run_id, frame_index)
        );
        """
    )
    connection.commit()


def _read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "fixtures" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _exposure(row: dict[str, str]) -> float | None:
    value = row.get("exposure_ms", "")
    if value is None or value == "":
        return None
    return float(value)


def import_fixtures(connection: sqlite3.Connection) -> dict[str, Any]:
    init_db(connection)
    connection.executescript("DELETE FROM run_frames; DELETE FROM runs; DELETE FROM events; DELETE FROM breaks; DELETE FROM frames;")
    frame_rows = _read_csv("timeseries.csv")
    connection.executemany(
        "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                int(row["frame_index"]),
                float(row["time_seconds"]),
                row["segment_id"],
                row["roi_id"],
                float(row["intensity"]),
                float(row["background"]),
                _exposure(row),
            )
            for row in frame_rows
        ],
    )
    connection.executemany(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        [
            (row["event_id"], row["segment_id"], int(row["start_frame"]), int(row["end_frame"]), row["label"])
            for row in _read_csv("events.csv")
        ],
    )
    connection.executemany(
        "INSERT INTO breaks VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["break_id"],
                row["kind"],
                int(row["after_frame"]),
                row["from_segment"],
                row["to_segment"],
                row["from_roi"],
                row["to_roi"],
            )
            for row in _read_csv("breaks.csv")
        ],
    )
    connection.commit()
    return {"frames": len(frame_rows), "events": 2, "breaks": 3, "runs": 0}


def frame_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute("SELECT * FROM frames ORDER BY frame_index")]


def event_records(connection: sqlite3.Connection, segment_id: str | None = None) -> list[dict[str, Any]]:
    if segment_id:
        rows = connection.execute("SELECT * FROM events WHERE segment_id = ? ORDER BY start_frame", (segment_id,))
    else:
        rows = connection.execute("SELECT * FROM events ORDER BY start_frame")
    return [dict(row) for row in rows]


def break_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute("SELECT * FROM breaks ORDER BY after_frame, break_id")]


def create_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    model: str,
    segment_id: str,
    fit_start: int,
    fit_end: int,
    factor_floor: float = 0.2,
) -> dict[str, Any]:
    result = analyze_run(
        frame_records(connection),
        break_records(connection),
        model,
        segment_id,
        fit_start,
        fit_end,
        event_intervals=event_records(connection, segment_id),
        factor_floor=factor_floor,
    )
    cursor = connection.execute(
        """
        INSERT INTO runs (name, model, segment_id, fit_start, fit_end, factor_floor, parameters_json, metrics_json, warnings_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            model,
            segment_id,
            fit_start,
            fit_end,
            factor_floor,
            json.dumps(result["parameters"], ensure_ascii=False),
            json.dumps(result["metrics"], ensure_ascii=False),
            json.dumps(result["warnings"], ensure_ascii=False),
        ),
    )
    run_id = int(cursor.lastrowid)
    connection.executemany(
        """
        INSERT INTO run_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                row["frame_index"],
                row["segment_id"],
                row["roi_id"],
                row["raw_intensity"],
                row["background"],
                row["exposure_ms"],
                row["background_corrected"],
                row["exposure_normalized"],
                row["bleach_factor"],
                row["noise_gain"],
                row["corrected_intensity"],
                row["residual"],
                int(row["in_fit_window"]),
                int(row["used_for_fit"]),
                int(row["is_extrapolation"]),
                json.dumps(row["diagnostics"], ensure_ascii=False),
            )
            for row in result["frames"]
        ],
    )
    connection.commit()
    return get_run(connection, run_id)


def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("parameters", "metrics", "warnings"):
        data[key] = json.loads(data.pop(f"{key}_json"))
    return data


def list_runs(connection: sqlite3.Connection, segment_id: str | None = None) -> list[dict[str, Any]]:
    if segment_id:
        rows = connection.execute("SELECT * FROM runs WHERE segment_id = ? ORDER BY metrics_json", (segment_id,))
    else:
        rows = connection.execute("SELECT * FROM runs ORDER BY run_id")
    return [_run_from_row(row) for row in rows]


def get_run(connection: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(run_id)
    run = _run_from_row(row)
    run["frames"] = []
    for frame in connection.execute("SELECT * FROM run_frames WHERE run_id = ? ORDER BY frame_index", (run_id,)):
        item = dict(frame)
        item["diagnostics"] = json.loads(item.pop("diagnostics_json"))
        for key in ("in_fit_window", "used_for_fit", "is_extrapolation"):
            item[key] = bool(item[key])
        run["frames"].append(item)
    return run


def replay_demo(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    import_fixtures(connection)
    specs = [
        ("S1-单指数", "single", "S1", 9, 13, 0.2),
        ("S1-双指数", "double", "S1", 9, 13, 0.2),
        ("S1-单调非参数", "monotone", "S1", 9, 13, 0.2),
        ("S2-小因子诊断", "single", "S2", 17, 23, 0.2),
    ]
    return [
        create_run(
            connection,
            name=name,
            model=model,
            segment_id=segment_id,
            fit_start=start,
            fit_end=end,
            factor_floor=floor,
        )
        for name, model, segment_id, start, end, floor in specs
    ]

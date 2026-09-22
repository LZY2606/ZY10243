import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PHOTOBLEACH_DB"] = ":memory:"

from database import break_records, connect, frame_records, import_fixtures, replay_demo  # noqa: E402


@pytest.fixture()
def connection():
    conn = connect()
    import_fixtures(conn)
    yield conn
    conn.close()


def test_import_fixtures_and_replay_branches(connection):
    assert len(frame_records(connection)) == 24
    runs = replay_demo(connection)
    assert len(runs) == 4
    assert {run["model"] for run in runs[:3]} == {"single", "double", "monotone"}


def test_fit_window_cannot_cross_pause_refocus_or_roi_change(connection):
    from correction import analyze_run

    frames = frame_records(connection)
    breaks = break_records(connection)
    with pytest.raises(ValueError, match="跨越采集段|跨越断点"):
        analyze_run(frames, breaks, "single", "S1", 7, 14, [])
        with pytest.raises(ValueError, match="拟合窗必须完全位于"):
            analyze_run(frames, breaks, "single", "S0", 0, 99, [])


def test_event_frames_are_excluded_but_still_corrected(connection):
    from correction import analyze_run

    result = analyze_run(
        frame_records(connection),
        break_records(connection),
        "single",
        "S1",
        9,
        14,
        [{"start_frame": 12, "end_frame": 12}],
    )
    fit_indices = [row["frame_index"] for row in result["frames"] if row["used_for_fit"]]
    event_row = next(row for row in result["frames"] if row["frame_index"] == 12)
    assert 12 not in fit_indices
    assert "biological_event_excluded" in event_row["diagnostics"]
    assert event_row["corrected_intensity"] > 1.15
    assert result["metrics"]["event_holdout_rmse"] is not None


def test_missing_and_zero_exposure_remain_unnormalized(connection):
    from correction import analyze_run

    result = analyze_run(frame_records(connection), break_records(connection), "single", "S0", 0, 5, [])
    by_frame = {row["frame_index"]: row for row in result["frames"]}
    assert by_frame[7]["exposure_ms"] is None
    assert by_frame[7]["corrected_intensity"] is None
    assert "exposure_missing" in by_frame[7]["diagnostics"]
    s1 = analyze_run(frame_records(connection), break_records(connection), "single", "S1", 9, 14, [])
    row = next(row for row in s1["frames"] if row["frame_index"] == 15)
    assert row["exposure_ms"] == 0
    assert row["corrected_intensity"] is None
    assert "exposure_zero" in row["diagnostics"]


def test_small_bleach_factor_warns_with_gain_cap_and_affected_frames(connection):
    from correction import analyze_run

    result = analyze_run(frame_records(connection), break_records(connection), "single", "S2", 17, 23, [])
    warning = next(item for item in result["warnings"] if item["code"] == "noise_amplification")
    assert warning["affected_frames"] == [22, 23]
    assert warning["max_noise_gain"] > 5
    by_frame = {row["frame_index"]: row for row in result["frames"]}
    assert by_frame[23]["bleach_factor"] < 0.2
    assert by_frame[23]["noise_gain"] > 5
    assert by_frame[23]["corrected_intensity"] is not None


def test_segments_are_independently_normalized(connection):
    from correction import analyze_run

    frames = frame_records(connection)
    s1 = analyze_run(frames, break_records(connection), "single", "S1", 9, 14, [])
    selected = next(row for row in s1["frames"] if row["frame_index"] == 9 and row["used_for_fit"])
    outside = next(row for row in s1["frames"] if row["frame_index"] == 0)
    assert selected["bleach_factor"] == pytest.approx(1.0)
    assert selected["corrected_intensity"] == pytest.approx(1.0, abs=0.02)
    assert outside["corrected_intensity"] is None
    assert "outside_selected_segment" in outside["diagnostics"]

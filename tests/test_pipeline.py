"""验收测试：断点分段、曝光异常、噪声放大、跨断点拒绝、可重放。"""
import os
import tempfile

os.environ["BLEACH_DB"] = tempfile.mktemp(suffix=".db")

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from core import (Breakpoint, Frame, ModelSpec, assign_segments,
                  fit_single_exp, predict_single_exp, run_pipeline,
                  FitWindowError)
from fixtures import make_fixture

client = TestClient(app_module.app)


@pytest.fixture(scope="module", autouse=True)
def fresh_db():
    client.post("/api/reset")
    yield


def import_fixture():
    r = client.post("/api/import-fixture")
    assert r.status_code == 200
    return r.json()["dataset_id"]


# ------------------------------------------------------------ 核心单元

def test_segments_assigned_by_breakpoints():
    frames, bps, _ = make_fixture()
    assign_segments(frames, bps)
    assert frames[0].segment == 0
    assert frames[39].segment == 0
    assert frames[40].segment == 1   # 暂停后进入新段
    assert frames[70].segment == 2   # 重新对焦
    assert frames[100].segment == 3  # ROI 身份变更
    assert len({f.segment for f in frames}) == 4


def test_single_exp_recovers_ground_truth():
    t = np.arange(0, 40, dtype=float)
    y = 1.0 * np.exp(-t / 90.0)
    p = fit_single_exp(t, y)
    assert abs(p["tau"] - 90.0) < 1.0
    assert abs(p["a"] - 1.0) < 0.02
    assert np.allclose(predict_single_exp(p, t), y, atol=1e-6)


def test_fit_window_crossing_breakpoint_rejected():
    frames, bps, _ = make_fixture()
    spec = ModelSpec("single_exp", fit_start=30, fit_end=50)  # 跨过 t=40 暂停
    with pytest.raises(FitWindowError, match="跨越断点"):
        run_pipeline(frames, bps, spec)


def test_exposure_zero_and_missing_not_divided():
    frames, bps, _ = make_fixture()
    spec = ModelSpec("single_exp", fit_start=0, fit_end=39)
    result = run_pipeline(frames, bps, spec)
    by_t = {r["t"]: r for r in result["frames"]}
    for t in (55, 56, 57):          # 曝光为 0
        assert by_t[t]["exp_norm"] is None
        assert by_t[t]["corrected"] is None
        assert "exposure_invalid" in by_t[t]["flags"]
    assert by_t[110]["exp_norm"] is None   # 曝光缺失
    diag = result["diagnostics"]["exposure_invalid_frames"]
    assert set(diag) == {55, 56, 57, 110}


def test_noise_amplification_flagged_not_truncated():
    frames, bps, _ = make_fixture()
    # 第 2 段漂白深(tau=25)，窗尾因子约 exp(-29/25)≈0.31，段尾更深
    spec = ModelSpec("single_exp", fit_start=70, fit_end=99, noise_cap=0.35)
    result = run_pipeline(frames, bps, spec)
    na = result["diagnostics"]["noise_amplification"]
    assert na["affected_frames"], "应检出噪声放大帧"
    assert na["max_amplification"] > 1 / 0.35
    for r in result["frames"]:
        if r["segment"] != 2 or r["exp_norm"] is None:
            continue
        if r["bleach_factor"] < 0.35:
            assert "noise_amplification" in r["flags"]
            # 不截断：corrected 仍保留真实除法结果
            assert r["corrected"] == pytest.approx(
                r["exp_norm"] / r["bleach_factor"], rel=1e-9)


def test_segments_not_stitched():
    """其他段不套用本段模型，漂白因子留空，绝不跨段拼绝对强度。"""
    frames, bps, _ = make_fixture()
    spec = ModelSpec("single_exp", fit_start=0, fit_end=39)
    result = run_pipeline(frames, bps, spec)
    for r in result["frames"]:
        if r["segment"] != result["segment"]:
            assert r["bleach_factor"] is None
            assert r["corrected"] is None
            assert "other_segment" in r["flags"]


def test_bio_event_excluded_from_fit():
    frames, bps, _ = make_fixture()
    spec = ModelSpec("single_exp", fit_start=40, fit_end=69,
                     exclude_events=[(48, 52)])
    result = run_pipeline(frames, bps, spec)
    assert set(result["diagnostics"]["excluded_event_frames"]) == set(range(48, 53))
    # 事件帧无残差（未参与拟合）
    for r in result["frames"]:
        if 48 <= r["t"] <= 52:
            assert r["residual"] is None
    # 排除事件后 tau 应接近真值 45
    assert abs(result["params"]["tau"] - 45.0) < 8.0


def test_monotone_and_double_exp_run():
    frames, bps, _ = make_fixture()
    for mt in ("double_exp", "monotone"):
        spec = ModelSpec(mt, fit_start=0, fit_end=39)
        result = run_pipeline(frames, bps, spec)
        assert result["params"]["type"] == mt
        assert all(r["bleach_factor"] is not None
                   for r in result["frames"] if r["segment"] == 0)


# ------------------------------------------------------------ API 端到端

def test_api_import_fit_compare_export():
    ds_id = import_fixture()
    r = client.get("/api/datasets")
    assert any(d["id"] == ds_id for d in r.json())

    # 跨断点拟合被 422 拒绝
    r = client.post(f"/api/datasets/{ds_id}/models", json={
        "model_type": "single_exp", "fit_start": 30, "fit_end": 50})
    assert r.status_code == 422
    assert "跨越断点" in r.json()["detail"]

    # 三个分支
    ids = []
    for mt in ("single_exp", "double_exp", "monotone"):
        r = client.post(f"/api/datasets/{ds_id}/models", json={
            "model_type": mt, "fit_start": 0, "fit_end": 39,
            "exclude_events": [[48, 52]]})
        assert r.status_code == 200, r.text
        ids.append(r.json()["model_id"])

    detail = client.get(f"/api/models/{ids[0]}").json()
    for key in ("raw", "bg_corrected", "exp_norm", "bleach_factor",
                "corrected", "residual", "flags"):
        assert key in detail["frames"][0]

    cmp_ = client.get(f"/api/datasets/{ds_id}/compare").json()
    assert len(cmp_["branches"]) == 3
    assert cmp_["branches"][0]["rank"] == 1
    assert all("relative_diff" in b for b in cmp_["branches"])

    runs = client.get("/api/runs/export").json()["runs"]
    actions = [r["action"] for r in runs]
    assert "import_dataset" in actions and "fit_model" in actions


def test_reset_and_reimport_replay():
    ds_id = import_fixture()
    client.post(f"/api/datasets/{ds_id}/models", json={
        "model_type": "single_exp", "fit_start": 0, "fit_end": 39})
    client.post("/api/reset")
    assert client.get("/api/datasets").json() == []
    ds_id2 = import_fixture()   # 清空后可重新导入复核
    d = client.get(f"/api/datasets/{ds_id2}").json()
    assert len(d["frames"]) == 120 and len(d["breakpoints"]) == 3

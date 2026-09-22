import math
from typing import Any

import numpy as np


def _sse(points: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((points - pred) ** 2))


def _anchor(values: np.ndarray, times: np.ndarray) -> float:
    return float(values[np.argmin(times)])


def fit_single(points: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    anchor = _anchor(points, times)
    if len(points) < 2 or anchor <= 0:
        raise ValueError("单指数模型至少需要两个正的拟合点")
    best: dict[str, Any] | None = None
    y = points / anchor
    for log_k in np.linspace(math.log(1e-4), math.log(5.0), 160):
        k = math.exp(log_k)
        e = np.exp(-k * times)
        a, b = np.linalg.lstsq(np.column_stack((e, np.ones_like(e))), y, rcond=None)[0]
        if not np.isfinite([a, b]).all() or a < 0 or b < 0:
            continue
        pred = anchor * (a * e + b)
        if np.any(pred <= 0):
            continue
        sse = _sse(points, pred)
        if best is None or sse < best["sse"]:
            best = {"params": {"k": float(k), "a": float(a), "b": float(b)}, "sse": sse}
    if best is None:
        raise ValueError("单指数拟合没有找到有效候选")
    return best


def fit_double(points: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    anchor = _anchor(points, times)
    if len(points) < 4 or anchor <= 0:
        raise ValueError("双指数模型至少需要四个正的拟合点")
    y = points / anchor
    best: dict[str, Any] | None = None
    rates = np.exp(np.linspace(math.log(1e-4), math.log(5.0), 24))
    for k1 in rates:
        for k2 in rates:
            if k2 <= k1:
                continue
            design = np.column_stack((np.exp(-k1 * times), np.exp(-k2 * times), np.ones_like(times)))
            coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
            a, c, b = coeff
            if not np.isfinite(coeff).all() or a < 0 or c < 0 or b < 0 or a + c > 1.05:
                continue
            pred = anchor * (design @ coeff)
            if np.any(pred <= 0):
                continue
            sse = _sse(points, pred)
            if best is None or sse < best["sse"]:
                best = {
                    "params": {"k1": float(k1), "k2": float(k2), "a": float(a), "c": float(c), "b": float(b)},
                    "sse": sse,
                }
    if best is not None:
        return best
    single = fit_single(points, times)
    p = single["params"]
    return {"params": {"k1": p["k"], "k2": p["k"] * 2.0, "a": p["a"], "c": 0.0, "b": p["b"]}, "sse": single["sse"]}


def fit_nonparametric(points: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    anchor_index = int(np.argmin(times))
    blocks: list[dict[str, Any]] = []
    for value, weight in zip(points, np.ones_like(points)):
        blocks.append({"value": float(value), "weight": float(weight), "times": [0.0]})
    for block, time_value in zip(blocks, times):
        block["times"] = [float(time_value)]
    merged = True
    while merged:
        merged = False
        for idx in range(len(blocks) - 1):
            if blocks[idx]["value"] < blocks[idx + 1]["value"]:
                left, right = blocks[idx], blocks[idx + 1]
                weight = left["weight"] + right["weight"]
                blocks[idx : idx + 2] = [
                    {"value": (left["value"] * left["weight"] + right["value"] * right["weight"]) / weight,
                     "weight": weight, "times": left["times"] + right["times"]}
                ]
                merged = True
                break
    anchor_level = next(block["value"] for block in blocks if float(times[anchor_index]) in block["times"])
    levels = [
        {"time": min(block["times"]), "end_time": max(block["times"]), "value": block["value"] / anchor_level}
        for block in blocks
    ]
    return {"params": {"levels": levels, "anchor": float(anchor_level)}, "sse": 0.0}


def predict_curve(model: str, params: dict[str, Any], times: np.ndarray) -> np.ndarray:
    if model == "single":
        e = np.exp(-params["k"] * times)
        return params["a"] * e + params["b"]
    if model == "double":
        return params["a"] * np.exp(-params["k1"] * times) + params["c"] * np.exp(-params["k2"] * times) + params["b"]
    levels = sorted(params["levels"], key=lambda item: item["time"])
    result = []
    for time_value in times:
        chosen = levels[0]
        for level in levels:
            if level["time"] <= time_value <= level["end_time"]:
                chosen = level
                break
            if abs(time_value - level["time"]) < abs(time_value - chosen["time"]):
                chosen = level
        result.append(chosen["value"])
    return np.asarray(result, dtype=float)


def analyze_run(
    frames: list[dict[str, Any]],
    breaks: list[dict[str, Any]],
    model: str,
    segment_id: str,
    fit_start: int,
    fit_end: int,
    event_intervals: list[dict[str, int]] | None = None,
    factor_floor: float = 0.2,
) -> dict[str, Any]:
    if model not in {"single", "double", "monotone"}:
        raise ValueError("model 必须是 single、double 或 monotone")
    if fit_start > fit_end:
        raise ValueError("拟合起点不能晚于终点")
    segment_frames = [frame for frame in frames if frame["segment_id"] == segment_id]
    if not segment_frames:
        raise ValueError("找不到指定采集段")
    segment_numbers = {frame["frame_index"] for frame in segment_frames}
    all_numbers = {frame["frame_index"] for frame in frames}
    if fit_start not in all_numbers or fit_end not in all_numbers:
        raise ValueError("拟合窗必须完全位于所选采集段内")
    in_same_segment = fit_start in segment_numbers and fit_end in segment_numbers
    if not in_same_segment:
        raise ValueError("拟合窗跨越采集段、暂停、重对焦或 ROI 身份变更，禁止拟合")
    crossed = [
        item for item in breaks
        if fit_start < int(item["after_frame"]) + 1 <= fit_end
    ]
    if crossed:
        kinds = ", ".join(sorted({item["kind"] for item in crossed}))
        raise ValueError(f"拟合窗跨越断点，禁止拟合：{kinds}")

    event_intervals = event_intervals or []
    event_frames = {
        frame_no
        for interval in event_intervals
        for frame_no in range(int(interval["start_frame"]), int(interval["end_frame"]) + 1)
    }
    fit_rows = [
        frame for frame in segment_frames
        if fit_start <= frame["frame_index"] <= fit_end
        and frame["exposure_ms"] is not None and frame["exposure_ms"] > 0
        and frame["frame_index"] not in event_frames
    ]
    if len(fit_rows) < (4 if model == "double" else 2):
        raise ValueError("拟合窗内可归一化且非事件的帧数不足")

    fit_time = np.asarray([row["time_seconds"] for row in fit_rows], dtype=float)
    t0 = float(fit_time[np.argmin(fit_time)])
    fit_time -= t0
    fit_values = np.asarray(
        [(row["intensity"] - row["background"]) / row["exposure_ms"] * 100.0 for row in fit_rows],
        dtype=float,
    )
    if np.any(fit_values <= 0):
        raise ValueError("背景修正后存在非正值，无法建立单调衰减模型")
    fit_name = "double" if model == "double" else "single" if model == "single" else "monotone"
    fitted = fit_double(fit_values, fit_time) if model == "double" else (
        fit_single(fit_values, fit_time) if model == "single" else fit_nonparametric(fit_values, fit_time)
    )
    anchor_value = float(fit_values[np.argmin(fit_time)])
    anchor_curve = float(predict_curve(fit_name, fitted["params"], np.array([0.0]))[0])
    output_frames = []
    for frame in sorted(segment_frames, key=lambda item: item["frame_index"]):
        time_value = float(frame["time_seconds"]) - t0
        factor = max(
            float(predict_curve(fit_name, fitted["params"], np.array([time_value]))[0] / anchor_curve),
            1e-8,
        )
        background_corrected = float(frame["intensity"] - frame["background"])
        exposure = frame["exposure_ms"]
        normalized = None if exposure is None or exposure <= 0 else background_corrected / exposure * 100.0
        corrected = None if normalized is None else normalized / factor
        in_fit = fit_start <= frame["frame_index"] <= fit_end
        diagnostic = []
        if exposure is None:
            diagnostic.append("exposure_missing")
        elif exposure <= 0:
            diagnostic.append("exposure_zero")
        if frame["frame_index"] in event_frames:
            diagnostic.append("biological_event_excluded")
        if factor < factor_floor:
            diagnostic.append("bleach_factor_below_floor")
        output_frames.append({
            "frame_index": frame["frame_index"],
            "time_seconds": frame["time_seconds"],
            "segment_id": segment_id,
            "roi_id": frame["roi_id"],
            "raw_intensity": float(frame["intensity"]),
            "background": float(frame["background"]),
            "exposure_ms": None if exposure is None else float(exposure),
            "background_corrected": background_corrected,
            "exposure_normalized": normalized,
            "bleach_factor": factor,
            "noise_gain": 1.0 / factor,
            "corrected_intensity": corrected,
            "residual": None if corrected is None else normalized - anchor_value * factor,
            "in_fit_window": in_fit,
            "used_for_fit": in_fit and normalized is not None and frame["frame_index"] not in event_frames,
            "is_extrapolation": not in_fit,
            "diagnostics": diagnostic,
        })

    for frame in sorted((item for item in frames if item["segment_id"] != segment_id), key=lambda item: item["frame_index"]):
        exposure = frame["exposure_ms"]
        output_frames.append({
            "frame_index": frame["frame_index"],
            "time_seconds": frame["time_seconds"],
            "segment_id": frame["segment_id"],
            "roi_id": frame["roi_id"],
            "raw_intensity": float(frame["intensity"]),
            "background": float(frame["background"]),
            "exposure_ms": None if exposure is None else float(exposure),
            "background_corrected": float(frame["intensity"] - frame["background"]),
            "exposure_normalized": None if exposure is None or exposure <= 0 else (frame["intensity"] - frame["background"]) / exposure * 100.0,
            "bleach_factor": None,
            "noise_gain": None,
            "corrected_intensity": None,
            "residual": None,
            "in_fit_window": False,
            "used_for_fit": False,
            "is_extrapolation": False,
            "diagnostics": ["outside_selected_segment"] + (
                ["exposure_missing"] if exposure is None else ["exposure_zero"] if exposure <= 0 else []
            ),
        })
    output_frames.sort(key=lambda item: item["frame_index"])

    usable_in_segment = [
        row for row in output_frames
        if row["segment_id"] == segment_id and row["exposure_normalized"] is not None
    ]
    fit_mask = np.asarray([row["used_for_fit"] for row in output_frames])
    residuals = np.asarray([row["residual"] for row in output_frames if row["residual"] is not None])
    fit_residuals = np.asarray([row["residual"] for row in output_frames if row["used_for_fit"] and row["residual"] is not None])
    extrap = [
        row for row in output_frames
        if row["frame_index"] > fit_end and row["segment_id"] == segment_id and row["corrected_intensity"] is not None
    ]
    event_rows = [row for row in output_frames if "biological_event_excluded" in row["diagnostics"] and row["corrected_intensity"] is not None]
    low = [
        row for row in output_frames
        if row["segment_id"] == segment_id and "bleach_factor_below_floor" in row["diagnostics"]
    ]
    fit_rmse = float(np.sqrt(np.mean(fit_residuals ** 2))) if len(fit_residuals) else None
    event_rmse = float(np.sqrt(np.mean([row["residual"] ** 2 for row in event_rows]))) if event_rows else None
    extrap_gain = max([row["noise_gain"] for row in extrap], default=1.0)
    extrap_span = float(extrap_gain / min([row["noise_gain"] for row in extrap], default=1.0)) if extrap else 1.0
    penalty = 0.25 if model == "double" else 0.0
    stability_score = fit_rmse + (0.15 * math.log1p(extrap_span) if fit_rmse is not None else 0.0) + penalty
    warnings = []
    if low:
        affected = [row["frame_index"] for row in low]
        warnings.append({
            "code": "noise_amplification",
            "message": f"漂白因子低于 {factor_floor:g}；噪声增益上限 {max(row['noise_gain'] for row in low):.2f}x，未截断曲线。",
            "affected_frames": affected,
            "max_noise_gain": max(row["noise_gain"] for row in low),
        })
    unnormalizable = [
        row["frame_index"]
        for row in output_frames
        if row["segment_id"] == segment_id and row["corrected_intensity"] is None
    ]
    if unnormalizable:
        warnings.append({"code": "not_normalized", "message": "曝光为零或缺失，不做除法并保留原诊断。", "affected_frames": unnormalizable})
    return {
        "model": model,
        "segment_id": segment_id,
        "fit_start": fit_start,
        "fit_end": fit_end,
        "t0_seconds": t0,
        "anchor_value": anchor_value,
        "parameters": fitted["params"],
        "frames": output_frames,
        "metrics": {
            "fit_rmse": fit_rmse,
            "event_holdout_rmse": event_rmse,
            "extrapolation_gain_max": extrap_gain,
            "extrapolation_span": extrap_span,
            "stability_score": stability_score,
            "usable_frames": int(len(usable_in_segment)),
            "unnormalizable_frames": unnormalizable,
        },
        "warnings": warnings,
    }

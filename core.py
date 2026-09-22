"""光衰校正室 - 核心校正流水线。

每一步都保留可解释中间量：
raw -> bg_corrected -> exp_norm -> bleach_factor -> corrected -> residual

约定：
- 曝光时间为 0 或缺失(None)时不做除法，该帧 exp_norm/corrected 为 None 并打标。
- 断点(暂停/对焦重置/ROI 身份变更)把时间轴切成若干段，拟合窗不得跨段；
  各段独立归一(曝光参考、漂白因子参考都在段内计算)，绝不跨段拼接绝对强度。
- 漂白因子低于 noise_cap 时，除法会把噪声放大 1/factor 倍：不截断、不粉饰，
  保留数值但逐帧打标并汇总受影响帧与最大放大倍数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BREAKPOINT_TYPES = ("pause", "refocus", "roi_change")
MODEL_TYPES = ("single_exp", "double_exp", "monotone")


@dataclass
class Frame:
    t: int
    roi: float
    background: float
    exposure: float | None  # None 或 0 表示不可用
    segment: int = 0
    event: str = ""


@dataclass
class Breakpoint:
    t: int          # 断点发生在帧 t 之前（帧 t 属于新的一段）
    type: str
    note: str = ""


@dataclass
class ModelSpec:
    model_type: str
    fit_start: int
    fit_end: int
    exclude_events: list[tuple[int, int]] = field(default_factory=list)
    noise_cap: float = 0.2


class FitWindowError(ValueError):
    """拟合窗跨越断点或非法时抛出。"""


def assign_segments(frames: list[Frame], breakpoints: list[Breakpoint]) -> None:
    """按断点给每帧分配段号（原地修改）。"""
    bps = sorted(breakpoints, key=lambda b: b.t)
    seg = 0
    bi = 0
    for f in frames:
        while bi < len(bps) and f.t >= bps[bi].t:
            seg += 1
            bi += 1
        f.segment = seg


def validate_fit_window(spec: ModelSpec, frames: list[Frame],
                        breakpoints: list[Breakpoint]) -> int:
    """校验拟合窗：非空、在时间轴内、不跨断点。返回所属段号。"""
    if spec.model_type not in MODEL_TYPES:
        raise FitWindowError(f"未知模型类型: {spec.model_type}")
    if spec.fit_end <= spec.fit_start:
        raise FitWindowError("拟合窗为空：fit_end 必须大于 fit_start")
    ts = [f.t for f in frames]
    if spec.fit_start < ts[0] or spec.fit_end > ts[-1]:
        raise FitWindowError(f"拟合窗 [{spec.fit_start},{spec.fit_end}] 超出数据范围 "
                             f"[{ts[0]},{ts[-1]}]")
    for bp in breakpoints:
        if spec.fit_start < bp.t <= spec.fit_end:
            raise FitWindowError(
                f"拟合窗 [{spec.fit_start},{spec.fit_end}] 跨越断点 "
                f"t={bp.t} ({bp.type}: {bp.note})，请缩回到单段内")
    seg = next(f.segment for f in frames if f.t == spec.fit_start)
    return seg


def _segment_exposure_ref(frames: list[Frame], seg: int) -> float | None:
    """段内有效曝光的中位数，作为该段曝光归一参考。"""
    vals = [f.exposure for f in frames
            if f.segment == seg and f.exposure is not None and f.exposure > 0]
    if not vals:
        return None
    return float(np.median(vals))


# ---------------------------------------------------------------- 模型拟合

def fit_single_exp(t: np.ndarray, y: np.ndarray) -> dict:
    """y = a * exp(-t/tau)，对正值取对数做线性最小二乘。"""
    mask = y > 0
    if mask.sum() < 2:
        raise ValueError("单指数拟合需要至少 2 个正值点")
    tt, yy = t[mask], y[mask]
    slope, intercept = np.polyfit(tt, np.log(yy), 1)
    tau = -1.0 / slope if slope < 0 else float("inf")
    return {"type": "single_exp", "a": float(math.exp(intercept)),
            "tau": float(tau)}


def predict_single_exp(params: dict, t: np.ndarray) -> np.ndarray:
    if math.isinf(params["tau"]):
        return np.full_like(t, params["a"], dtype=float)
    return params["a"] * np.exp(-t / params["tau"])


def fit_double_exp(t: np.ndarray, y: np.ndarray) -> dict:
    """y = a1*exp(-t/tau1) + a2*exp(-t/tau2)。

    无 scipy：对 (tau1, tau2) 做网格搜索 + 变投影(振幅由线性最小二乘解出)，
    再做坐标下降细化。
    """
    mask = y > 0
    if mask.sum() < 4:
        raise ValueError("双指数拟合需要至少 4 个正值点")
    tt, yy = t[mask].astype(float), y[mask].astype(float)
    t0 = tt.min()
    tt = tt - t0
    span = max(tt.max(), 1.0)

    def solve_amps(tau1, tau2):
        e1 = np.exp(-tt / tau1)
        e2 = np.exp(-tt / tau2)
        A = np.column_stack([e1, e2])
        amps, *_ = np.linalg.lstsq(A, yy, rcond=None)
        resid = yy - A @ amps
        return float(resid @ resid), amps

    def ssr(tau1, tau2):
        if tau1 <= 0 or tau2 <= 0 or abs(tau1 - tau2) < 1e-9:
            return float("inf"), None
        return solve_amps(min(tau1, tau2), max(tau1, tau2))

    best = (float("inf"), None, None)
    grid = np.geomspace(span / 50.0, span * 20.0, 24)
    for t1 in grid:
        for t2 in grid:
            if t1 >= t2:
                continue
            s, amps = ssr(t1, t2)
            if s < best[0]:
                best = (s, (t1, t2), amps)
    (t1, t2) = best[1]
    step = t1 / 4.0
    for _ in range(40):
        improved = False
        for d1, d2 in ((step, 0), (-step, 0), (0, step), (0, -step)):
            s, amps = ssr(t1 + d1, t2 + d2)
            if s < best[0]:
                best = (s, (t1 + d1, t2 + d2), amps)
                t1, t2 = best[1]
                improved = True
        if not improved:
            step /= 2.0
            if step < 1e-3:
                break
    a1, a2 = best[2]
    return {"type": "double_exp", "a1": float(a1), "tau1": float(t1),
            "a2": float(a2), "tau2": float(t2), "t0": float(t0)}


def predict_double_exp(params: dict, t: np.ndarray) -> np.ndarray:
    tt = t.astype(float) - params.get("t0", 0.0)
    return (params["a1"] * np.exp(-tt / params["tau1"])
            + params["a2"] * np.exp(-tt / params["tau2"]))


def _pava_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators：单调非增等渗回归。"""
    y = y.astype(float)
    levels = y.copy()
    weights = np.ones_like(y)
    starts = np.arange(len(y))
    ends = np.arange(len(y))
    n = len(y)
    i = 0
    # 用栈式实现
    stack = []
    for idx in range(n):
        stack.append([idx, idx, y[idx], 1.0])
        while len(stack) >= 2 and stack[-2][2] < stack[-1][2]:
            b = stack.pop()
            a = stack.pop()
            w = a[3] + b[3]
            lvl = (a[2] * a[3] + b[2] * b[3]) / w
            stack.append([a[0], b[1], lvl, w])
    out = np.empty_like(y)
    for s, e, lvl, _w in stack:
        out[s:e + 1] = lvl
    return out


def fit_monotone(t: np.ndarray, y: np.ndarray) -> dict:
    """单调非参数衰减：窗内等渗回归，窗外常数外推。"""
    order = np.argsort(t)
    tt, yy = t[order], y[order]
    iso = _pava_decreasing(yy)
    return {"type": "monotone", "t": tt.tolist(), "y": iso.tolist()}


def predict_monotone(params: dict, t: np.ndarray) -> np.ndarray:
    tt = np.asarray(params["t"], dtype=float)
    yy = np.asarray(params["y"], dtype=float)
    t = np.asarray(t, dtype=float)
    # 窗外常数外推（不臆造衰减形状）
    return np.interp(t, tt, yy, left=yy[0], right=yy[-1])


def fit_model(model_type: str, t: np.ndarray, y: np.ndarray) -> dict:
    if model_type == "single_exp":
        return fit_single_exp(t, y)
    if model_type == "double_exp":
        return fit_double_exp(t, y)
    if model_type == "monotone":
        return fit_monotone(t, y)
    raise ValueError(f"未知模型类型: {model_type}")


def predict_model(params: dict, t: np.ndarray) -> np.ndarray:
    tp = params["type"]
    if tp == "single_exp":
        return predict_single_exp(params, t)
    if tp == "double_exp":
        return predict_double_exp(params, t)
    if tp == "monotone":
        return predict_monotone(params, t)
    raise ValueError(f"未知模型类型: {tp}")


# ---------------------------------------------------------------- 主流程

def _in_events(t: int, events: list[tuple[int, int]]) -> bool:
    return any(s <= t <= e for s, e in events)


def run_pipeline(frames: list[Frame], breakpoints: list[Breakpoint],
                 spec: ModelSpec) -> dict:
    """执行完整校正，返回每帧中间量与诊断。"""
    assign_segments(frames, breakpoints)
    seg = validate_fit_window(spec, frames, breakpoints)

    # 1) 背景修正 + 2) 曝光归一（各段独立参考）
    refs = {s: _segment_exposure_ref(frames, s)
            for s in sorted({f.segment for f in frames})}
    rows = []
    for f in frames:
        bg_corr = f.roi - f.background
        exp_valid = f.exposure is not None and f.exposure > 0
        ref = refs[f.segment]
        exp_norm = (bg_corr * ref / f.exposure) if (exp_valid and ref) else None
        rows.append({
            "t": f.t, "segment": f.segment, "raw": f.roi,
            "background": f.background, "bg_corrected": bg_corr,
            "exposure": f.exposure, "exposure_valid": bool(exp_valid),
            "exp_norm": exp_norm, "flags": ([] if exp_valid else ["exposure_invalid"]),
        })

    # 3) 拟合：窗内、本段、曝光有效、且不在生物事件区间内的帧
    fit_rows = [r for r in rows
                if r["segment"] == seg
                and spec.fit_start <= r["t"] <= spec.fit_end
                and r["exp_norm"] is not None
                and not _in_events(r["t"], spec.exclude_events)]
    if len(fit_rows) < 3:
        raise FitWindowError(
            f"拟合窗内可用点不足（{len(fit_rows)} < 3）："
            "请检查曝光无效帧与生物事件排除区间")
    ft = np.array([r["t"] for r in fit_rows], dtype=float)
    fy = np.array([r["exp_norm"] for r in fit_rows], dtype=float)
    params = fit_model(spec.model_type, ft, fy)

    # 4) 漂白因子：以拟合窗起点处的模型值为参考（段内归一，不跨段拼接）
    ref_val = float(predict_model(params, np.array([float(spec.fit_start)]))[0])
    if ref_val <= 0:
        raise FitWindowError("模型在拟合窗起点处的值非正，无法定义漂白因子")
    all_t = np.array([r["t"] for r in rows], dtype=float)
    model_all = predict_model(params, all_t)

    noise_frames = []
    max_amp = 1.0
    for i, r in enumerate(rows):
        if r["segment"] != seg:
            # 其他段：不套用本段模型，漂白因子留空（各段独立归一，绝不偷拼）
            r["bleach_factor"] = None
            r["corrected"] = None
            r["residual"] = None
            r["flags"].append("other_segment")
            continue
        factor = float(model_all[i] / ref_val)
        r["bleach_factor"] = factor
        if r["exp_norm"] is None:
            r["corrected"] = None
            r["residual"] = None
            continue
        amp = 1.0 / factor if factor > 0 else float("inf")
        if factor < spec.noise_cap:
            r["flags"].append("noise_amplification")
            noise_frames.append({"t": r["t"], "factor": factor,
                                 "amplification": amp})
            max_amp = max(max_amp, amp)
        # 不截断、不粉饰：保留真实除法结果，靠打标提示
        r["corrected"] = r["exp_norm"] / factor if factor > 0 else None
        if spec.fit_start <= r["t"] <= spec.fit_end \
                and not _in_events(r["t"], spec.exclude_events):
            r["residual"] = r["exp_norm"] - float(model_all[i])
        else:
            r["residual"] = None

    # 5) 外推稳定性：用窗前 60% 重拟合，比较段尾外推因子
    seg_end = max(r["t"] for r in rows if r["segment"] == seg)
    stability = _extrapolation_stability(spec, ft, fy, seg_end, ref_val)

    return {
        "segment": seg,
        "params": params,
        "exposure_ref": refs,
        "frames": rows,
        "diagnostics": {
            "exposure_invalid_frames": [r["t"] for r in rows
                                        if not r["exposure_valid"]],
            "noise_cap": spec.noise_cap,
            "noise_amplification": {
                "affected_frames": noise_frames,
                "max_amplification": (max_amp if noise_frames else 1.0),
            },
            "excluded_event_frames": [r["t"] for r in rows
                                      if _in_events(r["t"], spec.exclude_events)],
            "fit_frame_count": len(fit_rows),
        },
        "stability": stability,
    }


def _extrapolation_stability(spec: ModelSpec, ft: np.ndarray, fy: np.ndarray,
                             seg_end: int, ref_val: float) -> dict:
    """前半窗重拟合 vs 全窗拟合，在段尾外推点的漂白因子相对差。"""
    n = len(ft)
    cut = max(3, int(n * 0.6))
    if n < 6:
        return {"segment_end": seg_end, "relative_diff": None,
                "note": "拟合点过少，无法评估外推稳定性"}
    try:
        full = fit_model(spec.model_type, ft, fy)
        part = fit_model(spec.model_type, ft[:cut], fy[:cut])
        f_full = float(predict_model(full, np.array([float(seg_end)]))[0]) / ref_val
        f_part = float(predict_model(part, np.array([float(seg_end)]))[0]) / ref_val
        if f_full <= 0:
            return {"segment_end": seg_end, "relative_diff": None,
                    "note": "全窗模型外推值非正"}
        return {"segment_end": seg_end,
                "relative_diff": abs(f_full - f_part) / f_full,
                "factor_full_at_end": f_full, "factor_partial_at_end": f_part}
    except Exception as exc:  # noqa: BLE001
        return {"segment_end": seg_end, "relative_diff": None, "note": str(exc)}

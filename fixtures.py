"""固定 fixture：确定性生成的时序数据，包含验收要求的全部边界情形。

- 4 段，由 3 个断点分隔：暂停采集(t=40)、重新对焦(t=70)、ROI 身份变更(t=100)
- 曝光异常：t=55..57 曝光为 0，t=110 曝光缺失(None)
- 曝光变化：第 2 段曝光 20ms（其余 10ms）
- 生物事件：t=48..52 荧光瞬时上升（应在拟合中排除）
- 第 3 段漂白较深，漂白因子会跌破默认上限，触发噪声放大诊断
"""
from __future__ import annotations

import numpy as np

from core import Frame, Breakpoint

N_FRAMES = 120
SEED = 20260922


def make_fixture() -> tuple[list[Frame], list[Breakpoint], dict]:
    rng = np.random.default_rng(SEED)
    breakpoints = [
        Breakpoint(t=40, type="pause", note="显微镜暂停 5 分钟"),
        Breakpoint(t=70, type="refocus", note="重新对焦，焦面微调"),
        Breakpoint(t=100, type="roi_change", note="ROI 重新圈选，身份变更"),
    ]

    def segment_of(t: int) -> int:
        return sum(t >= b.t for b in breakpoints)

    frames: list[Frame] = []
    for t in range(N_FRAMES):
        seg = segment_of(t)
        # 各段真实漂白曲线（段内定义，段间不连续）
        if seg == 0:
            true_bleach = 1.00 * np.exp(-t / 90.0)
            base, exposure = 800.0, 10.0
        elif seg == 1:
            true_bleach = 0.95 * np.exp(-(t - 40) / 45.0)
            base, exposure = 780.0, 10.0
        elif seg == 2:
            true_bleach = 0.90 * np.exp(-(t - 70) / 25.0)  # 漂白较深
            base, exposure = 760.0, 20.0
        else:
            true_bleach = 0.85 * np.exp(-(t - 100) / 60.0)
            base, exposure = 500.0, 10.0  # ROI 身份变更：绝对强度不同

        background = 120.0 + 8.0 * np.sin(t / 30.0) + 0.15 * t  # 背景漂移
        signal = base * true_bleach
        if 48 <= t <= 52:  # 生物事件：真实荧光上升，非漂白
            signal *= 1.35
        roi = background + signal * (exposure / 10.0) \
            + rng.normal(0, 6.0)
        background_obs = background + rng.normal(0, 2.0)

        exp: float | None = exposure
        if 55 <= t <= 57:
            exp = 0.0          # 曝光时间为零
        if t == 110:
            exp = None         # 曝光元数据缺失

        frames.append(Frame(t=t, roi=float(roi),
                            background=float(background_obs),
                            exposure=exp,
                            event="bio_event" if 48 <= t <= 52 else ""))

    meta = {
        "description": "固定 fixture：4 段 / 3 断点 / 曝光 0 与缺失 / 生物事件",
        "seed": SEED,
        "ground_truth": {
            "seg0": "1.00*exp(-t/90)",
            "seg1": "0.95*exp(-(t-40)/45)",
            "seg2": "0.90*exp(-(t-70)/25)",
            "seg3": "0.85*exp(-(t-100)/60)",
            "bio_event": "t=48..52 荧光 x1.35",
        },
    }
    return frames, breakpoints, meta


def fixture_payload() -> dict:
    frames, bps, meta = make_fixture()
    return {
        "name": "fixture_photobleach_v1",
        "meta": meta,
        "frames": [{"t": f.t, "roi": f.roi, "background": f.background,
                    "exposure": f.exposure, "event": f.event} for f in frames],
        "breakpoints": [{"t": b.t, "type": b.type, "note": b.note}
                        for b in bps],
    }

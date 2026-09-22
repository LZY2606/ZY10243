# 光衰校正室

长时间显微成像荧光强度的光漂白校正服务。导入时序 ROI 强度、局部背景、
曝光元数据与采集断点，逐层保留可解释中间量，支持多模型分支对比。

## 安装与运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q && .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5583
```

访问 http://127.0.0.1:5583 即可看到“光衰校正室”操作页面。

## 数据口径

每帧一条记录：`t`（帧号）、`roi`（ROI 积分强度）、`background`
（局部背景，与 ROI 同单位）、`exposure`（曝光时间 ms；`0` 或缺失表示不可用）。
断点记录：`t`（断点发生在该帧之前）、`type`（`pause` 暂停 / `refocus`
重新对焦 / `roi_change` ROI 身份变更）、`note`。

校正流水线（每层的中间量都入库可查）：

1. **背景修正** `bg_corrected = roi - background`
2. **曝光归一** `exp_norm = bg_corrected × exposure_ref / exposure`，
   其中 `exposure_ref` 为**本段**有效曝光的中位数。曝光为 0 或缺失时
   **不做除法**，该帧 `exp_norm/corrected` 置空并打 `exposure_invalid` 标。
3. **漂白模型**：在拟合窗内、本段、曝光有效且不在生物事件区间的帧上拟合：
   - `single_exp`：`a·exp(-t/τ)`（对数线性最小二乘）
   - `double_exp`：`a1·exp(-t/τ1)+a2·exp(-t/τ2)`（网格搜索 + 变投影 + 坐标下降）
   - `monotone`：单调非增等渗回归（PAVA），窗外常数外推
4. **漂白因子** `factor(t) = model(t) / model(fit_start)`，以拟合窗起点为参考，
   **段内归一**；其他段不套用本段模型（`other_segment`），绝不跨段拼接绝对强度。
5. **最终序列** `corrected = exp_norm / factor`；**残差**仅在参与拟合的帧上给出。

### 硬性约束

- 拟合窗不得跨越任何断点（暂停 / 对焦重置 / ROI 身份变更），否则 422 拒绝。
- 漂白因子低于 `noise_cap`（默认 0.2）时，除法把噪声放大 `1/factor` 倍：
  逐帧打 `noise_amplification` 标，诊断中给出上限、受影响帧清单与最大放大
  倍数；**数值不截断、不粉饰**，由用户决定是否采信。
- 生物事件区间（如 fixture 的 t=48–52 荧光上升）从拟合中排除，不参与残差统计。

### 外推稳定性

每个分支用“前 60% 窗重拟合 vs 全窗拟合”在**段尾外推点**的漂白因子相对差
衡量外推稳定性，`/api/datasets/{id}/compare` 按此升序排名（越小越稳）。

## 固定 fixture

`fixtures.py` 用固定种子生成 120 帧、4 段、3 个断点，覆盖全部边界情形：
曝光为 0（t=55–57）、曝光缺失（t=110）、曝光变化（第 2 段 20ms）、
暂停（t=40）、重新对焦（t=70）、ROI 身份变更（t=100，绝对强度改变）、
生物事件（t=48–52）、深漂白段（第 2 段 τ=25，触发噪声放大诊断）。
真值衰减曲线记录在数据集 `meta.ground_truth` 中。

## 重放与复核

- 所有导入 / 拟合 / 重置操作写入 `runs` 表，
  `GET /api/runs/export`（或页面“导出运行记录”按钮）导出 JSON。
- `POST /api/reset` 清空业务数据（保留运行记录），随后
  `POST /api/import-fixture` 可重新导入同一固定 fixture 复核，
  数据逐帧一致（确定性种子）。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 操作页面 |
| POST | `/api/import-fixture` | 导入固定 fixture |
| POST | `/api/reset` | 清空数据库 |
| GET | `/api/datasets` / `/api/datasets/{id}` | 数据集列表 / 详情 |
| POST | `/api/datasets/{id}/models` | 新建模型分支并拟合 |
| GET | `/api/models/{id}` | 分支逐帧中间量与诊断 |
| GET | `/api/datasets/{id}/compare` | 分支外推稳定性排名 |
| GET | `/api/runs/export` | 导出运行记录 |

## 测试

```bash
.venv/bin/python -m pytest -q
```

覆盖：断点分段、跨断点拟合拒绝、曝光 0/缺失不除法、噪声放大打标不截断、
段间不拼接、生物事件排除、三种模型可拟合、单指数真值恢复、
清空后重导入复核。

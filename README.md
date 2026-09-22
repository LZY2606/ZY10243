# 光衰校正室

本地显微时序光漂白校正服务。服务导入固定 CSV fixture，将每帧依次保留为：原始 ROI 强度、局部背景、背景修正、曝光归一、漂白因子、最终校正值和残差。

## 数据口径

- `fixtures/timeseries.csv`：24 帧，包含三个采集段和 3 个断点标记。
- `fixtures/breaks.csv`：第 7 帧后暂停且 ROI 从 A 到 B；第 15 帧后同时记录重对焦和 ROI 身份变化。
- `fixtures/events.csv`：第 6、12 帧为生物事件，可校正但不参与拟合。
- 曝光时间缺失或为 0 时不做除法，最终值保留为空并输出 `exposure_missing` 或 `exposure_zero`。
- 拟合窗只允许落在一个采集段；窗口进入断点后的帧会被拒绝。
- 每个分支只对所选段归一，其他段保留原值和诊断，不跨段重拼绝对强度。
- 漂白因子以拟合窗首个有效点为 1；支持 `single`、`double`、`monotone`。
- 因子低于阈值时不截断平滑曲线；页面和导出给出噪声增益上限、受影响帧和逐帧标记。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 自动化测试

```bash
.venv/bin/python -m pytest -q
```

## 演示

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5583
```

访问 <http://127.0.0.1:5583>，页面标题为“光衰校正室”。点击“清空并重放 fixture”会清空 SQLite、重新导入固定数据并生成四个模型分支；S1 段三种模型使用第 9–13 帧，第 14 帧用于后向外推比较。“导出运行记录”下载完整 JSON 审计包。清空数据库后也可以点击“仅重新导入”复核原始数据。

默认数据库为 `data/photobleach_room.sqlite3`；测试使用独立临时数据库。

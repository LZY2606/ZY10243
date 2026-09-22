let state = { frames: [], events: [], breaks: [], runs: [] };
let selectedRunId = null;

const $ = (selector) => document.querySelector(selector);
const modelNames = { single: '单指数', double: '双指数', monotone: '单调非参数' };
const colors = {
  raw: '#64748b', backgroundCorrected: '#0ea5e9', normalized: '#a855f7',
  factor: '#f59e0b', corrected: '#22c55e', residual: '#ef4444'
};

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `请求失败：${response.status}`);
  }
  return response.json();
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

async function refresh() {
  state = await request('/api/state');
  const segments = [...new Set(state.frames.map(frame => frame.segment_id))];
  const select = $('#segment-select');
  select.innerHTML = segments.map(id => `<option value="${id}">${id}</option>`).join('');
  if (!segments.includes(select.value)) select.value = segments[0] || '';
  renderChips();
  renderComparison();
}

function renderChips() {
  $('#breakpoints').innerHTML = state.breaks.map(item =>
    `<span class="chip">${item.break_id} · ${item.kind} · 帧 ${item.after_frame} 后 · ${item.from_segment}/${item.from_roi} → ${item.to_segment}/${item.to_roi}</span>`
  ).join('') || '<span class="chip">无数据，先重放 fixture</span>';
  $('#event-chips').innerHTML = state.events.map(item =>
    `<span class="chip event">${item.event_id} · ${item.segment_id} · 帧 ${item.start_frame}-${item.end_frame} · ${item.label}</span>`
  ).join('') || '<span class="chip event">无事件数据</span>';
}

function currentRuns() {
  const segmentId = $('#segment-select').value;
  return state.runs.filter(run => run.segment_id === segmentId);
}

function renderComparison() {
  const runs = currentRuns();
  if (!runs.some(run => run.run_id === selectedRunId)) {
    selectedRunId = runs[0] ? runs[0].run_id : null;
  }
  $('#compare-table tbody').innerHTML = runs.map((run, index) => {
    const metrics = run.metrics;
    return `<tr>
      <td>${index + 1}</td><td>${run.name}</td><td>${modelNames[run.model]}</td>
      <td>${run.fit_start}–${run.fit_end}</td><td>${fmt(metrics.fit_rmse)}</td>
      <td>${fmt(metrics.event_holdout_rmse)}</td><td>${fmt(metrics.extrapolation_gain_max, 2)}×</td>
      <td>${fmt(metrics.stability_score)}</td>
      <td><button data-run="${run.run_id}">查看</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="9">该段尚无分支；重放会创建示例分支。</td></tr>';
  document.querySelectorAll('[data-run]').forEach(button => {
    button.addEventListener('click', () => { selectedRunId = Number(button.dataset.run); renderRun(); });
  });
  renderRun();
}

function renderRun() {
  const run = state.runs.find(item => item.run_id === selectedRunId);
  if (!run) {
    $('#warnings').innerHTML = '';
    $('#frames-table tbody').innerHTML = '<tr><td colspan="12">请选择或创建模型分支。</td></tr>';
    const canvas = $('#chart');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  $('#warnings').innerHTML = run.warnings.map(warning =>
    `<div class="warning"><strong>${warning.code}</strong>：${warning.message}<br>受影响帧：${(warning.affected_frames || []).join(', ') || '无'}</div>`
  ).join('');
  $('#frames-table tbody').innerHTML = run.frames.map(frame => {
    const badges = frame.diagnostics.map(tag => {
      const bad = tag.includes('below') || tag.includes('zero') || tag.includes('missing') || tag.includes('outside');
      return `<span class="badge ${bad ? 'bad' : 'good'}">${tag}</span>`;
    }).join('');
    return `<tr>
      <td>${frame.frame_index}</td><td>${frame.segment_id}/${frame.roi_id}</td>
      <td>${fmt(frame.raw_intensity, 2)}</td><td>${fmt(frame.background, 2)}</td>
      <td>${fmt(frame.exposure_ms, 1)}</td><td>${fmt(frame.background_corrected, 2)}</td>
      <td>${fmt(frame.exposure_normalized)}</td><td>${fmt(frame.bleach_factor)}</td>
      <td>${frame.noise_gain ? fmt(frame.noise_gain, 2) + '×' : '—'}</td>
      <td>${fmt(frame.corrected_intensity)}</td><td>${fmt(frame.residual)}</td><td>${badges}</td>
    </tr>`;
  }).join('');
  drawChart(run);
}

function line(ctx, frames, key, color, band, minValue, maxValue) {
  const pad = 58, width = 1180 - pad * 2, height = 124, top = 16 + band * 136;
  const frames2 = frames.filter(f => f[key] !== null && f[key] !== undefined);
  if (!frames2.length) return;
  const span = maxValue - minValue || 1;
  const xs = (frame) => pad + frame.frame_index / 23 * width;
  const ys = (value) => top + 22 + (1 - (value - minValue) / span) * (height - 42);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  let moving = false;
  frames.forEach(frame => {
    const value = frame[key];
    if (value === null || value === undefined) { moving = false; return; }
    const x = xs(frame), y = ys(value);
    moving ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    moving = true;
  });
  ctx.stroke();
  frames2.forEach(frame => {
    ctx.fillStyle = frame.used_for_fit ? color : '#ffffff';
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(xs(frame), ys(frame[key]), 3.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  });
  ctx.fillStyle = '#475569';
  ctx.font = '12px sans-serif';
  ctx.fillText(`${minValue.toFixed(2)} / ${maxValue.toFixed(2)}`, 4, top + height - 24);
}

function drawChart(run) {
  const canvas = $('#chart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fcfdff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const pad = 58, width = 1180 - pad * 2;
  state.breaks.forEach(item => {
    const x = pad + (item.after_frame + 0.5) / 23 * width;
    [16, 152, 288].forEach(top => {
      ctx.strokeStyle = '#ef4444';
      ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + 110); ctx.stroke();
      ctx.setLineDash([]);
    });
  });
  ['原始强度与背景修正', '曝光归一、漂白因子与最终值', '残差（观测归一值 − 模型预测）'].forEach((label, i) => {
    const top = 16 + i * 136;
    ctx.strokeStyle = '#e2e8f0'; ctx.strokeRect(pad, top + 22, width, 82);
    ctx.fillStyle = '#334155'; ctx.font = '13px sans-serif'; ctx.fillText(label, pad, top + 12);
  });
  const frames = run.frames;
  line(ctx, frames, 'raw_intensity', colors.raw, 0, 20, 48);
  line(ctx, frames, 'background_corrected', colors.backgroundCorrected, 0, -6, 28);
  line(ctx, frames, 'exposure_normalized', colors.normalized, 1, -0.2, 1.5);
  line(ctx, frames, 'bleach_factor', colors.factor, 1, 0, 1);
  line(ctx, frames, 'corrected_intensity', colors.corrected, 1, -0.2, 1.5);
  const residuals = frames.map(f => f.residual).filter(v => v !== null);
  const limit = Math.max(0.08, ...residuals.map(v => Math.abs(v)));
  line(ctx, frames, 'residual', colors.residual, 2, -limit, limit);
  ctx.strokeStyle = '#94a3b8';
  ctx.beginPath();
  const yZero = 288 + 22 + (1 - (0 + limit) / (2 * limit)) * 82;
  ctx.moveTo(pad, yZero); ctx.lineTo(pad + width, yZero); ctx.stroke();
  ctx.fillStyle = '#64748b';
  for (let frame = 0; frame < 24; frame++) {
    const x = pad + frame / 23 * width;
    ctx.fillText(String(frame), x - 4, 416);
  }
}

$('#segment-select')?.addEventListener('change', renderComparison);

$('#run-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('#form-error').textContent = '';
  const form = new FormData(event.currentTarget);
  try {
    const run = await request('/api/runs', {
      method: 'POST',
      body: JSON.stringify(Object.fromEntries(form.entries())),
    });
    selectedRunId = run.run_id;
    await refresh();
  } catch (error) {
    $('#form-error').textContent = error.message;
  }
});

$('#replay').addEventListener('click', async () => { await request('/api/replay', { method: 'POST' }); await refresh(); });
$('#import').addEventListener('click', async () => { await request('/api/import', { method: 'POST' }); selectedRunId = null; await refresh(); });
$('#clear').addEventListener('click', async () => { await request('/api/state', { method: 'DELETE' }); selectedRunId = null; await refresh(); });

refresh().catch(error => { $('#warnings').textContent = error.message; });

#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard for one MoE expert-statistics NPZ.

Example:
    python offline_exp/vis/moe_expert_dashboard.py \
        --input offline_exp/runs/.../stats/math500.npz \
        --output offline_exp/vis/res/math500.html

Serve every dataset below ``runs/<tag>/stats`` at ``/<tag>/<dataset>``:
    python offline_exp/vis/moe_expert_dashboard.py --serve --root offline_exp/runs --port 8765
"""

from __future__ import annotations

import argparse
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np


def _array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise ValueError(f"NPZ is missing one of: {', '.join(names)}")


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    raw = data["metadata"] if "metadata" in data.files else "{}"
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.ndim == 0 else raw.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": raw}
    return raw if isinstance(raw, dict) else {}


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        route = _array(data, "route_count")
        importance_all_sum = _array(data, "expert_importance_all", "expert_importance_all_mean")
        importance_topk_sum = _array(data, "expert_importance_topk", "expert_importance_topk_mean")
        distance = _array(data, "distance_mean", "distance_sum")
        token_count = _array(data, "token_count")
        importance_all = _array(data, "expert_importance_all_mean", "expert_importance_all")
        importance_topk = _array(data, "expert_importance_topk_mean", "expert_importance_topk")
        layer_ids = _array(data, "layer_ids") if "layer_ids" in data.files else np.arange(route.shape[0])
        num_experts = int(np.asarray(data["num_experts"]).item()) if "num_experts" in data.files else route.shape[-1]
        metadata = _metadata(data)

    if route.ndim != 2 or route.shape[1] != num_experts:
        raise ValueError(f"route_count must be [layers, experts], got {route.shape}")
    if distance.ndim != 3 or distance.shape[0] != route.shape[0] or distance.shape[1:] != (num_experts, num_experts):
        raise ValueError(f"distance must be [layers, experts, experts], got {distance.shape}")
    if importance_all.shape != route.shape or importance_topk.shape != route.shape:
        raise ValueError("importance arrays must have the same [layers, experts] shape as route_count")

    # JSON cannot represent NaN/Inf. Encode them as null so JSON.parse remains valid.
    def clean(a: np.ndarray) -> list:
        def sanitize(value):
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            return float(value) if np.isfinite(value) else None

        return sanitize(a.astype(float).tolist())

    return {
        "route": clean(route),
        "importanceAll": clean(importance_all.astype(float)),
        "importanceTopk": clean(importance_topk.astype(float)),
        "importanceAllSum": clean(
            importance_all_sum.astype(float)
            if "expert_importance_all" in data.files
            else (importance_all.astype(float) * token_count[:, None])
        ),
        "importanceTopkSum": clean(
            importance_topk_sum.astype(float)
            if "expert_importance_topk" in data.files
            else (importance_topk.astype(float) * token_count[:, None])
        ),
        "distance": clean(distance.astype(float)),
        "tokenCount": token_count.astype(float).tolist(),
        "layerIds": layer_ids.astype(int).tolist(),
        "numExperts": num_experts,
        "metadata": metadata,
        "distanceKind": "distance_mean" if "distance_mean" in data.files else "distance_sum",
    }


HTML_HEAD = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoE Expert Behavior Dashboard</title>
<style>
:root { --bg:#f5f7fb; --ink:#18202b; --muted:#677386; --line:#dce2eb; --blue:#2563eb; --teal:#0f9d8a; --amber:#e58b24; --red:#d9485f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
header { background:#101827; color:#fff; padding:22px 28px; }
header h1 { margin:0 0 6px; font-size:22px; letter-spacing:0; }
header p { margin:0; color:#b8c3d4; }
main { max-width:1700px; margin:0 auto; padding:20px 24px 48px; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:16px; }
label, select, button { font:inherit; }
select, button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:8px 11px; }
button { cursor:pointer; } button:hover, button.active { border-color:var(--blue); color:var(--blue); background:#eff5ff; }
.toolbar .spacer { flex:1; }
.importance-controls { display:grid; grid-template-columns:auto 1fr; gap:6px 12px; align-items:center; margin:0 0 12px; }
.importance-controls .control-label { color:var(--muted); font-size:12px; white-space:nowrap; }
.importance-controls .control-row { display:flex; flex-wrap:wrap; gap:7px; }
.statline { color:var(--muted); font-size:12px; }
.grid { display:grid; grid-template-columns:minmax(520px,1.35fr) minmax(420px,1fr); gap:16px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; min-width:0; box-shadow:0 1px 2px #1b2a4110; }
.panel h2 { margin:0 0 3px; font-size:16px; } .panel .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
.chart-wrap { overflow-x:hidden; } canvas.chart { display:block; width:100%; min-width:0; height:300px; }
.legend { display:flex; gap:16px; color:var(--muted); font-size:12px; margin-top:7px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; vertical-align:0; }
.rank-list { display:grid; grid-template-columns:1fr 1fr; gap:6px 18px; max-height:306px; overflow:auto; padding-right:4px; }
.rank-row { display:grid; grid-template-columns:32px 72px 1fr; gap:7px; align-items:center; border-bottom:1px solid #eef1f5; padding:5px 0; cursor:pointer; }
.rank-row.selected { background:#fff8e8; outline:2px solid #f0a129; outline-offset:-2px; border-radius:4px; }
.rank-row .rank { color:var(--muted); text-align:right; } .rank-row .expert { font-weight:650; }
.mini-bar { height:8px; background:#edf1f7; border-radius:3px; overflow:hidden; } .mini-bar i { display:block; height:100%; background:var(--blue); }
.layer-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:7px; max-height:180px; overflow:auto; }
.layer-btn { display:flex; justify-content:space-between; align-items:center; padding:8px 10px; text-align:left; }
.layer-btn strong { font-size:13px; } .layer-btn small { color:var(--muted); }
.heatmap-box { display:grid; grid-template-columns:minmax(320px,1fr) 290px; gap:18px; align-items:start; }
canvas.heatmap { width:100%; max-width:720px; aspect-ratio:1; display:block; image-rendering:auto; border:1px solid var(--line); }
.distance-summary { min-width:0; } .distance-summary h3 { margin:0 0 8px; font-size:14px; }
.near-cards { display:grid; grid-template-columns:1fr 1fr; gap:7px; max-height:310px; overflow:auto; }
.near-card { border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:6px; padding:8px 9px; background:#fbfdfd; }
.near-card.closest { border-color:#f0a129; border-left-color:#f0a129; background:#fff8e8; }
.near-card strong { display:block; } .near-card span { color:var(--muted); font-size:12px; }
.selection-banner { display:flex; align-items:center; gap:12px; margin:16px 0; padding:12px 15px; border:2px solid #f0a129; border-radius:8px; background:#fff8e8; color:#6b4307; }
.selection-banner b { font-size:18px; color:#9a5b00; } .selection-banner span { color:#795b2a; }
.metric-strip { display:grid; grid-template-columns:repeat(5,1fr); gap:7px; margin:0 0 12px; }
.metric { border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#f8fafc; }
.metric small { display:block; color:var(--muted); } .metric strong { font-size:13px; }
.distance-table-wrap { max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:6px; }
table { border-collapse:collapse; width:100%; font-size:12px; } th,td { border-bottom:1px solid #edf0f4; padding:7px 8px; text-align:right; white-space:nowrap; } th { position:sticky; top:0; background:#f8fafc; color:var(--muted); z-index:1; } th:first-child,td:first-child { text-align:left; }
tr.selected { background:#fff8ea; } .distance-pill { display:inline-block; min-width:65px; padding:3px 5px; border-radius:4px; background:#eef7f6; color:#087d6e; }
.full { grid-column:1/-1; } .hint { color:var(--muted); font-size:12px; }
.meta { white-space:pre-wrap; word-break:break-word; background:#f8fafc; padding:10px; border-radius:6px; color:#536174; font-size:11px; max-height:120px; overflow:auto; }
@media (max-width:1000px) { main { padding:14px; } .grid { grid-template-columns:1fr; } .heatmap-box { grid-template-columns:1fr; } .full { grid-column:auto; } }
@media (max-width:640px) { header { padding:18px; } .rank-list { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header><h1>MoE Expert Behavior Dashboard</h1><p id="fileTitle"></p></header>
<main>
<div class="toolbar"><label for="layerSelect">当前层</label><select id="layerSelect"></select><button id="prevLayer">上一层</button><button id="nextLayer">下一层</button><span class="spacer"></span><span class="statline" id="layerStats"></span></div>
<section class="panel"><h2>层选择</h2><div class="sub">每个层都保留，点击卡片切换；不隐藏任何专家。</div><div class="layer-grid" id="layerGrid"></div></section>
<div class="selection-banner" id="selectionBanner"><b>当前选中：Expert 0</b><span id="selectionDetails">距离视图、排名和两张柱状图会围绕这个专家联动。</span></div>
<div class="grid" style="margin-top:16px">
<section class="panel"><h2>专家路由次数</h2><div class="sub">当前层的 128 个专家，按专家编号排列。柱高使用线性比例。</div><div class="chart-wrap"><canvas class="chart" id="routeChart"></canvas></div><div class="legend"><span><i class="dot" style="background:#2563eb"></i>route_count</span><span class="hint">悬停可查看精确值</span></div></section>
<section class="panel"><h2>专家重要性（全部 expert）</h2><div class="sub">先选择统计范围，再选择平均或累积口径，最后选择显示尺度。占比是当前层真实总量中的比例，坐标轴自动取当前显示数据的最大值，不把最大专家改写成 100%。</div><div class="importance-controls"><span class="control-label">统计范围</span><div class="control-row"><button class="importanceMode active" data-mode="topk">top-k</button><button class="importanceMode" data-mode="all">all</button></div><span class="control-label">聚合口径</span><div class="control-row"><button class="importanceAgg active" data-agg="mean">token 平均</button><button class="importanceAgg" data-agg="sum">累积总量</button></div><span class="control-label">显示尺度</span><div class="control-row"><button class="importanceScale active" data-scale="share">层内占比</button><button class="importanceScale" data-scale="raw">原始值</button><button class="importanceScale" data-scale="log">log1p</button></div></div><div class="chart-wrap"><canvas class="chart" id="importanceChart"></canvas></div><div class="legend"><span><i class="dot" style="background:#0f9d8a"></i>importance</span><span class="hint">图表绘制全部 expert；悬停可查看原始值和显示值。</span></div></section>
<section class="panel"><h2>Top 64 专家排序</h2><div class="sub" id="rankingTitle"></div><div class="toolbar"><button class="rankMode active" data-mode="topk">top-k importance</button><button class="rankMode" data-mode="all">all importance</button><button class="rankMode" data-mode="route">route count</button><button class="rankAgg active" data-agg="mean">token 平均</button><button class="rankAgg" data-agg="sum">累积总量</button></div><div class="hint" id="rankingExplain"></div><div class="rank-list" id="rankList"></div></section>
<section class="panel"><h2>选中专家的近邻</h2><div class="sub">按距离从小到大排序；距离越小表示输出越相近。点击表格行可选择专家。</div><div class="near-cards" id="nearCards"></div></section>
<section class="panel full"><h2>专家输出距离</h2><div class="sub" id="distanceTitle"></div><div class="metric-strip" id="distanceMetrics"></div><div class="heatmap-box"><div><canvas class="heatmap" id="heatmap"></canvas><div class="hint" style="margin-top:7px">横纵轴均为专家编号。颜色越偏绿表示距离越小；色阶按 P1-P99 截断，避免极端值淹没细节。点击单元格可选择行专家。</div></div><div class="distance-summary"><h3 id="selectedTitle"></h3><div class="distance-table-wrap"><table><thead><tr><th>专家</th><th>距离</th><th>当前层专家对百分位</th></tr></thead><tbody id="distanceRows"></tbody></table></div></div></div></section>
<section class="panel full"><h2>数据元信息</h2><div class="meta" id="metadata"></div></section>
</div></main>
<script>
const DATA = __DATA__;
const fileName = __FILE_NAME__;
let layer = 0, selectedExpert = 0, importanceMode = 'topk', importanceAgg = 'mean', importanceScale = 'share', rankMode = 'topk', rankAgg = 'mean';
const $ = id => document.getElementById(id);
const fmt = (x, digits=4) => Number.isFinite(x) ? x.toLocaleString(undefined,{maximumFractionDigits:digits}) : 'N/A';
const css = getComputedStyle(document.documentElement);
function setup() {
  $('fileTitle').textContent = `${fileName}  |  ${DATA.route.length} 层 × ${DATA.numExperts} 专家  |  距离字段: ${DATA.distanceKind}`;
  $('metadata').textContent = JSON.stringify(DATA.metadata, null, 2);
  const sel = $('layerSelect'), grid = $('layerGrid');
  DATA.layerIds.forEach((id,i) => { const o=document.createElement('option'); o.value=i; o.textContent=`Layer ${id}`; sel.appendChild(o); const b=document.createElement('button'); b.className='layer-btn'; b.innerHTML=`<strong>Layer ${id}</strong><small>${fmt(DATA.tokenCount[i],0)} tokens</small>`; b.onclick=()=>setLayer(i); b.dataset.layer=i; grid.appendChild(b); });
  sel.onchange=()=>setLayer(Number(sel.value)); $('prevLayer').onclick=()=>setLayer((layer+DATA.route.length-1)%DATA.route.length); $('nextLayer').onclick=()=>setLayer((layer+1)%DATA.route.length);
  document.querySelectorAll('.importanceMode').forEach(b=>b.onclick=()=>{ importanceMode=b.dataset.mode; document.querySelectorAll('.importanceMode').forEach(x=>x.classList.toggle('active',x===b)); render(); });
  document.querySelectorAll('.importanceAgg').forEach(b=>b.onclick=()=>{ importanceAgg=b.dataset.agg; document.querySelectorAll('.importanceAgg').forEach(x=>x.classList.toggle('active',x===b)); render(); });
  document.querySelectorAll('.importanceScale').forEach(b=>b.onclick=()=>{ importanceScale=b.dataset.scale; document.querySelectorAll('.importanceScale').forEach(x=>x.classList.toggle('active',x===b)); render(); });
  document.querySelectorAll('.rankMode').forEach(b=>b.onclick=()=>{ rankMode=b.dataset.mode; document.querySelectorAll('.rankMode').forEach(x=>x.classList.toggle('active',x===b)); renderRanking(); });
  document.querySelectorAll('.rankAgg').forEach(b=>b.onclick=()=>{ rankAgg=b.dataset.agg; document.querySelectorAll('.rankAgg').forEach(x=>x.classList.toggle('active',x===b)); renderRanking(); });
  setLayer(0); window.addEventListener('resize',render);
}
function setLayer(i) { layer=Math.max(0,Math.min(DATA.route.length-1,i)); $('layerSelect').value=layer; document.querySelectorAll('.layer-btn').forEach(b=>b.classList.toggle('active',Number(b.dataset.layer)===layer)); render(); }
function prepCanvas(canvas, cssWidth, cssHeight) { const dpr=window.devicePixelRatio||1; const w=Math.max(10,Math.floor(cssWidth*dpr)),h=Math.max(10,Math.floor(cssHeight*dpr)); canvas.width=w;canvas.height=h;canvas.style.height=cssHeight+'px'; const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return ctx; }
function drawBars(id, values, color, rawValues=values, mode='raw') { const c=$(id), width=Math.max(c.parentElement.clientWidth,10), height=300, ctx=prepCanvas(c,width,height); const pad={l:56,r:12,t:15,b:31}, pw=Math.max(1,width-pad.l-pad.r),ph=height-pad.t-pad.b; const finite=values.filter(Number.isFinite), max=mode==='share'?Math.max(...finite,1e-12):Math.max(...finite,1), n=values.length, gap=Math.max(.5,pw/n*.16), bw=Math.max(.5,pw/n-gap); ctx.clearRect(0,0,width,height);ctx.strokeStyle='#dce2eb';ctx.fillStyle='#677386';ctx.font='11px system-ui';[0,.25,.5,.75,1].forEach(t=>{let y=pad.t+ph*(1-t);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();ctx.fillText(mode==='share'?`${(max*t*100).toFixed(2)}%`:fmt(max*t,2),3,y+4);}); values.forEach((v,i)=>{let x=pad.l+i*pw/n+gap/2, h=Number.isFinite(v)?(v/max)*ph:0;ctx.fillStyle=i===selectedExpert?'#f0a129':color;ctx.fillRect(x,pad.t+ph-h,bw,h); if(i===selectedExpert){ctx.strokeStyle='#8d5300';ctx.lineWidth=2.5;ctx.strokeRect(x-2,pad.t+ph-h-2,bw+4,h+4);} if(i%16===0||i===n-1){ctx.fillStyle='#677386';ctx.fillText(i,x-4,height-10);}}); c.onmousemove=e=>{const rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*(width/rect.width);const i=Math.max(0,Math.min(n-1,Math.floor((x-pad.l)/(pw/n))));c.title=`Expert ${i}: ${mode==='share'?(Number.isFinite(rawValues[i])?fmt(rawValues[i],6):'N/A')+' (share '+fmt(values[i]*100,3)+'%)':fmt(rawValues[i],6)}`;}; c.onclick=e=>{const rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*(width/rect.width);const i=Math.max(0,Math.min(n-1,Math.floor((x-pad.l)/(pw/n))));selectedExpert=i;render();}; }
function pearson(a,b) { const pairs=a.map((x,i)=>[x,b[i]]).filter(x=>x.every(Number.isFinite)); if(pairs.length<2)return NaN; const ax=pairs.reduce((s,x)=>s+x[0],0)/pairs.length,bx=pairs.reduce((s,x)=>s+x[1],0)/pairs.length;let n=0,da=0,db=0;pairs.forEach(x=>{const u=x[0]-ax,v=x[1]-bx;n+=u*v;da+=u*u;db+=v*v;});return n/Math.sqrt(da*db); }
function importanceRaw() { if(importanceMode==='topk') return importanceAgg==='sum'?DATA.importanceTopkSum[layer]:DATA.importanceTopk[layer]; return importanceAgg==='sum'?DATA.importanceAllSum[layer]:DATA.importanceAll[layer]; }
function importanceDisplay(raw) { const finite=raw.filter(Number.isFinite), total=finite.reduce((a,b)=>a+b,0); if(importanceScale==='share')return raw.map(x=>Number.isFinite(x)&&total?x/total:NaN); if(importanceScale==='log')return raw.map(x=>Number.isFinite(x)?Math.log1p(Math.max(0,x)):NaN); return raw; }
function render() { const r=DATA.route[layer], imp=importanceRaw(), shown=importanceDisplay(imp), corr=pearson(r,imp); $('layerStats').textContent=`Layer ${DATA.layerIds[layer]} · ${fmt(DATA.tokenCount[layer],0)} tokens · route total ${fmt(r.reduce((a,b)=>a+b,0),0)} · route/importance Pearson r=${fmt(corr,3)}`; $('selectionBanner').innerHTML=`<b>当前选中：Expert ${selectedExpert}</b><span>Layer ${DATA.layerIds[layer]} · route ${fmt(r[selectedExpert],0)} · importance ${fmt(imp[selectedExpert],6)} · ${importanceAgg==='mean'?'token 平均':'累积总量'} · 点击任意图表、热力图单元格或距离行切换。</span>`; drawBars('routeChart',r,'#2563eb',r,'raw'); drawBars('importanceChart',shown,'#0f9d8a',imp,importanceScale); renderRanking(); renderDistance(); }
function renderRanking() { let values, label; if(rankMode==='route'){ values=rankAgg==='sum'?DATA.route[layer]:DATA.route[layer].map(x=>x); label='route count（无 token 平均/累积区别，原始即累计次数）'; } else { const topk=rankMode==='topk'; values=topk?(rankAgg==='sum'?DATA.importanceTopkSum[layer]:DATA.importanceTopk[layer]):(rankAgg==='sum'?DATA.importanceAllSum[layer]:DATA.importanceAll[layer]); label=`${rankMode} importance · ${rankAgg==='mean'?'token 平均':'累积总量'}`; } $('rankingTitle').textContent=`按 ${label} 排序 · 当前 Layer ${DATA.layerIds[layer]}`; $('rankingExplain').textContent=`排名数字是真实 ${rankMode==='route'?'route count':rankMode+' importance '+(rankAgg==='mean'?'mean':'sum')}；小柱长度仅表示相对当前 Top 64 中最大值的视觉比例。`; const order=values.map((v,i)=>({i,v:Number.isFinite(v)?v:-Infinity})).sort((a,b)=>b.v-a.v).slice(0,64), max=Math.max(...order.map(x=>x.v),1); $('rankList').innerHTML=order.map((x,n)=>`<div class="rank-row ${x.i===selectedExpert?'selected':''}" data-expert="${x.i}" title="点击查看该专家的距离"><span class="rank">${n+1}</span><span class="expert">E${x.i}${x.i===selectedExpert?'  ←':''}</span><span><div class="mini-bar"><i style="width:${Math.max(0,x.v/max*100)}%"></i></div><small>${fmt(x.v,6)}</small></span></div>`).join(''); document.querySelectorAll('.rank-row').forEach(x=>x.onclick=()=>{selectedExpert=Number(x.dataset.expert);render();}); }
function colorFor(t) { t=Math.max(0,Math.min(1,t)); const a=[20,184,166],b=[239,68,68], q=t; return `rgb(${Math.round(a[0]*(1-q)+b[0]*q)},${Math.round(a[1]*(1-q)+b[1]*q)},${Math.round(a[2]*(1-q)+b[2]*q)})`; }
function quantile(sorted,q) { if(!sorted.length)return NaN;const p=(sorted.length-1)*q,i=Math.floor(p),f=p-i;return sorted[i]+(sorted[Math.min(i+1,sorted.length-1)]-sorted[i])*f; }
function layerDistances() { const d=DATA.distance[layer], out=[];for(let i=0;i<d.length;i++)for(let j=i+1;j<d[i].length;j++)if(Number.isFinite(d[i][j]))out.push(d[i][j]);return out.sort((a,b)=>a-b); }
function percentile(sorted,v) { let lo=0,hi=sorted.length;while(lo<hi){const m=(lo+hi)>>1;if(sorted[m]<=v)lo=m+1;else hi=m;}return lo/sorted.length*100; }
function drawHeatmap(all) { const c=$('heatmap'), box=c.parentElement, size=Math.min(Math.max(box.clientWidth,320),720), ctx=prepCanvas(c,size,size), d=DATA.distance[layer], n=DATA.numExperts, cell=(size-30)/n;ctx.clearRect(0,0,size,size); let min=quantile(all,.01),max=quantile(all,.99), span=max-min||1; for(let i=0;i<n;i++)for(let j=0;j<n;j++){let v=d[i][j];const picked=i===selectedExpert||j===selectedExpert;ctx.fillStyle=i===j?'#111827':Number.isFinite(v)?colorFor((v-min)/span):'#e5e7eb';ctx.globalAlpha=picked?(i===selectedExpert&&j===selectedExpert?1:.32):1;ctx.fillRect(25+j*cell,5+i*cell,cell+.5,cell+.5);ctx.globalAlpha=1;if(picked&&i!==j){ctx.strokeStyle='#f0a129';ctx.lineWidth=1.5;ctx.strokeRect(25+j*cell,5+i*cell,cell+.5,cell+.5);}} ctx.fillStyle='#536174';ctx.font='10px system-ui';for(let i=0;i<n;i+=16){ctx.fillText(i,25+i*cell,size-10);ctx.fillText(i,2,13+i*cell);} ctx.fillStyle='#9a5b00';ctx.font='bold 11px system-ui';ctx.fillText(`E${selectedExpert}`,25+selectedExpert*cell,size-20);ctx.fillText(`E${selectedExpert}`,2,13+selectedExpert*cell); c.onclick=e=>{const rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*(size/rect.width),y=(e.clientY-rect.top)*(size/rect.height);const j=Math.floor((x-25)/cell),i=Math.floor((y-5)/cell);if(i>=0&&i<n&&j>=0&&j<n){selectedExpert=i;render();}}; $('distanceTitle').textContent=`Layer ${DATA.layerIds[layer]} · ${n}×${n} 距离矩阵 · ${fmt(all.length,0)} 个无重复专家对`;
   const qs=[['P1',.01],['P5',.05],['P10',.10],['Median',.5],['P90',.9]]; $('distanceMetrics').innerHTML=qs.map(x=>`<div class="metric"><small>${x[0]} distance</small><strong>${fmt(quantile(all,x[1]),6)}</strong></div>`).join(''); }
function renderDistance() { const all=layerDistances();drawHeatmap(all); const row=DATA.distance[layer][selectedExpert]||[], vals=row.map((v,i)=>({i,v})).filter(x=>x.i!==selectedExpert&&Number.isFinite(x.v)).sort((a,b)=>a.v-b.v); $('selectedTitle').textContent=`Expert ${selectedExpert} 的完整 127 个距离（越小越相近）`; $('nearCards').innerHTML=vals.slice(0,12).map((x,n)=>{const p=percentile(all,x.v);return `<div class="near-card ${n===0?'closest':''}"><strong>#${n+1} · Expert ${x.i}${n===0?'  ← 最近':''}</strong><span>distance ${fmt(x.v,6)} · 全层 P${p.toFixed(1)}</span></div>`;}).join(''); $('distanceRows').innerHTML=vals.map(x=>{const p=percentile(all,x.v);return `<tr class="${x.i===selectedExpert?'selected':''}" data-expert="${x.i}"><td>Expert ${x.i}</td><td><span class="distance-pill">${fmt(x.v,7)}</span></td><td>P${p.toFixed(2)}（${p<=5?'非常接近':p<=10?'接近':'一般'}）</td></tr>`;}).join(''); document.querySelectorAll('#distanceRows tr').forEach(x=>x.onclick=()=>{selectedExpert=Number(x.dataset.expert);render();}); }
setup();
</script>
</body></html>
'''


def generate(input_path: Path, output_path: Path) -> None:
    payload = load_npz(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload, input_path.name), encoding="utf-8")


def render_html(payload: dict, file_name: str) -> str:
    """Render the existing dashboard template for either CLI or HTTP use."""
    rendered = HTML_HEAD.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return rendered.replace("__FILE_NAME__", json.dumps(file_name, ensure_ascii=False))


class DatasetCatalog:
    """Map stable URL names to NPZ files under one or more run directories."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self._lock = threading.RLock()
        self._files: dict[str, dict[str, Path]] = {}
        self._payload_cache: dict[Path, tuple[int, dict]] = {}
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            run_dirs = [self.root] if (self.root / "stats").is_dir() else [
                p for p in sorted(self.root.iterdir()) if p.is_dir() and (p / "stats").is_dir()
            ] if self.root.is_dir() else []
            files: dict[str, dict[str, Path]] = {}
            for run_dir in run_dirs:
                tag = run_dir.name
                datasets: dict[str, Path] = {}
                for path in sorted((run_dir / "stats").rglob("*.npz")):
                    name = path.relative_to(run_dir / "stats").with_suffix("").as_posix()
                    datasets[name] = path
                if datasets:
                    files[tag] = datasets
            self._files = files

    def catalog(self) -> dict:
        self.refresh()
        with self._lock:
            return {
                "root": str(self.root),
                "tags": [
                    {"tag": tag, "datasets": sorted(datasets)}
                    for tag, datasets in sorted(self._files.items())
                ],
            }

    def resolve(self, tag: str, dataset: str) -> Path | None:
        self.refresh()
        with self._lock:
            return self._files.get(tag, {}).get(dataset)

    def payload(self, path: Path) -> dict:
        stamp = path.stat().st_mtime_ns
        with self._lock:
            cached = self._payload_cache.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
        payload = load_npz(path)
        with self._lock:
            self._payload_cache[path] = (stamp, payload)
        return payload


INDEX_HEAD = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoE Expert Dashboards</title>
<style>
body{margin:0;background:#f5f7fb;color:#18202b;font:14px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:28px 20px 50px}header{background:#101827;color:white;padding:24px 20px}h1{max-width:1100px;margin:0 auto;font-size:22px}h2{font-size:16px;margin:24px 0 8px}.tag{background:white;border:1px solid #dce2eb;border-radius:8px;padding:14px 16px;margin:10px 0}.tag strong{display:block;margin-bottom:8px}.links{display:flex;flex-wrap:wrap;gap:7px}.links a{color:#2563eb;text-decoration:none;border:1px solid #dce2eb;border-radius:6px;padding:6px 9px;background:#fff}.links a:hover{background:#eff5ff;border-color:#2563eb}.empty{color:#677386}
</style></head><body><header><h1>MoE Expert Behavior Dashboards</h1></header><main>
"""


def render_index(catalog: dict) -> str:
    sections = []
    for item in catalog["tags"]:
        tag = item["tag"]
        links = "".join(
            f'<a href="/{quote(tag, safe="")}/{quote(dataset, safe="/")}">'
            f"{html.escape(dataset)}</a>"
            for dataset in item["datasets"]
        )
        sections.append(f'<section class="tag"><strong>{html.escape(tag)}</strong><div class="links">{links}</div></section>')
    body = "".join(sections) or '<p class="empty">No .npz datasets found under the configured root.</p>'
    return INDEX_HEAD + body + "</main></body></html>"


class DashboardHandler(BaseHTTPRequestHandler):
    catalog: DatasetCatalog

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(value, ensure_ascii=False).encode())

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        try:
            if not parts:
                self._send(200, "text/html; charset=utf-8", render_index(self.catalog.catalog()).encode())
                return
            if parts == ["healthz"]:
                self._json(200, {"ok": True})
                return
            if parts == ["api", "catalog"]:
                self._json(200, self.catalog.catalog())
                return
            if len(parts) >= 4 and parts[:2] == ["api", "data"]:
                tag, dataset = parts[2], "/".join(parts[3:])
                path = self.catalog.resolve(tag, dataset)
                if path is None:
                    self._json(404, {"error": "dataset not found", "tag": tag, "dataset": dataset})
                    return
                self._json(200, self.catalog.payload(path))
                return
            if len(parts) >= 2:
                tag, dataset = parts[0], "/".join(parts[1:])
                path = self.catalog.resolve(tag, dataset)
                if path is None:
                    self._send(404, "text/plain; charset=utf-8", b"dataset not found\n")
                    return
                page = render_html(self.catalog.payload(path), f"{tag}/{dataset}")
                self._send(200, "text/html; charset=utf-8", page.encode())
                return
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
        except (FileNotFoundError, ValueError, OSError) as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string}] {format % args}")


def serve(root: Path, host: str, port: int) -> None:
    catalog = DatasetCatalog(root)
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"catalog": catalog})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"serving {root.resolve()} at http://{host}:{port}/")
    print("dataset URLs: /<tag>/<dataset>  |  API: /api/catalog, /api/data/<tag>/<dataset>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone MoE expert behavior HTML dashboard")
    parser.add_argument("--input", help="input .npz file for standalone HTML generation")
    parser.add_argument("--output", help="output .html path (default: offline_exp/vis/res/<input>.html)")
    parser.add_argument("--serve", action="store_true", help="serve dashboards over HTTP")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "runs"), help="runs root or one run directory")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port")
    args = parser.parse_args()
    if args.serve:
        if args.input:
            parser.error("--input and --serve are mutually exclusive")
        serve(Path(args.root), args.host, args.port)
        return
    if not args.input:
        parser.error("either --input or --serve is required")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    output_path = Path(args.output).expanduser() if args.output else Path(__file__).parent / "res" / (input_path.stem + ".html")
    output_path = output_path.resolve()
    try:
        generate(input_path, output_path)
    except Exception as exc:
        parser.error(str(exc))
    print(f"generated {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()

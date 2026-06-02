#!/usr/bin/env python3
"""Simple curses TUI for configuring and running buun llama-bench comparisons.

- Finds GGUF models under ./models
- Lets you configure two model/parameter profiles
- Saves an INI config before running so the benchmark is reproducible/editable
- Can re-run a saved config with: scripts/bench-buun-tui.py --config path.ini --run
"""

from __future__ import annotations

import argparse
import configparser
import csv
import curses
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
import copy
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "versions/buun/src/buun-llama-cpp/build/bin/llama-bench"
CONFIG_DIR = ROOT / "bench-configs/buun"
LAST_CONFIG = CONFIG_DIR / ".last-config"
RESULT_DIR = ROOT / "bench-results/buun-tui"
GGUF_PY = ROOT / "versions/buun/src/buun-llama-cpp/gguf-py"

CACHE_STRATEGIES = {
    "turbo": ("turbo4", "turbo3"),
    "q8": ("q8_0", "q8_0"),
    "f16": ("f16", "f16"),
    "unset": ("", ""),  # omit --cache-type-k/v and let llama-bench use its default
}

FIELD_DESCRIPTIONS = {
    "model": "GGUF model file to benchmark. Models are discovered under ./models.",
    "cache_strategy": "KV cache quantization strategy. turbo=buun optimized turbo4/turbo3; q8=q8_0/q8_0; f16=full precision KV cache, usually larger/slower to fit.",
    "depths": "llama-bench --n-depth. Prefills this many tokens into KV before measuring. This is expensive at high values; use small lists for exploration.",
    "batch_size": "llama-bench -b/--batch-size. Logical prompt-processing batch size. Larger can improve prefill throughput but increases memory/graph pressure.",
    "ubatch_size": "llama-bench -ub/--ubatch-size. Physical micro-batch size. Lower values reduce memory pressure; higher values may improve throughput if VRAM allows.",
    "threads": "llama-bench -t/--threads. CPU threads used during eval/decode. More is not always better; oversubscription can hurt latency.",
    "n_cpu_moe": "buun -ncmoe/--n-cpu-moe. Number of first MoE layers whose expert weights stay on CPU, not a thread count. Higher uses less VRAM but more CPU/RAM; lower may OOM on 8GB VRAM.",
    "gpu_layers": "llama-bench -ngl/--n-gpu-layers. Number of layers to offload to GPU. 99 usually means offload as much as possible; lower values reduce VRAM use but may slow compute.",
    "prompt": "llama-bench -p/--n-prompt. Number of prompt/prefill tokens to measure after depth. Use 0 for generation-only depth tests.",
    "gen": "llama-bench -n/--n-gen. Number of generated tokens to measure. Smaller is faster but noisier; 128 is steadier than 32.",
    "repetitions": "llama-bench -r/--repetitions. Number of repeats per test case. Higher is more stable but much slower.",
    "warmup": "llama-bench warmup toggle. false adds --no-warmup for faster exploration; true gives more representative/stable numbers.",
    "device": "llama-bench -dev/--device. CUDA0 targets the RTX GPU explicitly; auto lets llama.cpp choose devices.",
    "flash_attn": "llama-bench -fa/--flash-attn. Equivalent to server preset flash-attn=on/off. 1 enables Flash Attention; unset uses llama-bench default, currently 0.",
}


PRESETS = {
    # llama-bench accepts comma-separated values for these parameters; the TUI
    # offers both single-value choices and small matrix presets.
    "depths": ["0,4096,8192,16384,32768", "0,8192,32768,65536", "0,32768,65536,131072", "custom", "unset"],
    "batch_size": ["1024", "512", "2048", "512,1024", "1024,2048", "custom", "unset"],
    "ubatch_size": ["256", "128", "512", "128,256", "256,512", "custom", "unset"],
    "threads": ["12", "8", "10", "8,10,12", "custom", "unset"],
    "n_cpu_moe": ["33", "40", "36", "32", "24", "33,36,40", "custom", "unset"],
    "gpu_layers": ["99", "80", "60", "99,80,60", "custom", "unset"],
    "prompt": ["0", "512", "0,512", "custom", "unset"],
    "gen": ["128", "32", "32,128", "custom", "unset"],
    # keep repetitions single-valued; it controls sampling stability, not a model parameter matrix.
    "repetitions": ["1", "2", "3", "unset"],
    "warmup": ["false", "true", "unset"],
    "device": ["CUDA0", "auto", "unset"],
    "flash_attn": ["1", "0", "unset"],
}


@dataclass
class Profile:
    name: str
    model: str = ""
    cache_strategy: str = "turbo"
    depths: str = "0,4096,8192,16384,32768"
    batch_size: str = "1024"
    ubatch_size: str = "256"
    threads: str = "12"
    n_cpu_moe: str = "33"
    gpu_layers: str = "unset"
    prompt: str = "512"
    gen: str = "128"
    repetitions: str = "1"
    warmup: str = "false"
    device: str = "CUDA0"
    flash_attn: str = "1"

    def items(self) -> list[tuple[str, str]]:
        return [
            ("model", self.model),
            ("cache_strategy", self.cache_strategy),
            ("depths", self.depths),
            ("batch_size", self.batch_size),
            ("ubatch_size", self.ubatch_size),
            ("threads", self.threads),
            ("n_cpu_moe", self.n_cpu_moe),
            ("gpu_layers", self.gpu_layers),
            ("prompt", self.prompt),
            ("gen", self.gen),
            ("repetitions", self.repetitions),
            ("warmup", self.warmup),
            ("device", self.device),
            ("flash_attn", self.flash_attn),
        ]


@dataclass
class BenchConfig:
    profiles: list[Profile] = field(default_factory=lambda: [Profile("profile_a"), Profile("profile_b")])
    output_format: str = "jsonl"
    run_prefix: str = "bench"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def human_size(path: Path) -> str:
    size = path.stat().st_size if path.exists() else 0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit in {"B", "KB", "MB"} else f"{size:.1f}{unit}"
        size /= 1024
    return "?"


def find_models() -> list[str]:
    models = sorted(rel(p) for p in (ROOT / "models").glob("**/*.gguf"))
    return models or [""]


def model_path(model: str) -> Path:
    p = Path(model)
    return p if p.is_absolute() else ROOT / p


def model_label(model: str) -> str:
    p = model_path(model)
    return f"{Path(model).name} ({human_size(p)})"


def model_options(models: list[str]) -> tuple[list[str], dict[str, str]]:
    mapping: dict[str, str] = {}
    options: list[str] = []
    for model in models:
        label = model_label(model)
        # Keep labels unique if duplicate basenames ever appear.
        if label in mapping:
            label = f"{model} ({human_size(model_path(model))})"
        mapping[label] = model
        options.append(label)
    return options, mapping


def gguf_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        if value and all(isinstance(x, int) and 0 <= x < 256 for x in value):
            try:
                return bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                return value
    return value


META_CACHE: dict[str, dict] = {}


def model_metadata(model: str) -> dict:
    if model in META_CACHE:
        return META_CACHE[model]
    meta = {"architecture": "?", "name": Path(model).stem, "block_count": None}
    try:
        if str(GGUF_PY) not in sys.path:
            sys.path.insert(0, str(GGUF_PY))
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(model_path(model), "r")
        for key, field in reader.fields.items():
            val = gguf_value(field.parts[-1])
            if key == "general.architecture":
                meta["architecture"] = val
            elif key == "general.name":
                meta["name"] = val
            elif key.endswith(".block_count"):
                meta["block_count"] = int(val)
    except Exception as exc:
        meta["error"] = repr(exc)
    META_CACHE[model] = meta
    return meta


def model_description(model: str) -> str:
    meta = model_metadata(model)
    parts = [f"Selected: {model_label(model)}"]
    if meta.get("name"):
        parts.append(f"name={meta['name']}")
    if meta.get("architecture"):
        parts.append(f"arch={meta['architecture']}")
    if meta.get("block_count"):
        parts.append(f"layers/block_count={meta['block_count']}")
    if meta.get("error"):
        parts.append(f"metadata read failed: {meta['error']}")
    return "; ".join(parts)


def gpu_layer_options(model: str) -> list[str]:
    n = model_metadata(model).get("block_count")
    if not isinstance(n, int) or n <= 0:
        return PRESETS["gpu_layers"]
    vals = []
    for ratio in [1.0, 0.8, 0.6, 0.4]:
        v = max(1, round(n * ratio))
        if v not in vals:
            vals.append(v)
    return [str(v) for v in vals] + [",".join(str(v) for v in vals[:3]), "custom", "unset"]


def choose_default_model(models: list[str]) -> str:
    for needle in ["IQ4_NL", "Q4", "IQ2_M"]:
        for model in models:
            if needle in model:
                return model
    return models[0]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return value or "bench"


def remember_config(path: Path) -> None:
    LAST_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LAST_CONFIG.write_text(str(path) + "\n")


def find_saved_configs() -> list[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)


def default_config_path() -> Path | None:
    if LAST_CONFIG.exists():
        p = Path(LAST_CONFIG.read_text().strip())
        if p.exists():
            return p
    configs = find_saved_configs()
    return configs[0] if configs else None


def save_config(cfg: BenchConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cp = configparser.ConfigParser()
    cp["run"] = {
        "bench": str(BENCH),
        "output_format": cfg.output_format,
        "run_prefix": cfg.run_prefix,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for profile in cfg.profiles:
        k, v = CACHE_STRATEGIES[profile.cache_strategy]
        cp[profile.name] = {key: value for key, value in profile.items()}
        cp[profile.name]["cache_type_k"] = k
        cp[profile.name]["cache_type_v"] = v
    with path.open("w") as f:
        cp.write(f)
    remember_config(path)


def load_config(path: Path) -> BenchConfig:
    cp = configparser.ConfigParser()
    cp.read(path)
    cfg = BenchConfig()
    if "run" in cp and "run_prefix" in cp["run"]:
        cfg.run_prefix = cp["run"]["run_prefix"]
    profiles: list[Profile] = []
    for section in cp.sections():
        if not section.startswith("profile"):
            continue
        p = Profile(section)
        for key, _ in p.items():
            if key in cp[section]:
                setattr(p, key, cp[section][key])
        profiles.append(p)
    if profiles:
        cfg.profiles = profiles
    return cfg


def is_set(value: str) -> bool:
    return bool(value.strip()) and value.strip().lower() not in {"unset", "none", "default"}


def add_arg(cmd: list[str], flag: str, value: str) -> None:
    if is_set(value):
        cmd.extend([flag, value])


def profile_command(profile: Profile, raw_path: Path) -> list[str]:
    ctk, ctv = CACHE_STRATEGIES.get(profile.cache_strategy, CACHE_STRATEGIES["turbo"])
    model_path = Path(profile.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    cmd = [
        str(BENCH),
        "--model", str(model_path),
        "--output", "jsonl",
        "--progress",
    ]
    add_arg(cmd, "--n-depth", profile.depths)
    add_arg(cmd, "--n-prompt", profile.prompt)
    add_arg(cmd, "--n-gen", profile.gen)
    add_arg(cmd, "--batch-size", profile.batch_size)
    add_arg(cmd, "--ubatch-size", profile.ubatch_size)
    add_arg(cmd, "--threads", profile.threads)
    add_arg(cmd, "--n-cpu-moe", profile.n_cpu_moe)
    add_arg(cmd, "--n-gpu-layers", profile.gpu_layers)
    if is_set(ctk):
        cmd.extend(["--cache-type-k", ctk])
    if is_set(ctv):
        cmd.extend(["--cache-type-v", ctv])
    add_arg(cmd, "--device", profile.device)
    add_arg(cmd, "--flash-attn", profile.flash_attn)
    add_arg(cmd, "--repetitions", profile.repetitions)
    if profile.warmup.lower() not in {"1", "true", "yes", "on", "unset", "none", "default"}:
        cmd.append("--no-warmup")
    return cmd


def parse_rows(text: str) -> list[dict]:
    """Parse either final JSON output or newline-delimited JSON output."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_name(row: dict) -> str:
    if int(row.get("n_prompt") or 0) > 0:
        return f"pp{row.get('n_prompt')}"
    if int(row.get("n_gen") or 0) > 0:
        return f"tg{row.get('n_gen')}"
    return "unknown"


def report_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        out = dict(row)
        model = out.get("model") or out.get("model_filename") or ""
        try:
            meta = model_metadata(str(model))
            if meta.get("block_count"):
                layers = int(meta["block_count"])
                out["model_layers"] = layers
                if "n_cpu_moe" in out and str(out.get("n_cpu_moe", "")).strip():
                    cpu_moe = max(0, min(layers, int(out["n_cpu_moe"])))
                    out["cpu_moe_layers"] = cpu_moe
                    out["remaining_moe_layers"] = max(0, layers - cpu_moe)
        except Exception:
            pass
        enriched.append(out)
    return enriched


def write_html_report(rows: list[dict], path: Path, errors: list[dict] | None = None) -> None:
    errors = errors or []
    rows = report_rows(rows)
    data = json.dumps(rows)
    errors_data = json.dumps(errors)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Buun llama-bench report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    svg {{ border: 1px solid #ddd; width: 100%; height: auto; display: block; }}
    table {{ border-collapse: collapse; margin-top: 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 4px 8px; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
    pre {{ white-space: pre-wrap; background: #fff4f4; border: 1px solid #f0b8b8; padding: 12px; }}
    .error {{ color: #8a0000; }}
    .tabs {{ display: flex; gap: 8px; margin: 18px 0; border-bottom: 1px solid #ddd; }}
    .tab-btn {{ border: 1px solid #ddd; border-bottom: none; background: #f6f6f6; padding: 8px 12px; cursor: pointer; border-radius: 6px 6px 0 0; }}
    .tab-btn.active {{ background: white; font-weight: 600; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 24px; align-items: start; }}
    .chart {{ min-width: 0; }}
    .chart h2 {{ margin-bottom: 8px; }}
    .legend {{ font-size: 12px; columns: 1; }}
    .legend span {{ display: inline-block; width: 0.9em; height: 0.9em; margin-right: 0.35em; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>Buun llama-bench report</h1>
  <p>Separate charts are used for prompt processing (<code>pp*</code>) and generation (<code>tg*</code>) so the much faster prompt numbers do not flatten generation lines.</p>
  <div class=\"tabs\">
    <button class=\"tab-btn active\" data-tab=\"charts-tab\">Charts</button>
    <button class=\"tab-btn\" data-tab=\"table-tab\">Raw table</button>
  </div>
  <section id=\"charts-tab\" class=\"tab-panel active\">
    <h2>Throughput</h2>
    <div id=\"charts\" class=\"charts-grid\"></div>
    <h2>CPU/GPU offload</h2>
    <div id=\"offload\" class=\"charts-grid\"></div>
    <h2 class=\"error\" id=\"error-title\"></h2>
    <div id=\"errors\"></div>
  </section>
  <section id=\"table-tab\" class=\"tab-panel\">
    <h2>Raw rows</h2>
    <table id=\"rows\"></table>
  </section>
<script>
const rows = {data};
const errors = {errors_data};
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(btn.dataset.tab).classList.add('active');
}}));
const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
const fields = ['profile','model','model_layers','cpu_moe_layers','remaining_moe_layers','cache_strategy','test','n_depth','n_prompt','n_gen','n_batch','n_ubatch','n_threads','n_cpu_moe','n_gpu_layers','flash_attn','type_k','type_v','avg_ts','stddev_ts'];
function num(x) {{ return Number(x || 0); }}
function testName(r) {{ return num(r.n_prompt) > 0 ? `pp${{r.n_prompt}}` : (num(r.n_gen) > 0 ? `tg${{r.n_gen}}` : 'unknown'); }}
function seriesKey(r) {{ return `${{r.profile}} ${{r.model?.split('/').pop() ?? ''}} ${{r.cache_strategy}} b=${{r.n_batch}} ub=${{r.n_ubatch}} t=${{r.n_threads}} moe=${{r.n_cpu_moe}} ngl=${{r.n_gpu_layers}} fa=${{r.flash_attn}}`; }}
function esc(s) {{ return String(s ?? '').replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function svgEl(svg, name, attrs) {{ const e = document.createElementNS('http://www.w3.org/2000/svg', name); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); svg.appendChild(e); return e; }}
function drawChart(title, chartRows) {{
  if (!chartRows.length) return;
  const wrap = document.createElement('div'); wrap.className = 'chart';
  wrap.innerHTML = `<h2>${{esc(title)}}</h2>`;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const W = 900, H = 460, L = 75, R = 25, T = 30, B = 65;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`); wrap.appendChild(svg);
  const xs = chartRows.map(r => num(r.n_depth));
  const ys = chartRows.map(r => num(r.avg_ts));
  const xmin = Math.min(...xs, 0), xmax = Math.max(...xs, 1);
  const ymin = 0, ymax = Math.max(...ys, 1) * 1.12;
  const sx = x => L + (num(x)-xmin) / (xmax-xmin || 1) * (W-L-R);
  const sy = y => H-B - (num(y)-ymin) / (ymax-ymin || 1) * (H-T-B);
  svgEl(svg, 'line', {{x1:L,y1:H-B,x2:W-R,y2:H-B,stroke:'#333'}});
  svgEl(svg, 'line', {{x1:L,y1:T,x2:L,y2:H-B,stroke:'#333'}});
  for (let i=0;i<=5;i++) {{
    const y = ymin + (ymax-ymin)*i/5, py = sy(y);
    svgEl(svg, 'line', {{x1:L,x2:W-R,y1:py,y2:py,stroke:'#eee'}});
    svgEl(svg, 'text', {{x:8,y:py+4,fill:'#333'}}).textContent = y.toFixed(1);
  }}
  const xTicks = [...new Set(xs)].sort((a,b)=>a-b);
  for (const x of xTicks) {{
    const px = sx(x);
    svgEl(svg, 'line', {{x1:px,x2:px,y1:H-B,y2:H-B+5,stroke:'#333'}});
    const t = svgEl(svg, 'text', {{x:px,y:H-B+22,fill:'#333','text-anchor':'middle'}}); t.textContent = x;
  }}
  const groups = new Map();
  for (const r of chartRows) {{ const k = seriesKey(r); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); }}
  const legend = document.createElement('div'); legend.className = 'legend';
  let ci = 0;
  for (const [k, rs] of groups) {{
    const color = colors[ci++ % colors.length];
    rs.sort((a,b)=>num(a.n_depth)-num(b.n_depth));
    const pts = rs.map(r => `${{sx(r.n_depth)}},${{sy(r.avg_ts)}}`).join(' ');
    svgEl(svg, 'polyline', {{points:pts, fill:'none', stroke:color, 'stroke-width':2}});
    for (const r of rs) {{
      svgEl(svg, 'circle', {{cx:sx(r.n_depth), cy:sy(r.avg_ts), r:4, fill:color}});
      svgEl(svg, 'title', {{}}).textContent = `${{k}} depth=${{r.n_depth}} avg_ts=${{r.avg_ts}}`;
    }}
    const item = document.createElement('div'); item.innerHTML = `<span style=\"background:${{color}}\"></span>${{esc(k)}}`; legend.appendChild(item);
  }}
  svgEl(svg, 'text', {{x:(W-R+L)/2, y:H-15, fill:'#333','text-anchor':'middle'}}).textContent = 'n_depth';
  svgEl(svg, 'text', {{x:8, y:18, fill:'#333'}}).textContent = 'tokens/s';
  wrap.appendChild(legend);
  document.getElementById('charts').appendChild(wrap);
}}
const rowsWithTest = rows.map(r => Object.assign({{}}, r, {{test: r.test ?? testName(r)}}));
for (const t of [...new Set(rowsWithTest.map(r => r.test))].sort()) drawChart(t, rowsWithTest.filter(r => r.test === t));
function drawOffloadChart() {{
  const seen = new Set();
  const configs = [];
  for (const r of rowsWithTest) {{
    const k = seriesKey(r);
    if (!seen.has(k) && num(r.model_layers) > 0) {{ seen.add(k); configs.push(r); }}
  }}
  if (!configs.length) return;
  const wrap = document.createElement('div'); wrap.className = 'chart';
  wrap.innerHTML = '<h2>MoE expert placement by layer count</h2><p>CPU MoE layers come from <code>--n-cpu-moe</code>; remaining MoE layers are not forced to CPU. <code>n_gpu_layers</code> is shown in labels as the llama.cpp GPU layer setting.</p>';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const W = 900, rowH = 34, L = 260, R = 35, T = 25, B = 30, H = T + B + configs.length * rowH;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`); wrap.appendChild(svg);
  const maxLayers = Math.max(...configs.map(r => num(r.model_layers)), 1);
  const sx = x => L + num(x) / maxLayers * (W-L-R);
  svgEl(svg, 'text', {{x:L,y:16,fill:'#333'}}).textContent = 'CPU MoE';
  svgEl(svg, 'text', {{x:L+95,y:16,fill:'#333'}}).textContent = 'remaining MoE';
  svgEl(svg, 'rect', {{x:L+70,y:6,width:14,height:10,fill:'#d62728'}});
  svgEl(svg, 'rect', {{x:L+180,y:6,width:14,height:10,fill:'#1f77b4'}});
  configs.forEach((r, i) => {{
    const y = T + i * rowH + 8;
    const layers = num(r.model_layers), cpu = Math.min(layers, num(r.cpu_moe_layers)), rem = Math.max(0, layers - cpu);
    svgEl(svg, 'text', {{x:4,y:y+13,fill:'#333'}}).textContent = `${{r.profile}} ${{(r.model ?? '').split('/').pop()}} ngl=${{r.n_gpu_layers}} ncmoe=${{r.n_cpu_moe}}`;
    svgEl(svg, 'rect', {{x:L,y:y,width:sx(cpu)-L,height:18,fill:'#d62728'}});
    svgEl(svg, 'rect', {{x:sx(cpu),y:y,width:sx(cpu+rem)-sx(cpu),height:18,fill:'#1f77b4'}});
    svgEl(svg, 'rect', {{x:L,y:y,width:sx(layers)-L,height:18,fill:'none',stroke:'#333'}});
    svgEl(svg, 'text', {{x:sx(layers)+5,y:y+13,fill:'#333'}}).textContent = `${{cpu}} CPU / ${{rem}} remaining / ${{layers}} total`;
  }});
  document.getElementById('offload').appendChild(wrap);
}}
drawOffloadChart();
if (errors.length) {{
  document.getElementById('error-title').textContent = `Errors (${{errors.length}})`;
  document.getElementById('errors').innerHTML = errors.map(e => `<h3>${{esc(e.profile ?? 'run')}} exit=${{esc(e.returncode ?? '')}}</h3><pre>${{esc(e.stderr_tail ?? e.error ?? '')}}</pre>`).join('');
}}
const table = document.getElementById('rows');
table.innerHTML = '<tr>' + fields.map(f => `<th>${{f}}</th>`).join('') + '</tr>' + rowsWithTest.map(r => '<tr>' + fields.map(f => `<td>${{esc(r[f])}}</td>`).join('') + '</tr>').join('');
</script>
</body>
</html>"""
    path.write_text(html)


SUMMARY_FIELDS = [
    "profile", "model", "cache_strategy", "test", "n_depth", "n_prompt", "n_gen",
    "n_batch", "n_ubatch", "n_threads", "n_cpu_moe", "n_gpu_layers", "flash_attn",
    "model_layers", "cpu_moe_layers", "remaining_moe_layers",
    "type_k", "type_v", "avg_ts", "stddev_ts", "avg_ns",
]


def summary_row(row: dict) -> dict:
    out = report_rows([row])[0]
    out["test"] = test_name(row)
    return out


def write_summary(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(summary_row(row))


def run_config(config_path: Path) -> int:
    cfg = load_config(config_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULT_DIR / f"{stamp}-{slugify(cfg.run_prefix)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    errors: list[dict] = []
    live_summary = out_dir / "summary.live.csv"
    with live_summary.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore").writeheader()

    for profile in cfg.profiles:
        raw_path = out_dir / f"{profile.name}.jsonl"
        err_path = out_dir / f"{profile.name}.stderr.log"
        cmd = profile_command(profile, raw_path)
        print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
        with raw_path.open("w") as raw, err_path.open("w") as err:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=err,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                raw.write(line)
                raw.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"partial non-JSON stdout from {profile.name}: {line[:160]}", file=sys.stderr)
                    continue
                row["profile"] = profile.name
                row["model"] = profile.model
                row["cache_strategy"] = profile.cache_strategy
                row.setdefault("flash_attn", profile.flash_attn if is_set(profile.flash_attn) else "")
                all_rows.append(row)
                out = summary_row(row)
                with live_summary.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore").writerow(out)
                print(
                    f"{profile.name} depth={row.get('n_depth')} {out['test']} "
                    f"b={row.get('n_batch')} ub={row.get('n_ubatch')} "
                    f"t={row.get('n_threads')} moe={row.get('n_cpu_moe')} "
                    f"ngl={row.get('n_gpu_layers')} fa={row.get('flash_attn')} {row.get('avg_ts')} tok/s",
                    flush=True,
                )
            returncode = proc.wait()
        if returncode != 0:
            stderr_text = err_path.read_text(errors="replace") if err_path.exists() else ""
            errors.append({
                "profile": profile.name,
                "model": profile.model,
                "returncode": returncode,
                "stderr_log": str(err_path),
                "stderr_tail": stderr_text[-8000:],
            })
            print(f"{profile.name} failed; see {err_path}; partial results: {live_summary}", file=sys.stderr)
            break

    summary = out_dir / "summary.csv"
    report = out_dir / "report.html"
    write_summary(all_rows, summary)
    write_html_report(all_rows, report, errors)
    print(f"summary: {summary}")
    print(f"live:    {live_summary}")
    print(f"report:  {report}")
    return 1 if errors else 0


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + (1 if cur else 0) > width:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_description_header(stdscr, start_row: int, description: str, max_lines: int = 6) -> int:
    """Draw a bordered description hint and return the next usable row."""
    if not description:
        return start_row
    width = max(20, curses.COLS - 1)
    inner_width = max(1, width - 4)
    lines = wrap_text(description, inner_width)[:max_lines]
    border = "+" + "-" * (width - 2) + "+"
    stdscr.addstr(start_row, 0, border[: curses.COLS - 1])
    row = start_row + 1
    for line in lines:
        stdscr.addstr(row, 0, ("| " + line.ljust(inner_width) + " |")[: curses.COLS - 1])
        row += 1
    stdscr.addstr(row, 0, border[: curses.COLS - 1])
    return row + 2


def prompt_text(stdscr, title: str, initial: str, description: str = "") -> str:
    """Small editable text prompt with a visible cursor."""
    value = list(initial)
    pos = len(value)
    curses.curs_set(1)
    try:
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, title[: curses.COLS - 1])
            row = draw_description_header(stdscr, 2, description)
            stdscr.addstr(row, 0, "Enter accept, Esc cancel, arrows move, Backspace delete"[: curses.COLS - 1])
            row += 2
            visible_width = max(1, curses.COLS - 1)
            start = max(0, pos - visible_width + 1)
            text = "".join(value[start : start + visible_width])
            stdscr.addstr(row, 0, text.ljust(visible_width))
            stdscr.move(row, pos - start)
            key = stdscr.getch()
            if key in (10, 13):
                result = "".join(value).strip()
                return result or initial
            if key in (27,):
                return initial
            if key in (curses.KEY_LEFT,):
                pos = max(0, pos - 1)
            elif key in (curses.KEY_RIGHT,):
                pos = min(len(value), pos + 1)
            elif key in (curses.KEY_HOME,):
                pos = 0
            elif key in (curses.KEY_END,):
                pos = len(value)
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                if pos > 0:
                    del value[pos - 1]
                    pos -= 1
            elif key == curses.KEY_DC:
                if pos < len(value):
                    del value[pos]
            elif 32 <= key <= 126:
                value.insert(pos, chr(key))
                pos += 1
    finally:
        curses.curs_set(0)


def choose_from(stdscr, title: str, options: list[str], current: str, description: str = "", allow_edit: bool = False) -> str:
    idx = options.index(current) if current in options else 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, title)
        row = draw_description_header(stdscr, 2, description)
        help_text = "↑/↓ select, Enter accept, q cancel"
        if allow_edit:
            help_text += ", Tab edit highlighted"
        stdscr.addstr(row, 0, help_text[: curses.COLS - 1])
        row += 2
        for i, opt in enumerate(options):
            marker = "> " if i == idx else "  "
            attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
            stdscr.addstr(row + i, 0, marker + opt[: curses.COLS - 4], attr)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(options)
        elif key in (10, 13):
            return options[idx]
        elif allow_edit and key in (9,):
            return prompt_text(stdscr, f"Edit {title}", options[idx], description)
        elif key in (ord("q"), 27):
            return current


def edit_profile(stdscr, profile: Profile, models: list[str]) -> None:
    fields = [k for k, _ in profile.items()]
    idx = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Editing {profile.name}")
        stdscr.addstr(1, 0, "↑/↓ move, Enter edit, q back")
        for i, key in enumerate(fields):
            value = getattr(profile, key)
            shown = model_label(value) if key == "model" and value else value
            attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
            stdscr.addstr(3 + i, 0, f"{key:15} {shown}"[: curses.COLS - 1], attr)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(fields)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(fields)
        elif key in (ord("q"), 27):
            return
        elif key in (10, 13):
            field = fields[idx]
            cur = getattr(profile, field)
            description = FIELD_DESCRIPTIONS.get(field, "")
            if field == "model":
                opts, mapping = model_options(models)
                current_label = model_label(cur) if cur else opts[0]
                picked = choose_from(stdscr, "Choose model", opts, current_label, description)
                setattr(profile, field, mapping.get(picked, cur))
            elif field == "cache_strategy":
                setattr(profile, field, choose_from(stdscr, "Choose cache strategy", list(CACHE_STRATEGIES), cur, description))
            elif field in PRESETS:
                options = gpu_layer_options(profile.model) if field == "gpu_layers" else PRESETS[field]
                if field == "gpu_layers":
                    description = f"{description} {model_description(profile.model)} Suggested values are based on GGUF block_count when readable."
                elif field == "n_cpu_moe":
                    description = f"{description} {model_description(profile.model)} For MoE models, choose up to the layer/block_count; e.g. 33 of 40 keeps most experts on CPU."
                picked = choose_from(stdscr, f"Choose {field}", options, cur, description, allow_edit=True)
                if picked == "custom":
                    picked = prompt_text(stdscr, f"Enter {field} value(s), comma-separated if supported", cur, description)
                setattr(profile, field, picked)
            else:
                setattr(profile, field, prompt_text(stdscr, f"Enter {field}", cur, description))


def new_default_config(models: list[str]) -> BenchConfig:
    cfg = BenchConfig()
    default_model = choose_default_model(models)
    cfg.profiles[0].model = default_model
    cfg.profiles[1] = copy.deepcopy(cfg.profiles[0])
    cfg.profiles[1].name = "profile_b"
    return cfg


def tui_main(stdscr, cfg: BenchConfig | None = None) -> Path | None:
    curses.curs_set(0)
    models = find_models()
    if cfg is None:
        cfg = new_default_config(models)
    selected = 0
    actions = ["Load saved config", "Edit run prefix", "Edit profile_a", "Edit profile_b", "Save config", "Save config and run", "Quit"]
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Buun llama-bench TUI")
        stdscr.addstr(1, 0, "Configure two reproducible benchmark profiles")
        stdscr.addstr(2, 0, f"Run prefix: {cfg.run_prefix}")
        for i, action in enumerate(actions):
            attr = curses.A_REVERSE if i == selected else curses.A_NORMAL
            stdscr.addstr(4 + i, 0, action, attr)
        stdscr.addstr(11, 0, "Profiles:")
        for i, p in enumerate(cfg.profiles):
            stdscr.addstr(12 + i, 0, f"{p.name}: {model_label(p.model)}, cache={p.cache_strategy}, depths={p.depths}, t={p.threads}, moe={p.n_cpu_moe}, fa={p.flash_attn}"[: curses.COLS - 1])
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(actions)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(actions)
        elif key in (ord("q"), 27):
            return None
        elif key in (10, 13):
            if selected == 0:
                configs = find_saved_configs()
                if not configs:
                    stdscr.clear(); stdscr.addstr(0, 0, "No saved configs found."); stdscr.addstr(2, 0, "Press any key"); stdscr.getch()
                else:
                    labels = [f"{p.name}" for p in configs]
                    mapping = dict(zip(labels, configs))
                    picked = choose_from(stdscr, "Load saved config", labels, labels[0], "Load params from any previous INI config. This does not run it until you choose Save config and run.")
                    if picked in mapping:
                        cfg = load_config(mapping[picked])
                        remember_config(mapping[picked])
            elif selected == 1:
                cfg.run_prefix = prompt_text(stdscr, "Enter run prefix, e.g. new-qwen-models", cfg.run_prefix, "Used in saved config and result folder names so you can identify benchmark runs later.")
            elif selected == 2:
                edit_profile(stdscr, cfg.profiles[0], models)
            elif selected == 3:
                edit_profile(stdscr, cfg.profiles[1], models)
            elif selected in (4, 5):
                path = CONFIG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(cfg.run_prefix)}.ini"
                save_config(cfg, path)
                if selected == 5:
                    return path
                stdscr.clear()
                stdscr.addstr(0, 0, f"Saved: {path}")
                stdscr.addstr(2, 0, "Press any key")
                stdscr.getch()
            elif selected == 6:
                return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="saved INI config to load")
    parser.add_argument("--run", action="store_true", help="run the saved config without opening the TUI")
    args = parser.parse_args()

    if args.config and args.run:
        return run_config(args.config)

    if args.config:
        cfg = load_config(args.config)
        remember_config(args.config)
    else:
        previous = default_config_path()
        cfg = load_config(previous) if previous else None
    last_rc = 0
    while True:
        path = curses.wrapper(tui_main, cfg)
        if not path:
            return last_rc
        print(f"Saved config: {path}")
        last_rc = run_config(path)
        cfg = load_config(path)
        print("\nRun finished. Returning to TUI; press Enter to continue...", flush=True)
        try:
            input()
        except EOFError:
            return last_rc


if __name__ == "__main__":
    raise SystemExit(main())

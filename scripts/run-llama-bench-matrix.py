#!/usr/bin/env python3
"""Run the full llama benchmark matrix.

This is the one-shot runner: it starts each server, waits for health, runs the
prompt suite, stops the server, and repeats for every permutation of:

  model x transport x context size x server

The three main knobs are:
  --models      label=path,label=path or just path,path
  --transports  curl,pi-blank,pi-current
  --contexts    32768,65536,131072,262144
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from bench_buun_tui.models import model_detail_brackets


DEFAULT_MODEL = "models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"
BUUN_LIB_REL = "versions/buun/src/buun-llama-cpp/build"


PROMPT_NAMES = {
    "regular_kv_cache": "KV cache explanation",
    "regular_debug": "Python CLI debug checklist",
    "internet_latest_release": "latest release refusal",
    "internet_news": "AI infra news refusal",
    "tool_weather": "weather tool prompt",
    "tool_search": "web-search tool prompt",
}


def slugify(value: str) -> str:
    value = Path(value).stem if "/" in value else value
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return value or "model"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_models(value: str, root: Path) -> list[dict]:
    models = []
    for item in split_csv(value):
        if "=" in item:
            label, path = item.split("=", 1)
            label = slugify(label)
        else:
            path = item
            label = slugify(path)
        model_path = Path(path).expanduser()
        if not model_path.is_absolute():
            model_path = root / model_path
        models.append({"label": label, "display_label": f"{label} {model_detail_brackets(str(model_path))}", "path": str(model_path)})
    return models


def wait_for_server(url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {url}")


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def server_command(root: Path, server: str) -> tuple[list[str], dict]:
    env = os.environ.copy()
    if server == "vanilla":
        return [str(root / "serve.sh")], env
    if server == "buun":
        build = root / BUUN_LIB_REL
        old_ld = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{build / 'bin'}:{build / 'src'}" + (f":{old_ld}" if old_ld else "")
        return [str(root / "serve-buun.sh")], env
    raise ValueError(f"unknown server: {server}")


def bench_args(args, transport: str, label: str, ctx: int, alias: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(args.root_dir) / "scripts/bench-llama-suite.py"),
        "--label",
        label,
        "--timeout",
        str(args.prompt_timeout),
    ]

    if transport == "curl":
        cmd.extend(["--transport", "http", "--model", alias])
    elif transport == "pi-blank":
        cmd.extend([
            "--transport",
            "pi",
            "--model",
            alias,
            "--pi-context-window",
            str(ctx),
        ])
    elif transport == "pi-current":
        cmd.extend([
            "--transport",
            "pi",
            "--pi-current-config",
        ])
        if args.pi_current_pass_model:
            cmd.extend(["--model", alias])
        if args.pi_current_no_session:
            cmd.append("--pi-no-session")
        if args.pi_current_no_tools:
            cmd.append("--pi-no-tools")
    else:
        raise ValueError(f"unknown transport: {transport}")

    if args.bench_extra_args:
        cmd.extend(split_csv(args.bench_extra_args) if "," in args.bench_extra_args else args.bench_extra_args.split())
    return cmd


def run_one(args, root: Path, raw_dir: Path, server: str, model: dict, ctx: int, transport: str) -> dict:
    alias = args.alias_template.format(model=model["label"], server=server, ctx=ctx)
    label = f"{server}-{model['label']}-ctx{ctx}-{transport}"
    server_log_path = raw_dir / f"{label}.server.log"
    result_path = raw_dir / f"{label}.results.json"

    cmd, env = server_command(root, server)
    env.update({
        "MODEL_FILE": model["path"],
        "CTX_SIZE": str(ctx),
        "ALIAS": alias,
        "PORT": str(args.port),
    })

    print(f"START {label} — model {model['display_label']}", file=sys.stderr, flush=True)
    with server_log_path.open("w") as server_log:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        try:
            wait_for_server(args.health_url, proc, args.startup_timeout)
            run_cmd = bench_args(args, transport, label, ctx, alias)
            with result_path.open("w") as result_file:
                completed = subprocess.run(
                    run_cmd,
                    cwd=root,
                    text=True,
                    stdout=result_file,
                    stderr=subprocess.PIPE,
                    timeout=args.suite_timeout,
                )
            if completed.stderr:
                (raw_dir / f"{label}.bench.stderr.log").write_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"bench client exited with code {completed.returncode}")
        except Exception as exc:
            error = {
                "label": label,
                "server": server,
                "model": model,
                "ctx": ctx,
                "transport": transport,
                "error": repr(exc),
            }
            result_path.write_text(json.dumps(error, indent=2) + "\n")
        finally:
            stop_process(proc)

    result = load_result(result_path)
    result.update({
        "server": server,
        "model_label": model["label"],
        "model_display": model["display_label"],
        "model_path": model["path"],
        "ctx": ctx,
        "transport_requested": transport,
        "server_log": str(server_log_path),
        "result_file": str(result_path),
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    time.sleep(args.cooldown)
    return result


def load_result(path: Path) -> dict:
    text = path.read_text(errors="replace")
    idx = text.find("{")
    if idx < 0:
        return {"error": "no JSON object in result file", "raw_preview": text[:500]}
    try:
        return json.loads(text[idx:])
    except Exception as exc:
        return {"error": f"could not parse JSON: {exc!r}", "raw_preview": text[:500]}


def compact_metrics(results: list[dict]) -> list[dict]:
    rows = []
    for item in results:
        tests = item.get("results", [])
        ok = [r for r in tests if "wall_s" in r and r.get("returncode", 0) == 0 and "error" not in r]
        errors = len(tests) - len(ok) + (1 if item.get("error") and not tests else 0)
        rows.append({
            "server": item.get("server"),
            "model": item.get("model_display") or item.get("model_label"),
            "context": item.get("ctx"),
            "transport": item.get("transport_requested"),
            "total_wall_s": round(sum(r["wall_s"] for r in ok), 3),
            "avg_wall_s": round(statistics.mean([r["wall_s"] for r in ok]), 3) if ok else None,
            "errors": errors,
            "prompts": [
                {
                    "id": r.get("id"),
                    "wall_s": r.get("wall_s"),
                    "status": r.get("finish_reason") or r.get("returncode") or ("error" if "error" in r else None),
                }
                for r in tests
            ],
        })
    return rows


def summarize(results: list[dict], llm_analysis: str | None = None) -> str:
    lines = [
        "# llama benchmark matrix",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "This file was generated by `scripts/run-llama-bench-matrix.py`.",
        "",
        "## Summary",
        "",
        "| server | model | context | transport | total wall | avg wall | errors |",
        "|---|---|---:|---|---:|---:|---:|",
    ]

    for item in results:
        tests = item.get("results", [])
        ok = [r for r in tests if "wall_s" in r and r.get("returncode", 0) == 0 and "error" not in r]
        errors = len(tests) - len(ok) + (1 if item.get("error") and not tests else 0)
        total = sum(r["wall_s"] for r in ok)
        avg = statistics.mean([r["wall_s"] for r in ok]) if ok else 0
        lines.append(
            f"| {item.get('server')} | {item.get('model_display') or item.get('model_label')} | {item.get('ctx') // 1024}k | "
            f"{item.get('transport_requested')} | {total:.2f}s | {avg:.2f}s | {errors} |"
        )

    lines += ["", "## Per-prompt wall times", ""]
    lines.append("| server | model | context | transport | prompt | wall | status |")
    lines.append("|---|---|---:|---|---|---:|---|")
    for item in results:
        for test in item.get("results", []):
            prompt = PROMPT_NAMES.get(test.get("id"), test.get("id", "unknown"))
            wall = f"{test['wall_s']:.2f}s" if "wall_s" in test else ""
            status = test.get("finish_reason") or test.get("returncode")
            if "error" in test:
                status = "error"
            lines.append(
                f"| {item.get('server')} | {item.get('model_display') or item.get('model_label')} | {item.get('ctx') // 1024}k | "
                f"{item.get('transport_requested')} | {prompt} | {wall} | {status} |"
            )
    lines.append("")
    if llm_analysis:
        lines += ["", "## LLM-generated observations and conclusion", "", llm_analysis.strip(), ""]
    return "\n".join(lines)


def call_chat_completion(url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You analyze benchmark results. Be concise, factual, and do not invent data."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return data["choices"][0]["message"].get("content") or ""


def generate_llm_analysis(args, root: Path, raw_dir: Path, results: list[dict], models: list[dict], contexts: list[int]) -> str:
    model = next((m for m in models if m["label"] == args.analysis_model), models[0]) if args.analysis_model else models[0]
    ctx = args.analysis_context or max(contexts)
    alias = args.analysis_alias
    label = f"analysis-{args.analysis_server}-{model['label']}-ctx{ctx}"
    server_log_path = raw_dir / f"{label}.server.log"

    cmd, env = server_command(root, args.analysis_server)
    env.update({
        "MODEL_FILE": model["path"],
        "CTX_SIZE": str(ctx),
        "ALIAS": alias,
        "PORT": str(args.port),
    })

    prompt = (
        "Generate markdown sections named exactly `## Observations` and `## Conclusion` for these benchmark results.\n"
        "Use only the data provided. Mention winners, slowdowns, errors, and trade-offs.\n"
        "Keep it concise: 4-8 bullets for observations and 1 short conclusion paragraph.\n\n"
        f"Compact benchmark metrics JSON:\n{json.dumps(compact_metrics(results), indent=2)}"
    )

    print(f"START {label} — model {model['display_label']}", file=sys.stderr, flush=True)
    with server_log_path.open("w") as server_log:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        try:
            wait_for_server(args.health_url, proc, args.startup_timeout)
            analysis = call_chat_completion(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                alias,
                prompt,
                args.analysis_max_tokens,
                args.analysis_timeout,
            )
            (raw_dir / f"{label}.analysis.md").write_text(analysis)
            return analysis
        finally:
            stop_process(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", default=None, help="default: <root>/docs/benchmarks/matrix/<timestamp>")
    parser.add_argument("--models", default=DEFAULT_MODEL, help="Comma list of model paths or label=path entries")
    parser.add_argument("--transports", default="curl,pi-blank,pi-current", help="Comma list: curl,pi-blank,pi-current")
    parser.add_argument("--contexts", default="32768,65536,131072,262144", help="Comma list of context sizes")
    parser.add_argument("--servers", default="vanilla,buun", help="Comma list: vanilla,buun")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--prompt-timeout", type=int, default=420, help="Timeout passed to each prompt call")
    parser.add_argument("--suite-timeout", type=int, default=3600, help="Timeout for one full prompt suite")
    parser.add_argument("--cooldown", type=float, default=3)
    parser.add_argument("--alias-template", default="selected", help="Server model alias; fields: {model}, {server}, {ctx}")
    parser.add_argument("--pi-current-pass-model", action="store_true", help="Pass --model <alias> to pi-current runs")
    parser.add_argument("--pi-current-no-session", action="store_true", help="Pass --pi-no-session to pi-current runs")
    parser.add_argument("--pi-current-no-tools", action="store_true", help="Pass --pi-no-tools to pi-current runs")
    parser.add_argument("--bench-extra-args", default="", help="Extra flags for bench-llama-suite.py")
    parser.add_argument("--llm-analysis", action=argparse.BooleanOptionalAction, default=True, help="After the matrix, start llama-server and ask it to generate observations/conclusion; enabled by default")
    parser.add_argument("--analysis-server", choices=["vanilla", "buun"], default="buun")
    parser.add_argument("--analysis-model", help="Model label to use for analysis; defaults to the first --models entry")
    parser.add_argument("--analysis-context", type=int, help="Context size for analysis; defaults to the largest --contexts value")
    parser.add_argument("--analysis-alias", default="analysis")
    parser.add_argument("--analysis-max-tokens", type=int, default=900)
    parser.add_argument("--analysis-timeout", type=int, default=900)
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    args.root_dir = str(root)
    models = parse_models(args.models, root)
    transports = split_csv(args.transports)
    contexts = [int(x) for x in split_csv(args.contexts)]
    servers = split_csv(args.servers)

    allowed_transports = {"curl", "pi-blank", "pi-current"}
    allowed_servers = {"vanilla", "buun"}
    bad_transports = set(transports) - allowed_transports
    bad_servers = set(servers) - allowed_servers
    if bad_transports:
        raise SystemExit(f"unknown transports: {sorted(bad_transports)}")
    if bad_servers:
        raise SystemExit(f"unknown servers: {sorted(bad_servers)}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root / "docs/benchmarks/matrix" / time.strftime("%Y%m%d-%H%M%S")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    matrix = [
        (server, model, ctx, transport)
        for model in models
        for ctx in contexts
        for server in servers
        for transport in transports
    ]

    print(f"runs: {len(matrix)}", file=sys.stderr)
    print(f"out: {out_dir}", file=sys.stderr)

    results = []
    for server, model, ctx, transport in matrix:
        results.append(run_one(args, root, raw_dir, server, model, ctx, transport))
        (out_dir / "all-results.json").write_text(json.dumps(results, indent=2) + "\n")
        (out_dir / "bench.md").write_text(summarize(results))

    llm_analysis = None
    if args.llm_analysis:
        try:
            llm_analysis = generate_llm_analysis(args, root, raw_dir, results, models, contexts)
        except Exception as exc:
            llm_analysis = f"LLM analysis generation failed: `{exc!r}`"
        (out_dir / "bench.md").write_text(summarize(results, llm_analysis))

    print(f"wrote {out_dir / 'bench.md'}")
    print(f"wrote {out_dir / 'all-results.json'}")
    print(f"wrote raw files under {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


TESTS = [
    {
        "id": "regular_kv_cache",
        "kind": "regular",
        "messages": [
            {
                "role": "user",
                "content": "In two sentences, explain why KV cache size grows with context length.",
            }
        ],
        "max_tokens": 140,
    },
    {
        "id": "regular_debug",
        "kind": "regular",
        "messages": [
            {
                "role": "user",
                "content": "A Python CLI exits with code 0 but writes no output. Give a concise debugging checklist with 4 likely causes.",
            }
        ],
        "max_tokens": 180,
    },
    {
        "id": "internet_latest_release",
        "kind": "internet_prompt",
        "messages": [
            {
                "role": "user",
                "content": "Use internet search to answer: what is the latest llama.cpp release today? If you cannot browse, say exactly that and do not guess.",
            }
        ],
        "max_tokens": 120,
    },
    {
        "id": "internet_news",
        "kind": "internet_prompt",
        "messages": [
            {
                "role": "user",
                "content": "Use internet search to summarize the top AI infrastructure news from today. If you cannot browse, say exactly that and do not invent news.",
            }
        ],
        "max_tokens": 160,
    },
    {
        "id": "tool_weather",
        "kind": "tool_call",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Stockholm right now? Use the available tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 120,
    },
    {
        "id": "tool_search",
        "kind": "tool_call",
        "messages": [
            {
                "role": "user",
                "content": "Search the web for the latest llama.cpp release and report the version. Use the available tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 120,
    },
]


def call_http(url, model, test, timeout):
    payload = {
        "model": model,
        "messages": test["messages"],
        "temperature": 0,
        "max_tokens": test["max_tokens"],
        "stream": False,
    }
    if "tools" in test:
        payload["tools"] = test["tools"]
        payload["tool_choice"] = test.get("tool_choice", "auto")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    wall = time.time() - started

    choice = data["choices"][0]
    message = choice.get("message", {})
    timings = data.get("timings", {})
    usage = data.get("usage", {})

    return {
        "id": test["id"],
        "kind": test["kind"],
        "transport": "http",
        "wall_s": round(wall, 3),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tps": round(timings.get("prompt_per_second", 0), 2) if timings else None,
        "decode_tps": round(timings.get("predicted_per_second", 0), 2) if timings else None,
        "tool_calls": message.get("tool_calls"),
        "content": (message.get("content") or "")[:500].replace("\n", " "),
    }


def test_prompt(test):
    return "\n\n".join(message["content"] for message in test["messages"] if message.get("role") == "user")


def write_pi_models_config(pi_dir, provider, model, base_url, context_window, max_tokens):
    pi_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "providers": {
            provider: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": "llama-server",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": model,
                        "name": f"local llama-server {model}",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": context_window,
                        "maxTokens": max_tokens,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }
    (pi_dir / "models.json").write_text(json.dumps(config, indent=2) + "\n")


def call_pi(args, test, pi_dir):
    env = os.environ.copy()
    if pi_dir is not None:
        env["PI_CODING_AGENT_DIR"] = str(pi_dir)

    cmd = [args.pi_bin]

    if args.pi_provider:
        cmd.extend(["--provider", args.pi_provider])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.pi_thinking:
        cmd.extend(["--thinking", args.pi_thinking])

    if args.pi_current_config:
        if args.pi_no_session:
            cmd.append("--no-session")
        if args.pi_no_tools:
            cmd.append("--no-tools")
    else:
        cmd.extend([
            "--no-session",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ])
        if not args.pi_tools:
            cmd.append("--no-tools")

    if args.pi_system_prompt:
        cmd.extend(["--system-prompt", args.pi_system_prompt])

    cmd.extend(["-p", test_prompt(test)])

    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=args.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    wall = time.time() - started

    return {
        "id": test["id"],
        "kind": test["kind"],
        "transport": "pi",
        "wall_s": round(wall, 3),
        "returncode": completed.returncode,
        "stdout_chars": len(completed.stdout),
        "stderr_chars": len(completed.stderr),
        "tool_schema_sent": False,
        "content": completed.stdout[:500].replace("\n", " "),
        "stderr": completed.stderr[:500].replace("\n", " "),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "pi"], default="http")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model")
    parser.add_argument("--label", default="llama-server")
    parser.add_argument("--timeout", type=int, default=420)

    parser.add_argument("--pi-bin", default="pi")
    parser.add_argument("--pi-current-config", action="store_true", help="Use your normal pi configuration instead of a temporary isolated local-provider config")
    parser.add_argument("--pi-provider")
    parser.add_argument("--pi-base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--pi-context-window", type=int, default=262144)
    parser.add_argument("--pi-max-tokens", type=int, default=2048)
    parser.add_argument("--pi-system-prompt")
    parser.add_argument("--pi-thinking", help="Thinking level to pass to pi; defaults to off in isolated pi mode and omitted with --pi-current-config")
    parser.add_argument("--pi-config-dir", help="Use an existing PI_CODING_AGENT_DIR instead of a temporary one")
    parser.add_argument("--pi-tools", action="store_true", help="Allow pi tools in isolated pi mode; default disables all pi tools for repeatability")
    parser.add_argument("--pi-no-tools", action="store_true", help="Disable tools when using --pi-current-config")
    parser.add_argument("--pi-no-session", action="store_true", help="Disable session saving when using --pi-current-config")
    parser.add_argument("--cwd", default=str(Path.cwd()))
    args = parser.parse_args()

    if args.model is None and args.transport == "http":
        args.model = "selected"
    if args.transport == "pi" and not args.pi_current_config:
        if args.model is None:
            args.model = "selected"
        if args.pi_provider is None:
            args.pi_provider = "llama-local"
        if args.pi_system_prompt is None:
            args.pi_system_prompt = "You are a concise assistant. Answer directly."
        if args.pi_thinking is None:
            args.pi_thinking = "off"

    pi_tmp = None
    pi_dir = None
    if args.transport == "pi" and not args.pi_current_config:
        if args.pi_config_dir:
            pi_dir = Path(args.pi_config_dir)
        else:
            pi_tmp = tempfile.mkdtemp(prefix="bench-llama-pi-")
            pi_dir = Path(pi_tmp)
            write_pi_models_config(
                pi_dir,
                args.pi_provider,
                args.model,
                args.pi_base_url,
                args.pi_context_window,
                args.pi_max_tokens,
            )

    results = []
    try:
        for test in TESTS:
            print(f"RUN {args.label} {test['id']}", file=sys.stderr, flush=True)
            try:
                if args.transport == "http":
                    results.append(call_http(args.url, args.model, test, args.timeout))
                else:
                    results.append(call_pi(args, test, pi_dir))
            except Exception as exc:
                results.append({"id": test["id"], "kind": test["kind"], "transport": args.transport, "error": repr(exc)})
    finally:
        if pi_tmp:
            shutil.rmtree(pi_tmp, ignore_errors=True)

    print(json.dumps({"label": args.label, "transport": args.transport, "results": results}, indent=2))


if __name__ == "__main__":
    main()

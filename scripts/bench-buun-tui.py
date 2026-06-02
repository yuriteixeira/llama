#!/usr/bin/env python3
"""Curses TUI for configuring and running buun llama-bench comparisons."""

from __future__ import annotations

import argparse
import curses
from pathlib import Path

from bench_buun_tui.config_store import default_config_path, load_config, remember_config
from bench_buun_tui.runner import run_config
from bench_buun_tui.tui import tui_main


def main() -> int:
    args = parse_args()
    if args.config and args.run:
        return run_config(args.config)
    return run_tui_loop(load_initial_config(args.config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="saved INI config to load")
    parser.add_argument("--run", action="store_true", help="run the saved config without opening the TUI")
    return parser.parse_args()


def load_initial_config(config_path: Path | None):
    if config_path:
        cfg = load_config(config_path)
        remember_config(config_path)
        return cfg
    previous = default_config_path()
    return load_config(previous) if previous else None


def run_tui_loop(cfg) -> int:
    last_rc = 0
    while True:
        path = curses.wrapper(tui_main, cfg)
        if not path:
            return last_rc
        print(f"Saved config: {path}")
        last_rc = run_config(path)
        cfg = load_config(path)
        if not wait_for_enter():
            return last_rc


def wait_for_enter() -> bool:
    print("\nRun finished. Returning to TUI; press Enter to continue...", flush=True)
    try:
        input()
        return True
    except EOFError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

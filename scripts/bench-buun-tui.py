#!/usr/bin/env python3
"""Curses TUI for configuring and running buun llama-bench comparisons."""

from __future__ import annotations

import argparse
import curses
import logging
import subprocess
import sys
from pathlib import Path

from bench_buun_tui.config_store import default_config_path, load_config, remember_config
from bench_buun_tui.report_server import start_report_server
from bench_buun_tui.paths import RESULT_DIR
from bench_buun_tui.runner import run_config
from bench_buun_tui.tui import tui_main

RANK_SCRIPT = Path(__file__).with_name("rank-buun-results.py")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("bench-buun-tui starting args=%s", args)
    if args.config and args.run:
        server = start_report_server()
        logging.info("report server started root=%s url=%s", server.root, server.url)
        print(f"Serving benchmark reports from {server.root}: {server.url}")
        try:
            rc = run_config(args.config)
            logging.info("run finished config=%s rc=%s", args.config, rc)
            return rc
        finally:
            logging.info("stopping report server")
            server.stop()
    return run_tui_loop(load_initial_config(args.config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="saved INI config to load")
    parser.add_argument("--run", action="store_true", help="run the saved config without opening the TUI")
    parser.add_argument("--log-file", type=Path, default=RESULT_DIR / "bench-buun-tui.log", help="TUI log file")
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.info("logging to %s", log_file)


def load_initial_config(config_path: Path | None):
    if config_path:
        logging.info("loading config %s", config_path)
        cfg = load_config(config_path)
        remember_config(config_path)
        return cfg
    previous = default_config_path()
    if previous:
        logging.info("loading previous config %s", previous)
    return load_config(previous) if previous else None


def run_tui_loop(cfg) -> int:
    last_rc = 0
    server = start_report_server()
    logging.info("report server started root=%s url=%s", server.root, server.url)
    print(f"Serving benchmark reports from {server.root}: {server.url}")
    try:
        while True:
            path = curses.wrapper(tui_main, cfg, server.url)
            if not path:
                logging.info("TUI exited rc=%s", last_rc)
                return last_rc
            logging.info("saved config from TUI: %s", path)
            print(f"Saved config: {path}")
            last_rc = run_config(path)
            logging.info("run finished config=%s rc=%s", path, last_rc)
            cfg = load_config(path)
            if prompt_update_ranking(server.url):
                update_ranking_reports()
            if not wait_for_enter(server.url):
                return last_rc
    finally:
        logging.info("stopping report server")
        server.stop()


def prompt_update_ranking(report_url: str) -> bool:
    print(f"\nReports served at: {report_url}")
    try:
        answer = input("Update ranking results now? [Y/n] ").strip().lower()
    except EOFError:
        return False
    return answer in {"", "y", "yes"}


def update_ranking_reports() -> bool:
    cmd = [
        sys.executable,
        str(RANK_SCRIPT),
        "--input",
        str(RESULT_DIR),
        "--output",
        str(RESULT_DIR / "ranking.md"),
    ]
    print("Running ranking update...", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    if proc.returncode == 0:
        print(f"Ranking updated: {RESULT_DIR / 'ranking.md'}")
        print(f"Browser report: {RESULT_DIR / 'index.html'}")
        return True
    print(f"Ranking update failed (exit {proc.returncode}).")
    return False


def wait_for_enter(report_url: str) -> bool:
    print(f"\nReports served at: {report_url}")
    print("Returning to TUI; press Enter to continue...", flush=True)
    try:
        input()
        return True
    except EOFError:
        return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("bench-buun-tui crashed")
        raise

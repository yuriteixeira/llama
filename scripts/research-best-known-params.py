#!/usr/bin/env python3
"""Generate best-known model parameter recommendations for the buun TUI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench_buun_tui.best_known import BEST_KNOWN_PARAMS, ResearchError, build_research_prompt, run_best_known_research
from bench_buun_tui.models import find_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BEST_KNOWN_PARAMS, help="INI file to write")
    parser.add_argument("--print-prompt", action="store_true", help="print the generated pi -p prompt and exit")
    parser.add_argument("--stream", action="store_true", help="run pi with --mode json and print assistant deltas as they arrive")
    args = parser.parse_args()

    if args.print_prompt:
        print(build_research_prompt(find_models()))
        return 0

    try:
        run = run_best_known_research(args.output, on_output=print_stream_chunk if args.stream else None, stream=args.stream)
    except ResearchError as exc:
        print(f"\nResearch errored: {exc}" + (f" (status {exc.returncode})" if exc.returncode is not None else ""))
        print(f"Command ran:\n{exc.command}\n")
        print(f"Response:\n{exc.response.rstrip()}\n")
        return 1
    print(f"\nCommand ran:\n{run.command}\n")
    print(f"Response:\n{run.response.rstrip()}\n")
    print(f"Saved best-known params: {run.path}")
    return 0


def print_stream_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

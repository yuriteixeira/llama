#!/usr/bin/env python3
"""Generate best-known model parameter recommendations for the buun TUI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench_buun_tui.best_known import BEST_KNOWN_PARAMS, build_research_prompt, generate_best_known_params
from bench_buun_tui.models import find_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BEST_KNOWN_PARAMS, help="INI file to write")
    parser.add_argument("--print-prompt", action="store_true", help="print the generated pi -p prompt and exit")
    args = parser.parse_args()

    if args.print_prompt:
        print(build_research_prompt(find_models()))
        return 0

    path = generate_best_known_params(args.output)
    print(f"Saved best-known params: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

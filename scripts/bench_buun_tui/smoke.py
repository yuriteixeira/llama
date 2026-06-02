from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .commands import profile_command
from .paths import ROOT
from .profile import Profile


@dataclass(frozen=True)
class SmokeTestResult:
    command: list[str]
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def smoke_test_context(profile: Profile) -> int:
    vals = [int(part.strip()) for part in profile.depths.split(",") if part.strip().isdigit()]
    return max(vals) if vals else 512


def smoke_test_profile(profile: Profile, desired_context: int) -> Profile:
    test_profile = copy.deepcopy(profile)
    test_profile.depths = str(desired_context)
    test_profile.prompt = "0"
    test_profile.gen = "1"
    test_profile.repetitions = "1"
    test_profile.warmup = "false"
    return test_profile


def smoke_test_command(profile: Profile, desired_context: int) -> list[str]:
    return profile_command(smoke_test_profile(profile, desired_context), Path("smoke-test.jsonl"))


def run_smoke_test(
    profile: Profile,
    desired_context: int,
    on_output: Callable[[str], None] | None = None,
    timeout: int | None = None,
) -> SmokeTestResult:
    cmd = smoke_test_command(profile, desired_context)
    if on_output is None:
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return SmokeTestResult(command=cmd, returncode=completed.returncode, output=completed.stdout or "")
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            output += f"\nSmoke test timed out after {timeout} seconds.\n"
            return SmokeTestResult(command=cmd, returncode=124, output=output)

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert proc.stdout is not None
    for chunk in proc.stdout:
        output_parts.append(chunk)
        on_output(chunk)
    return SmokeTestResult(command=cmd, returncode=proc.wait(), output="".join(output_parts))

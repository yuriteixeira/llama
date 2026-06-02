from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .commands import profile_command
from .paths import ROOT
from .process_stream import StreamResult, StreamStatus, run_streaming_process
from .profile import Profile


@dataclass(frozen=True)
class SmokeTestResult:
    command: list[str]
    returncode: int
    output: str
    reason: str = ""
    timed_out: bool = False
    idle_timed_out: bool = False
    cancelled: bool = False

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
    idle_timeout: int | None = 120,
    on_heartbeat: Callable[[StreamStatus], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    heartbeat_interval: float = 1.0,
) -> SmokeTestResult:
    cmd = smoke_test_command(profile, desired_context)
    result: StreamResult = run_streaming_process(
        cmd,
        cwd=ROOT,
        on_output=on_output,
        on_heartbeat=on_heartbeat,
        cancel_check=cancel_check,
        total_timeout=timeout,
        idle_timeout=idle_timeout,
        heartbeat_interval=heartbeat_interval,
    )
    return SmokeTestResult(
        command=result.command,
        returncode=result.returncode,
        output=result.output,
        reason=result.reason,
        timed_out=result.timed_out,
        idle_timed_out=result.idle_timed_out,
        cancelled=result.cancelled,
    )

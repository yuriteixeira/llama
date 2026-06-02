from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import ROOT


@dataclass(frozen=True)
class StreamStatus:
    command: list[str]
    elapsed_seconds: float
    idle_seconds: float
    output: str
    output_lines: int


@dataclass(frozen=True)
class StreamResult:
    command: list[str]
    returncode: int
    output: str
    elapsed_seconds: float
    reason: str = ""
    timed_out: bool = False
    idle_timed_out: bool = False
    cancelled: bool = False


_SENTINEL = object()


def run_streaming_process(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    on_output: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[StreamStatus], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    total_timeout: int | float | None = None,
    idle_timeout: int | float | None = None,
    heartbeat_interval: float = 1.0,
    terminate_grace_seconds: float = 3.0,
) -> StreamResult:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        bufsize=1,
        start_new_session=True,
    )
    q: queue.Queue[str | object] = queue.Queue()
    started = time.monotonic()
    last_output = started
    output_parts: list[str] = []
    reason = ""
    timed_out = False
    idle_timed_out = False
    cancelled = False

    def reader() -> None:
        assert proc.stdout is not None
        try:
            for chunk in proc.stdout:
                q.put(chunk)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    done = False
    next_heartbeat = started
    sentinel_seen = False

    try:
        while True:
            now = time.monotonic()
            wait = max(0.0, min(heartbeat_interval, max(0.0, next_heartbeat - now)))
            try:
                item = q.get(timeout=wait)
            except queue.Empty:
                item = None

            now = time.monotonic()
            if item is _SENTINEL:
                sentinel_seen = True
            elif isinstance(item, str):
                output_parts.append(item)
                last_output = now
                if on_output:
                    on_output(item)
            if on_heartbeat and now >= next_heartbeat:
                on_heartbeat(stream_status(cmd, started, last_output, output_parts))
                next_heartbeat = now + heartbeat_interval

            if cancel_check and cancel_check():
                cancelled = True
                reason = "cancelled"
                break
            if total_timeout is not None and now - started >= total_timeout:
                timed_out = True
                reason = f"timed out after {int(total_timeout)}s"
                break
            if idle_timeout is not None and now - last_output >= idle_timeout:
                idle_timed_out = True
                reason = f"idle timeout after {int(idle_timeout)}s without output"
                break
            if proc.poll() is not None and sentinel_seen and q.empty():
                done = True
                break
    finally:
        if not done:
            terminate_process(proc, terminate_grace_seconds)
        thread.join(timeout=max(terminate_grace_seconds, 1.0))

    if not done:
        returncode = proc.poll()
        if returncode is None:
            returncode = 130 if cancelled else 124 if (timed_out or idle_timed_out) else 1
    else:
        returncode = proc.returncode if proc.returncode is not None else proc.wait()

    output = "".join(output_parts)
    if reason:
        output += f"\n{reason}.\n"
    return StreamResult(
        command=cmd,
        returncode=returncode,
        output=output,
        elapsed_seconds=time.monotonic() - started,
        reason=reason,
        timed_out=timed_out,
        idle_timed_out=idle_timed_out,
        cancelled=cancelled,
    )


def stream_status(cmd: list[str], started: float, last_output: float, output_parts: list[str]) -> StreamStatus:
    now = time.monotonic()
    return StreamStatus(
        command=cmd,
        elapsed_seconds=now - started,
        idle_seconds=now - last_output,
        output="".join(output_parts),
        output_lines=len(output_parts),
    )


def terminate_process(proc: subprocess.Popen, grace_seconds: float) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

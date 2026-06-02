# PLAN.md — Mitigate Buun Benchmark TUI Hangs

## Goal

Make `./scripts/bench-buun-tui.py` feel responsive during long-running operations and prevent silent subprocesses from making the curses UI appear frozen.

## Problem summary

The TUI sometimes appears to hang because long-running work is executed synchronously on the curses/UI thread. The screen only updates when a child process emits stdout, so a still-running but silent process looks like a frozen UI.

Known cases:

- `scripts/bench_buun_tui/tui.py`
  - `run_best_known_research_with_popup()` blocks while `pi --mode json ...` runs.
  - `run_smoke_test_and_repair_with_popup()` blocks while `llama-bench` runs.
  - `draw_output_popup()` is only called from subprocess output callbacks, so there is no heartbeat during output gaps.
- `scripts/bench_buun_tui/smoke.py`
  - `run_smoke_test(..., on_output=...)` uses `subprocess.Popen()` and loops over `proc.stdout` with no timeout in streaming mode.
  - The `timeout` argument only applies to the non-streaming `subprocess.run()` path.
- `scripts/bench_buun_tui/best_known.py`
  - `run_research_command()` and `run_research_command_json()` stream `Popen.stdout` with no timeout, idle-timeout, or cancel mechanism.
- `scripts/bench-buun-tui.py`
  - After **Save config and run**, curses exits and `run_config(path)` runs in the foreground. This is expected, but can look like the UI disappeared if the user expects an in-TUI runner.
- `scripts/bench_buun_tui/tui.py`
  - Profile field selection can move beyond visible rows on small terminals because `draw_profile_fields_panel()` renders only `fields[: panel.inner_height]` without scrolling.

## Proposed changes

### 1. Add a reusable async streaming subprocess helper

Create a helper module, for example:

- `scripts/bench_buun_tui/process_stream.py`

Preferred implementation:

- Use `asyncio.create_subprocess_exec()` for subprocesses.
- Read stdout/stderr asynchronously with `await proc.stdout.readline()` or short `asyncio.wait_for()` polling intervals.
- Drive curses redraws from an async loop with a heartbeat tick, instead of only redrawing when child output arrives.
- Poll keyboard input using curses non-blocking mode (`stdscr.nodelay(True)`) or `asyncio.to_thread(stdscr.getch)` where appropriate.

Fallback implementation if asyncio/curses integration becomes awkward:

- Use a reader thread + `queue.Queue`, or `select`/non-blocking I/O, behind the same helper API.

Responsibilities:

- Start a subprocess with stdout/stderr merged or separated as needed.
- Stream output without blocking the UI thread forever.
- Support:
  - total timeout
  - idle timeout when no output is received
  - periodic heartbeat callback even with no output
  - cancellation callback, e.g. Esc/Ctrl-C from curses
  - graceful terminate, then kill after a short grace period
- Return command, return code, collected output, and timeout/cancel reason.

### 2. Apply streaming helper to smoke tests

Update:

- `scripts/bench_buun_tui/smoke.py`
- `scripts/bench_buun_tui/tui.py`

Desired behavior:

- Use the same timeout behavior in streaming and non-streaming smoke-test modes.
- Show a live popup footer such as:
  - `Running... 42s elapsed • Esc cancel`
  - `No output for 18s, still waiting... • Esc cancel`
- If timeout/cancel occurs, terminate `llama-bench` and show the partial output and reason.

### 3. Apply streaming helper to best-known research

Update:

- `scripts/bench_buun_tui/best_known.py`
- `scripts/bench_buun_tui/tui.py`

Desired behavior:

- Keep streaming pi output when available.
- Redraw heartbeat/status while pi is thinking or using tools silently.
- Allow Esc/Ctrl-C to stop the research subprocess.
- Add a generous total timeout or configurable timeout to avoid indefinite waits.

### 4. Improve foreground benchmark run UX

Update:

- `scripts/bench-buun-tui.py`
- optionally `scripts/bench_buun_tui/runner.py`

Options:

1. Minimal: before running, print a clear message after curses exits:
   - `Running benchmark outside TUI. This may take a long time. Progress will print below.`
2. Better: add an in-TUI run progress screen that wraps `run_config()` output using the async streaming helper or refactors `runner.py` to emit async/callback progress events.

### 5. Add profile editor scrolling

Update:

- `scripts/bench_buun_tui/tui.py`

Desired behavior:

- Track a scroll offset in `edit_profile()`.
- Render `fields[offset : offset + panel.inner_height]`.
- Keep selected field visible as the user moves.
- Show a small indicator like `3/27` or `↑ more` / `↓ more`.

## Acceptance criteria

- Silent `llama-bench` or `pi` subprocesses no longer make the popup look frozen; a heartbeat updates at least once per second.
- Users can cancel smoke tests and best-known research from the TUI.
- Streaming smoke tests honor timeout/idle-timeout settings.
- Timed-out or cancelled runs show partial output and the command that was run.
- Profile editor selection always remains visible, even in a small terminal.
- **Save config and run** clearly communicates that the benchmark is running outside the curses UI, or provides an in-TUI progress screen.

## Notes

This is primarily a responsiveness and UX issue, not a curses rendering bug. Async subprocess handling is the preferred direction because it gives the TUI a natural heartbeat/cancel loop while child processes run. The main risk is subprocess management: make sure cancelling or timing out does not leave orphaned `llama-bench`, `pi`, or child tool processes running.

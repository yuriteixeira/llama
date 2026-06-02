from __future__ import annotations

import copy
import curses
import json
import logging
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from .best_known import (
    BEST_KNOWN_PARAMS,
    ResearchError,
    apply_recommendation,
    build_apply_plan,
    find_recommendations,
    load_research_meta,
    run_best_known_research,
)
from .commands import profile_server_command
from .config_store import find_saved_configs, load_config, remember_config, save_config, slugify
from .models import (
    cache_type_is_compatible,
    choose_default_model,
    compatible_cache_type_options,
    find_models,
    gpu_layer_options,
    model_description,
    model_detail_brackets,
    model_label,
    model_metadata,
    model_options,
)
from .offload import OffloadError, calculate_offload, format_bytes
from .process_stream import StreamStatus
from .repair import diagnose_failure, next_repair_step
from .paths import CONFIG_DIR
from .profile import BenchConfig, CACHE_STRATEGIES, FIELD_DESCRIPTIONS, PRESETS, Profile, apply_cache_strategy, cache_types_text, resolve_cache_types
from .smoke import run_smoke_test, smoke_test_command, smoke_test_context


def new_default_config(models: list[str]) -> BenchConfig:
    cfg = BenchConfig()
    default_model = choose_default_model(models)
    cfg.profiles[0].model = default_model
    cfg.profiles[1] = copy.deepcopy(cfg.profiles[0])
    cfg.profiles[1].name = "profile_b"
    return cfg


def tui_main(stdscr, cfg: BenchConfig | None = None, report_url: str = "") -> Path | None:
    curses.curs_set(0)
    draw_loading_screen(stdscr, "Loading Buun TUI", "Scanning model files...")
    models = find_models()
    state = TuiState(cfg or new_default_config(models), report_url, models)
    return main_menu(stdscr, state)


class TuiState:
    def __init__(self, cfg: BenchConfig, report_url: str = "", models: list[str] | None = None):
        self.cfg = cfg
        self.report_url = report_url
        self.models = models or find_models()
        self.selected = 0
        self.actions = [
            "Load saved config",
            "Scan models and research best-known params",
            "Edit run prefix",
            "Edit profile_a",
            "Edit profile_b",
            "Save config",
            "Save config and run",
            "Quit",
        ]


def main_menu(stdscr, state: TuiState) -> Path | None:
    while True:
        draw_main_menu(stdscr, state)
        result = handle_main_menu_key(stdscr, state, stdscr.getch())
        if result is not False:
            return result


def draw_main_menu(stdscr, state: TuiState) -> None:
    stdscr.clear()
    layout = main_menu_layout()
    draw_header_panel(stdscr, layout["header"], state.cfg.run_prefix, state.report_url)
    draw_actions_panel(stdscr, layout["actions"], state.actions, state.selected)
    draw_profiles_panel(stdscr, layout["profiles"], state.cfg.profiles)
    draw_help_panel(stdscr, layout["help"], "↑/↓ move  •  Enter select  •  q/Esc quit")


def draw_header_panel(stdscr, panel: "Panel", run_prefix: str, report_url: str = "") -> None:
    draw_panel(stdscr, panel, "Buun llama-bench TUI")
    write_in_panel(stdscr, panel, 0, "Configure two reproducible benchmark profiles")
    write_in_panel(stdscr, panel, 1, f"Run prefix: {run_prefix}")
    if report_url:
        write_in_panel(stdscr, panel, 2, f"Reports: {report_url}")


def draw_actions_panel(stdscr, panel: "Panel", actions: list[str], selected: int) -> None:
    draw_panel(stdscr, panel, "Actions")
    row = 0
    for i, action in enumerate(actions):
        if row >= panel.inner_height:
            return
        attr = curses.A_REVERSE if i == selected else curses.A_NORMAL
        write_in_panel(stdscr, panel, row, f"> {action}" if i == selected else f"  {action}", attr)
        row += 1
        if i in {1, 4} and row < panel.inner_height:
            write_action_separator(stdscr, panel, row)
            row += 1


def write_action_separator(stdscr, panel: "Panel", row: int) -> None:
    write_in_panel(stdscr, panel, row, "─" * panel.inner_width)


def draw_profiles_panel(stdscr, panel: "Panel", profiles: list[Profile]) -> None:
    draw_panel(stdscr, panel, "Profiles")
    row = 0
    for profile in profiles:
        row = draw_profile_summary(stdscr, panel, profile, row)


def draw_profile_summary(stdscr, panel: "Panel", profile: Profile, row: int) -> int:
    lines = [
        f"{profile.name}: {model_label(profile.model)}",
        f"cache={profile.cache_strategy} kv={cache_types_text(profile)} depths={profile.depths}",
        f"batch={profile.batch_size} ubatch={profile.ubatch_size} threads={profile.threads}",
        f"moe={profile.n_cpu_moe} ngl={profile.gpu_layers} fa={profile.flash_attn} mmap={profile.no_mmap}",
    ]
    for line in lines:
        if row >= panel.inner_height:
            return row
        write_in_panel(stdscr, panel, row, line)
        row += 1
    if row < panel.inner_height:
        write_in_panel(stdscr, panel, row, "")
        row += 1
    return row


def draw_help_panel(stdscr, panel: "Panel", text: str) -> None:
    draw_panel(stdscr, panel, "Keys")
    write_in_panel(stdscr, panel, 0, text)


def handle_main_menu_key(stdscr, state: TuiState, key: int):
    if key in (curses.KEY_UP, ord("k")):
        state.selected = (state.selected - 1) % len(state.actions)
    elif key in (curses.KEY_DOWN, ord("j")):
        state.selected = (state.selected + 1) % len(state.actions)
    elif key in (ord("q"), 27):
        return None
    elif key in (10, 13):
        return handle_main_action(stdscr, state)
    return False


def handle_main_action(stdscr, state: TuiState):
    selected = state.selected
    if selected == 0:
        load_saved_config_action(stdscr, state)
    elif selected == 1:
        research_best_known_action(stdscr)
    elif selected == 2:
        edit_run_prefix_action(stdscr, state)
    elif selected in (3, 4):
        profile_idx = selected - 3
        other_idx = 1 - profile_idx
        edit_profile(stdscr, state.cfg.profiles[profile_idx], state.models, state.cfg.profiles[other_idx])
    elif selected in (5, 6):
        return save_config_action(stdscr, state.cfg, run_after_save=selected == 6)
    elif selected == 7:
        return None
    return False


def research_best_known_action(stdscr) -> None:
    run_best_known_research_with_popup(stdscr)


def run_best_known_research_with_popup(stdscr, models: list[str] | None = None) -> bool:
    scope = f" for {model_label(models[0])}" if models and len(models) == 1 else " for all models"
    output_lines = [f"Starting pi streaming research{scope} with: pi --mode json <research prompt>"]
    stdscr.nodelay(True)
    try:
        def on_output(chunk: str) -> None:
            append_output_lines(output_lines, chunk)
            draw_output_popup(stdscr, "Researching best-known params", output_lines, "Streaming... Esc cancels")

        def on_heartbeat(status: StreamStatus) -> None:
            draw_output_popup(stdscr, "Researching best-known params", output_lines, stream_footer("pi", status, cancel_hint=True))

        def cancel_check() -> bool:
            return popup_cancel_requested(stdscr)

        draw_output_popup(stdscr, "Researching best-known params", output_lines, "Starting pi --mode json... Esc cancels")
        try:
            run = run_best_known_research(on_output=on_output, stream=True, models=models, on_heartbeat=on_heartbeat, cancel_check=cancel_check)
        except ResearchError as exc:
            lines = research_result_lines(
                saved_path=None,
                command=exc.command,
                response=exc.response,
                error=f"{exc}" + (f" (status {exc.returncode})" if exc.returncode is not None else ""),
            )
            show_scroll_text(stdscr, "Best-known params research errored", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
            return False
        except Exception as exc:
            lines = research_result_lines(saved_path=None, command="pi --mode json <research prompt>", response="\n".join(output_lines), error=str(exc))
            show_scroll_text(stdscr, "Best-known params research errored", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
            return False
        lines = research_result_lines(saved_path=run.path, command=run.command, response=run.response)
        show_scroll_text(stdscr, "Best-known params research result", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
        return True
    finally:
        stdscr.nodelay(False)


def append_output_lines(lines: list[str], chunk: str) -> None:
    parts = chunk.replace("\r", "").split("\n")
    if len(parts) == 1:
        lines[-1] = lines[-1] + parts[0]
        return
    lines[-1] = lines[-1] + parts[0]
    lines.extend(parts[1:])
    if lines and lines[-1] == "":
        lines.pop()
    del lines[:-500]


def draw_output_popup(stdscr, title: str, lines: list[str], footer: str) -> None:
    stdscr.clear()
    rows, cols = screen_size()
    height = max(3, min(max(3, rows - 2), max(3, rows * 3 // 4)))
    width = max(10, min(max(10, cols - 4), max(10, cols * 4 // 5)))
    top = max(0, (rows - height) // 2)
    left = max(0, (cols - width) // 2)
    win = curses.newwin(height, width, top, left)
    win.box()
    safe_addstr(win, 0, 2, f" {title} "[: max(0, width - 4)], curses.A_BOLD)
    body_height = max(1, height - 4)
    body_width = max(1, width - 4)
    visible = wrap_lines_for_screen(lines, body_width)[-body_height:]
    for row, line in enumerate(visible, start=1):
        safe_addstr(win, row, 2, line[:body_width])
    safe_addstr(win, height - 2, 2, footer[:body_width])
    stdscr.refresh()
    win.refresh()


def popup_cancel_requested(stdscr) -> bool:
    while True:
        key = stdscr.getch()
        if key == -1:
            return False
        if key in (27, 3, ord("q")):
            return True


def stream_footer(label: str, status: StreamStatus, cancel_hint: bool = True) -> str:
    parts = [f"{label}: {int(status.elapsed_seconds)}s elapsed"]
    if status.idle_seconds >= 3:
        parts.append(f"no output for {int(status.idle_seconds)}s")
    if cancel_hint:
        parts.append("Esc cancel")
    return " • ".join(parts)


def smoke_footer(status: StreamStatus, cancel_hint: bool = True) -> str:
    parts = [f"Running... {int(status.elapsed_seconds)}s elapsed"]
    if status.idle_seconds >= 3:
        parts.append(f"No output for {int(status.idle_seconds)}s, still waiting")
    if cancel_hint:
        parts.append("Esc cancel")
    return " • ".join(parts)


def research_result_lines(saved_path: Path | None, command: str, response: str, error: str = "") -> list[str]:
    lines = []
    if error:
        lines.extend(["Research errored:", error, ""])
    if saved_path:
        lines.extend([f"Saved best-known params: {saved_path}", ""])
    lines.extend(["Command ran:", command or "(unknown)", "", "Response:", response.strip() or "(empty response)"])
    return lines


def show_status(stdscr, title: str, detail: str = "") -> None:
    stdscr.clear()
    stdscr.addstr(0, 0, title[: curses.COLS - 1])
    if detail:
        stdscr.addstr(2, 0, detail[: curses.COLS - 1])
    stdscr.refresh()


def draw_loading_screen(stdscr, title: str, detail: str) -> None:
    stdscr.clear()
    rows, cols = screen_size()
    safe_addstr(stdscr, max(0, rows // 2 - 1), max(0, (cols - len(title)) // 2), title, curses.A_BOLD)
    safe_addstr(stdscr, max(0, rows // 2 + 1), max(0, (cols - len(detail)) // 2), detail)
    stdscr.refresh()


def load_saved_config_action(stdscr, state: TuiState) -> None:
    configs = find_saved_configs()
    if not configs:
        show_message(stdscr, "No saved configs found.")
        return
    labels = [f"{p.name}" for p in configs]
    mapping = dict(zip(labels, configs))
    picked = choose_from(
        stdscr,
        "Load saved config",
        labels,
        labels[0],
        "Load params from any previous INI config. This does not run it until you choose Save config and run.",
    )
    if picked in mapping:
        state.cfg = load_config(mapping[picked])
        remember_config(mapping[picked])


def edit_run_prefix_action(stdscr, state: TuiState) -> None:
    state.cfg.run_prefix = prompt_text(
        stdscr,
        "Enter run prefix, e.g. new-qwen-models",
        state.cfg.run_prefix,
        "Used in saved config and result folder names so you can identify benchmark runs later.",
    )


def save_config_action(stdscr, cfg: BenchConfig, run_after_save: bool) -> Path | bool:
    path = CONFIG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(cfg.run_prefix)}.ini"
    save_config(cfg, path)
    if run_after_save:
        return path
    show_message(stdscr, f"Saved: {path}")
    return False


def show_message(stdscr, message: str) -> None:
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.addstr(0, 0, message[: curses.COLS - 1])
    stdscr.addstr(2, 0, "Press any key")
    stdscr.getch()


def show_scroll_text(stdscr, title: str, lines: list[str], accept_text: str, enter_accepts: bool = False) -> bool:
    stdscr.nodelay(False)
    top = 0
    while True:
        stdscr.clear()
        rows, cols = screen_size()
        body_height = max(1, rows - 4)
        safe_addstr(stdscr, 0, 0, title[: cols - 1], curses.A_BOLD)
        visible_lines = wrap_lines_for_screen(lines, cols - 1)
        top = max(0, min(top, max(0, len(visible_lines) - body_height)))
        for row, line in enumerate(visible_lines[top : top + body_height], start=1):
            safe_addstr(stdscr, row, 0, line)
        safe_addstr(stdscr, rows - 2, 0, accept_text[: cols - 1])
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            top -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            top += 1
        elif key in (curses.KEY_NPAGE, ord(" ")):
            top += body_height
        elif key in (curses.KEY_PPAGE,):
            top -= body_height
        elif key in (ord("q"), 27):
            return False
        elif key in (10, 13):
            return True if not enter_accepts else False


def wrap_lines_for_screen(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(wrap_text(line, width))
    return wrapped


PROFILE_ACTIONS = {
    "__load_best_known": "Load best-known params for this model",
    "__auto_offload": "Auto-calculate GPU/MoE offload",
    "__smoke_test": "Smoke-test current profile load",
    "__copy_from_other": "Copy bench params from other profile",
    "__show_server_command": "Show equivalent llama-server command",
}


def edit_profile(stdscr, profile: Profile, models: list[str], other_profile: Profile | None = None) -> None:
    fields = list(PROFILE_ACTIONS) + [k for k, _ in profile.items()]
    idx = 0
    scroll = 0
    while True:
        layout = profile_editor_layout()
        visible_rows = layout["fields"].inner_height - 1 if len(fields) > layout["fields"].inner_height else layout["fields"].inner_height
        draw_profile_editor(stdscr, profile, fields, idx, scroll)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(fields)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(fields)
        elif key in (ord("q"), 27):
            return
        elif key in (10, 13):
            handle_profile_editor_selection(stdscr, profile, fields[idx], models, other_profile)
        scroll = adjust_profile_scroll(idx, scroll, fields, visible_rows)


def draw_profile_editor(stdscr, profile: Profile, fields: list[str], idx: int, scroll: int) -> None:
    stdscr.clear()
    layout = profile_editor_layout()
    draw_profile_editor_header(stdscr, layout["header"], profile)
    draw_profile_fields_panel(stdscr, layout["fields"], profile, fields, idx, scroll)
    draw_selected_field_panel(stdscr, layout["details"], profile, fields[idx])
    draw_help_panel(stdscr, layout["help"], "↑/↓ move  •  Enter edit  •  q/Esc back")


def draw_profile_editor_header(stdscr, panel: "Panel", profile: Profile) -> None:
    draw_panel(stdscr, panel, f"Editing {profile.name}")
    write_in_panel(stdscr, panel, 0, model_label(profile.model))
    write_in_panel(stdscr, panel, 1, f"cache={profile.cache_strategy} kv={cache_types_text(profile)} depths={profile.depths}")


def draw_profile_fields_panel(stdscr, panel: "Panel", profile: Profile, fields: list[str], idx: int, scroll: int) -> None:
    draw_panel(stdscr, panel, "Parameters")
    indicator_needed = len(fields) > panel.inner_height
    visible_rows = panel.inner_height - 1 if indicator_needed else panel.inner_height
    start = max(0, min(scroll, max(0, len(fields) - visible_rows)))
    visible = fields[start : start + visible_rows]
    for row, key in enumerate(visible):
        field_idx = start + row
        attr = curses.A_REVERSE if field_idx == idx else curses.A_NORMAL
        if key in PROFILE_ACTIONS:
            write_in_panel(stdscr, panel, row, f"▶ {PROFILE_ACTIONS[key]}", attr)
            continue
        value = getattr(profile, key)
        shown = model_label(value) if key == "model" and value else value
        write_in_panel(stdscr, panel, row, f"{key:15} {shown}", attr)
    if indicator_needed:
        indicator = f"{start + 1}-{min(len(fields), start + visible_rows)}/{len(fields)}"
        write_in_panel(stdscr, panel, panel.inner_height - 1, indicator.rjust(panel.inner_width), curses.A_DIM)


def adjust_profile_scroll(idx: int, scroll: int, fields: list[str], window_size: int) -> int:
    if len(fields) <= window_size:
        return 0
    if idx < scroll:
        return idx
    if idx >= scroll + window_size:
        return idx - window_size + 1
    return max(0, min(scroll, len(fields) - window_size))


def draw_selected_field_panel(stdscr, panel: "Panel", profile: Profile, field: str) -> None:
    draw_panel(stdscr, panel, "Selected parameter")
    lines = selected_field_lines(profile, field, panel.inner_width)
    for i, line in enumerate(lines[: panel.inner_height]):
        write_in_panel(stdscr, panel, i, line)


def selected_field_lines(profile: Profile, field: str, width: int) -> list[str]:
    if field in PROFILE_ACTIONS:
        return profile_action_lines(profile, field, width)
    value = getattr(profile, field)
    shown = model_label(value) if field == "model" and value else value
    lines = [f"{field} = {shown}", ""]
    description = FIELD_DESCRIPTIONS.get(field, "")
    if field in {"gpu_layers", "n_cpu_moe"}:
        description = enrich_preset_description(profile, field, description)
    for paragraph in description.split(". "):
        lines.extend(wrap_text(paragraph.strip(), width))
        lines.append("")
    return lines


def profile_action_lines(profile: Profile, field: str, width: int) -> list[str]:
    if field == "__load_best_known":
        lines = ["Load best-known params for this model", ""]
        lines.extend(wrap_text(f"Reads {BEST_KNOWN_PARAMS} and applies validated recommendations for the selected model.", width))
        lines.append("")
        lines.extend(wrap_text("A preview is shown before values are changed. The profile model itself is kept unchanged.", width))
        return lines
    if field == "__auto_offload":
        lines = ["Auto-calculate GPU/MoE offload", ""]
        lines.extend(wrap_text("Calculates gpu_layers/ngl and n_cpu_moe from GGUF layer sizes, detected MoE support, VRAM, RAM, and the profile's desired max context.", width))
        lines.append("")
        lines.extend(wrap_text("A preview shows the estimated GPU/CPU split before values are changed.", width))
        return lines
    if field == "__smoke_test":
        lines = ["Smoke-test current profile load", ""]
        lines.extend(wrap_text("Runs a brief llama-bench pass with the current profile values to verify the model/context actually loads. It uses the profile's max depth as the test context, with prompt=0, gen=1, repetitions=1.", width))
        lines.append("")
        lines.extend(wrap_text("This is useful after auto-offload or any manual ngl/n_cpu_moe change.", width))
        return lines
    if field == "__copy_from_other":
        lines = ["Copy bench params from other profile", ""]
        lines.extend(wrap_text("Copies all benchmark/profile parameters from the other profile, but keeps this profile's model unchanged.", width))
        lines.append("")
        lines.extend(wrap_text("Useful when comparing two models with identical runtime settings.", width))
        return lines
    if field == "__show_server_command":
        lines = ["Show equivalent llama-server command", ""]
        lines.extend(wrap_text("Renders the current benchmark profile as a buun llama-server command where fields map cleanly to server flags.", width))
        lines.append("")
        lines.extend(wrap_text("Benchmark-only fields such as depths/prompt/gen/repetitions are not passed except depths is used to choose a context size.", width))
        return lines
    return [""]


def handle_profile_editor_selection(stdscr, profile: Profile, field: str, models: list[str], other_profile: Profile | None = None) -> None:
    if field == "__load_best_known":
        load_best_known_for_profile(stdscr, profile)
        return
    if field == "__auto_offload":
        auto_calculate_offload_for_profile(stdscr, profile)
        return
    if field == "__smoke_test":
        smoke_test_current_profile(stdscr, profile)
        return
    if field == "__copy_from_other":
        copy_from_other_profile(stdscr, profile, other_profile)
        return
    if field == "__show_server_command":
        show_server_command(stdscr, profile)
        return
    edit_profile_field(stdscr, profile, field, models)


def auto_calculate_offload_for_profile(stdscr, profile: Profile) -> None:
    context_text = prompt_text(
        stdscr,
        "Desired max context for offload sizing",
        default_offload_context(profile),
        "Used only for this calculation to estimate KV/cache runtime memory. Use your intended server --ctx-size, or the largest benchmark depth.",
    )
    try:
        desired_context = int(context_text)
        if desired_context <= 0:
            raise ValueError
    except ValueError:
        show_message(stdscr, f"Invalid desired context: {context_text}")
        return
    try:
        rec = calculate_offload(profile, desired_context=desired_context)
    except OffloadError as exc:
        show_message(stdscr, str(exc))
        return
    except Exception as exc:
        show_message(stdscr, f"Failed to calculate offload: {exc}")
        return
    if confirm_auto_offload(stdscr, profile, rec):
        profile.gpu_layers = rec.gpu_layers
        profile.n_cpu_moe = rec.n_cpu_moe
        show_message(stdscr, f"Applied gpu_layers={rec.gpu_layers}, n_cpu_moe={rec.n_cpu_moe}.")
        if show_scroll_text(stdscr, "Smoke-test now?", ["Run a brief load test with the updated profile right now?", "", f"Context: {desired_context} tokens", f"Model: {model_label(profile.model)}"], accept_text="Enter run smoke test  •  q/Esc skip"):
            run_smoke_test_and_repair_with_popup(stdscr, profile, desired_context=desired_context)


def smoke_test_current_profile(stdscr, profile: Profile) -> None:
    if not profile.model:
        show_message(stdscr, "Select a model before running a smoke test.")
        return
    desired_context = smoke_test_context(profile)
    if show_scroll_text(stdscr, "Smoke-test current profile?", smoke_test_preview_lines(profile, desired_context), accept_text="Enter run smoke test  •  q/Esc cancel"):
        run_smoke_test_and_repair_with_popup(stdscr, profile, desired_context=desired_context)


def default_offload_context(profile: Profile) -> str:
    vals = [int(part.strip()) for part in profile.depths.split(",") if part.strip().isdigit()]
    if vals:
        return str(max(vals))
    return "262144"


def confirm_auto_offload(stdscr, profile: Profile, rec) -> bool:
    lines = auto_offload_preview_lines(profile, rec)
    return show_scroll_text(stdscr, "Apply auto-calculated offload?", lines, accept_text="Enter apply  •  q/Esc cancel")


def auto_offload_preview_lines(profile: Profile, rec) -> list[str]:
    model = rec.model
    budget = rec.budget
    candidate = rec.candidate
    supports_moe = "unknown" if model.supports_moe is None else "yes" if model.supports_moe else "no"
    lines = [
        f"Model: {model_label(profile.model)}",
        f"Path: {profile.model}",
        f"Name: {model.name} {model_detail_brackets(profile.model) if profile.model else ''}",
        f"Architecture: {model.architecture}",
        f"Layers/block_count: {model.block_count}",
        f"MoE support: {supports_moe}",
        f"Experts: count={model.expert_count or 'unknown'} used={model.expert_used_count or 'unknown'}",
        f"Tensor sizing: {'exact GGUF tensors' if model.exact_tensor_sizes else 'fallback estimate'}",
        f"Confidence: {rec.confidence}",
        "",
        "Recommended changes:",
        f"  gpu_layers/ngl: {profile.gpu_layers} -> {rec.gpu_layers}",
        f"  n_cpu_moe:     {profile.n_cpu_moe} -> {rec.n_cpu_moe}",
        "",
        "Budgets and target:",
        f"  Desired max context: {budget.desired_context} tokens",
        f"  VRAM usable budget: {format_bytes(budget.vram_total_bytes)}",
        f"  VRAM free now: {format_bytes(budget.vram_free_bytes) if budget.vram_free_bytes is not None else 'unknown'}",
        f"  RAM MemAvailable: {format_bytes(budget.ram_available_bytes)}",
        f"  GPU runtime workspace: {format_bytes(budget.gpu_workspace_bytes)}",
        f"  CPU runtime scratch: {format_bytes(budget.cpu_runtime_bytes)}",
        "",
        "Estimated selected split:",
        f"  GPU weights: {format_bytes(candidate.gpu_weight_bytes)}",
        f"  CPU/RAM weights: {format_bytes(candidate.cpu_weight_bytes)}",
        f"  GPU KV cache: {format_bytes(candidate.gpu_kv_bytes)}",
        f"  CPU KV cache: {format_bytes(candidate.cpu_kv_bytes)}",
        f"  GPU total: {format_bytes(candidate.gpu_total_bytes)} / {format_bytes(budget.vram_total_bytes)} usable",
        f"  RAM total: {format_bytes(candidate.cpu_total_bytes)} / {format_bytes(budget.ram_available_bytes)}",
    ]
    if candidate.gpu_over_bytes or candidate.cpu_over_bytes:
        lines.extend([
            "",
            "Warning:",
            f"  VRAM overage: {format_bytes(candidate.gpu_over_bytes)}",
            f"  RAM overage: {format_bytes(candidate.cpu_over_bytes)}",
        ])
    lines.extend(["", "Explanation:"])
    for line in rec.explanations:
        lines.append(f"  {line}")
    lines.extend(["", "Press Enter to apply these values, or q/Esc to cancel."])
    return lines


def smoke_test_preview_lines(profile: Profile, desired_context: int) -> list[str]:
    return [
        "Smoke-test current profile load",
        "",
        f"Model: {model_label(profile.model)}",
        f"Path: {profile.model}",
        f"Context used for test: {desired_context} tokens",
        f"gpu_layers/ngl: {profile.gpu_layers}",
        f"n_cpu_moe: {profile.n_cpu_moe}",
        f"batch_size: {profile.batch_size}  ubatch_size: {profile.ubatch_size}",
        f"threads: {profile.threads}",
        f"cache_strategy: {profile.cache_strategy}  cache_k/v: {cache_types_text(profile)}  flash_attn: {profile.flash_attn}  no_mmap: {profile.no_mmap}",
        "",
        "This runs a minimal llama-bench pass to validate context creation and model load.",
        "",
        "Press Enter to run, or q/Esc to cancel.",
    ]


def smoke_test_output_tail(output: str, max_lines: int = 40) -> list[str]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return lines[-max_lines:]


def smoke_test_last_row(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    return None


def smoke_test_metric_lines(result) -> list[str]:
    row = smoke_test_last_row(result.output)
    lines: list[str] = []
    if isinstance(row, dict):
        avg_ts = row.get("avg_ts")
        avg_ns = row.get("avg_ns")
        if avg_ts is not None:
            lines.append(f"Benchmark throughput: {float(avg_ts):.3f} tok/s")
        if avg_ns is not None:
            try:
                lines.append(f"Benchmark wall: {float(avg_ns) / 1e9:.3f}s")
            except (TypeError, ValueError):
                pass
    if getattr(result, "elapsed_seconds", None) is not None:
        lines.append(f"Process wall: {float(result.elapsed_seconds):.3f}s")
    return lines


def run_smoke_test_and_repair_with_popup(stdscr, profile: Profile, desired_context: int) -> bool:
    if not profile.model:
        show_message(stdscr, "Select a model before running a smoke test.")
        return False
    tried: set[tuple[str, str]] = set()
    stdscr.nodelay(True)
    try:
        while True:
            output_lines = [f"Starting smoke test for {model_label(profile.model)}"]

            def on_output(chunk: str) -> None:
                append_output_lines(output_lines, chunk)
                draw_output_popup(stdscr, "Smoke-testing current profile", output_lines, "Streaming... Esc cancels")

            def on_heartbeat(status: StreamStatus) -> None:
                draw_output_popup(stdscr, "Smoke-testing current profile", output_lines, smoke_footer(status))

            def cancel_check() -> bool:
                return popup_cancel_requested(stdscr)

            draw_output_popup(stdscr, "Smoke-testing current profile", output_lines, "Starting smoke test... Esc cancels")
            result = run_smoke_test(profile, desired_context, on_output=on_output, timeout=1800, idle_timeout=180, on_heartbeat=on_heartbeat, cancel_check=cancel_check)
            cmd_text = shlex.join(result.command)
            diagnosis = diagnose_failure(result.output)
            output_tail = smoke_test_output_tail(result.output)
            if not result.passed:
                logging.info(
                    "smoke test failed model=%s diagnosis=%s rc=%s reason=%s cmd=%s tail:\n%s",
                    model_label(profile.model),
                    diagnosis,
                    result.returncode,
                    result.reason or "",
                    cmd_text,
                    "\n".join(output_tail),
                )
                if result.timed_out or result.idle_timed_out or result.cancelled:
                    reason = result.reason or ("cancelled" if result.cancelled else "timed out")
                    lines = [f"Smoke test command: {cmd_text}", f"Return code: {result.returncode}", f"Reason: {reason}"]
                    metric_lines = smoke_test_metric_lines(result)
                    if metric_lines:
                        lines.extend(["", *metric_lines])
                    lines.extend(["", "Output:"])
                    if result.output.strip():
                        lines.extend(result.output.strip().splitlines())
                    show_scroll_text(stdscr, "Smoke test result", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
                    return False
            lines = [f"Smoke test command: {cmd_text}", f"Return code: {result.returncode}", f"Diagnosis: {diagnosis}"]
            metric_lines = smoke_test_metric_lines(result)
            if metric_lines:
                lines.extend(["", *metric_lines])
            lines.extend(["", "Output:"])
            if output_tail:
                lines.extend(output_tail)
            if result.passed:
                lines.extend(["", "Smoke test passed."])
                show_scroll_text(stdscr, "Smoke test result", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
                return True
            lines.extend(["", f"Smoke test failed ({diagnosis})."])
            step = next_repair_step(profile, smoke_test_block_count(profile), smoke_test_supports_moe(profile), diagnosis, tried)
            if step is None:
                show_scroll_text(stdscr, "Smoke test result", lines, accept_text="q/Esc/Enter back", enter_accepts=True)
                return False
            lines.extend(["", f"Next repair step: {step.label}", f"  gpu_layers -> {step.gpu_layers}", f"  n_cpu_moe -> {step.n_cpu_moe}"])
            if not show_scroll_text(stdscr, "Smoke test result", lines, accept_text="Enter apply repair  •  q/Esc stop", enter_accepts=False):
                return False
            tried.add((profile.gpu_layers, profile.n_cpu_moe))
            profile.gpu_layers = step.gpu_layers
            profile.n_cpu_moe = step.n_cpu_moe
    finally:
        stdscr.nodelay(False)


def smoke_test_block_count(profile: Profile) -> int:
    meta = model_metadata(profile.model)
    block_count = meta.get("block_count")
    return int(block_count) if isinstance(block_count, int) and block_count > 0 else 40


def smoke_test_supports_moe(profile: Profile) -> bool | None:
    meta = model_metadata(profile.model)
    haystack = f"{profile.model} {meta.get('name', '')} {meta.get('architecture', '')}".lower()
    if any(token in haystack for token in ["moe", "mixtral", "qwen3moe", "qwen35moe", "a3b"]):
        return True
    return False


def copy_from_other_profile(stdscr, profile: Profile, other_profile: Profile | None) -> None:
    if other_profile is None:
        show_message(stdscr, "No other profile available to copy from.")
        return
    if confirm_copy_from_other_profile(stdscr, profile, other_profile):
        copied = copy_profile_params(profile, other_profile)
        show_message(stdscr, f"Copied {len(copied)} params from {other_profile.name}; kept model unchanged.")


def confirm_copy_from_other_profile(stdscr, profile: Profile, other_profile: Profile) -> bool:
    lines = [
        f"Copy bench params from {other_profile.name} to {profile.name}?",
        "",
        f"Current model kept: {model_label(profile.model)}",
        f"Source model ignored: {model_label(other_profile.model)}",
        "",
        "Changes:",
    ]
    for key in copy_profile_param_keys(profile):
        lines.append(f"  {key}: {getattr(profile, key)} -> {getattr(other_profile, key)}")
    lines.extend(["", "Press Enter to copy these params, or q/Esc to cancel."])
    return show_scroll_text(stdscr, "Copy bench params?", lines, accept_text="Enter copy  •  q/Esc cancel")


def copy_profile_params(profile: Profile, other_profile: Profile) -> list[str]:
    keys = copy_profile_param_keys(profile)
    for key in keys:
        setattr(profile, key, getattr(other_profile, key))
    return keys


def copy_profile_param_keys(profile: Profile) -> list[str]:
    return [key for key, _ in profile.items() if key != "model"]


def load_best_known_for_profile(stdscr, profile: Profile) -> None:
    if not profile.model:
        show_message(stdscr, "Select a model before loading best-known params.")
        return
    recs = find_recommendations(profile.model)
    if not recs:
        if not confirm_research_missing_recommendation(stdscr, profile):
            return
        if not run_best_known_research_with_popup(stdscr, models=[profile.model]):
            return
        recs = find_recommendations(profile.model)
    if not recs:
        show_message(stdscr, f"Research completed, but no recommendation matched {model_label(profile.model)} in {BEST_KNOWN_PARAMS}")
        return
    rec = choose_recommendation(stdscr, profile, recs)
    if rec is None:
        return
    if confirm_recommendation(stdscr, profile, rec):
        plan = apply_recommendation(profile, rec)
        show_message(stdscr, f"Applied {len(plan.applied)} params from {rec.section}.")


def confirm_research_missing_recommendation(stdscr, profile: Profile) -> bool:
    lines = [
        "No saved recommendation matched this model.",
        "",
        f"Model: {model_label(profile.model)}",
        f"Path: {profile.model}",
        f"Recommendation file: {BEST_KNOWN_PARAMS}",
        "",
        "Press Enter to run streaming pi research for this model only and watch the output live.",
    ]
    return show_scroll_text(stdscr, "Research best-known params now?", lines, accept_text="Enter research  •  q/Esc cancel")


def choose_recommendation(stdscr, profile: Profile, recs):
    if len(recs) == 1:
        return recs[0]
    labels = [f"{model_label(r.model)} ({r.metadata.get('confidence', 'unknown')})" for r in recs]
    picked = choose_from(stdscr, "Choose best-known params", labels, labels[0], "Multiple recommendations matched this model.")
    if picked not in labels:
        return None
    return recs[labels.index(picked)]


def confirm_recommendation(stdscr, profile: Profile, rec) -> bool:
    plan = build_apply_plan(rec)
    lines = recommendation_preview_lines(profile, rec, plan)
    return show_scroll_text(stdscr, "Apply best-known params?", lines, accept_text="Enter apply  •  q/Esc cancel")


def recommendation_preview_lines(profile: Profile, rec, plan) -> list[str]:
    lines = [
        f"Recommendation: {rec.section}",
        f"Recommendation model: {model_label(rec.model)}",
        f"Current model: {model_label(profile.model)}",
        f"Path: {profile.model}",
        f"Confidence: {rec.metadata.get('confidence', 'unknown')}",
        "",
        "Summary:",
        rec.metadata.get("research_summary", "No summary."),
        "",
        "Changes:",
    ]
    if not plan.applied:
        lines.append("  No valid profile params to apply.")
    for key, value in plan.applied.items():
        lines.append(f"  {key}: {getattr(profile, key)} -> {value}")
    if plan.invalid:
        lines.extend(["", "Invalid/not applied:"])
        lines.extend(f"  {k}: {v}" for k, v in plan.invalid.items())
    if plan.skipped:
        lines.extend(["", "Unknown/skipped:"])
        lines.extend(f"  {k}: {v}" for k, v in plan.skipped.items())
    refs = rec.metadata.get("references")
    if refs:
        lines.extend(["", "References:", refs])
    research_meta = load_research_meta()
    command = research_meta.get("source")
    response = research_meta.get("response")
    if command or response:
        lines.extend(["", "pi research run:"])
        if command:
            lines.extend(["  Command ran:", command])
        if response:
            lines.extend(["", "  Response:", response.strip() or "(empty response)"])
    lines.extend(["", "Press Enter to apply these valid params, or q/Esc to cancel."])
    return lines


def show_server_command(stdscr, profile: Profile) -> None:
    cmd = shlex.join(profile_server_command(profile))
    lines = [
        "Equivalent buun llama-server command:",
        "",
        cmd,
        "",
        "Notes:",
        "- depths/prompt/gen/repetitions/warmup are benchmark-only and are not passed directly.",
        "- depths is used only to choose --ctx-size.",
        "- threads is reused for --threads and --threads-batch.",
    ]
    show_scroll_text(stdscr, "Equivalent llama-server command", lines, accept_text="q/Esc/Enter back", enter_accepts=True)


def edit_profile_field(stdscr, profile: Profile, field: str, models: list[str]) -> None:
    cur = getattr(profile, field)
    description = FIELD_DESCRIPTIONS.get(field, "")
    new_value = choose_field_value(stdscr, profile, field, cur, description, models)
    if field == "cache_strategy":
        if new_value != profile.cache_strategy:
            apply_cache_strategy(profile, new_value)
        return
    if field == "model":
        setattr(profile, field, new_value)
        reconcile_cache_types_for_model(profile)
        return
    setattr(profile, field, new_value)


def choose_field_value(stdscr, profile: Profile, field: str, cur: str, description: str, models: list[str]) -> str:
    if field == "model":
        return choose_model(stdscr, cur, description, models)
    if field == "cache_strategy":
        return choose_from(stdscr, "Choose cache strategy", list(CACHE_STRATEGIES), cur, description)
    if field in PRESETS:
        return choose_preset_value(stdscr, profile, field, cur, description)
    return prompt_text(stdscr, f"Enter {field}", cur, description)


def choose_model(stdscr, current: str, description: str, models: list[str]) -> str:
    opts, mapping = model_options(models)
    current_label = next((label for label, model in mapping.items() if model == current), opts[0] if opts else "")
    picked = choose_from(stdscr, "Choose model", opts, current_label, description)
    return mapping.get(picked, current)


def choose_preset_value(stdscr, profile: Profile, field: str, cur: str, description: str) -> str:
    if field == "gpu_layers":
        options = gpu_layer_options(profile.model)
    elif field in {"cache_type_k", "cache_type_v"}:
        options = compatible_cache_type_options(profile.model, field)
    else:
        options = PRESETS[field]
    description = enrich_preset_description(profile, field, description)
    picked = choose_from(stdscr, f"Choose {field}", options, cur, description, allow_edit=True)
    if picked == "custom":
        picked = prompt_text(stdscr, f"Enter {field} value(s), comma-separated if supported", cur, description)
    return picked


def reconcile_cache_types_for_model(profile: Profile) -> None:
    if not profile.model:
        return
    k_opts = compatible_cache_type_options(profile.model, "cache_type_k")
    v_opts = compatible_cache_type_options(profile.model, "cache_type_v")
    meta = model_metadata(profile.model)
    key_len = meta.get("key_length")
    val_len = meta.get("value_length")
    resolved_k, resolved_v = resolve_cache_types(profile)
    if cache_type_is_compatible(resolved_k, key_len) and cache_type_is_compatible(resolved_v, val_len):
        return
    if "q8_0" in k_opts and "q8_0" in v_opts:
        profile.cache_strategy = "q8"
        profile.cache_type_k = ""
        profile.cache_type_v = ""
        return
    if "f16" in k_opts and "f16" in v_opts:
        profile.cache_strategy = "f16"
        profile.cache_type_k = ""
        profile.cache_type_v = ""
        return
    profile.cache_strategy = "unset"
    profile.cache_type_k = k_opts[0] if k_opts else ""
    profile.cache_type_v = v_opts[0] if v_opts else ""


def enrich_preset_description(profile: Profile, field: str, description: str) -> str:
    if field == "gpu_layers":
        return f"{description} {model_description(profile.model)} Suggested values are based on GGUF block_count when readable."
    if field == "n_cpu_moe":
        return f"{description} {model_description(profile.model)} For MoE models, choose up to the layer/block_count; e.g. 33 of 40 keeps most experts on CPU."
    if field in {"cache_type_k", "cache_type_v"}:
        head_key = "key_length" if field == "cache_type_k" else "value_length"
        head_dim = model_metadata(profile.model).get(head_key) if profile.model else None
        compat = ", ".join(compatible_cache_type_options(profile.model, field)[:8])
        extra = f" Compatible with this model's {head_key.replace('_', ' ')}={head_dim}." if isinstance(head_dim, int) and head_dim > 0 else ""
        return f"{description} {model_description(profile.model)}{extra} Compatible options: {compat}."
    return description


def choose_from(stdscr, title: str, options: list[str], current: str, description: str = "", allow_edit: bool = False) -> str:
    idx = options.index(current) if current in options else 0
    while True:
        draw_choice_screen(stdscr, title, options, idx, description, allow_edit)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(options)
        elif key in (10, 13):
            return options[idx]
        elif allow_edit and key in (9,):
            return prompt_text(stdscr, f"Edit {title}", options[idx], description)
        elif key in (ord("q"), 27):
            return current


def draw_choice_screen(stdscr, title: str, options: list[str], idx: int, description: str, allow_edit: bool) -> None:
    stdscr.clear()
    layout = choice_layout()
    draw_panel(stdscr, layout["title"], title)
    write_in_panel(stdscr, layout["title"], 0, title)
    draw_description_panel(stdscr, layout["description"], description)
    draw_options_panel(stdscr, layout["options"], options, idx)
    help_text = "↑/↓ select  •  Enter accept  •  q/Esc cancel"
    if allow_edit:
        help_text += "  •  Tab edit highlighted"
    draw_help_panel(stdscr, layout["help"], help_text)


def draw_description_panel(stdscr, panel: "Panel", description: str) -> None:
    draw_panel(stdscr, panel, "Description")
    lines = wrap_text(description, panel.inner_width) if description else ["No description available."]
    for i, line in enumerate(lines[: panel.inner_height]):
        write_in_panel(stdscr, panel, i, line)


def draw_options_panel(stdscr, panel: "Panel", options: list[str], idx: int) -> None:
    draw_panel(stdscr, panel, "Options")
    start = visible_window_start(idx, panel.inner_height, len(options))
    for row, opt_idx in enumerate(range(start, min(len(options), start + panel.inner_height))):
        marker = "> " if opt_idx == idx else "  "
        attr = curses.A_REVERSE if opt_idx == idx else curses.A_NORMAL
        write_in_panel(stdscr, panel, row, marker + options[opt_idx], attr)


def prompt_text(stdscr, title: str, initial: str, description: str = "") -> str:
    stdscr.nodelay(False)
    value = list(initial)
    pos = len(value)
    curses.curs_set(1)
    try:
        while True:
            row = draw_text_prompt(stdscr, title, description, value, pos)
            key = stdscr.getch()
            result = handle_text_key(key, value, pos, initial)
            if isinstance(result, str):
                return result
            pos = result
    finally:
        curses.curs_set(0)


def draw_text_prompt(stdscr, title: str, description: str, value: list[str], pos: int) -> int:
    stdscr.clear()
    layout = text_prompt_layout()
    draw_panel(stdscr, layout["title"], title)
    write_in_panel(stdscr, layout["title"], 0, title)
    draw_description_panel(stdscr, layout["description"], description)
    cursor_row = draw_input_panel(stdscr, layout["input"], value, pos)
    draw_help_panel(stdscr, layout["help"], "Enter accept  •  Esc cancel  •  arrows move  •  Backspace delete")
    return cursor_row


def draw_input_panel(stdscr, panel: "Panel", value: list[str], pos: int) -> int:
    draw_panel(stdscr, panel, "Value")
    visible_width = max(1, panel.inner_width)
    start = max(0, pos - visible_width + 1)
    text = "".join(value[start : start + visible_width])
    write_in_panel(stdscr, panel, 0, text.ljust(visible_width))
    cursor_row = panel.row + 1
    cursor_col = panel.col + 2 + pos - start
    safe_move(stdscr, cursor_row, cursor_col)
    return cursor_row


def handle_text_key(key: int, value: list[str], pos: int, initial: str) -> int | str:
    if key in (10, 13):
        result = "".join(value).strip()
        return result or initial
    if key in (27,):
        return initial
    if key in (curses.KEY_LEFT,):
        return max(0, pos - 1)
    if key in (curses.KEY_RIGHT,):
        return min(len(value), pos + 1)
    if key in (curses.KEY_HOME,):
        return 0
    if key in (curses.KEY_END,):
        return len(value)
    if key in (curses.KEY_BACKSPACE, 127, 8) and pos > 0:
        del value[pos - 1]
        return pos - 1
    if key == curses.KEY_DC and pos < len(value):
        del value[pos]
    elif 32 <= key <= 126:
        value.insert(pos, chr(key))
        return pos + 1
    return pos


@dataclass(frozen=True)
class Panel:
    row: int
    col: int
    height: int
    width: int

    @property
    def inner_height(self) -> int:
        return max(0, self.height - 2)

    @property
    def inner_width(self) -> int:
        return max(1, self.width - 4)


def main_menu_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    header = Panel(0, 0, min(5, max(3, rows - 5)), cols)
    help_panel = Panel(max(0, rows - 3), 0, min(3, rows), cols)
    body_height = max(1, rows - header.height - help_panel.height)
    left_width = responsive_split_width(cols, preferred_min=28, preferred_max=38)
    return {
        "header": header,
        "actions": Panel(header.height, 0, body_height, left_width),
        "profiles": Panel(header.height, left_width, body_height, max(1, cols - left_width)),
        "help": help_panel,
    }


def profile_editor_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    header = Panel(0, 0, min(4, max(3, rows - 5)), cols)
    help_panel = Panel(max(0, rows - 3), 0, min(3, rows), cols)
    body_height = max(1, rows - header.height - help_panel.height)
    left_width = responsive_split_width(cols, preferred_min=36, preferred_max=56)
    return {
        "header": header,
        "fields": Panel(header.height, 0, body_height, left_width),
        "details": Panel(header.height, left_width, body_height, max(1, cols - left_width)),
        "help": help_panel,
    }


def choice_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    title = Panel(0, 0, min(3, rows), cols)
    desc_height = min(9, max(2, rows // 4), max(1, rows - 6))
    help_panel = Panel(max(0, rows - 3), 0, min(3, rows), cols)
    options_row = title.height + desc_height
    options_height = max(1, rows - options_row - help_panel.height)
    return {
        "title": title,
        "description": Panel(title.height, 0, desc_height, cols),
        "options": Panel(options_row, 0, options_height, cols),
        "help": help_panel,
    }


def text_prompt_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    title = Panel(0, 0, min(3, rows), cols)
    desc_height = min(9, max(2, rows // 4), max(1, rows - 9))
    input_panel = Panel(title.height + desc_height, 0, min(3, max(1, rows - title.height - desc_height)), cols)
    help_panel = Panel(max(input_panel.row + input_panel.height, rows - 3), 0, min(3, rows), cols)
    return {
        "title": title,
        "description": Panel(title.height, 0, desc_height, cols),
        "input": input_panel,
        "help": help_panel,
    }


def responsive_split_width(cols: int, preferred_min: int, preferred_max: int) -> int:
    if cols <= 2:
        return 1
    if cols < preferred_min * 2:
        return max(1, cols // 2)
    return min(preferred_max, max(preferred_min, cols // 3), max(1, cols - preferred_min))


def screen_size() -> tuple[int, int]:
    # Keep curses dimensions current after terminal/tmux pane resizes.
    if hasattr(curses, "update_lines_cols"):
        curses.update_lines_cols()
    # Keep one column unused. Many curses terminals raise or clip when writing
    # into the final screen column, which would hide the right panel border.
    return max(5, curses.LINES), max(1, curses.COLS - 1)


def draw_panel(stdscr, panel: Panel, title: str = "") -> None:
    if panel.height <= 0 or panel.width <= 1:
        return
    horizontal = "─" * max(0, panel.width - 2)
    safe_addstr(stdscr, panel.row, panel.col, "┌" + horizontal + "┐")
    for row in range(panel.row + 1, panel.row + panel.height - 1):
        safe_addstr(stdscr, row, panel.col, "│" + " " * max(0, panel.width - 2) + "│")
    safe_addstr(stdscr, panel.row + panel.height - 1, panel.col, "└" + horizontal + "┘")
    if title:
        safe_addstr(stdscr, panel.row, panel.col + 2, f" {title} "[: max(0, panel.width - 4)], curses.A_BOLD)


def write_in_panel(stdscr, panel: Panel, row: int, text: str, attr: int = curses.A_NORMAL) -> None:
    if row < 0 or row >= panel.inner_height:
        return
    safe_addstr(stdscr, panel.row + 1 + row, panel.col + 2, text[: panel.inner_width].ljust(panel.inner_width), attr)


def safe_addstr(stdscr, row: int, col: int, text: str, attr: int = curses.A_NORMAL) -> None:
    if row < 0 or col < 0 or row >= curses.LINES or col >= curses.COLS:
        return
    try:
        stdscr.addstr(row, col, text[: max(0, curses.COLS - col - 1)], attr)
    except curses.error:
        pass


def safe_move(stdscr, row: int, col: int) -> None:
    try:
        stdscr.move(min(row, curses.LINES - 1), min(col, curses.COLS - 1))
    except curses.error:
        pass


def visible_window_start(selected: int, window_size: int, item_count: int) -> int:
    if item_count <= window_size:
        return 0
    half = max(1, window_size // 2)
    return max(0, min(selected - half, item_count - window_size))


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + (1 if cur else 0) > width:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines or [""]

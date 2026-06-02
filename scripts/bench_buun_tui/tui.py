from __future__ import annotations

import copy
import curses
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
    choose_default_model,
    find_models,
    gpu_layer_options,
    model_description,
    model_label,
    model_options,
)
from .offload import OffloadError, calculate_offload, format_bytes
from .paths import CONFIG_DIR
from .profile import BenchConfig, CACHE_STRATEGIES, FIELD_DESCRIPTIONS, PRESETS, Profile


def new_default_config(models: list[str]) -> BenchConfig:
    cfg = BenchConfig()
    default_model = choose_default_model(models)
    cfg.profiles[0].model = default_model
    cfg.profiles[1] = copy.deepcopy(cfg.profiles[0])
    cfg.profiles[1].name = "profile_b"
    return cfg


def tui_main(stdscr, cfg: BenchConfig | None = None, report_url: str = "") -> Path | None:
    curses.curs_set(0)
    state = TuiState(cfg or new_default_config(find_models()), report_url)
    return main_menu(stdscr, state)


class TuiState:
    def __init__(self, cfg: BenchConfig, report_url: str = ""):
        self.cfg = cfg
        self.report_url = report_url
        self.models = find_models()
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
        f"{profile.name}: {Path(profile.model).name}",
        f"cache={profile.cache_strategy} depths={profile.depths}",
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
    scope = f" for {models[0]}" if models and len(models) == 1 else " for all models"
    output_lines = [f"Starting pi streaming research{scope} with: pi --mode json <research prompt>"]

    def on_output(chunk: str) -> None:
        append_output_lines(output_lines, chunk)
        draw_output_popup(stdscr, "Researching best-known params", output_lines, "Streaming pi output... wait for completion")

    draw_output_popup(stdscr, "Researching best-known params", output_lines, "Starting pi --mode json...")
    try:
        run = run_best_known_research(on_output=on_output, stream=True, models=models)
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
    stdscr.clear()
    stdscr.addstr(0, 0, message[: curses.COLS - 1])
    stdscr.addstr(2, 0, "Press any key")
    stdscr.getch()


def show_scroll_text(stdscr, title: str, lines: list[str], accept_text: str, enter_accepts: bool = False) -> bool:
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
    "__copy_from_other": "Copy bench params from other profile",
    "__show_server_command": "Show equivalent llama-server command",
}


def edit_profile(stdscr, profile: Profile, models: list[str], other_profile: Profile | None = None) -> None:
    fields = list(PROFILE_ACTIONS) + [k for k, _ in profile.items()]
    idx = 0
    while True:
        draw_profile_editor(stdscr, profile, fields, idx)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(fields)
        elif key in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(fields)
        elif key in (ord("q"), 27):
            return
        elif key in (10, 13):
            handle_profile_editor_selection(stdscr, profile, fields[idx], models, other_profile)


def draw_profile_editor(stdscr, profile: Profile, fields: list[str], idx: int) -> None:
    stdscr.clear()
    layout = profile_editor_layout()
    draw_profile_editor_header(stdscr, layout["header"], profile)
    draw_profile_fields_panel(stdscr, layout["fields"], profile, fields, idx)
    draw_selected_field_panel(stdscr, layout["details"], profile, fields[idx])
    draw_help_panel(stdscr, layout["help"], "↑/↓ move  •  Enter edit  •  q/Esc back")


def draw_profile_editor_header(stdscr, panel: "Panel", profile: Profile) -> None:
    draw_panel(stdscr, panel, f"Editing {profile.name}")
    write_in_panel(stdscr, panel, 0, Path(profile.model).name or "No model selected")
    write_in_panel(stdscr, panel, 1, f"cache={profile.cache_strategy} depths={profile.depths}")


def draw_profile_fields_panel(stdscr, panel: "Panel", profile: Profile, fields: list[str], idx: int) -> None:
    draw_panel(stdscr, panel, "Parameters")
    for i, key in enumerate(fields[: panel.inner_height]):
        attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
        if key in PROFILE_ACTIONS:
            write_in_panel(stdscr, panel, i, f"▶ {PROFILE_ACTIONS[key]}", attr)
            continue
        value = getattr(profile, key)
        shown = model_label(value) if key == "model" and value else value
        write_in_panel(stdscr, panel, i, f"{key:15} {shown}", attr)


def draw_selected_field_panel(stdscr, panel: "Panel", profile: Profile, field: str) -> None:
    draw_panel(stdscr, panel, "Selected parameter")
    lines = selected_field_lines(profile, field, panel.inner_width)
    for i, line in enumerate(lines[: panel.inner_height]):
        write_in_panel(stdscr, panel, i, line)


def selected_field_lines(profile: Profile, field: str, width: int) -> list[str]:
    if field in PROFILE_ACTIONS:
        return profile_action_lines(profile, field, width)
    value = getattr(profile, field)
    lines = [f"{field} = {value}", ""]
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
        f"Model: {profile.model}",
        f"Name: {model.name}",
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
        f"  VRAM total: {format_bytes(budget.vram_total_bytes)}",
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
        f"  GPU total: {format_bytes(candidate.gpu_total_bytes)} / {format_bytes(budget.vram_total_bytes)}",
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
        f"Current model kept: {profile.model}",
        f"Source model ignored: {other_profile.model}",
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
        show_message(stdscr, f"Research completed, but no recommendation matched {profile.model} in {BEST_KNOWN_PARAMS}")
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
        f"Model: {profile.model}",
        f"Recommendation file: {BEST_KNOWN_PARAMS}",
        "",
        "Press Enter to run streaming pi research for this model only and watch the output live.",
    ]
    return show_scroll_text(stdscr, "Research best-known params now?", lines, accept_text="Enter research  •  q/Esc cancel")


def choose_recommendation(stdscr, profile: Profile, recs):
    if len(recs) == 1:
        return recs[0]
    labels = [f"{r.model} ({r.metadata.get('confidence', 'unknown')})" for r in recs]
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
        f"Current model: {profile.model}",
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
    current_label = model_label(current) if current else opts[0]
    picked = choose_from(stdscr, "Choose model", opts, current_label, description)
    return mapping.get(picked, current)


def choose_preset_value(stdscr, profile: Profile, field: str, cur: str, description: str) -> str:
    options = gpu_layer_options(profile.model) if field == "gpu_layers" else PRESETS[field]
    description = enrich_preset_description(profile, field, description)
    picked = choose_from(stdscr, f"Choose {field}", options, cur, description, allow_edit=True)
    if picked == "custom":
        picked = prompt_text(stdscr, f"Enter {field} value(s), comma-separated if supported", cur, description)
    return picked


def enrich_preset_description(profile: Profile, field: str, description: str) -> str:
    if field == "gpu_layers":
        return f"{description} {model_description(profile.model)} Suggested values are based on GGUF block_count when readable."
    if field == "n_cpu_moe":
        return f"{description} {model_description(profile.model)} For MoE models, choose up to the layer/block_count; e.g. 33 of 40 keeps most experts on CPU."
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

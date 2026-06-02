from __future__ import annotations

import copy
import curses
import time
from pathlib import Path

from .config_store import find_saved_configs, load_config, remember_config, save_config, slugify
from .models import (
    choose_default_model,
    find_models,
    gpu_layer_options,
    model_description,
    model_label,
    model_options,
)
from .paths import CONFIG_DIR
from .profile import BenchConfig, CACHE_STRATEGIES, FIELD_DESCRIPTIONS, PRESETS, Profile


def new_default_config(models: list[str]) -> BenchConfig:
    cfg = BenchConfig()
    default_model = choose_default_model(models)
    cfg.profiles[0].model = default_model
    cfg.profiles[1] = copy.deepcopy(cfg.profiles[0])
    cfg.profiles[1].name = "profile_b"
    return cfg


def tui_main(stdscr, cfg: BenchConfig | None = None) -> Path | None:
    curses.curs_set(0)
    state = TuiState(cfg or new_default_config(find_models()))
    return main_menu(stdscr, state)


class TuiState:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.models = find_models()
        self.selected = 0
        self.actions = [
            "Load saved config",
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
    stdscr.addstr(0, 0, "Buun llama-bench TUI")
    stdscr.addstr(1, 0, "Configure two reproducible benchmark profiles")
    stdscr.addstr(2, 0, f"Run prefix: {state.cfg.run_prefix}")
    for i, action in enumerate(state.actions):
        attr = curses.A_REVERSE if i == state.selected else curses.A_NORMAL
        stdscr.addstr(4 + i, 0, action, attr)
    draw_profile_summaries(stdscr, state.cfg.profiles)


def draw_profile_summaries(stdscr, profiles: list[Profile]) -> None:
    stdscr.addstr(11, 0, "Profiles:")
    for i, profile in enumerate(profiles):
        stdscr.addstr(12 + i, 0, profile_summary(profile)[: curses.COLS - 1])


def profile_summary(profile: Profile) -> str:
    return (
        f"{profile.name}: {model_label(profile.model)}, cache={profile.cache_strategy}, "
        f"depths={profile.depths}, t={profile.threads}, moe={profile.n_cpu_moe}, fa={profile.flash_attn}"
    )


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
        edit_run_prefix_action(stdscr, state)
    elif selected in (2, 3):
        edit_profile(stdscr, state.cfg.profiles[selected - 2], state.models)
    elif selected in (4, 5):
        return save_config_action(stdscr, state.cfg, run_after_save=selected == 5)
    elif selected == 6:
        return None
    return False


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


def edit_profile(stdscr, profile: Profile, models: list[str]) -> None:
    fields = [k for k, _ in profile.items()]
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
            edit_profile_field(stdscr, profile, fields[idx], models)


def draw_profile_editor(stdscr, profile: Profile, fields: list[str], idx: int) -> None:
    stdscr.clear()
    stdscr.addstr(0, 0, f"Editing {profile.name}")
    stdscr.addstr(1, 0, "↑/↓ move, Enter edit, q back")
    for i, key in enumerate(fields):
        value = getattr(profile, key)
        shown = model_label(value) if key == "model" and value else value
        attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
        stdscr.addstr(3 + i, 0, f"{key:15} {shown}"[: curses.COLS - 1], attr)


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
    stdscr.addstr(0, 0, title)
    row = draw_description_header(stdscr, 2, description)
    help_text = "↑/↓ select, Enter accept, q cancel"
    if allow_edit:
        help_text += ", Tab edit highlighted"
    stdscr.addstr(row, 0, help_text[: curses.COLS - 1])
    row += 2
    for i, opt in enumerate(options):
        marker = "> " if i == idx else "  "
        attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
        stdscr.addstr(row + i, 0, marker + opt[: curses.COLS - 4], attr)


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
    stdscr.addstr(0, 0, title[: curses.COLS - 1])
    row = draw_description_header(stdscr, 2, description)
    stdscr.addstr(row, 0, "Enter accept, Esc cancel, arrows move, Backspace delete"[: curses.COLS - 1])
    row += 2
    visible_width = max(1, curses.COLS - 1)
    start = max(0, pos - visible_width + 1)
    text = "".join(value[start : start + visible_width])
    stdscr.addstr(row, 0, text.ljust(visible_width))
    stdscr.move(row, pos - start)
    return row


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


def draw_description_header(stdscr, start_row: int, description: str, max_lines: int = 6) -> int:
    if not description:
        return start_row
    width = max(20, curses.COLS - 1)
    inner_width = max(1, width - 4)
    lines = wrap_text(description, inner_width)[:max_lines]
    border = "+" + "-" * (width - 2) + "+"
    stdscr.addstr(start_row, 0, border[: curses.COLS - 1])
    row = start_row + 1
    for line in lines:
        stdscr.addstr(row, 0, ("| " + line.ljust(inner_width) + " |")[: curses.COLS - 1])
        row += 1
    stdscr.addstr(row, 0, border[: curses.COLS - 1])
    return row + 2


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

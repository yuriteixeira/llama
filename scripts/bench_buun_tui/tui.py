from __future__ import annotations

import copy
import curses
import time
from dataclasses import dataclass
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
    layout = main_menu_layout()
    draw_header_panel(stdscr, layout["header"], state.cfg.run_prefix)
    draw_actions_panel(stdscr, layout["actions"], state.actions, state.selected)
    draw_profiles_panel(stdscr, layout["profiles"], state.cfg.profiles)
    draw_help_panel(stdscr, layout["help"], "↑/↓ move  •  Enter select  •  q/Esc quit")


def draw_header_panel(stdscr, panel: "Panel", run_prefix: str) -> None:
    draw_panel(stdscr, panel, "Buun llama-bench TUI")
    write_in_panel(stdscr, panel, 0, "Configure two reproducible benchmark profiles")
    write_in_panel(stdscr, panel, 1, f"Run prefix: {run_prefix}")


def draw_actions_panel(stdscr, panel: "Panel", actions: list[str], selected: int) -> None:
    draw_panel(stdscr, panel, "Actions")
    row = 0
    for i, action in enumerate(actions):
        if row >= panel.inner_height:
            return
        attr = curses.A_REVERSE if i == selected else curses.A_NORMAL
        write_in_panel(stdscr, panel, row, f"> {action}" if i == selected else f"  {action}", attr)
        row += 1
        if i in {1, 3} and row < panel.inner_height:
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
        value = getattr(profile, key)
        shown = model_label(value) if key == "model" and value else value
        attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
        write_in_panel(stdscr, panel, i, f"{key:15} {shown}", attr)


def draw_selected_field_panel(stdscr, panel: "Panel", profile: Profile, field: str) -> None:
    draw_panel(stdscr, panel, "Selected parameter")
    lines = selected_field_lines(profile, field, panel.inner_width)
    for i, line in enumerate(lines[: panel.inner_height]):
        write_in_panel(stdscr, panel, i, line)


def selected_field_lines(profile: Profile, field: str, width: int) -> list[str]:
    value = getattr(profile, field)
    lines = [f"{field} = {value}", ""]
    description = FIELD_DESCRIPTIONS.get(field, "")
    if field in {"gpu_layers", "n_cpu_moe"}:
        description = enrich_preset_description(profile, field, description)
    for paragraph in description.split(". "):
        lines.extend(wrap_text(paragraph.strip(), width))
        lines.append("")
    return lines


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
    header = Panel(0, 0, 4, cols)
    help_panel = Panel(max(0, rows - 3), 0, 3, cols)
    body_height = max(3, rows - header.height - help_panel.height)
    left_width = max(28, min(38, cols // 3))
    return {
        "header": header,
        "actions": Panel(header.height, 0, body_height, left_width),
        "profiles": Panel(header.height, left_width, body_height, cols - left_width),
        "help": help_panel,
    }


def profile_editor_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    header = Panel(0, 0, 4, cols)
    help_panel = Panel(max(0, rows - 3), 0, 3, cols)
    body_height = max(3, rows - header.height - help_panel.height)
    left_width = max(36, min(56, cols // 2))
    return {
        "header": header,
        "fields": Panel(header.height, 0, body_height, left_width),
        "details": Panel(header.height, left_width, body_height, cols - left_width),
        "help": help_panel,
    }


def choice_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    title = Panel(0, 0, 3, cols)
    desc_height = min(9, max(4, rows // 4))
    help_panel = Panel(max(0, rows - 3), 0, 3, cols)
    options_row = title.height + desc_height
    options_height = max(3, rows - options_row - help_panel.height)
    return {
        "title": title,
        "description": Panel(title.height, 0, desc_height, cols),
        "options": Panel(options_row, 0, options_height, cols),
        "help": help_panel,
    }


def text_prompt_layout() -> dict[str, Panel]:
    rows, cols = screen_size()
    title = Panel(0, 0, 3, cols)
    desc_height = min(9, max(4, rows // 4))
    input_panel = Panel(title.height + desc_height, 0, 3, cols)
    help_panel = Panel(max(input_panel.row + input_panel.height, rows - 3), 0, 3, cols)
    return {
        "title": title,
        "description": Panel(title.height, 0, desc_height, cols),
        "input": input_panel,
        "help": help_panel,
    }


def screen_size() -> tuple[int, int]:
    # Keep one column unused. Many curses terminals raise or clip when writing
    # into the final screen column, which would hide the right panel border.
    return max(10, curses.LINES), max(1, curses.COLS - 1)


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

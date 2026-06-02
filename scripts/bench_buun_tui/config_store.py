from __future__ import annotations

import configparser
import re
import time
from pathlib import Path

from .paths import BENCH, CONFIG_DIR, LAST_CONFIG
from .profile import BenchConfig, CACHE_STRATEGIES, Profile


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return value or "bench"


def remember_config(path: Path) -> None:
    LAST_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LAST_CONFIG.write_text(str(path) + "\n")


def find_saved_configs() -> list[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)


def default_config_path() -> Path | None:
    if LAST_CONFIG.exists():
        p = Path(LAST_CONFIG.read_text().strip())
        if p.exists():
            return p
    configs = find_saved_configs()
    return configs[0] if configs else None


def save_config(cfg: BenchConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cp = configparser.ConfigParser()
    cp["run"] = {
        "bench": str(BENCH),
        "output_format": cfg.output_format,
        "run_prefix": cfg.run_prefix,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for profile in cfg.profiles:
        k, v = CACHE_STRATEGIES[profile.cache_strategy]
        cp[profile.name] = {key: value for key, value in profile.items()}
        cp[profile.name]["cache_type_k"] = k
        cp[profile.name]["cache_type_v"] = v
    with path.open("w") as f:
        cp.write(f)
    remember_config(path)


def load_config(path: Path) -> BenchConfig:
    cp = configparser.ConfigParser()
    cp.read(path)
    cfg = BenchConfig()
    if "run" in cp and "run_prefix" in cp["run"]:
        cfg.run_prefix = cp["run"]["run_prefix"]
    profiles: list[Profile] = []
    for section in cp.sections():
        if not section.startswith("profile"):
            continue
        p = Profile(section)
        for key, _ in p.items():
            if key in cp[section]:
                setattr(p, key, cp[section][key])
        profiles.append(p)
    if profiles:
        cfg.profiles = profiles
    return cfg

from __future__ import annotations

import configparser
import re
import time
from pathlib import Path

from .paths import BENCH, CONFIG_DIR, LAST_CONFIG
from .profile import BenchConfig, Profile, normalize_cache_type, resolve_cache_types


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
        "run_profiles": cfg.run_profiles,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for profile in cfg.profiles:
        k, v = resolve_cache_types(profile)
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
    if "run" in cp and "run_profiles" in cp["run"]:
        cfg.run_profiles = cp["run"].get("run_profiles", "both")
    profiles: list[Profile] = []
    for section in cp.sections():
        if not section.startswith("profile"):
            continue
        p = Profile(section)
        for key, _ in p.items():
            if key in cp[section]:
                setattr(p, key, cp[section][key])
        k_explicit = normalize_cache_type(p.cache_type_k)
        v_explicit = normalize_cache_type(p.cache_type_v)
        if k_explicit:
            p.cache_type_k = k_explicit
        if v_explicit:
            p.cache_type_v = v_explicit
        k, v = resolve_cache_types(p)
        if not k_explicit:
            p.cache_type_k = k
        if not v_explicit:
            p.cache_type_v = v
        profiles.append(p)
    if profiles:
        cfg.profiles = profiles
    return cfg

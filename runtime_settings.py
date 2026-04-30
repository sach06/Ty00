"""Shared runtime settings persisted to disk for cross-process updates."""
import json
import os
from typing import Any

SETTINGS_FILE = "runtime_settings.json"


def load_settings() -> dict[str, Any]:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict[str, Any]) -> bool:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        return True
    except OSError:
        return False


def get_budget(default_value: float) -> float:
    settings = load_settings()
    try:
        raw = settings.get("trading_budget", default_value)
        val = float(raw)
        if val >= 1:
            return val
    except (TypeError, ValueError):
        pass
    return default_value


def set_budget(value: float) -> bool:
    settings = load_settings()
    settings["trading_budget"] = float(value)
    return save_settings(settings)

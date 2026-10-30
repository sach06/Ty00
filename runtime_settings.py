"""Shared runtime settings persisted to disk for cross-process updates."""
import json
import os
from datetime import datetime, timedelta
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


def get_countdown_target(default: str = "18:40") -> str:
    """Return stored HH:MM countdown target, e.g. '18:40'."""
    settings = load_settings()
    val = settings.get("countdown_target", default)
    # Validate format HH:MM
    try:
        h, m = str(val).split(":")
        if 0 <= int(h) <= 23 and 0 <= int(m) <= 59:
            return f"{int(h):02d}:{int(m):02d}"
    except (ValueError, AttributeError):
        pass
    return default


def set_countdown_target(hhmm: str) -> bool:
    """Persist a HH:MM countdown target."""
    settings = load_settings()
    settings["countdown_target"] = hhmm
    return save_settings(settings)


def _hhmm_to_next_epoch(hhmm: str) -> float:
    """Return the Unix timestamp (seconds) for the next occurrence of HH:MM."""
    now = datetime.now()
    try:
        h, m = map(int, str(hhmm).split(":"))
    except (ValueError, AttributeError):
        h, m = 18, 40
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t.timestamp() <= now.timestamp():
        t += timedelta(days=1)
    return t.timestamp()


def get_countdown_epoch(default_hhmm: str = "18:40") -> float:
    """Return stored countdown epoch (Unix seconds). Migrates old HH:MM format."""
    settings = load_settings()
    if "countdown_epoch" in settings:
        try:
            return float(settings["countdown_epoch"])
        except (TypeError, ValueError):
            pass
    hhmm = settings.get("countdown_target", default_hhmm)
    return _hhmm_to_next_epoch(hhmm)


def set_countdown_epoch(epoch: float) -> bool:
    """Persist countdown as a Unix epoch (seconds)."""
    settings = load_settings()
    settings["countdown_epoch"] = float(epoch)
    return save_settings(settings)

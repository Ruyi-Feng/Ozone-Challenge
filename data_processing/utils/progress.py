"""Shared tqdm settings with elapsed / remaining-time bars."""

from __future__ import annotations

from typing import Any

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
)


def tqdm_bar(**kwargs: Any) -> dict[str, Any]:
    """Default kwargs so every processing bar shows elapsed and ETA."""
    defaults: dict[str, Any] = {
        "dynamic_ncols": True,
        "mininterval": 0.5,
        "smoothing": 0.08,
        "bar_format": TQDM_BAR_FORMAT,
    }
    defaults.update(kwargs)
    return defaults


def format_hms(seconds: float | None) -> str:
    if isinstance(seconds, str):
        return seconds or "--:--"
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"

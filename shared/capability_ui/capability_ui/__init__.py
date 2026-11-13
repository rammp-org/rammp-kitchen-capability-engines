"""Panels a capability shows on the UI while its action runs: Stop on every panel, bounded waits. See `panels`."""

from .panels import (
    DONE,
    NOTICE_TIMEOUT_S,
    PROMPT_TIMEOUT_S,
    STOP,
    STOPPED_MESSAGE,
    WRAP_CHARS,
    RunUI,
    cancel_own_run,
    panel_open,
    wrap,
)

__all__ = [
    "DONE",
    "NOTICE_TIMEOUT_S",
    "PROMPT_TIMEOUT_S",
    "STOP",
    "STOPPED_MESSAGE",
    "RunUI",
    "cancel_own_run",
    "panel_open",
    "WRAP_CHARS",
    "wrap",
]

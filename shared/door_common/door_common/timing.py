"""In-memory monotonic spans for action diagnostics."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_current = ContextVar("door_timing", default=None)


class Trace:
    def __init__(self):
        self.start = time.monotonic()
        self.spans = []

    def diagnostics(self):
        return {
            "start_s": self.start,
            "end_s": time.monotonic(),
            "spans": sorted(self.spans, key=lambda s: s["start_s"]),
        }


@contextmanager
def record():
    trace = Trace()
    token = _current.set((trace, ""))
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def span(name):
    current = _current.get()
    if current is None:
        yield
        return
    trace, parent = current
    path = f"{parent}/{name}" if parent else name
    token = _current.set((trace, path))
    start = time.monotonic()
    try:
        yield
    finally:
        trace.spans.append({"name": path, "start_s": start, "end_s": time.monotonic()})
        _current.reset(token)


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _current.get() is None:
                return function(*args, **kwargs)
            with span(name if isinstance(name, str) else name(*args, **kwargs)):
                return function(*args, **kwargs)

        return wrapped

    return decorate

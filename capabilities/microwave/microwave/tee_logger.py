"""The node's logger, each line also in the running goal's console.log (recording.to_console)."""

import os

from rclpy.impl import rcutils_logger

from . import recording

# rclpy keys every log call's severity and filters (throttle, once, skip_first) by the line it is called from, found by
# walking out of the files listed here, and refuses a different severity or filter set on the same line. Listed, a call
# through the tee counts as its caller's own line, as it would without the tee. Unlisted, every caller shared the
# tee's lines: 11-04, one shared line crashed every executed run at its first error; one line per severity still
# crashed the node at start-up under the orchestrator (engine.py's throttled "waiting for the orchestrator" info).
rcutils_logger._internal_callers.append(os.path.realpath(__file__))


class TeeLogger:
    """The node's logger, each line also in the running goal's console.log (recording.to_console): what T2 shows, from
    the planner, self.arm and the engine alike, kept with the run."""

    def __init__(self, logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def info(self, message, **kwargs):
        recording.to_console("INFO", str(message))
        return self._logger.info(message, **kwargs)

    def warning(self, message, **kwargs):
        recording.to_console("WARN", str(message))
        return self._logger.warning(message, **kwargs)

    warn = warning

    def error(self, message, **kwargs):
        recording.to_console("ERROR", str(message))
        return self._logger.error(message, **kwargs)

    def fatal(self, message, **kwargs):
        recording.to_console("FATAL", str(message))
        return self._logger.fatal(message, **kwargs)

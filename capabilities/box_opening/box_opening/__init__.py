"""Box opening: press the container's button, take its lid off and set it down beside it.

Every module logs to LOGGER (`from box_opening import LOGGER as log`). Its handler writes each
message alone on a line to the stdout of the moment, as the print() calls it replaced did, so
an attended run's console reads as before (and pytest's capsys sees it). The engine node
replaces that handler with the node's own logger (node.use_node_logger).

LOGGER is one Logger made here, not logging.getLogger()'s: once ROS 2's `launch` is imported,
getLogger() makes LaunchLoggers, which do not propagate to their parent, and launch's
logging.reset() strips the handlers from every one of them.
"""

import logging
import sys


class ConsoleHandler(logging.Handler):
    """Each record's message on its own line, on sys.stdout as it is when the record comes."""

    def emit(self, record):
        try:
            sys.stdout.write(record.getMessage() + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


LOGGER = logging.Logger(__name__, logging.INFO)
LOGGER.addHandler(ConsoleHandler())

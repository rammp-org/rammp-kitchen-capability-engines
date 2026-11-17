"""open_box_terminal's main() whole, against the node: the real entry point as a process of its own (its
keyboard a pipe, its Ctrl+C a SIGINT) and the real engine node, mock, as another, talking over DDS inside
the test container. Nothing here reaches a robot: the mock moves nothing, and the container has no network.

Why it is here: the terminal's Session was tested and its main() never run, and at the first run against a
node (the review of 2026-11-17) it ended every action with a traceback in place of its result, and a
Ctrl+C with "no result from the node within 45 s" 0.3 s later.

A ROS node can be made inside the test container, and the two processes find each other over its
loopback, only with the harness's DDS setting, as for test_node.py:

    EXTRA="-e ROS_LOCALHOST_ONLY=0 -e CYCLONEDDS_URI=file:///cyc.xml -v .../cyclone_lo.xml:/cyc.xml:ro"
"""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

WAIT_S = 90.0  # for a line or an exit: the two processes start under `nice`, beside the arm's driver
NO_GRAPH = (
    "no ROS graph in this container: this file needs the harness's EXTRA with CYCLONEDDS_URI, as test_node.py "
    "does (see this file's docstring)"
)


class Proc:
    """A process, its lines kept as they come: (seconds since it started, line)."""

    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            start_new_session=True,  # a signal sent to it is its own, never the test's
        )
        self.t0 = time.monotonic()
        self.lines = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.append((time.monotonic() - self.t0, line.rstrip("\n")))

    def said(self):
        return [line for _t, line in self.lines]

    def wait_for(self, text, timeout=WAIT_S):
        """Whether a line with `text` came within `timeout` (or before it ended)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if any(text in line for line in self.said()):
                return True
            if self.p.poll() is not None:
                time.sleep(0.3)  # its reader's last lines
                return any(text in line for line in self.said())
            time.sleep(0.05)
        return False

    def type(self, text):
        self.p.stdin.write(text)
        self.p.stdin.flush()

    def exit_code(self, timeout=WAIT_S):
        try:
            code = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        time.sleep(0.3)
        return code

    def end(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()

    def tail(self):
        return "\n".join("%6.2f %s" % (t, line[:200]) for t, line in self.lines[-40:])


@pytest.fixture
def graph():
    """start(mock_phase_s) -> (the mock node lining up on its console, the terminal), both ended after."""
    procs = []

    def start(phase_s):
        argv = [sys.executable, "-c", "from box_opening.node import main; main()", "--ros-args"]
        for param in ("mock:=true", "use_orchestrator:=false", "line_up:=true", "mock_phase_s:=%s" % phase_s):
            argv += ["-p", param]
        node = Proc(argv)
        procs.append(node)
        terminal = Proc([sys.executable, "-u", "-m", "box_opening.tasks.open_box_terminal", "--wait", "60"])
        procs.append(terminal)
        assert terminal.wait_for("open_box sent"), NO_GRAPH + "\n" + terminal.tail() + "\n-- node:\n" + node.tail()
        return node, terminal

    yield start
    for proc in reversed(procs):
        proc.end()


def _report(node, terminal):
    return "\n-- terminal:\n%s\n-- node:\n%s" % (terminal.tail(), node.tail())


def test_a_y_to_the_question_opens_and_the_terminal_ends_with_the_nodes_result(graph):
    """A (2026-11-17): the goal sent, the node's instructions shown, the question asked and answered y, and
    the action's result shown: DONE, exit 0. As first written the last step was a traceback and exit 1."""
    node, terminal = graph(0.3)
    terminal.type("y\n")  # typed ahead: nothing is asked yet, and it is not sent
    assert terminal.wait_for("Ready. Open the container?   y / n, then Enter"), _report(node, terminal)
    terminal.type("y\n")
    code = terminal.exit_code()
    said = terminal.said()
    assert code == 0, _report(node, terminal)
    assert "   (nothing is asked yet: y is for the question, when it comes)" in said
    assert "   sent: yes" in said
    assert said[-1].startswith("DONE: the lid is off, set down at [0.60, -0.07]"), _report(node, terminal)
    assert not any("Traceback" in line or "Error" in line for line in said), _report(node, terminal)
    # the question is asked once the mock's box has come into reach, each instruction a line
    assert any(line.startswith(">> Move ") for line in said)
    # ... and on the node's own console each is there once, the flight view's line: until the review of
    # 2026-11-17 the node's "[INFO] ... line-up: Ready. Open the container?" stood beside it
    assert node.wait_for("LINE UP   ok    yes", 30), _report(node, terminal)
    console = node.said()
    assert len([line for line in console if "Ready. Open the container?" in line]) == 1, _report(node, terminal)
    assert "             ask   Ready. Open the container?   y / n" in console
    assert not [line for line in console if "line-up: " in line], _report(node, terminal)


def test_an_n_is_the_cancel_and_the_terminal_ends_with_the_nodes_result(graph):
    node, terminal = graph(3.0)  # the mock's box comes into reach over 18 s: the question is not up yet
    assert terminal.wait_for(">> "), _report(node, terminal)
    terminal.type("n\n")
    code = terminal.exit_code()
    said = terminal.said()
    assert code == 1, _report(node, terminal)
    assert "   sent: no: cancelling" in said
    assert said[-1] == "CANCELLED: the run was stopped where it was", _report(node, terminal)
    assert not any("Traceback" in line for line in said)


def test_ctrl_c_cancels_the_action_and_the_terminal_waits_for_and_shows_the_nodes_result(graph):
    """The review, 2026-11-17: `Ctrl+C: cancelling, waiting for the node to stop the run`, then 0.3 s later
    `no result from the node within 45 s of the cancel: see its console`, exit. rclpy's own Ctrl+C handler
    had shut ROS down under the wait. Now the cancel is the action's own, the node ends the action
    cancelled, and that result is what the terminal ends with."""
    node, terminal = graph(3.0)
    assert terminal.wait_for(">> "), _report(node, terminal)
    terminal.p.send_signal(signal.SIGINT)
    code = terminal.exit_code()
    said = terminal.said()
    assert code == 1, _report(node, terminal)
    assert said[-1] == "CANCELLED: the run was stopped where it was", _report(node, terminal)
    assert not any("no result" in line or "Traceback" in line for line in said), _report(node, terminal)
    assert any(line.startswith("   Ctrl+C: cancelling. Waiting for the node") for line in said)
    # ... and it was the node that ended it so: its own console says the action was cancelled
    assert node.wait_for("RESULT  CANCELLED", 30), _report(node, terminal)


def test_main_with_no_node_on_the_graph_says_so_and_ends_without_a_traceback():
    """main() itself, as `ros2 run box_opening open_box_terminal` runs it, as far as it goes with nothing
    to talk to: ROS up without its own Ctrl+C handler, the node, the action client on the interface's
    real type, the answer's publisher, and the wait for a server that is not there."""
    ran = subprocess.run(
        [sys.executable, "-m", "box_opening.tasks.open_box_terminal", "--node", "nobody_there", "--wait", "2"],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert ran.returncode == 2, NO_GRAPH + "\n" + ran.stdout + ran.stderr
    assert ran.stdout.splitlines() == [
        "waiting for the node (/nobody_there/execute_action) ...",
        "no /nobody_there/execute_action within 2 s: is the node up?",
    ]
    assert "Traceback" not in ran.stderr


def test_a_ctrl_c_before_the_goal_is_sent_just_leaves():
    """Nothing was asked of the node: there is nothing to cancel and no result to wait for."""
    terminal = Proc(
        [sys.executable, "-u", "-m", "box_opening.tasks.open_box_terminal", "--node", "nobody_there", "--wait", "60"]
    )
    try:
        assert terminal.wait_for("waiting for the node"), NO_GRAPH + "\n" + terminal.tail()
        terminal.p.send_signal(signal.SIGINT)
        assert terminal.exit_code(30) == 130, terminal.tail()
        assert terminal.said()[-1] == "stopped before the goal was sent: nothing was asked of the node"
        assert not any("Traceback" in line for line in terminal.said())
    finally:
        terminal.end()

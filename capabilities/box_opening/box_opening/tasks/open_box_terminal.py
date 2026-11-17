"""open_box_terminal — open_box from a terminal, for a goal with no UI.

    ros2 run box_opening open_box_terminal

The node has no terminal of its own: it runs in a container under sheppy, and a goal with no UI is
sent to it by a one-shot `ros2 action send_goal`, which can say nothing back. So where such a run
must ask its person something (the line-up: "Move closer, about 15 cm" ... "Ready. Open the
container?"), this is where they are asked. The owner, 2026-11-17: "don't end, just tell you how far
to move until it's close enough then asking if ready y/n."

It sends the open_box goal to the node as a goal sent by hand is sent (no token: the node then waits
for the person, node.CONSOLE_LINE_UP_S, and no orchestrator cancels it), shows what the node says as
it goes (the action's progress: a line each time it changes), and sends what is typed:

    y + Enter     yes, to the question (only while it is asked: a y typed at any other time is not sent)
    n + Enter     no: the cancel, at any moment (the node stops the run its own way)
    Ctrl+C        the cancel too (the action's own): it then WAITS for the node's result and shows it,
                  as after any other end. A second Ctrl+C leaves without it

The answer goes out on /box_opening/answer (std_msgs/String, "y" or "n"), which the node hears while
the action runs; `ros2 topic pub --once /box_opening/answer std_msgs/msg/String "{data: y}"` says the
same from any other terminal. It ends with the action's result: what the node's Result says (its status,
its error code, whether it can be started again), read out of what the action client hands back
(read_result).

It moves nothing itself and decides nothing: the node takes a y only on a fresh "Ready" (guide.Guide),
whoever sent it. What is said and what is sent is Session's, which knows no ROS; read_result and follow()
are the two places that read what ROS hands back, each tested on the real types or a stand-in of their
shape, and main() is the wiring, run whole against the mock node (test_open_box_terminal).
"""

import argparse
import json
import queue
import signal
import sys
import threading
import time

from box_opening import guide, line_up

SUCCESS, FAILED, CANCELLED = 0, 1, 2  # rammp_orchestrator_interfaces/Result's status
GOAL_SUCCEEDED = 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED: how the action itself ended, where it succeeded
NODE = "box_opening"
ACTION = "open_box"
SERVER_WAIT_S = 10.0  # the node's action server is waited for this long
# after a Ctrl+C: the node's own stop (the run's Ctrl+C, a second node.CANCEL_ESCALATE_S later, a kill
# node.CANCEL_KILL_S after that, then its last world-state writes) ends inside this
RESULT_WAIT_S = 45.0
SPIN_S = 0.1  # how long one turn of the loop waits on ROS: a typed line or a Ctrl+C is read this often
NO_RESULT = "no_result"  # the error code said where the action did not succeed and its Result says nothing
STOPPED_EARLY = 130  # the exit code of a Ctrl+C before the goal was sent (a shell's own for it)
# an error code in a person's words, where the code alone says too little (docs/interface.md has them all)
ERRORS = {
    "not_lined_up": "not in place in time: nothing was pressed and the wrist is back at rest. Start it again",
    "not_found": "no container found: nothing was pressed",
    "not_home": "the arm is not at its rest pose: home it, then start again",
    "not_ready": "the robot is not ready (driver, planner, cameras or TF): nothing moved",
    "not_allowed": "the node does not allow the arm to move (allow_execute is off)",
    "arm_holds": "the arm holds where it stopped: clear it, attended, before the next run",
    "arm_halted": "the arm was halted (the e-stop, or control lost): clear it, attended",
    "timeout": "the run took too long and was stopped where it was",
    NO_RESULT: "the action ended without a result of its own: the node's console says why",
}


class Session:
    """One open_box as its person sees it. `write(line)`: a line on the terminal. `answer(word)`: "y" or
    "n" out to the node. Fed by main(): accepted(), progress() for each feedback, typed() for each line
    from the keyboard, result() once. `asking`: the node's newest word is the question."""

    def __init__(self, write, answer):
        self._write, self._answer = write, answer
        self._said = None  # (phase, message) as last written
        self.asking = False
        self.done = False

    def accepted(self, ok):
        """`ok`: the node took the goal; False, it refused it; None, it gave no answer in the time waited."""
        if ok is None:
            self._write("the node did not answer the goal in time: its console says whether it runs it")
            return
        self._write(
            "open_box sent: the node runs it. Type n and Enter to cancel at any time."
            if ok
            else "the node refused the goal (another action is running, or it is shutting down)"
        )

    def progress(self, phase, pct, message):
        """The node's progress: written where it says something new. The question is written with how to
        answer it, and y is sent only while it stands."""
        key = (str(phase), str(message))
        if key == self._said or not str(message):
            return
        self._said = key
        self.asking = key == (line_up.PHASE, guide.ASK)
        if self.asking:
            self._write(">> %s   y / n, then Enter" % message)
        elif key[0] == line_up.PHASE:
            self._write(">> %s" % message)
        else:
            self._write("   %3.0f%%  %s" % (100.0 * float(pct), message))

    def typed(self, line):
        """A line from the keyboard: n is sent at any time, y only to the question; anything else is said
        back, and nothing is sent."""
        word = line_up.read_answer(line)
        if self.done:
            return
        if word is None:
            if str(line).strip():
                self._write('   (type y or n, then Enter: "%s" is neither)' % str(line).strip()[:20])
            return
        if word == line_up.YES and not self.asking:
            self._write("   (nothing is asked yet: y is for the question, when it comes)")
            return
        self._answer(word)
        self._write("   sent: %s" % ("yes" if word == line_up.YES else "no: cancelling"))

    def interrupted(self):
        self._write(
            "   Ctrl+C: cancelling. Waiting for the node to stop the run and say how it ended (Ctrl+C again "
            "leaves without that)"
        )

    def no_result(self, wait_s=None):
        """The cancel was sent and the node's result is not shown: none came in `wait_s`, or (None) the
        person pressed Ctrl+C again. Either way the node goes on stopping the run, and its console says how."""
        self.done, self.asking = True, False
        self._write(
            "left without the node's result: the cancel was sent, and the node's console says how the run ended"
            if wait_s is None
            else "no result from the node within %.0f s of the cancel: see its console" % wait_s
        )

    def result(self, status, error_code="", recoverable=False, payload_json=""):
        """The action's result, in a line or two. Returns the terminal's exit code."""
        self.done, self.asking = True, False
        if status == SUCCESS:
            try:
                payload = json.loads(payload_json or "{}")
            except ValueError:
                payload = {}
            xyz = payload.get("lid_xyz")
            where = "" if not xyz else ", set down at [%.2f, %.2f]" % (xyz[0], xyz[1])
            took = "" if payload.get("duration_s") is None else " (%.0f s)" % payload["duration_s"]
            self._write("DONE: %s%s%s" % ("the lid is off" if payload.get("lid_off") else "the run ended", where, took))
            return 0
        if status == CANCELLED:
            self._write("CANCELLED: the run was stopped where it was")
            return 1
        said = ERRORS.get(error_code, "")
        self._write("FAILED: %s%s" % (error_code or "no error code", ": " + said if said else ""))
        self._write(
            "   it can be started again"
            if recoverable
            else "   not recoverable as it stands: the node's console says where the arm was left"
        )
        return 1


def read_result(wrapped):
    """(status, error_code, recoverable, payload_json) of the node's Result, out of what the action client
    hands back when a goal ends (the future of ClientGoalHandle.get_result_async()).

    That is a WRAPPER, rammp_orchestrator_interfaces' ExecuteAction_GetResult_Response, of two fields that
    are not the same thing: `.status`, how the ACTION ended (action_msgs/GoalStatus: succeeded 4, canceled
    5, aborted 6), and `.result`, the action's ExecuteAction.Result, whose one field `.result` is the
    node's rammp_orchestrator_interfaces/Result (status 0 success / 1 failed / 2 cancelled, error_code,
    recoverable, payload_json): the engine returns ExecuteAction.Result(result=result). As first written
    this read `.result.status`, a field ExecuteAction.Result does not have, and the terminal ended every
    run, the successful ones too, with a traceback in place of its result (review of 2026-11-17, run
    against the mock node; nothing had exercised main()).

    A Result nobody filled in reads as success (status 0, no payload), so success is said only of an
    action that itself succeeded: any other end with a "successful" Result is a failure with no word of
    its own (NO_RESULT), never DONE."""
    result = wrapped.result.result
    status = int(result.status)
    if status == SUCCESS and int(wrapped.status) != GOAL_SUCCEEDED:
        return FAILED, str(result.error_code) or NO_RESULT, False, ""
    return status, str(result.error_code), bool(result.recoverable), str(result.payload_json)


class Stops:
    """The Ctrl+C's (and SIGTERM's) this process has had, counted: main() installs it as their handler.
    rclpy's own handler is not installed (main: SignalHandlerOptions.NO): it shuts ROS down at the first
    Ctrl+C, after which nothing waits on the node any more. The review of 2026-11-17: the terminal said
    "no result from the node within 45 s of the cancel" 0.3 s after the Ctrl+C and left, and the cancel
    itself went out as the process was already tearing its node down."""

    def __init__(self):
        self.n = 0

    def __call__(self, *_signal):
        self.n += 1


def follow(session, handle, spin, typed, stops, clock=time.monotonic, wait_s=RESULT_WAIT_S):
    """An accepted goal followed to its end: what is typed goes to the session, and a Ctrl+C (`stops`)
    goes on as the action's own cancel, ONCE, after which this goes on waiting: the node stops the run its
    own way (the motion in flight cancelled, the driver's answer awaited), and how that ended is what the
    person needs to read. Returns what the action client hands back (read_result's `wrapped`); or None,
    said by the session, where no result came within `wait_s` of the cancel or the person pressed Ctrl+C
    again after it was sent (one pressed twice before that is still the one cancel, waited for).
    `spin(seconds)`: one turn of ROS; `typed`: the keyboard's lines, a queue; `handle`: the accepted goal
    (rclpy's ClientGoalHandle)."""
    ending = handle.get_result_async()
    cancelled_at = stops_then = None
    while not ending.done():
        spin(SPIN_S)
        while not typed.empty():
            session.typed(typed.get())
        if ending.done():
            break
        if stops.n and cancelled_at is None:
            session.interrupted()
            handle.cancel_goal_async()  # answered in the turns of `spin` that follow
            cancelled_at, stops_then = clock(), stops.n
        elif cancelled_at is not None and stops.n > stops_then:
            session.no_result()
            return None
        elif cancelled_at is not None and clock() - cancelled_at > wait_s:
            session.no_result(wait_s)
            return None
    return ending.result()


def main(argv=None):
    import rclpy
    from rammp_orchestrator_interfaces.action import ExecuteAction
    from rclpy.action import ActionClient
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--node", default=NODE, help="the engine's node name (its /<node>/execute_action and /<node>/answer)"
    )
    ap.add_argument("--wait", type=float, default=SERVER_WAIT_S, help="seconds to wait for the node's action server")
    args = ap.parse_args(argv)

    def write(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    # Ctrl+C is this program's to act on (Stops): with rclpy's own handler ROS is gone at the first one
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stops = Stops()
    before = {sig: signal.signal(sig, stops) for sig in (signal.SIGINT, signal.SIGTERM)}
    node = rclpy.create_node("open_box_terminal")
    try:
        client = ActionClient(node, ExecuteAction, "/%s/execute_action" % args.node)
        answers = node.create_publisher(String, "/%s/answer" % args.node, 10)
        session = Session(write, lambda word: answers.publish(String(data=word)))

        def spin(seconds):
            rclpy.spin_once(node, timeout_sec=seconds)

        t0, told = time.monotonic(), False
        while not client.server_is_ready():
            waited = time.monotonic() - t0
            if stops.n:  # nothing was sent: there is nothing to cancel and nothing to wait for
                write("stopped before the goal was sent: nothing was asked of the node")
                return STOPPED_EARLY
            if waited >= args.wait:
                write("no /%s/execute_action within %.0f s: is the node up?" % (args.node, args.wait))
                return 2
            if waited >= 1.0 and not told:  # (a node that is up is found within the second)
                write("waiting for the node (/%s/execute_action) ..." % args.node)
                told = True
            spin(SPIN_S)
        goal = ExecuteAction.Goal(engine_id=args.node, action_id=ACTION, params_json="{}")

        def heard(msg):
            p = msg.feedback.progress
            session.progress(p.phase, p.pct, p.message)

        # from here a Ctrl+C is never just an exit: the goal is out, and what was sent is cancelled (follow)
        sending = client.send_goal_async(goal, feedback_callback=heard)
        t0 = time.monotonic()
        while not sending.done() and time.monotonic() - t0 < args.wait:
            spin(SPIN_S)
        handle = sending.result() if sending.done() else None
        session.accepted(None if handle is None else bool(handle.accepted))
        if handle is None or not handle.accepted:
            return 1
        # the keyboard on a thread of its own, its lines handed to this one: everything that talks to ROS
        # (the answer's publish, the spin) stays on one thread
        typed = queue.Queue()

        def keyboard():
            for line in sys.stdin:
                typed.put(line)

        threading.Thread(target=keyboard, name="keyboard", daemon=True).start()
        wrapped = follow(session, handle, spin, typed, stops)
        if wrapped is None:
            return 1
        return session.result(*read_result(wrapped))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        for sig, handler in before.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())

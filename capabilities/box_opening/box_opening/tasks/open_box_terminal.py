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
    Ctrl+C        the cancel too (the action's own), then it waits for the node's result

The answer goes out on /box_opening/answer (std_msgs/String, "y" or "n"), which the node hears while
the action runs; `ros2 topic pub --once /box_opening/answer std_msgs/msg/String "{data: y}"` says the
same from any other terminal. It ends with the action's result.

It moves nothing itself and decides nothing: the node takes a y only on a fresh "Ready" (guide.Guide),
whoever sent it. What is said and what is sent is Session's, which knows no ROS (test_open_box_terminal);
main() is the wiring.
"""

import argparse
import json
import queue
import sys
import threading

from box_opening import guide, line_up

SUCCESS, FAILED, CANCELLED = 0, 1, 2  # rammp_orchestrator_interfaces/Result's status
NODE = "box_opening"
ACTION = "open_box"
SERVER_WAIT_S = 10.0  # the node's action server is waited for this long
RESULT_WAIT_S = 45.0  # after a Ctrl+C: the node's own stop (its Ctrl+C, then a second, then a kill) ends inside this
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
        self._write("   Ctrl+C: cancelling, waiting for the node to stop the run")

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


def main(argv=None):
    import rclpy
    from rammp_orchestrator_interfaces.action import ExecuteAction
    from rclpy.action import ActionClient
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

    rclpy.init()
    node = rclpy.create_node("open_box_terminal")
    code = 1
    try:
        client = ActionClient(node, ExecuteAction, "/%s/execute_action" % args.node)
        answers = node.create_publisher(String, "/%s/answer" % args.node, 10)
        session = Session(write, lambda word: answers.publish(String(data=word)))
        if not client.wait_for_server(timeout_sec=args.wait):
            write("no /%s/execute_action within %.0f s: is the node up?" % (args.node, args.wait))
            return 2
        goal = ExecuteAction.Goal(engine_id=args.node, action_id=ACTION, params_json="{}")

        def heard(msg):
            p = msg.feedback.progress
            session.progress(p.phase, p.pct, p.message)

        sending = client.send_goal_async(goal, feedback_callback=heard)
        rclpy.spin_until_future_complete(node, sending, timeout_sec=args.wait)
        handle = sending.result() if sending.done() else None
        session.accepted(handle is not None and handle.accepted)
        if handle is None or not handle.accepted:
            return 1
        ending = handle.get_result_async()
        # the keyboard on a thread of its own, its lines handed to this one: everything that talks to ROS
        # (the answer's publish, the spin) stays on one thread
        typed = queue.Queue()

        def keyboard():
            for line in sys.stdin:
                typed.put(line)

        threading.Thread(target=keyboard, name="keyboard", daemon=True).start()
        try:
            while not ending.done():
                rclpy.spin_once(node, timeout_sec=0.1)
                while not typed.empty():
                    session.typed(typed.get())
        except KeyboardInterrupt:
            session.interrupted()
            handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, ending, timeout_sec=RESULT_WAIT_S)
        if not ending.done():
            write("no result from the node within %.0f s of the cancel: see its console" % RESULT_WAIT_S)
            return 1
        r = ending.result().result
        code = session.result(r.status, r.error_code, r.recoverable, r.payload_json)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())

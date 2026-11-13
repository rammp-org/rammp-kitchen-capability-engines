"""What the microwave's UI panels say when an open or close doesn't finish: what happened in plain words (numbers
kept), the state the arm and hand are left in, and what to do. The log keeps the full message; the panel never shows
planner or driver codes or parameter names, and never hides the reason behind a generic sentence.

Text only: nothing here decides whether to retry or what the action returns. No ROS imports (the UI tests load it).
"""

import re

from capability_ui import wording as W

WHAT = {"open": "open the microwave door", "close": "close the microwave door"}

# Lines the run logs (say) that name why a close or open ended False ("stopped short"), newest first: (pattern,
# what happened, what to do)
SAID = (
    (
        r"need \S*door\S*|door file has no door_open_deg",
        "There's no record of the microwave door being opened, so the arm doesn't know where the door is.",
        "Open the door with Open Microwave Door first: the close starts from where the open left the door.",
    ),
    (
        r"no plan passes every gate",
        "The arm can't find a safe way to push the door shut from here. Nothing moved.",
        "Check that the door is open about as far as the open left it and nothing is in front of it.",
    ),
    (
        r"the hand would tilt",
        "Closing the door from here would tilt the hand, so the arm didn't start. Nothing moved.",
        None,
    ),
)


def _num(text):
    return text.rstrip(".")


def _placement(raw):
    """The door's own placement search, in the wheelchair's words: (sentence, what to do) or None."""
    m = re.search(r"Would work with the microwave moved: (.+?)(?: \(tilt [^)]*\))?(?:; or |\.\s*$|\.\s)", raw)
    if m:
        where = m.group(1)
        return (
            f"It would work with the microwave {where}.",
            f"Move the microwave {where} (or move the wheelchair the same distance the other way).",
        )
    if re.search(r"No shift of the microwave tried \(up to (\d+) cm\)", raw):
        n = re.search(r"up to (\d+) cm", raw).group(1)
        return (
            f"Moving the microwave by up to {n} cm wouldn't help either.",
            "Move the wheelchair so the microwave is straight ahead, at the usual distance, facing it squarely.",
        )
    return None


def reason(action_id, raw, said=(), last_error=""):
    """(what happened, what to do or None) for the failure message `raw` of `action_id` ("open" / "close"). `said`:
    the run's logged lines (for a run that ended False without a reason in `raw`); `last_error`: the arm's last
    failure during this run ("" if none)."""
    text = " ".join(str(raw or "").split())
    advice = _placement(text)
    todo_move = advice[1] if advice else None

    def why(inner):  # a planner / driver failure inside the message, in plain words
        return W.explain(inner)[0]

    rules = (
        (
            r"No microwave box from boxer \((\S+)\) in ([\d.]+) s",
            lambda m: (
                f"The microwave detector sent no microwave in {m[2]} s.",
                "Check that the microwave is in the scene camera's view and the detector (boxer) is running.",
            ),
        ),
        (
            r"No scene camera frames on .* in ([\d.]+) s|No scene camera to place the view pose",
            lambda m: ("The scene camera isn't sending images.", "Check that the scene camera is running."),
        ),
        (
            r"No door in the scene camera after (\d+) tries",
            lambda m: (
                f"The scene camera didn't find the microwave door ({m[1]} tries).",
                "Make sure the microwave is in the scene camera's view and nothing is in front of it.",
            ),
        ),
        (
            r"No arm base <- camera transform",
            lambda m: (
                "The robot doesn't know where the camera is (no camera position on the robot).",
                "Check that the cameras and the robot's frame publisher are running.",
            ),
        ),
        (
            r"No view pose \((\d+) tried",
            lambda m: (
                f"None of the {m[1]} places the arm could look at the microwave from is safe to reach.",
                todo_move or "Move the wheelchair so the microwave is straight ahead and within the arm's " "reach.",
            ),
        ),
        (
            r"fixed view pose .* keep-out",
            lambda m: (
                "The arm's fixed viewing position would come too close to you or the wheelchair.",
                "An operator has to check the microwave's viewing position setting.",
            ),
        ),
        (
            r"cuRobo can't plan to the fixed view pose[^:]*: (.*)",
            lambda m: (
                f"The arm can't get to its viewing position in front of the microwave: {why(m[1])[0].lower()}"
                f"{why(m[1])[1:]}",
                W.explain(m[1])[1] or "Move the arm a little clear (teleop).",
            ),
        ),
        (
            r"[Dd]id not reach the view pose",
            lambda m: (
                "The arm didn't reach its viewing position in front of the microwave."
                + (f" {why(last_error)}" if last_error else ""),
                W.explain(last_error)[1] if last_error else None,
            ),
        ),
        (
            r"No RGB-D frames|no wrist camera",
            lambda m: ("The wrist camera isn't sending images.", "Check that the wrist camera is running."),
        ),
        (
            r"NO DETECTION \(\w+\) after (\d+) retries",
            lambda m: (
                f"The wrist camera couldn't see the microwave handle ({m[1]} tries).",
                "Make sure the handle is in view of the wrist camera and nothing covers it.",
            ),
        ),
        (
            r"No two of (\d+) detections agreed within (\d+) cm",
            lambda m: (
                f"The camera's {m[1]} looks at the handle didn't agree (they were more than {m[2]} cm apart).",
                "Keep the wheelchair and the microwave still.",
            ),
        ),
        (
            r"hinge detections differ by more than (\d+) cm",
            lambda m: (
                f"Two looks at the door's hinge were more than {m[1]} cm apart.",
                "Keep the wheelchair and the microwave still.",
            ),
        ),
        (
            r"handle ([xyz])=(-?[\d.]+) outside plausible \((-?[\d.]+), (-?[\d.]+)\)",
            lambda m: (
                f"The handle was seen where it can't be ({m[1]} = {m[2]} m, expected {m[3]} to {m[4]} m).",
                "Check that the wrist camera sees the microwave's handle and not something else.",
            ),
        ),
        (
            r"Face yaw (-?[\d.]+) deg > ([\d.]+) deg",
            lambda m: (
                f"The microwave door looks turned {_num(m[1])} degrees away from the arm (at most {m[2]}).",
                "Turn the wheelchair (or the microwave) so the door faces the arm squarely.",
            ),
        ),
        (
            r"GATE FAILED at grasp: range ([\d.]+) \(max ([\d.]+)\), z (-?[\d.]+) \(([^)]*)\)",
            lambda m: (
                (
                    f"The handle is out of the arm's reach: {float(m[1]):.2f} m away, at most {float(m[2]):.2f} m."
                    if float(m[1]) > float(m[2])
                    else f"The handle is at a height the arm can't use ({float(m[3]):.2f} m; {m[4]})."
                ),
                "Move the wheelchair closer to the microwave."
                if float(m[1]) > float(m[2])
                else "The microwave has to sit at a height the arm can reach.",
            ),
        ),
        (
            r"hinge radius ([\d.]+) cm is implausible",
            lambda m: (
                f"The door's hinge was measured {m[1]} cm from the handle, which can't be right (about 35 cm "
                "expected).",
                "Make sure the whole door is in view of the wrist camera.",
            ),
        ),
        (
            r"Gripper still closed after opening",
            lambda m: (
                "The gripper didn't open before reaching for the handle.",
                "Check the gripper: it may be holding something.",
            ),
        ),
        (
            r"Did not reach the line-up point",
            lambda m: (
                "The arm didn't reach the point in front of the handle; it never touched the handle."
                + (f" {why(last_error)}" if last_error else ""),
                W.explain(last_error)[1] if last_error else None,
            ),
        ),
        (
            r"line-up \+ straight-in stopped \((.*?)\) -- ABORT",
            lambda m: (
                f"The reach in to the handle stopped: {why(m[1])[0].lower()}{why(m[1])[1:]} The gripper "
                "didn't close.",
                W.explain(m[1])[1],
            ),
        ),
        (
            r"Tracking ([\d.]+) cm at the grasp",
            lambda m: (
                f"The hand ended {m[1]} cm from the handle, too far to grab it safely, so the gripper didn't " "close.",
                None,
            ),
        ),
        (
            r"Gripper is OPEN -- nothing grasped",
            lambda m: ("The gripper closed on nothing: the handle wasn't grabbed, so the door wasn't pulled.", None),
        ),
        (
            r"swing re-check from the real grasp failed \((.*?)\)",
            lambda m: (
                "From where the hand actually grabbed the handle, the door swing can't be planned safely.",
                None,
            ),
        ),
        (
            r"push-open: (?:'([^']+)'|([^(]+?)) stopped(?: \((.*?)\))?(?: -- |\.|$)",
            lambda m: (
                "Pushing the door further open stopped"
                + (f" ({(m[1] or m[2]).strip()})" if (m[1] or m[2]) else "")
                + (f": {why(m[3])[0].lower()}{why(m[3])[1:]}" if m[3] else "."),
                W.explain(m[3])[1] if m[3] else None,
            ),
        ),
        (
            r"(.*?) from where the swing ended -- still holding the handle",
            lambda m: ("From where the door swing ended, pushing it further open can't be planned safely.", None),
        ),
        (
            r"(.*?) -- refusing before any motion",
            lambda m: (
                f"The arm can't find a safe way to {WHAT.get(action_id, 'do this')} from here: "
                f"{W.clean(m[1])[0].lower()}{W.clean(m[1])[1:]}",
                todo_move or "Move the wheelchair so the microwave is straight ahead at a comfortable reach.",
            ),
        ),
    )
    for pattern, make in rules:
        m = re.search(pattern, text)
        if m:
            what, todo = make(m)
            if advice and advice[0] not in what and "refusing before any motion" in text:
                what = f"{what} {advice[0]}"
            return what, todo
    if re.search(r"stopped short", text):  # ended False: the reason is in the run's log lines or the arm's error
        for line in reversed(list(said)):
            for pattern, what, todo in SAID:
                if re.search(pattern, str(line)):
                    return what, todo
        if last_error:
            what, todo = W.explain(last_error)
            return (
                f"The arm stopped before the door was {'open' if action_id == 'open' else 'shut'}: "
                f"{what[0].lower()}{what[1:]}",
                todo,
            )
        return f"The arm stopped before the door was {'open' if action_id == 'open' else 'shut'}.", None
    return W.explain(text)


def state(holding):
    if holding:
        return (
            "The hand is still holding the handle: an operator has to open the gripper and move the arm off "
            "the handle before anything else."
        )
    return "The arm is holding still."


def failure(action_id, raw, holding=False, said=(), last_error=""):
    """The failed panel's message (Done only)."""
    what, todo = reason(action_id, raw, said, last_error)
    return W.failure_text(what, state(holding), todo)


def retry(action_id, raw, said=(), last_error=""):
    """The retry panel's message (RunUI.retry adds "Press <label> to try again.")."""
    what, todo = reason(action_id, raw, said, last_error)
    return W.failure_text(what, state(False), todo)


def disabled(title):
    return (
        f"{title} is turned off: this robot's microwave capability was started with arm motion off. An operator "
        "has to restart it with arm motion on, for a supervised run."
    )


def unexpected(action_id, error, holding=False):
    """A failure that isn't one of the action's own (an arm error such as the e-stop, or a bug)."""
    what, todo = W.explain(f"{type(error).__name__}: {error}")
    return W.failure_text(what, state(holding), todo)

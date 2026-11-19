"""The box_opening capability: press the food container's button, take its lid off, set the lid
down beside the container, and go home.

One action, open_box: the whole run (tasks/press_demo). It checks the robot is ready (driver,
planner, TF, gripper, cameras: nothing moves if not), brings the arm to its rest pose if it is
not there, then locates the container, looks, presses, grips, lifts, carries, sets the lid
down and goes home.

The action runs box opening's command line as a child process, exactly as it ran on flatbot
until now: its own process and executor, and Ctrl+C's stop path for a cancel (the motion in
flight is cancelled and the driver's answer awaited before it exits, runtime/abort.py). Its
console lines become this node's log lines, and the rows its run log gains become the
action's progress. A run is one motion from the press to the set-down, so
it cannot pause once it starts; a cancel stops it at once.

The run finds the container with sheppy's scene segmenter (/perception/object_detections),
the perception already running on the robot: only the object the orchestrator selected (its
instance_id, passed to the run as --instance-id), and a lid box stacked on it. No detector
runs in this node.

Started from the UI (the orchestrator's token for the execution in hand), the run shows a panel
for as long as it goes: what it is doing, a bar, and one button, Cancel, which is the
orchestrator's cancel (Panel). No Pause: a run cannot pause.

A box the arm cannot press from where the chair stands does not end the run (owner 2026-11-17: "it
should NOT just fail EVER"): the run asks to be guided, the arm at rest and its wrist turned to look,
and the user is told which way and how far to move the chair until the box is where the arm opens it,
asked, and the run goes on at their yes (Capability._follow, _run_lines_up). With `line_up` on the same
line-up comes before the run's first move. It is said and asked on the UI's panel where the goal has
one; where it has none (a goal sent by hand), on this node's console through the flight view, and
answered y or n on ~/answer (Asking; tasks/open_box_terminal sends the goal, shows the lines and sends
the answer typed).

SAFETY: the arm moves only on a node with allow_execute:=true, set for an attended run (someone
at the e-stop). With it false, open_box is reported unavailable and a goal that comes anyway is
refused before anything runs. The mock moves nothing and is always available.

    ros2 launch box_opening box_opening.launch.py use_orchestrator:=false
    make action CAP=box_opening ACTION=open_box
"""

import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from capability_engine import Action, Cancelled, CapabilityEngine, EngineError, run
from capability_engine.action_scope import ScopeClosed
from rammp_orchestrator_interfaces.msg import ActionAvailability
from rammp_orchestrator_interfaces.srv import Control

from . import LOGGER, guide, line_up
from .constants import AIR_CONTROL_ENV, state_dir
from .flight_view import FAR_SEEN, FlightView, no_press_reason, not_seen_reason
from .runtime.abort import EXIT_HALTED, EXIT_LINE_UP_HOLDS, EXIT_NOT_LINED_UP
from .worlds import PLANNER_WORLDS_AS_ENV, PLANNER_WORLDS_DIR_ENV

# ROS parameter -> default. Container and world name files in the image's config/ (or are
# absolute paths); the robot's per-run environment comes from its cameras, never from here.
PARAMS = {
    "allow_execute": False,
    "container": "ankou_pink.yaml",
    "world": "world_flatbot.yaml",
    # where runs keep their logs, captures and videos; "" = ~/.ros/box_opening
    "state_dir": "",
    # a video of both cameras from the grip to the set-down: for attended data collection
    "video": False,
    "viz": False,
    # debug: each --execute run keeps its camera frames (~10 MB; the newest 20 kept), and its
    # run record's flight files and room maps
    "mission_frames": False,
    "speed_scale": 1.0,
    # the cuRobo planner config whose layers (the chair and its user) and robot spheres every
    # plan-free line is held out of (runtime/body); unreadable, the run stops before it moves;
    # "" turns the check off (the bench). BOX_OPENING_BODY_LAYERS in the node's environment overrides
    "body_layers": "/config/robot/curobo/planner.yaml",
    # where the scene camera finds no container, look for it with the wrist camera from one or two
    # poses over the arm's side (a chair parked side-on to a counter: tasks/survey); off, such a
    # run stops with NO BOX before anything moves, as the owner had it (2026-10-22). Never with the
    # orchestrator's selection: that is the scene camera's detection
    "wrist_survey": False,
    # the wrist camera finds the container: the wrist turns where the arm rests so its camera sees
    # the table in front (up to four views, first where the scene camera has the box; a lid a view
    # cuts is looked at again), and what it finds there is the box; the scene camera's only where
    # the wrist finds none (a table faced from the front: the scene camera sees no lid whole where
    # the arm presses, tasks/survey). With the orchestrator's selection too: the wrist's box is
    # taken only where it stands in the direction the scene camera has the selected container
    # (press_demo.SelectionTie); the instance marked opened is the selected one either way
    "wrist_first": False,
    # with wrist_first, for a selection that is NO detection of the scene camera's (the selected id in none of
    # its frames: the UI's object list typed by hand): the wrist looks from its own views, tied to nothing, and
    # what it finds is taken only where it is the ONE container it sees; two or more, NO BOX. Off (the default):
    # such a run ends NO BOX without turning the wrist. On, the arm opens whichever single container stands in
    # front of it under the selected id: the owner's to turn on (press_demo.wrist_look)
    "wrist_alone": False,
    # a run's full lines ("[press_demo] ...") on this node's console too, between the flight view's, as before
    # 2026-11-17. Off (the default): they are kept in runs/console-<stamp>.log, and the console is the flight view
    "full_console": False,
    # how the arm flies the free air before the press (constants.AIR_CONTROL_ENV): "impedance" (the default),
    # every leg in the driver's joint impedance, one motion to under the button, as flown on 2026-11-12;
    # "position", the look and the way in's free-air lines in position control and slower, the arm then
    # standing about a second while the springs come in, and the press on them as ever (owner 2026-11-16:
    # the arm "a little floppy" on its way to the button). In position it follows its path closely and does
    # not give way to what it meets in the air. Not flown
    "air_control": "impedance",
    # the folder box opening hands the planner worlds from (worlds.PlannerTable): the run's measured table,
    # while a way HOME is planned. `planner_worlds_as` is that folder as the PLANNER's container mounts it
    # (read-only); "" (the default) and box opening sends the planner no world. `planner_worlds_dir` is the
    # same folder as this node writes it; "" = planner_worlds under state_dir
    "planner_worlds_as": "",
    "planner_worlds_dir": "",
    # how long (s) a line-up of a goal that came from the UI waits for the person, the arm at rest with its
    # wrist turned the while. 0 (the default): the orchestrator ends every action 120 s after its goal, and a
    # line-up has what those leave (LINE_UP_S). More than 0 is the DEPLOYMENT's word that its orchestrator
    # restarts its 120 s at every progress item (rammp-orchestrator e17c625 of 2026-11-16 and later: the one
    # in ~/orch_latest_ws): this node then sends its newest progress again every KEEP_ALIVE_S and the run has
    # its whole budget from the person's yes. The package's deployment fragment says 300 (UI_LINE_UP_S: the
    # owner's "5 min", 2026-11-19). The node cannot ask the orchestrator which it is, so nothing here is
    # assumed: it is 0 on a build whose goal brings no token (the protocol before that orchestrator), for a
    # goal that has no UI as it comes, and from the moment a goal is cancelled 120 s after it came although
    # its progress was being sent (Capability._clock_from_goal). At most UI_LINE_UP_MAX_S
    "ui_line_up_s": 0.0,
    # the line-up before a run (line_up, guide): the user is told which way to move until the container is
    # where the arm opens it, then asked, and the arm moves only on their yes. On: every run lines up before
    # its first move. Off (the default): open_box starts its run at once, as it always has, and lines up
    # only where it must: where the run finds no press from where the chair stands (run 2 of 2026-11-17
    # ended there; the owner: "it should NOT just fail EVER"). Either way the arm never flies on from a
    # line-up without that yes: Open on the UI's panel, or, for a goal with no UI (sent by hand, or the UI
    # bridge not there), y on ~/answer to the question on this node's console (Asking).
    # With wrist_first the wrist turns to look while the user lines up (the run itself, --line-up: gently,
    # and only where the arm stands at rest). Without it the SCENE camera alone follows the container, which
    # on flatbot does not see a lid whole in front of the arm: one it cannot see is never asked of
    "line_up": False,
    # how far from the arm's base (m) the line-up trusts the boundary: a container further out is told to come
    # closer, though the run's proof plans a press for it there. 0 (the default): as far as the proof plans.
    # A deployment's own: on flatbot the proof plans presses to about 0.90 m, and the arm has opened boxes
    # at 0.63 and 0.68 m (2026-11-12)
    "line_up_far_m": 0.0,
    # the mock only: how long each of its phases lasts (s). 0, the default, and its run is over at once, as it
    # always was; a second or two and the UI can be rehearsed on it (the panel, its bar, Cancel) with no arm
    "mock_phase_s": 0.0,
}
NOT_ALLOWED = "allow_execute is off: open_box moves the arm only on a node set for an attended run"
# what `air_control` may be. The run reads anything but "position" as impedance (tasks/reach_press), so a
# misspelt value would fly as impedance without a word: the node refuses it instead (Capability._air_control)
AIR_CONTROLS = ("impedance", "position")


def check_params(params):
    """`params` against open_box's schema (REVIEW_RULES O4: the base class does not enforce
    it): open_box takes none, so anything but an empty JSON object is EngineError("bad_params")
    (the dry run's `execute` included: it is gone)."""
    if not isinstance(params, dict):
        raise EngineError("bad_params", f"params must be a JSON object, not {type(params).__name__}")
    if params:
        raise EngineError("bad_params", f"open_box takes no params, not {', '.join(sorted(map(str, params)))}")


# the command line each action runs: the package's executable, as `ros2 run box_opening <it>`
EXECUTABLES = {"open_box": "press_demo"}
POLL_S = 0.05
# a cancel is Ctrl+C's: one SIGINT, which cancels the motion in flight and waits for the
# driver to confirm. A second, after this long, is its escalation; a kill after that.
CANCEL_ESCALATE_S = 15.0
CANCEL_KILL_S = 10.0
SHUTDOWN_WAIT_S = 3.0  # on SIGTERM: the run's own stop first, inside docker stop's grace period
# REVIEW_RULES O3: the orchestrator cancels any action still running after 120 s (fixed in
# rammp-orchestrator; an engine cannot ask for more), releases the arm and hibernates the engine.
# A run still going at this budget is stopped its own way (one Ctrl+C) and fails with `timeout`,
# so the end is the run's: its stop's worst (a second Ctrl+C after CANCEL_ESCALATE_S, a kill
# CANCEL_KILL_S after that) and the last world-state writes (WORLD_WAIT_S) still end inside the
# 120 s. The longest of 48 logged missions took 45 s, plus ~5 s to start.
ORCHESTRATOR_TIMEOUT_S = 120.0
WORLD_WAIT_S = 3.0
RUN_BUDGET_S = ORCHESTRATOR_TIMEOUT_S - CANCEL_ESCALATE_S - CANCEL_KILL_S - WORLD_WAIT_S - 2.0  # 90 s
# The line-up's share of those 120 s (LINE_UP_S): what the run's budget leaves once a long run has its time,
# so a run started on the line-up's last second still ends before its own stop (which leaves the arm where it
# is, the lid in its fingers if it comes then). A user not in place by then is told to press open again.
# A long run after its go, by the two runs of 2026-11-12 and the pacing since: ~1.5 s for the watch to hear
# the go, the first scene look (2 s where the wrist held the box, else 15 s), ~25 s of flight as flown, ~8 s
# more for the gentler looks and air_control=position (re-timed, not flown), ~10 s where a press or a grip is
# made twice: 37 s clean, 47-50 s with one of the long cases, more with both. 50 s (the longest of 48 logged
# missions with its start, before any of this) left such a run nothing: 55 until a lined-up run is timed
RUN_LONGEST_S = 55.0
LINE_UP_S = RUN_BUDGET_S - RUN_LONGEST_S  # 35 s
# A line-up in the MIDDLE of a run (no press plans from where the chair stands: Capability._follow) has what
# is left of those same 35 s, counted from the goal's start: the run that goes on at its yes finds the box
# again and flies as a lined-up run does (RUN_LONGEST_S), and must still end before its own stop at
# RUN_BUDGET_S. Run 2 of 2026-11-17 said "no press plans" 16 s after its goal (1.5 s to start, the wrist's
# look 4.2 s, the proof of four tilts 7.7 s): about 19 s are left, of which the wrist's turn and the first
# word of the boundary take 3-7. With less than this left nothing is guided: the run ends at once,
# not_lined_up and recoverable, saying why no press planned, and the next open_box lines up FIRST
# (LINE_UP_FIRST_S), with the whole 35 s
GUIDE_MIN_S = 8.0
# After an open_box that ended not_lined_up from a line-up in the middle of its run, the next one within
# this long lines up before its first move, whatever `line_up` says: "press open again" then goes on from
# the line-up, not from another look and another proof (16 s of the 35 on run 2's numbers). Once: a line-up
# that then never sees the container (the scene camera alone, which may not see what the run's own look
# found) does not ask for another
LINE_UP_FIRST_S = 300.0
# A goal that came WITHOUT the orchestrator's token (sent by hand: `ros2 action send_goal`, the terminal
# program) has no orchestrator to cancel it at 120 s, and the owner's word for it is "don't end, just tell
# you how far to move until it's close enough then asking if ready y/n" (2026-11-17). Each of its line-ups
# waits for the person this long, the arm at rest with its wrist turned the while, and then ends as a
# timed one does (the wrist back to rest, not_lined_up, recoverable). Ten minutes: long enough to bring a
# chair round a table, short enough that a goal nobody is answering does not hold the arm's wrist turned
# and the node busy for the afternoon. The time a person takes is no part of such a run's own budget: it
# has RUN_BUDGET_S from its last go (Capability._follow), so no wait can cut a motion short
CONSOLE_LINE_UP_S = 600.0
# A goal from the UI is bound by the orchestrator's 120 s only where the orchestrator counts them from the
# goal. rammp-orchestrator e17c625 (2026-11-16, the commit after the one that put the token in the goal)
# counts them from the newest progress item instead: "only a stretch of total silence trips this"
# (orchestrator_node._drive_remote_execute_action; read in ~/orch_latest_ws, which the owner starts the UI
# from, 2026-11-19). Under such an orchestrator a line-up waits for the person as one sent by hand does, the
# run has RUN_BUDGET_S from its go, and no wait can cut a motion short. How long it waits is the owner's word
# (2026-11-19, asked how long the guidance should wait on the UI: "5 min"), the deployment's to change
# (`ui_line_up_s`; 0 is the 120 s rule above). First run against the real orchestrator and UI bridge with
# the app stood in for on 2026-11-19 (night-1116/ui1119/rig); not yet with the app itself
UI_LINE_UP_S = 300.0
# ... and no deployment's typing slip makes it for ever: a wait longer than a goal sent by hand has
# (CONSOLE_LINE_UP_S, where nobody is there to cancel) is cut to this
UI_LINE_UP_MAX_S = 600.0
# The orchestrator that does NOT restart its clock cancels a goal 120 s after it came, whatever progress it
# was sent. A cancel this node did not ask for, this close to that moment, of a goal whose progress was
# being kept alive, is taken for that: `ui_line_up_s` is 0 from then until the node is started again, and
# the log says so. It cannot undo that first cut; it keeps it from happening twice (review of 2026-11-19)
CLOCK_FROM_GOAL_S = (ORCHESTRATOR_TIMEOUT_S - 3.0, ORCHESTRATOR_TIMEOUT_S + 6.0)
# What keeps such a goal alive while nothing new is said (a person holding still while the reach is proved,
# the wait for their answer, a long carry): its newest progress, sent again once this long has passed with
# none (KeptAlive). A quarter of the orchestrator's 120 s: three may be lost before it would cancel
KEEP_ALIVE_S = 30.0
# How the action ended stays on the panel this long before the action itself ends. The orchestrator shows no
# result of its own: on the rig's first run (2026-11-19) the screen went from "Going home" straight to the
# orchestrator's "select an action", and a run that failed would have told the person nothing at all. Only
# under an orchestrator whose clock is kept alive: under the 120 s rule a run stopped at its budget has no
# seconds left to say it in
END_SHOW_S = 5.0
# ... given up after this long where the bridge does not hold it (a press takes a panel down for under a
# second; a bridge that refuses the goal's token never takes it): the action is not kept 5 s for nobody
END_UNSHOWN_S = 1.5
# A line-up that waits minutes for its person must not wait them out on a screen that no longer shows it. On
# the rig (2026-11-19, bridge_restart: the real orchestrator and UI bridge, the bridge stopped and started
# again while the chair was being guided) the new bridge refused this goal's token from then on ("token
# doesn't match the current ui_control grant"), the orchestrator pushed no new one, the app went on showing
# the last "Move closer" with a Cancel that reached nobody, and the node asked "Ready?" of no one until the
# line-up's time was up: 35 s under the 120 s rule, but five minutes now, the wrist turned all the while.
# So: a line-up of a goal with the UI whose panel the bridge has not held for this long ends, not_lined_up
# and recoverable, and its wrist turns back as at any line-up's end. Twenty seconds: a press takes the
# bridge's panel down for under a second (the next send opens it again), and a bridge that restarted does
# not come back for this goal however long it is given
UI_GONE_S = 20.0
# What "held" cannot tell: a bridge that hangs keeps the goal, and so does one whose app has gone (the
# tablet off, its network down). The bridge says once a second whether the app is connected
# (rammp_ui_bridge: ~/ue_connection, std_msgs/Bool). Heard at all in this action, it is part of "shown":
# none for this long, or its newest false, and the screen is not showing the panel
UE_SILENT_S = 5.0
UE_CONNECTION_TOPIC = "/%s/ue_connection"  # the bridge's node name (the base class's ui_bridge_node_name)
UI_GONE = "The screen is not showing this any more"  # the panel's words for it, should anything show them
UI_WENT = "the UI stopped showing the line-up's panel (none on it for %.0f s): nothing was pressed (start again)"
# ... in the person's words, one short message (the panel shows one; the whole reason is the result's and
# the log's). By the error code the action ends with (error_for, STOP_CODES, and the engine's own)
END_DONE = "Done. The lid is off, on the table"
END_NOT_OFF = "Finished. The lid was not taken off"
# Each says only what its error code always means (review of 2026-11-19: "the arm stopped where it is" was
# said of a run that had to be killed, a motion perhaps still in flight; "could not reach the button" of
# fingers that had not shut; "nothing moved" of a home that may have begun)
END_CANCELLED = "Cancelled"
END_STOPPED = "Stopped"
END_HOLDING = "The lid is still in the fingers"  # after either: the one thing the person must be told
END_FAILED = "The container was not opened"
END_SAYS = {
    "not_lined_up": "Not in place in time. Nothing was pressed",
    "not_found": "The arm could not find the container. Nothing was pressed",
    "not_home": "The arm could not get to its rest position",
    "not_ready": "The robot is not ready. Nothing moved",
    "not_allowed": "Opening is switched off on this robot. Nothing moved",
    "line_up_unavailable": "The arm could not watch the container. Nothing was pressed",
    "timeout": "Taking too long. The run was stopped",
    "lid_put_back": "The lid could not be set down. It is back on the container",
    "contact_early": "Something was in the way of the button. Nothing was pressed",
    "contact_late": "The container was not where the arm saw it. Nothing was pressed",
    "box_moved": "The container moved. The arm stopped",
    "not_popped": "The button did not open the lid",
    "no_button": "The arm could not find the button. Nothing was pressed",
    "no_press": "The button was not pressed",
    "lid_dropped": "The lid slipped out of the fingers",
    "arm_holds": "The arm is holding where it stopped",
    "arm_halted": "The arm was halted",
}
# ... and two ends of a line-up that share not_lined_up with "not in place" and are not that (the error
# carries them: its `says`)
END_NO_ANSWER = "No answer in time. Nothing was pressed"
END_UI_GONE = "The screen lost the arm's panel. Nothing was pressed"
END_NOT_SEEN = "The arm never saw the container. Nothing was pressed"
END_RETRY = "You can try again"
END_HELP = "Ask someone to check the arm"


def goal_brings_token():
    """Whether this build's execute_action goal brings the UI's token (rammp-interfaces-ros2 2e02b09). The
    orchestrator that reads such a goal is the one whose 120 s restart at every progress item, bar the one
    commit between the two changes (8635b80, e17c625: the same afternoon). A build without it talks to the
    orchestrator before both, which ends every action 120 s after its goal: `ui_line_up_s` is 0 there."""
    try:
        from rammp_orchestrator_interfaces.action import ExecuteAction

        return "token" in ExecuteAction.Goal.get_fields_and_field_types()
    except Exception:  # noqa: BLE001 - no such interface here: nothing to wait longer under
        return False


def end_says(result=None, error=None, cancelled=False, holding=False):
    """How the action ended, for the panel: (words, whether it failed). `holding`: it ended with the lid in
    the fingers, which is then what is said, whatever else."""
    if holding:
        return "%s. %s. %s" % (END_CANCELLED if cancelled else END_STOPPED, END_HOLDING, END_HELP), True
    if cancelled:
        return END_CANCELLED, False
    if error is None:
        return (END_DONE if (result or {}).get("lid_off") else END_NOT_OFF), False
    what = getattr(error, "says", None) or END_SAYS.get(getattr(error, "error_code", ""), END_FAILED)
    return "%s. %s" % (what, END_RETRY if getattr(error, "recoverable", False) else END_HELP), True


def saying(error, says):
    """`error`, with the panel's own words for its end (end_says) where its code's are not true of it."""
    error.says = says
    return error


class PanelWatch:
    """A line-up's source for a goal with the UI, which also says when the UI has gone: everything is the
    source's own, but `unable`, which the line-up reads on every tick and ends on (line_up.UNABLE), is also
    set once `shown()` (the bridge holds this goal's panel: Capability._shown) has been false for `gone_s`.
    `lost` then tells the engine which of the two it was."""

    def __init__(self, source, shown, gone_s=None, clock=time.monotonic):
        self._source, self._shown, self._clock = source, shown, clock
        self._gone_s = UI_GONE_S if gone_s is None else float(gone_s)
        self._since = clock()  # when the panel was last seen up (from the start: its first send is still to come)
        self.lost = False

    def __getattr__(self, name):
        return getattr(self._source, name)

    @property
    def unable(self):
        why = getattr(self._source, "unable", "")
        if why:
            return why
        if not self.lost:
            now = self._clock()
            if self._shown():
                self._since = now
            elif now - self._since >= self._gone_s:
                self.lost = True
        return UI_GONE if self.lost else ""


class KeptAlive:
    """An action's context under an orchestrator that restarts its 120 s at every progress item: all it is
    asked goes to the context as it is, and its newest progress is sent again wherever KEEP_ALIVE_S have
    passed with none. It rides on what every wait of this node already does every POLL_S, reading the cancel
    and the panel's press (cancelled, poll_ui_event), so no loop has a call to forget; a wait that reads
    neither is one of the run's own stops, seconds long. It proves that this node follows its run, no more:
    how long a run or a line-up may take is this node's own to end (RUN_BUDGET_S, Asking.left_s)."""

    def __init__(self, ctx, period_s=None, clock=time.monotonic):
        self._ctx, self._clock = ctx, clock
        self._period_s = KEEP_ALIVE_S if period_s is None else float(period_s)
        self._said, self._at = ("locate", 0.0, "starting", False), clock()

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def progress(self, phase, pct, message="", pausable=True):
        self._said, self._at = (phase, pct, message, pausable), self._clock()
        return self._ctx.progress(phase, pct, message, pausable=pausable)

    def beat(self):
        if self._clock() - self._at >= self._period_s:
            phase, pct, message, pausable = self._said
            self.progress(phase, pct, message, pausable=pausable)

    def cancelled(self):
        self.beat()
        return self._ctx.cancelled()

    def poll_ui_event(self):
        self.beat()
        return self._ctx.poll_ui_event()


# an instruction whose distance alone changed is said no oftener than this (line_up.LineUp's say_min_s): on
# the console, in the orchestrator's progress and in the node's log. The UI's panel is sent as it always was
SAY_MIN_S = 1.0
# A run told its line-up is over turns its wrist back and ends within this, or is stopped. A stop that comes
# while the wrist is still turning OUT (a reach_probe that ended as the line-up began) has the whole turn out,
# its settle, the turn back (a look each, about 3-4 s) and the run's own end (its threads, its log, its
# mission frames) before it: 11-13 s by the constants, which the 12 s this was until 2026-11-16 cut into, and
# a run stopped then is the arm holding. A cancel is read all through the wait (_turned_back), so a longer
# one costs nothing: out of time the action still ends inside the orchestrator's 120 s
# (LINE_UP_S + line_up.LINGER_S + the source's close + this + CANCEL_ESCALATE_S + WORLD_WAIT_S)
LINE_UP_END_S = 25.0
# "nothing was pressed", not "nothing moved": with the wrist look the wrist turned to look, and back to rest
NOT_LINED_UP = (
    "the container was not where the arm opens it within %.0f s: nothing was pressed (press open again in place)"
)
# ... of a line-up in the middle of a run that was not given at all: what the orchestrator's 120 s left it
NO_TIME_TO_GUIDE = (
    "%s, and this run had no time left to guide the chair (%.0f s of the %.0f s a line-up has were left, under "
    "%.0f): nothing was pressed. Press open again: it lines up first"
)
# the y or n of a person with no panel to press on: std_msgs/String, heard only while an action runs
ANSWER_TOPIC = "~/answer"
CONSOLE_ASK = "y / n"  # after the question on the console
UI_ASK = "(Open on the UI)"  # ... where it is the panel that is asked on
CONSOLE_HOW = "answer in open_box_terminal, or publish y or n on /box_opening/answer (std_msgs/String)"
CONSOLE_AGAIN = "Out of time. Start open_box again when you are in place"
# what the flight view's line-up step says it is: before the run's first move, and in the middle of it
LINE_UP_DOING = "lining the chair up with the box: move it as told"
GUIDE_DOING = "the arm rests and the wrist looks: move the chair as told"
GUIDE_SAYS = ". Move the chair as told"  # after why no press plans, in the progress of a run that is guided
# What a run that found no press from where the chair stands says, on the panel and in the progress, from
# that moment until it asks to be guided: it has not ended, and the arm may still be on its way to rest (the
# aimed button's case homes from the look pose, about 5 s on 2026-11-17's run 1). Nothing in it tells the
# person to move: until 2026-11-17's review these read "no way in from here: the chair is to move" and
# "going to rest: the chair is to move", for the whole home flight, and a person who drives on that word
# moves the chair under an arm still out over the table. The first word about moving is the line-up's own
# (GUIDE_SAYS, then "Move closer, about 15 cm"), which is said only once the run has asked: the arm at rest
NO_WAY_IN_WAIT = "no press plans from here: wait, the arm comes to rest first"
TO_REST = "going to rest"
# ... and what it says from the person's yes to a line-up in the middle of it, until its next row: the run
# has its go and looks for the box again. Nothing of before the yes is said after it (Tracker.went)
LOOKING_AGAIN = "looking for the container again"
# A run may ask to be guided for two more things than a press that does not plan (the "what" of its request
# and of its reach_refused row: press_demo.GUIDE_FAR, GUIDE_UNSEEN): a container its wrist sees out of the
# arm's reach, and (owner 2026-11-19) a container no camera sees at all. Each is said as what it is, from
# that row until the run asks, as NO_WAY_IN_WAIT is said of a press with no way in (Tracker.add); of a
# container that was never placed, "no press plans from here" is not what happened. Until 2026-11-19 the
# far top's row read NO_WAY_IN_WAIT too, and its reason went through the words for a press that does not
# plan ("no press plans from here: the way in does not plan")
FAR, UNSEEN = "far", "unseen"
WAITS = {
    FAR: "out of reach from here: wait, the arm comes to rest first",
    UNSEEN: "no container in view: wait, the arm comes to rest first",
}
# ... what this node's own log line calls each as the chair is guided for it (_follow)
ASKED_FOR = {
    FAR: "the wrist sees the container out of the arm's reach",
    UNSEEN: "no camera sees the container",
}
# ... and what the not_lined_up of a line-up that ran out of time says the run had asked for (_line_up)
GUIDED_BECAUSE = {
    FAR: " -- the chair was guided because the wrist saw the container out of the arm's reach",
    UNSEEN: " -- the chair was guided because the run's own look had found no container",
}
# A line-up in which NO camera ever saw the container ends as one whose container never came into reach
# does (out of time: not_lined_up, recoverable, the wrist back at rest), with the words for what was
# missing: it was not that the container stood in the wrong place
NOT_SEEN = (
    "no camera saw the container within %.0f s: nothing was pressed (move the chair so that the container is "
    "in front of the arm, then press open again)"
)
CONSOLE_AGAIN_UNSEEN = "Out of time: the container was never seen. Start open_box again in front of it"
# what is said of an arm that holds where it stopped (error_for's arm_holds)
ARM_HOLDS = "the arm holds where it stopped: clear it, attended, before the next run"
LINE_UP_LATE = (
    "the run did not end within %.0f s of being told its line-up was over and was stopped (Ctrl+C): the wrist "
    "may stand part-turned -- " + ARM_HOLDS
)
# press_demo's last word of a line-up that ended without a go, its wrist turned back (line_up_watch)
LINE_UP_ENDED = "LINE-UP: ended without a go"
# ... its word as its wrist starts to turn to the view, and where the wrist then does not look after all
LINE_UP_TURNS = "LINE-UP: the wrist turns to look"
LINE_UP_NO_LOOK = "LINE-UP: the wrist does not look"
# A CANCEL that comes with no action running is for the action about to start (the user's cancel overtook
# the goal: the orchestrator answers the press at once and takes a moment to send the goal). A goal that
# starts within this long of it does not run. Longer would cancel a deliberate press that follows a stale
# cancel; that press is then made again, which is the cheap mistake of the two
CANCEL_EARLY_S = 3.0
TAIL_LINES = 40

# a leg's row -> (phase, the fraction of the run done it ends, what the run is doing then, for
# the user) (REVIEW_RULES U2: progress runs from 0 to 1, its message a person's words, not a leg's
# name); rows are written as their legs end. First match wins.
LEG_PHASES = (
    ("survey:", "locate", 0.03, "looking for the container with the wrist camera"),
    ("reach:look", "look", 0.1, "looking for the button"),
    ("press:close", "look", 0.12, "reaching for the button"),
    ("reach:", "press", 0.2, "reaching for the button"),
    ("depth:approach", "press", 0.25, "pressing the button"),
    ("depth:", "press", 0.35, "pressing the button"),
    ("grip:", "grip", 0.5, "gripping the lid"),
    ("lid:", "carry", 0.7, "lifting the lid off"),
    ("place:down", "carry", 0.8, "carrying the lid to the table"),
    ("place:touch", "carry", 0.85, "setting the lid down"),
    ("place:", "release", 0.9, "letting go of the lid"),
    ("home", "home", 1.0, "going home"),
)
# notes that say a run is past its start: it has looked for the container, or given up on it. A home row before
# any of them (and before any leg above but the last) is the run's FIRST motion, to its start pose where the arm
# does not rest there; from one of them on, a home row is the way home (Tracker.add)
PAST_START = ("scene_fix", "wrist_survey", "wrist_first", "counter_edges", "reach_plan", "reach_refused")
# the mock's run: the real run's phases, in its words. "grip" holds the lid, "release" lets it go
# on the table, as the real run's rows say (Tracker)
MOCK_PHASES = {
    "open_box": (
        ("locate", 0.05, "found the container"),
        ("look", 0.1, "looking for the button"),
        ("press", 0.35, "pressing the button"),
        ("grip", 0.5, "holding the lid"),
        ("carry", 0.85, "setting the lid down"),
        ("release", 0.9, "letting go of the lid"),
        ("home", 1.0, "going home"),
    ),
}
MOCK_APPROACH_PHASES = 6  # the mock's line-up: its box comes into reach over this many mock_phase_s
# where the mock says it set the lid down (arm_mount, m): a stand-in, the mock sees nothing.
# flatbot's 2026-10-30 13:28 run set it down here, beside the box
MOCK_LID_XYZ = [0.597, -0.071, 0.209]
VIDEO_LINE = re.compile(r"\] VIDEO: .*: (\S+)$")
# a line the run's own rclpy logger wrote: "[ERROR] [1793385477.635189323] [box_opening_run]: ..."
ROS_LINE = re.compile(r"^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[[0-9.]+\] \[([^\]]*)\]: (.*)$")
LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}


def share(*parts):
    from ament_index_python.packages import get_package_share_directory

    return str(Path(get_package_share_directory("box_opening"), *parts))


def executable(name):
    """The installed executable `name` of this package (what `ros2 run box_opening <name>` runs)."""
    from ament_index_python.packages import get_package_prefix

    return os.path.join(get_package_prefix("box_opening"), "lib", "box_opening", name)


def config_path(name, *where):
    """A config file by its name in the image's config/<where>, or an absolute path as it is."""
    return name if os.path.isabs(name) else share("config", *where, name)


class Run:
    """One command line's run as a child process: its console lines logged (`log(level, text)`)
    and kept (the last TAIL_LINES), the video it announced, its exit code."""

    def __init__(self, argv, env, log, talk=False):
        self.lines = deque(maxlen=TAIL_LINES)
        self.video = None
        self.started = time.monotonic()
        self.timed_out = False  # stopped by the node at RUN_BUDGET_S
        self._log = log
        # `talk` (a run that may be lined up: --line-up before it starts, --guide in the middle of it): its
        # lines that begin with line_up.MARK are messages for the line-up, kept apart from its log
        # (messages()), and it is answered on its stdin (send())
        self._got, self._lock = [], threading.Lock()
        self._guide = None  # its request to be guided ({"guide": {...}}), until it is taken (take_guide)
        # its own session: a Ctrl+C at this node's terminal reaches it once, from here. Its
        # lines are decoded whatever they hold: a reader that died would leave the run
        # blocked on a full pipe, mid-motion.
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if talk else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._read, name="box-opening-run", daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(line_up.MARK):  # for the line-up, not the log: five a second while it watches
                try:
                    msg = json.loads(line[len(line_up.MARK) :])
                except ValueError:
                    continue
                with self._lock:
                    if isinstance(msg, dict) and "guide" in msg:
                        # no press plans from where the chair stands (press_demo.guide_chair): a new line-up
                        # begins with this line, and what it said of an earlier one (the fixes of the frames
                        # between that one's yes and its go) is no word on where the box is now
                        self._guide = msg["guide"] if isinstance(msg["guide"], dict) else {}
                        self._got = []
                    else:
                        self._got.append((time.monotonic(), msg))
                continue
            ros = ROS_LINE.match(line)
            level, text = (LEVELS[ros.group(1)], f"[{ros.group(2)}] {ros.group(3)}") if ros else (logging.INFO, line)
            self.lines.append(text)
            m = VIDEO_LINE.search(text)
            if m:
                self.video = m.group(1)
            self._log(level, text)

    def messages(self):
        """The line-up's lines it has said since the last call: (when read, message), oldest first."""
        with self._lock:
            got, self._got = self._got, []
        return got

    def take_guide(self):
        """Its request to be guided, the once: {"n", "box": where the box stood, "why": why no press plans
        there}; None where it has made none since the last one taken."""
        with self._lock:
            asked, self._guide = self._guide, None
        return asked

    def send(self, msg):
        """A line on its stdin (a run started with `talk`): the line-up's go, or stop."""
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError, AttributeError):  # it has ended, or was never to be talked to
            pass

    def alive(self):
        return self.proc.poll() is None

    def interrupt(self):
        """Ctrl+C, to the run alone (not to the ffmpeg its video may be encoding with)."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)

    def stop(self, wait_s, interrupt=True):
        """Its own stop (`interrupt` False: it was sent already, and a second Ctrl+C is its escalation), then
        a kill if it has not ended within `wait_s`."""
        if interrupt:
            self.interrupt()
        try:
            self.proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def finish(self):
        self._reader.join(2.0)
        return self.proc.returncode


class RunLog:
    """The rows of the run log a run creates in `runs` (the first run-*.jsonl that was not
    there before it started), as they are written."""

    def __init__(self, runs, before):
        self.runs = Path(runs)
        self.before = before
        self.path = None
        self._at = 0
        self._partial = ""

    @staticmethod
    def existing(runs):
        return set(Path(runs).glob("run-*.jsonl")) if Path(runs).is_dir() else set()

    def rows(self):
        if self.path is None:
            new = sorted(self.existing(self.runs) - self.before)
            if not new:
                return []
            self.path = new[0]
        try:
            with open(self.path) as f:
                f.seek(self._at)
                text = f.read()
                self._at = f.tell()
        except OSError:
            return []
        lines = (self._partial + text).split("\n")
        self._partial = lines.pop()  # a row not yet finished
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


class WorldWriter:
    """A run's world-state writes, made in order on a thread of their own (REVIEW_RULES R4): an
    orchestrator call waits up to 10 s for its service and 10 s for the answer, and the poll loop
    that follows the run (its budget, its cancel) must never wait on one. `put` queues a write,
    a callable that returns whether it was taken; `wait` joins what is queued, at most `timeout_s`."""

    def __init__(self, log):
        self._log = log
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._work, name="box-opening-world-state", daemon=True)
        self._thread.start()

    def put(self, what, write, done=None):
        """Queue `write` (said as `what` where it is refused); `done(ok)` once it is answered."""
        self._q.put((what, write, done))

    def _work(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            what, write, done = item
            try:
                ok = bool(write())
            except Exception as e:  # noqa: BLE001 -- Cancelled included: no write once cancelled
                self._log.error(f"the world state was not told {what}: {type(e).__name__}: {e}")
                ok = False
            else:
                if not ok:
                    self._log.error(f"the world state was not told {what}")
            if done is not None:
                done(ok)

    def wait(self, timeout_s):
        """Every write queued so far answered (True), or `timeout_s` gone (False: said)."""
        self._q.put(None)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            self._log.error(f"the world-state writes did not finish in {timeout_s:.0f} s: some may be missing")
            return False
        return True


# The run's panel on the UI (Panel). The bar's scale is the UI's, 0 to 100 (docs/UI_JSON_SCHEMA.md, the
# base class's example); ctx.progress keeps REVIEW_RULES U2's fraction from 0 to 1.
PANEL_TITLE = "Opening the container"
PANEL_CANCEL = "cancel"  # the Cancel button's id
# with its style, as the orchestrator's own progress panel sends its Cancel (engine_node._progress_panel:
# "warning") and the bridge's schema asks of every button (rammp-UI-bridge docs/USER_INPUT_JSON_SCHEMA.md 6.3)
PANEL_CANCEL_BUTTON = {"button_id": PANEL_CANCEL, "label": "Cancel", "style": "warning"}
PANEL_PCT = 100.0
# an unchanged panel is sent again this often: a goal the bridge ended on its own (its abort, a refusal) is opened
# again by the next send, and Cancel is back on the screen (the microwave engine's period)
PANEL_REFRESH_S = 1.0
PANEL_CANCELLED = "Cancelled: the arm is stopping where it is"
PANEL_TIMED_OUT = "Taking too long: the arm is stopping where it is"
# a line-up that ended without its Open: the run turns its wrist back to rest, and Cancel stays on the screen
PANEL_TURNING_BACK = "The wrist is turning back to rest"
PANEL_LINE_UP_OVER = "Ending"  # the run told its line-up is over, where its wrist never turned


class Panel:
    """open_box's panel on the UI for as long as it runs (REVIEW_RULES U3): what the run is doing (the phase and
    the message its progress reports), a bar, and ONE button, Cancel. No Pause, and `pausable` false: a run is one
    motion and cannot pause. Once the run is being stopped (`stopping`: a cancel, its budget) it says so and has
    no button left: there is nothing more to ask of it.

    It asks the user nothing and waits for nothing: `tick()` (the follow loop's, every POLL_S) reads a press and
    sends the panel where it changed or PANEL_REFRESH_S has passed, both calls the base class answers at once.
    Its life is the run's: the base class closes it as execute() ends (at most RUN_BUDGET_S and the run's stop),
    with the token it was shown by. `live()`: whether this execution has the UI at all (Capability._ui_live);
    without it nothing is sent, and a run is as it was."""

    def __init__(self, ctx, live, clock=time.monotonic):
        self._ctx = ctx
        self._live = live
        self._clock = clock
        self._said = ("locate", 0.0, "starting")
        self._sent = None  # the panel as last sent, None where the last send did not go out
        self._tried_at = None  # when a send was last tried
        self.pressed = False  # Cancel was pressed on it (it stays said)
        self.stopping = ""  # the run is being stopped, in the user's words: no button is offered

    def say(self, phase, pct, message=""):
        """What the run does now: the progress it just reported."""
        self._said = (str(phase), float(pct), str(message))

    def again(self):
        """The next tick sends the panel whatever was sent last: a line-up in the middle of the run had the
        screen meanwhile, and its last panel is what the bridge shows."""
        self._sent = self._tried_at = None

    def panel(self):
        phase, pct, message = self._said
        text = self.stopping or (message[:1].upper() + message[1:])
        return {
            "title": PANEL_TITLE,
            "message": text,
            "progress": {"phase": phase, "pct": round(PANEL_PCT * pct, 1), "message": text, "pausable": False},
            "buttons": [] if self.stopping else [dict(PANEL_CANCEL_BUTTON)],
        }

    def tick(self):
        """Read a press, and send the panel where it is due: at once where it changed (or a press took the
        bridge's panel down), every PANEL_REFRESH_S where it did not, and no oftener than that where the last
        send did not go out. True the once Cancel is pressed: the panel says the arm is stopping from then on."""
        event = self._ctx.poll_ui_event()
        pressed = False
        if event is not None and event.get("type") == "button":
            # any press ends the bridge's goal, and its panel with it: the next send opens it again
            self._sent = self._tried_at = None
            pressed = event.get("button_id") == PANEL_CANCEL and not self.pressed and not self.stopping
            if pressed:
                self.pressed, self.stopping = True, PANEL_CANCELLED
        panel, now = self.panel(), self._clock()
        due = self._tried_at is None or now - self._tried_at >= PANEL_REFRESH_S
        if due or (self._sent is not None and panel != self._sent):
            self._tried_at = now
            self._sent = panel if self._live() and self._ctx.update_user_panel(panel_json=panel) else None
        return pressed


class Asking:
    """How one goal's line-ups are said and asked, and how long each may take: decided ONCE, as the goal
    comes (execute), so that nothing later takes another view of it.

    `ui`: the goal has the UI (Capability._ui_live): the panel says and asks, Open is the yes. Without it
    the flight view says each line on this node's console and the question is answered y or n on ~/answer
    (the answers are read either way: a y is Open's press, an n is Cancel's).
    `timed`: the orchestrator ends the goal 120 s after it came, so its line-ups must end LINE_UP_S after
    the goal began, whichever of them it is (the orchestrator's goal, where its clock is not kept alive:
    Capability._ui_wait_s). Not timed, each line-up waits for the person: `wait_s`, the UI's wait for the
    orchestrator's goal (UI_LINE_UP_S, the deployment's `ui_line_up_s`), and CONSOLE_LINE_UP_S (None) for a
    goal sent by hand, which nothing cancels.
    `view`: the goal's flight view (None: nothing is written). `find`: whether a find follows a yes on it
    (the mock's does not: it has no flight view of its run). `carried`: this goal lines up first because
    the one before it ended in the middle of a line-up (Capability._lines_up).
    `refused`: every place ((x, y, lid z), the arm's frame) where a run found no press and asked to be
    guided, for as long as the goal lasts and into the goal a line-up is carried to: each line-up's source
    is told them all (line_up.BoxSource). As first written only the place of the request in hand was held,
    for that one line-up: a second guidance forgot the first place, and the carried line-up knew none, so
    either could ask "Ready?" on the very spot a run had refused (review of 2026-11-17)."""

    def __init__(self, ui, timed, started, view=None, find=True, carried=False, refused=(), wait_s=None):
        self.ui, self.timed, self.started, self.view, self.find = bool(ui), bool(timed), started, view, find
        self.wait_s = None if wait_s is None else float(wait_s)
        self.carried = bool(carried)
        self.refused = [list(place) for place in refused or ()]

    def left_s(self, now=None):
        """How long a line-up that begins now may take."""
        if not self.timed:
            return CONSOLE_LINE_UP_S if self.wait_s is None else self.wait_s
        return LINE_UP_S - ((time.monotonic() if now is None else now) - self.started)


class Tracker:
    """A run's rows as it goes: the progress each says, and what the result reports.
    `instance_id`: the container the orchestrator selected when the goal came, kept for the
    whole run (a hibernate clears the engine's own mid-run; REVIEW_RULES O3)."""

    def __init__(self, action_id, instance_id=""):
        self.action_id = action_id
        self.instance_id = instance_id
        self.pct = 0.0
        self.looking = False  # said once, as the run's log appears: it is up and looking for the container
        self.past_start = False  # a row past the run's start was read: a home row from here on is the way home
        # the run found no press from where the chair stands and is to be guided (its reach_refused row says
        # so): a home row from here to its request (its "guide" row) is the arm going to rest for the chair's
        # move, not the way home, and no more of the run is done by it
        self.to_rest = False
        # when the run last asked to be guided and was let go on (went): a row it wrote before that is of
        # before the person's yes, whenever it is read
        self.asked_at = None
        self.presses = 0
        self.grips = 0
        self.lid_spot = None
        self.set_down = False
        self.released = False
        self.held = False  # the lid in the fingers: from the grip that held it to its release or put-back
        self.held_said = False  # ... as the world state was last told (Capability._sync_held)
        self.opened = None  # the object record's `opened`: None not asked, then the orchestrator's answer

    def add(self, row):
        """The (phase, %, message) this row says, or None."""
        leg, kind = str(row.get("leg", "")), row.get("kind")
        if self._of_before_the_yes(row):
            # read late: the chair was guided and the person has said yes since (went). What the row says
            # (that no press plans, that the arm goes to rest) was true before that, and is not said after it
            return None
        if kind == "note":
            self.past_start = self.past_start or leg in PAST_START
            if leg == "reach_refused" and row.get("guided"):
                self.to_rest = True
                # (never that the chair is to move: NO_WAY_IN_WAIT; what the run is guided for, where it
                # says: a container out of reach, or one no camera sees)
                return "locate", self.pct, WAITS.get(row.get("what"), NO_WAY_IN_WAIT)
            if leg == "guide":
                self.to_rest = False
            return self._note(leg, row)
        ok = bool(row.get("ok"))
        if leg == "depth:press" and ok:
            self.presses += 1
        if leg == "grip:close":
            self.grips += 1
        if leg == "place:open" and ok:
            self.released = True
        if leg in ("place:open", "lid:release") and ok:
            self.held = False
        for start, phase, pct, doing in LEG_PHASES:
            if leg.startswith(start):
                if phase == "home" and not self.past_start:
                    # the run's first motion, to its start pose (press_demo.home_first): not the way home, and
                    # no part of the run done (flatbot 2026-11-11 18:06 began so, and reported 100 % at once)
                    phase, pct, doing = "locate", self.pct, "going to the start pose"
                elif phase == "home" and self.to_rest:
                    # the arm goes to rest before the chair is guided (reach_press.no_way_in, press_demo.
                    # guide_chair): the run goes on from the find after it, and its bar must still have room
                    phase, pct, doing = "locate", self.pct, TO_REST
                else:
                    self.past_start = True
                self.pct = max(self.pct, pct)
                return phase, self.pct, doing if ok else f"{doing}: stopped"
        return None

    def went(self, asked_at=None):
        """The person's yes to a line-up in the middle of the run (Capability._follow), and the progress
        that says where the run is from here: ("locate", %, LOOKING_AGAIN). The run has its go, so whatever
        was said of the refusal the chair was guided for is over: with the UI, the run's panel came back
        after Open with its last word still on it, "No way in from here: the chair is to move" beside
        Cancel, for as long as the run took to find the box again (the scene look's 2 s and the wrist's
        frames, up to 15 s more), while the arm was about to reach out (review of 2026-11-17). Nothing may
        tell the person to move once they have said yes. `asked_at`: when the run asked to be guided (this
        host's time.time(), which stamps the run's rows too): a row it wrote before that says nothing
        from here on, whenever it is read (add)."""
        self.asked_at = time.time() if asked_at is None else asked_at
        self.to_rest = False
        return "locate", self.pct, LOOKING_AGAIN

    def _of_before_the_yes(self, row):
        """Whether the run wrote `row` before it last asked to be guided and was let go on (its "t": the run
        stamps every row; one with none is taken as it comes)."""
        t = row.get("t")
        return self.asked_at is not None and isinstance(t, int | float) and t < self.asked_at

    def up(self):
        """("locate", %, message) the once, as the run's log appears: the run is up (its start-up and readiness
        take a second or two) and looks for the container, which with the scene camera can take 15 s in which
        no row is written."""
        if self.looking:
            return None
        self.looking = True
        return "locate", self.pct, "looking for the container"

    def _note(self, leg, row):
        if leg == "scene_fix" and row.get("origin_xyz"):
            x, y = row["origin_xyz"][:2]
            self.pct = max(self.pct, 0.05)
            return "locate", self.pct, f"found the container at [{x:.2f}, {y:.2f}]"
        if leg == "lid_spot" and row.get("xyz"):
            self.lid_spot = [round(float(v), 3) for v in row["xyz"]]
            x, y = self.lid_spot[:2]
            return "grip", self.pct, f"the lid will go to [{x:.2f}, {y:.2f}] on the table"
        if leg == "reach_setdown":
            self.set_down = True
        if leg == "lid_put_back" and row.get("ok"):
            self.held = False
            return "release", self.pct, "put the lid back on the container"
        if leg == "reach_grip_result":
            self.held = self.held or bool(row.get("ok"))
            return "grip", self.pct, "holding the lid" if row.get("ok") else "the grip missed the lid"
        if leg == "reach_refused":
            return "locate", self.pct, "stopping: the run refused to go on"
        return None

    def lid_down(self):
        """The lid set down on the table and let go (the container is open, whatever follows)."""
        return self.set_down and self.released

    def lid_off(self, code):
        return self.action_id == "open_box" and code == 0 and self.lid_down()


# The run's last STOP line -> (error code, recoverable), first match wins (REVIEW_RULES U4).
# Recoverable only where the run says nothing moved, or that the arm got home with nothing
# held: a retry may then succeed. A STOP no rule names, and an exit with none (a traceback),
# is `failed`, not recoverable: nothing says where the arm is.
STOP_CODES = (
    (re.compile(r"not starting"), "not_ready", True),
    (re.compile(r"NO BOX"), "not_found", True),
    # "— the arm holds", "it holds", "The arm holds over where it was": a person clears it
    (re.compile(r"\bholds\b|did not get home"), "arm_holds", False),
    # the press's contact off the band around the cameras' top, none, or its watch failed
    # (tasks/press_contact): stopped there, back up and home
    (re.compile(r"something else is in the way"), "contact_early", True),
    (re.compile(r"the box is not where it was seen"), "contact_late", True),
    (re.compile(r"contact watch failed"), "watch_failed", False),
    (re.compile(r"MOVED the box"), "box_moved", True),
    (re.compile(r"slipped out of the fingers"), "lid_dropped", False),
    (re.compile(r"the lid is back on the box"), "lid_put_back", True),
    (re.compile(r"did not pop|the knob is not up|pressed nothing"), "not_popped", True),
    (re.compile(r"no button"), "no_button", True),
    (re.compile(r"nothing was pressed|nothing moved"), "no_press", True),
)


# the run's lines that say no press plans from where the chair stands and that the engine is asked to guide
# (reach_press.no_way_in, press_demo's wrist look), or that no camera sees the container (press_demo's main):
# never a run's last word (error_for)
NOT_AN_END = ("NO WAY IN:", "OUT OF REACH:", "NOT SEEN:")


def guide_reason(asked, default="no reason given"):
    """Why a run asked to be guided ({"what", "why"}: its request, press_demo.guide_chair), in a few plain
    words for a person: what its cameras were missing (flight_view.not_seen_reason), that the container
    stands out of reach, or each tilt's reason why no press plans (no_press_reason; `default` where the
    request gives none)."""
    what, why = asked.get("what"), str(asked.get("why") or "")
    if what == UNSEEN:
        return not_seen_reason(why)
    if what == FAR:
        return FAR_SEEN
    return no_press_reason(why or default)


def error_for(action_id, code, lines):
    """The EngineError a run that ended with `code` reports, from what it said last. Not from a line that
    said the chair was to be guided (NOT_AN_END): the run went on from it, and its reasons (a planner's
    "IK_FAIL", a camera's "NO BOX") are not how the run ended."""
    said = [ln for ln in lines if "STOP" in ln or "not starting" in ln or "FAIL" in ln or "NO BOX" in ln]
    said = [ln for ln in said if not any(mark in ln for mark in NOT_AN_END)]
    why = (said or list(lines) or [f"exit {code}"])[-1]
    if code == EXIT_HALTED:
        # the e-stop, or control lost: the run sent nothing after it (REVIEW_RULES C5)
        return EngineError("arm_halted", why, recoverable=False)
    if code == 2:
        return EngineError("not_found", why, recoverable=True)
    if code == 3:
        return EngineError("not_home", why, recoverable=True)
    if code == EXIT_NOT_LINED_UP:  # its line-up ended with no go (the engine's own stop is said by the engine)
        return EngineError("not_lined_up", why, recoverable=True)
    if code == EXIT_LINE_UP_HOLDS:  # its line-up's own wrist turn stopped part way, out or back: the code says so
        return EngineError("arm_holds", f"{why} -- {ARM_HOLDS}", recoverable=False)
    if code is not None and code < 0:  # killed: whatever it said last, it did not end its own way
        return EngineError("stopped", f"the run was ended by signal {-code}: {why}", recoverable=False)
    if said:
        for pattern, error_code, recoverable in STOP_CODES:
            if pattern.search(why):
                if error_code == "arm_holds":
                    why = f"{why} -- {ARM_HOLDS}"
                return EngineError(error_code, why, recoverable=recoverable)
    return EngineError("failed", why, recoverable=False)


class Capability(CapabilityEngine):
    display_name = "Open the container"
    # "food container" is the class the scene segmenter publishes. "food_container" is the same class without
    # its space, for a selection typed by hand: the orchestrator's interactive_perception_node (the UI's object
    # list on snorlax) reads `add <instance_id> <perception_class>` split on spaces, so the first cannot be typed
    perception_classes = ["food container", "food_container"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action(
            "open_box",
            "Open the container",
            params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("box_opening", **node_kwargs)
        for name, default in PARAMS.items():
            if name == "ui_line_up_s":
                # a deployment that writes 0 or 300 means 0.0 or 300.0: an integer must not stop the node
                # from starting, least of all the 0 that says "my orchestrator is the older one"
                from rcl_interfaces.msg import ParameterDescriptor

                self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
                continue
            self.declare_parameter(name, default)
        self._run = None  # the Run under way
        self._kept = None  # this goal's context where its orchestrator's clock is kept alive (_execute)
        self._ue_heard = None  # the bridge's newest word on its app in this action (_hear_bridge)
        self._clock_from_goal = False  # this node's orchestrator was seen not to restart its 120 s (execute)
        self._person_cancelled = False  # this goal's cancel was the person's (the panel's Cancel, an n)
        self._ends_holding = False  # this goal ends with the lid in the fingers (_execute)
        self._shutting_down = False
        self._cancelled_token = None  # the execution a CANCEL came for before its goal did (_on_control)
        self._cancelled_at = None  # ... and when it came (the clock's), where it came with no token to keep it by
        self._answers = line_up.Answers()  # the y or n heard on ~/answer while an action runs (_hear_answers)
        # (when an open_box ended not_lined_up in the middle of its run, the places its runs refused): _lines_up
        self._line_up_first = None
        self._mock_pubs = self._mock_interface() if self.mock else []
        use_node_logger(self)
        self.context.on_shutdown(self._on_shutdown)
        self.get_logger().info(
            "allow_execute is on: open_box moves the arm"
            if self._allowed()
            else NOT_ALLOWED + " (open_box unavailable)"
        )

    def _mock_interface(self):
        """The mock's share of the real alternative's interface: a publisher, idle, on every topic
        the real run publishes (rammp-alternative.yaml `publishes`), with the run's own types and
        QoS — the gripper commands its run sends the driver, and with viz what its perception
        sees and decides. The real node holds none of them: its run does, and press_demo
        refuses to move an arm another program on the graph publishes gripper commands to. Nothing
        is ever sent on them (scripts/smoke.py checks that every declared topic appears)."""
        from rammp_arm_interfaces.msg import GripperSetpoint
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        from box_opening.constants import GRIPPER_SETPOINT_TOPIC
        from box_opening.perception import viz

        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        return [
            self.create_publisher(GripperSetpoint, GRIPPER_SETPOINT_TOPIC, best_effort),
            self.create_publisher(CompressedImage, viz.SCENE_TOPIC, 1),
            self.create_publisher(CompressedImage, viz.WRIST_TOPIC, 1),
            self.create_publisher(MarkerArray, viz.MARKER_TOPIC, 1),
            self.create_publisher(String, viz.STATUS_TOPIC, 1),
        ]

    def _param(self, name):
        """A PARAMS value as it is now: each goal reads its own (a deployment, or the sheppy TUI,
        may set one while the node runs), never a copy taken at start-up."""
        return self.get_parameter(name).value

    def _allowed(self):
        """allow_execute as it is now (a deployment may set it while the node runs)."""
        return bool(self._param("allow_execute"))

    def _ui_live(self):
        """Whether this execution may show its panel: the orchestrator pushed its token for it (standalone: never)
        and the bridge's request_user_input is there. Asked before every send, of the base class's own state,
        which has no call for it: as of feat/engine_update 8e7f0ab update_user_panel() logs a warning for every
        send without a token, and a run whose first send finds the bridge's server not there never opens its
        panel again (its ui_opening stays set)."""
        return self._assigned_token != bytes(16) and self._ui_request_client.server_is_ready()

    def _shown(self, ctx):
        """Whether the bridge holds this goal's panel now: the UI is there (_ui_live) and the base class has a
        goal of this run's accepted on it (its run's `ui_goal_handle`: None from a refusal, or a press, until
        the next send is taken). Where the base class cannot be asked (a context that is not its own), the
        UI being there is the answer."""
        if not self._ui_live():
            return False
        heard = self._ue_heard  # the bridge's newest word on its app, in this action: (when, connected)
        if heard is not None and (not heard[1] or time.monotonic() - heard[0] > UE_SILENT_S):
            return False
        run_ = getattr(ctx, "_run", None)
        if run_ is None or not hasattr(run_, "ui_goal_handle"):
            return True
        return run_.ui_goal_handle is not None

    def _hear_bridge(self):
        """From here to this action's end the bridge's word on whether its app is connected is heard
        (UE_CONNECTION_TOPIC), for _shown. The action's own subscription, as the answers' is."""
        self._ue_heard = None
        try:
            from std_msgs.msg import Bool

            bridge = self.get_parameter("ui_bridge_node_name").value
            self.create_action_subscription(Bool, UE_CONNECTION_TOPIC % bridge, self._on_ue_connection, 10)
        except ScopeClosed:
            pass  # no goal is running (a test drives execute() itself)
        except Exception as e:  # noqa: BLE001 - without it "shown" is the goal the bridge holds, as before
            self.get_logger().warning(f"the UI bridge's connection is not heard: {e!r}")

    def _on_ue_connection(self, msg):
        self._ue_heard = (time.monotonic(), bool(getattr(msg, "data", False)))

    def _cancel_from_ui(self, by="Cancel pressed on the UI"):
        """The panel's Cancel is the orchestrator's: the base class's own Control.CANCEL, so the run ends the
        one way a cancel ends it (its Ctrl+C, the result cancelled, no world-state write after it). `by`:
        who cancelled, for the log (the person's n on ~/answer is the same cancel)."""
        self._person_cancelled = True  # the end is said on the panel: the screen is still this goal's
        self.get_logger().warning(f"{by}: the action is cancelled")
        CapabilityEngine._on_control(self, Control.Request(command=Control.Request.CANCEL), Control.Response())

    def _timed(self, ui=False):
        """Whether this goal is the orchestrator's: it came with its token (`ui`: it has the UI, which is
        shown by that token). The orchestrator cancels it after ORCHESTRATOR_TIMEOUT_S: of the goal, and
        everything it waits for must fit inside that; or of silence, where its clock restarts at every
        progress item (_ui_wait_s). A goal with no token (sent by hand) has nothing to cancel it, and
        waits for the person (CONSOLE_LINE_UP_S)."""
        return bool(ui) or self._assigned_token != bytes(16)

    def _ui_wait_s(self):
        """How long a line-up of the orchestrator's goal waits for the person (`ui_line_up_s`, as it is now,
        at most UI_LINE_UP_MAX_S), its orchestrator's 120 s kept alive the while (KeptAlive). 0: the
        orchestrator counts its 120 s from the goal, and line-ups have what those leave. Always 0 on a build
        whose goal brings no token (the orchestrator it talks to is the one before the restart:
        goal_brings_token), for a value that is no number, and once this node has seen its orchestrator cut
        a goal at 120 s (_clock_from_goal)."""
        if self._clock_from_goal or not goal_brings_token():
            return 0.0
        try:
            wait = float(self._param("ui_line_up_s") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return min(wait, UI_LINE_UP_MAX_S) if wait > 0.0 else 0.0  # (a NaN is not > 0)

    def _hear_answers(self):
        """From here to this action's end the person's y or n is heard on ~/answer (std_msgs/String): the
        answer to a line-up's question where the goal has no panel to press on, an n the cancel at any
        moment. The subscription is the action's own (REVIEW_RULES: nothing is subscribed while idle), and
        an answer said before it began, to another action, is not kept."""
        self._answers.clear()
        try:
            from std_msgs.msg import String

            self.create_action_subscription(String, ANSWER_TOPIC, self._on_answer, 10)
        except ScopeClosed:
            pass  # no goal is running (a test drives execute() itself): answers are put straight in

    def _on_answer(self, msg):
        """A message on ~/answer. Never blocks, raises or ends the action (it runs on the action's gate)."""
        word = self._answers.put(getattr(msg, "data", ""))
        if word is None:
            self.get_logger().warning("an answer that is neither y nor n is no answer: %r" % str(msg.data)[:40])
        else:
            self.get_logger().info(f"answer heard: {word}")

    def _said_no(self, panel=None):
        """Whether the person's n came (read where no line-up is asking: while the run flies, while a wrist
        turns back): the cancel, as Cancel on the panel is, which `panel` then says. A y heard here answers
        nothing: no question is up."""
        while True:
            answer = self._answers.poll()
            if answer is None:
                return False
            if answer == line_up.NO:
                if panel is not None and not panel.pressed:
                    panel.pressed, panel.stopping = True, PANEL_CANCELLED
                self._cancel_from_ui("n answered")
                return True

    def _on_control(self, request, response):
        """The base class's, and one case it loses: a CANCEL that comes after the orchestrator has pushed its
        token for an execution and before that execution's goal (it pushes the token to the engine first, then
        to the bridge and the drivers, waiting 2 s for each that is not there, then activates and sends the
        goal). The base class has no run to cancel then, answers "ok", and the action ran to its end. It is
        kept for the execution that token is for, which ends cancelled before anything runs (execute)."""
        with self._lifecycle:
            idle = self._current is None
        response = super()._on_control(request, response)
        token = self._assigned_token
        if idle and request.command == Control.Request.CANCEL:
            # With the token pushed ahead of the goal the cancel is kept for that token's execution. With
            # the token in the goal itself (the engine update, 2026-11-16) there is none yet to keep it by:
            # it is kept by when it came, for a goal that starts within CANCEL_EARLY_S of it
            self._cancelled_token = token if token != bytes(16) else None
            self._cancelled_at = time.monotonic()
            response.message = "ok (an action that starts now is cancelled)"
            self.get_logger().warning(
                f"cancelled with no action running: one that starts within {CANCEL_EARLY_S:.0f} s will not run"
            )
        return response

    def _cancelled_early(self):
        """Whether a CANCEL came for THIS execution before its goal did (_on_control); asked once, as it
        starts: one kept for the token this execution has, or, kept with no token, one that came no longer
        than CANCEL_EARLY_S ago."""
        token, self._cancelled_token = self._cancelled_token, None
        at, self._cancelled_at = self._cancelled_at, None
        if token is not None:
            return token == self._assigned_token
        return at is not None and time.monotonic() - at <= CANCEL_EARLY_S

    def _far_m(self):
        """line_up_far_m as it is now, None where it sets no bound (0 or less)."""
        far = float(self._param("line_up_far_m") or 0.0)
        return far if far > 0.0 else None

    def _air_control(self):
        """air_control as it is now: "impedance" (unset too, as flown) or "position". Any other value is
        refused, EngineError("bad_config"), before anything runs: whoever set it meant one of the two."""
        value = str(self._param("air_control") or "impedance")
        if value not in AIR_CONTROLS:
            raise EngineError(
                "bad_config",
                f'air_control is {value!r}: it must be "impedance" or "position". Nothing ran',
                recoverable=False,
            )
        return value

    def _lines_up(self):
        """(whether this execution lines up before its first move, whether that is carried over, the places
        carried with it): the `line_up` param as its goal comes; or, carried, the open_box before it ended
        not_lined_up from a line-up in the middle of its run no longer than LINE_UP_FIRST_S ago, so that
        "press open again" goes on from the line-up and not from another look and proof. With it come the
        places that goal's run refused (Asking.refused): the carried line-up does not ask "Ready?" on one of
        them, as the line-up it takes up would not have. Asked ONCE, by execute(), and handed down from
        there. A goal with no UI lines up too (owner 2026-11-17): on the console (Asking). Until then it was
        refused, line_up_unavailable."""
        first, self._line_up_first = self._line_up_first, None
        carried = first is not None and time.monotonic() - first[0] <= LINE_UP_FIRST_S
        if carried and not self._param("line_up"):
            self.get_logger().info(
                "the open_box before this one ended in the middle of a line-up: this one lines up first"
            )
        return bool(self._param("line_up")) or carried, carried, (list(first[1]) if carried else [])

    def _line_up_again(self, ask):
        """The next open_box lines up first (_lines_up), and knows every place this goal's runs refused."""
        self._line_up_first = (time.monotonic(), [list(place) for place in ask.refused])

    def _logged(self, text, view=None, level=logging.INFO):
        """A line of this node's own about a line-up. Where the goal's flight view says the same thing on
        the console (`view`: it is there), the line is kept in the goal's file alone (NodeLogHandler.keep),
        and nowhere where the goal has no file (the mock, full_console): the console has each thing once.
        Until 2026-11-17's review every instruction stood there twice, "[INFO] ... line-up: Move closer,
        about 25 cm" and under it the flight view's "say   Move closer, about 25 cm". With no view it is
        this node's log line, as it always was."""
        if view is None:
            logger = self.get_logger()
            (logger.warning if level >= logging.WARNING else logger.info)(text)
            return
        lines = getattr(self, "lines", None)
        if lines is not None:
            lines.keep(text, logging.getLevelName(level))

    def _lined_up_run(self, ctx, env, action_id, instance_id, ask):
        """The run started with --line-up and taken through its line-up (_run_lines_up): the Run, told to
        go on the user's yes. Until then it only watches (the wrist turned to look, gently, where the arm
        stands at rest)."""
        argv = self._argv(action_id, instance_id) + ["--line-up", "--guide"]
        run_ = Run(argv, env, ask.view.relay(LOGGER.log), talk=True)
        self._run = run_  # a shutdown stops it (_on_shutdown)
        self._run_lines_up(ctx, env, run_, action_id, instance_id, ask)
        return run_

    def _run_lines_up(self, ctx, env, run_, action_id, instance_id, ask, asked=None):
        """`run_`, which watches the container, taken through one line-up; returns once it is told to go,
        on the user's yes alone. Before its first move (--line-up), or in the middle of it (`asked`: its
        request to be guided, where no press plans from where the chair stands: press_demo.guide_chair).

        A line-up that ends any other way ends the run before this raises: a cancel is its Ctrl+C, as every
        cancel is (the arm stops where it is); out of time it is told so, turns its wrist back to rest and
        ends, a cancel read all through that turn (_turned_back). Only a run that then ended "nothing
        pressed, the wrist back at rest" (EXIT_NOT_LINED_UP) leaves the line-up's own end as the error. One
        that ended any other way (not ready, its wrist's turn stopped part way: EXIT_LINE_UP_HOLDS, the
        e-stop) raises what it said, and one that had to be stopped is the arm holding (LINE_UP_LATE); where
        that is not recoverable the world state is told the arm is in error, as a run's own stop tells it.
        The flight view's result is written here, with where the arm was left.

        In the middle of a run under the orchestrator's 120 s the line-up has only what the goal's first
        LINE_UP_S leave (Asking.left_s). With less than GUIDE_MIN_S of them nothing is guided: the run is
        told to stop at once and the action ends not_lined_up, recoverable, saying why no press planned,
        and the next open_box lines up first (_lines_up)."""
        view = ask.view
        stopped = []  # a cancel's Ctrl+C was sent

        def stop_first():
            """A cancel: the run's own stop at once, BEFORE the line-up closes its source. Closing gives
            reach_probe a moment to keep its grid (line_up.PROBE_QUIT_S), and the arm's stop never waits on
            that."""
            self.get_logger().warning("cancelled while lining up: the run's own stop (Ctrl+C) sent")
            run_.interrupt()
            stopped.append(True)

        made = []  # the line-up's source, once made: it knows whether the run's wrist turned (its "view")
        # where the run has just found no press, and wherever a run of this goal (or of the goal this
        # line-up is carried from) found none before: those places do not work for this line-up (BoxSource)
        if asked is not None and asked.get("box"):
            ask.refused.append(list(asked["box"]))
        refused = [list(place) for place in ask.refused]

        # where the run asks because its own look found no container, that look's word on why: the line-up
        # says at once that the arm cannot see it, before its watch has said a word of its own (BoxSource)
        unseen = {}
        if asked is not None and asked.get("what") == UNSEEN:
            unseen = {"unseen": str(asked.get("why") or "no camera sees the container")}

        def source():
            made.append(self._box_source(env, instance_id, watch=run_, refused=refused, **unseen))
            return made[-1]

        try:
            left = ask.left_s()
            if asked is not None and ask.timed and left < GUIDE_MIN_S:
                why = guide_reason(asked, "no press plans from where the chair stands")
                self._line_up_again(ask)  # the next open_box lines up first, with its whole time
                if view is not None:
                    view.guide_end(False, "no time left to guide the chair (%.0f s): open again" % max(0.0, left))
                raise EngineError(
                    "not_lined_up",
                    NO_TIME_TO_GUIDE % (why, max(0.0, left), LINE_UP_S, ORCHESTRATOR_TIMEOUT_S),
                    recoverable=True,
                )
            self._line_up(ctx, source, ask, on_cancel=stop_first, asked=asked)
            if self._cancelled_before_go(ctx):
                raise Cancelled()  # ended as any cancel of the line-up is, below: the run is never told to go
        except BaseException as e:
            cancelled, late, turned = isinstance(e, Cancelled), False, False
            try:
                if cancelled:
                    if not stopped:
                        stop_first()
                    run_.stop(CANCEL_ESCALATE_S, interrupt=False)  # the one Ctrl+C is sent: its end, or a kill
                else:
                    run_.send({"stop": True})
                    try:
                        # its wrist turned where it said its view, or said it was turning and has not said
                        # since that it does not look: a line-up that ends in its first seconds (a
                        # reach_probe that died as it started) is stopped while that turn is still in flight
                        said = list(run_.lines)
                        turned = bool(made and getattr(made[-1], "viewed", False)) or (
                            any(LINE_UP_TURNS in ln for ln in said) and not any(LINE_UP_NO_LOOK in ln for ln in said)
                        )
                        # (in the middle of a run the wrist its own look turned is turned back too)
                        turned = turned or (asked is not None and bool(self._param("wrist_first")))
                        cancelled, late = self._turned_back(ctx, run_, turned, ask)
                    except BaseException:
                        # the engine failed while the wrist turned back (the panel, the progress): the run's
                        # own stop before anything else, or it would go on turning with nothing following it
                        run_.stop(CANCEL_ESCALATE_S)
                        raise
                code = run_.finish()
                if late and any(LINE_UP_ENDED in line for line in run_.lines):
                    # stopped late, but it had said its line-up was over and the wrist back at rest: only its
                    # own end was slow (its threads, its log), and that is no arm left part-turned
                    late, code = False, EXIT_NOT_LINED_UP
            finally:
                self._run = None  # only now: until it has ended, a shutdown must still be able to stop it
            if view is not None:
                # where the arm was left, for the result: every end of a line-up says it
                if cancelled:
                    view.arm = "stopped where it was (a wrist that looked stands turned)"
                elif late or code == EXIT_LINE_UP_HOLDS:
                    view.arm = "holding where it stopped"
                elif code == EXIT_NOT_LINED_UP:
                    view.arm = "at rest, the wrist turned back" if turned else "at rest"
                view.finish(code, cancelled=cancelled, timed_out=False, held=False)
            if cancelled and not isinstance(e, Cancelled):
                raise Cancelled() from e  # cancelled while its wrist turned back
            if cancelled or (code == EXIT_NOT_LINED_UP and not late):
                raise  # nothing was pressed and the wrist is back at rest: how the line-up itself ended
            if late:
                error = EngineError("arm_holds", LINE_UP_LATE % LINE_UP_END_S, recoverable=False)
            else:
                error = error_for(action_id, code, run_.lines)  # it ended its own way: what it said
            if not error.recoverable and self.orchestrated:  # standalone: no world state to tell
                world = WorldWriter(self.get_logger())
                self._report_arm_error(error, ctx, world)
                world.wait(WORLD_WAIT_S)
            raise error from e
        # _line_up returns on the user's yes alone (line_up.GO): every other end of a line-up raised above.
        # The go is this function's last act: nothing after it can raise with the run flying and unfollowed
        self._logged("line-up: the user's yes: the run goes on", view)
        run_.send({"go": True})

    def _cancelled_before_go(self, ctx):
        """Whether a cancel came between the line-up's Open and the run's go. The line-up's source is closed
        in between (reach_probe ended at once on an Open, its terminate keeping its grid: a moment, and up to
        line_up.STOP_WAIT_S for one that does not end), the panel already says the container is being opened
        and has its Cancel, and
        nothing reads the press or the orchestrator's cancel in that time: read here, once, so a cancel that
        came then is acted on before the arm is told to go and not a moment after."""
        event = ctx.poll_ui_event()
        if event is not None and event.get("type") == "button" and event.get("button_id") == PANEL_CANCEL:
            self._cancel_from_ui()
            return True
        if self._said_no():  # the person's n, where there is no panel
            return True
        return bool(ctx.cancelled())

    def _turned_back(self, ctx, run_, turned=True, ask=None):
        """A run told its line-up is over ({"stop": true}), followed until it has ended. Where its wrist
        turned (`turned`: it said its view) it turns it
        back to rest, beside the user: a motion as any other, so a cancel is read all through it (every
        line_up.POLL_S) and the panel says what the arm does and keeps its Cancel, which is the orchestrator's
        cancel as the run's own panel's is. A cancel is the run's Ctrl+C (the arm stops where it is). A run
        still going LINE_UP_END_S after it was told is stopped that way too, and killed if it has not ended
        CANCEL_ESCALATE_S after that. Returns (cancelled, late): a cancel came; it had to be stopped, so its
        wrist may stand part-turned. `ask`: the goal's Asking, whose flight view says the turn too; the
        person's n is the cancel here as the panel's is."""
        says = PANEL_TURNING_BACK if turned else PANEL_LINE_UP_OVER  # never "turning back" of a wrist that did not turn
        panel = Panel(ctx, self._ui_live)
        panel.say(line_up.PHASE, 0.0, says)
        told = time.monotonic()
        cancelled, late, sent = False, False, None
        if run_.alive():  # one that ended of itself already has nothing to turn back
            ctx.progress(line_up.PHASE, 0.0, says, pausable=False)
            if turned and ask is not None and ask.view is not None:
                ask.view.guide_say(says)
        while run_.proc.poll() is None:
            if panel.tick():
                self._cancel_from_ui()
            self._said_no(panel)
            now = time.monotonic()
            if sent is None and (ctx.cancelled() or panel.pressed):
                self.get_logger().warning("cancelled while the wrist turned back: the run's own stop (Ctrl+C) sent")
                cancelled, panel.stopping = True, PANEL_CANCELLED
                run_.interrupt()
                sent = now
            elif sent is None and now - told > LINE_UP_END_S:
                self.get_logger().error(f"the run did not end its line-up in {LINE_UP_END_S:.0f} s: its stop")
                late, panel.stopping = True, PANEL_TIMED_OUT
                run_.interrupt()
                sent = now
            elif sent is not None and now - sent > CANCEL_ESCALATE_S:
                self.get_logger().error("the run did not stop: killed (a motion may still be in flight)")
                run_.proc.kill()
            time.sleep(line_up.POLL_S)
        return cancelled, late

    def _box_source(self, env, instance_id, watch=None, refused=(), unseen=""):
        """A real run's line-up source (line_up.BoxSource): a watch follows the container and reach_probe judges
        where it stands, for as long as the panel is up. `watch`: the run itself, started with --line-up (its
        wrist looks; _lined_up_run); None, and box_watch is run for it, which reads the scene camera alone and
        holds no client for the arm. reach_probe never can move it. `refused`: places where the run itself has
        just found no press (a line-up in the middle of it), which do not work for this line-up. `unseen`:
        the run's own word on why its look found no container, where that is why it asked (BoxSource)."""
        container = config_path(self._param("container"), "containers")
        if watch is None:
            watch = [executable("box_watch"), "--container", container]
            watch += ["--bench-world", config_path(self._param("world"))]
            if instance_id:
                watch += ["--instance-id", instance_id]
        layers = self._param("body_layers")
        layers = PARAMS["body_layers"] if layers is None else str(layers)
        far = self._far_m()

        def probe(urdf, root):
            argv = [executable("reach_probe"), "--serve", "--urdf", urdf, "--container", container]
            argv += ["--root", root or "arm_mount", "--body-layers", layers]
            # the deployment's bound, for the probe too: its proofs are not spent on cells the line-up will
            # not trust (reach_probe --far-m; none said, it asks as far as the proof plans)
            return argv + (["--far-m", str(far)] if far is not None else [])

        return line_up.BoxSource(
            watch,
            probe,
            env,
            far_m=far,
            log=lambda text: LOGGER.log(logging.INFO, f"line-up: {text}"),
            refused=refused,
            **({"unseen": unseen} if unseen else {}),  # (said only of a run that saw none: every other as it was)
        )

    def _line_up(self, ctx, make_source, ask, on_cancel=None, asked=None):
        """One line-up (line_up.LineUp): the user is told which way to move and asked, and this returns only
        on their yes (line_up.GO), with the seconds it took. `ask` (Asking, decided by execute(): neither the
        param nor the UI is asked after again here): where it is said and asked, and how long it may take.
        With the UI, on its panel, and the yes is Open. Without it, on this node's console through the flight
        view, one line each time what it says changes, and the yes is a y on ~/answer; an n there is Cancel's
        press, with or without the UI. `asked`: the run's own request, where this is a line-up in the middle
        of it (None: before its first move).

        Cancel on it is the orchestrator's cancel (Cancelled); out of time, EngineError("not_lined_up"),
        recoverable, with the newest instruction the user was given and the watch's newest word on why it
        saw no container where that is why (why_none): nothing was pressed. A UI that goes while lining up
        shows no panel and takes no press, so it ends that way too. A source that cannot start, or cannot go
        on (its reach_probe not there to start, or ended: BoxSource.gone), is
        EngineError("line_up_unavailable"), at once and with why: no run goes on without its yes.
        `make_source()`: what sees the box and judges its place (BoxSource; the mock's MockApproach), made
        only where it is used, and closed before this returns: no run starts while it lives. `on_cancel()`:
        called as the line-up ends cancelled, before the source is closed (closing can take
        line_up.PROBE_QUIT_S): what a cancel must not keep waiting."""
        t0 = time.monotonic()
        budget_s = ask.left_s(t0)
        view = ask.view

        states = []  # what the line-up has said it is in, newest last (for its end's words)

        def said(say):
            if not states or states[-1] != say.state:
                states.append(say.state)
            ctx.progress(line_up.PHASE, 0.0, say.message, pausable=False)
            # on the console once: the flight view's line where the goal has one (its yes is the step's own
            # end: guide_end), and then this node's line is the goal's file's alone (_logged)
            self._logged(f"line-up: {say.message}", view)
            if view is not None and say.state != guide.GO:
                asks = say.state == guide.READY
                view.guide_say(
                    say.message,
                    ask=(UI_ASK if ask.ui else CONSOLE_ASK) if asks else None,
                    how=None if ask.ui else CONSOLE_HOW,
                )

        def unasked():
            # typed ahead, or as the box rolled out of place: never a yes (guide.Guide.press), and said, so
            # that the person does not wait on an answer that was not taken
            self._logged(
                "a y came with nothing asked (or asked of a box no longer seen): not a yes", view, logging.WARNING
            )
            if view is not None:
                view.guide_say("that y was not to a question: answer once it asks")

        def ended(text):
            """The flight view's word on a line-up that ends without its yes: the run's end."""
            if view is not None:
                view.guide_end(False, text)

        if view is not None:
            view.guide_start(GUIDE_DOING if asked is not None else LINE_UP_DOING)
        try:
            source = make_source()
        except OSError as e:
            ended("could not start (%s)" % e)
            raise EngineError(
                "line_up_unavailable", f"the line-up could not start ({e}): nothing moved", recoverable=False
            ) from e
        if ask.ui and not ask.timed:
            # minutes of waiting are not waited out on a screen that no longer shows them (UI_GONE_S). Under
            # the 120 s rule the line-up's own 35 s are the bound, as they always were
            source = PanelWatch(source, lambda: self._shown(ctx))
        went = False
        try:
            lining = line_up.LineUp(
                ctx,
                self._ui_live,
                source,
                budget_s,
                on_say=said,
                answers=self._answers,
                say_min_s=SAY_MIN_S,
                again=None if ask.ui else CONSOLE_AGAIN,
                on_unasked=unasked,
                again_unseen=None if ask.ui else CONSOLE_AGAIN_UNSEEN,
            )
            outcome = lining.run()
            went = outcome == line_up.GO
            if outcome == line_up.CANCELLED and on_cancel is not None:
                on_cancel()
        except OSError as e:
            # with the wrist look the probe is started inside the loop, once the run has said which robot this is
            ended("could not go on (%s)" % e)
            raise EngineError(
                "line_up_unavailable", f"the line-up could not go on ({e}): nothing was pressed", recoverable=False
            ) from e
        finally:
            close = getattr(source, "close", None)
            if close is not None and went:
                close(quit_s=0.0)  # the go waits on this: the probe is ended at once (it keeps its grid so too)
            elif close is not None:
                close()
        if outcome == line_up.CANCELLED:
            ended("cancelled: n" if lining.said_no else "cancelled")
            if lining.said_no:
                self._cancel_from_ui("n answered")
            elif lining.cancel_pressed:
                self._cancel_from_ui()
            raise Cancelled()
        if outcome == line_up.GONE:
            why = getattr(source, "why_gone", "")  # the probe, where it was the probe that ended
            ended("what watched the box, or judged its place, ended")
            raise EngineError(
                "line_up_unavailable",
                f"the line-up could not go on: {why}: nothing was pressed"
                if why
                else "what watched the container ended while lining up: nothing ran",
                recoverable=False,
            )
        if outcome == line_up.TIMED_OUT:
            why = getattr(source, "why_none", "")
            told = lining.guide.instruction  # the newest "Move ...", where one was given
            if (asked is not None or getattr(ask, "carried", False)) and lining.guide.saw_box:
                # "press open again" goes on from the line-up, with its whole time (_lines_up)
                self._line_up_again(ask)
            if lining.guide.saw_box:
                ended("not in place within %.0f s%s" % (budget_s, f" (last told: {told})" if told else ""))
                message = NOT_LINED_UP % budget_s + (f" -- last told: {told}" if told else "")
            else:
                # the same end, and what was missing said: no camera had the container at any time
                ended("the container never came into view within %.0f s" % budget_s)
                message = NOT_SEEN % budget_s
            if asked is not None and asked.get("what") in GUIDED_BECAUSE:
                message += GUIDED_BECAUSE[asked["what"]]
            elif asked is not None:
                message += " -- no press planned from where the chair stood: " + no_press_reason(
                    str(asked.get("why") or "no reason given")
                )
            error = EngineError("not_lined_up", message + (f" -- {why}" if why else ""), recoverable=True)
            # the person was in place and asked, and did not answer: not "not in place" (the panel's end)
            asked_last = [st for st in states if st != guide.TIMED_OUT][-1:] == [guide.READY]
            if not lining.guide.saw_box:
                raise saying(error, END_NOT_SEEN)  # ... nor one no camera ever saw
            raise saying(error, END_NO_ANSWER) if asked_last else error
        if outcome == line_up.UNABLE and getattr(source, "lost", False):
            # the UI went, not the arm: ended as a line-up out of time is, the wrist turned back (the caller)
            if (asked is not None or getattr(ask, "carried", False)) and lining.guide.saw_box:
                self._line_up_again(ask)  # as a line-up out of time: the next open_box goes on from the line-up
            ended("the UI stopped showing it")
            raise saying(EngineError("not_lined_up", UI_WENT % UI_GONE_S, recoverable=True), END_UI_GONE)
        if outcome == line_up.UNABLE:
            # what no waiting and no driving the chair cures (the arm not at its rest pose: a line-up moves
            # nothing but the wrist of an arm at rest). Said on the panel, and ended at once: the user is not
            # left "Looking for the container" for the whole line-up, to be told to press open again
            ended(str(source.unable))
            raise EngineError("not_home", f"{source.unable}: nothing moved", recoverable=True)
        if outcome != line_up.GO:  # nothing but the user's yes lets a run start
            ended(f"ended {outcome!r}")
            raise EngineError("line_up_unavailable", f"the line-up ended {outcome!r}: nothing ran", recoverable=False)
        if view is not None:
            view.guide_end(True, "yes" if lining.said_yes else "Open pressed", again=asked is not None, find=ask.find)
        return time.monotonic() - t0

    def availability(self, instance_id, world_state):
        """open_box for a container the world state does not already hold opened (its object
        record's `opened`, which a run writes as it lets the lid go on the table), while the
        gripper holds nothing: a run that stopped with the lid in the fingers says so in
        `gripper.holding`, and a run started from there would start holding it."""
        opened = ((world_state.get("objects") or {}).get(instance_id) or {}).get("opened") is True
        holding = (world_state.get("gripper") or {}).get("holding")
        if holding:
            why = f"the gripper holds {holding}: empty it, attended, before the next run"
        elif opened:
            why = "the container is already opened"
        else:  # REVIEW_RULES O1: offered only where it will run. The mock moves nothing: always offered
            why = "" if self.mock or self._allowed() else NOT_ALLOWED
        return [ActionAvailability(action_id=a.action_id, available=not why, reason=why) for a in self.actions]

    def execute(self, action_id, params, ctx):
        self._kept = None  # this goal's context where its orchestrator's clock is kept alive (_execute)
        self._person_cancelled = self._ends_holding = False
        ended, began = None, time.monotonic()  # how it ended, for the panel: (words, failed)
        try:
            result = self._execute(action_id, params, ctx)
            ended = end_says(result=result)
            return result
        except Cancelled:
            took = time.monotonic() - began
            if self._kept is not None and not self._person_cancelled and not self._shutting_down:
                if CLOCK_FROM_GOAL_S[0] <= took <= CLOCK_FROM_GOAL_S[1]:
                    self._clock_from_goal = True
                    self.get_logger().error(
                        "this goal was cancelled %.0f s after it came, its progress sent all the while: the "
                        "orchestrator does not restart its 120 s at each progress item. ui_line_up_s is taken "
                        "as 0 from now until this node is started again (set it 0 in the deployment)" % took
                    )
            elif self._person_cancelled:
                # the person's own cancel is said on the panel. The orchestrator's is not: the screen is its
                # own again from that moment, and the bridge refuses this goal's panel
                ended = end_says(cancelled=True, holding=self._ends_holding)
            raise
        except EngineError as e:
            ended = end_says(error=e, holding=self._ends_holding)
            raise
        finally:
            try:
                if ended is not None:
                    self._show_end(*ended)
            finally:
                lines = getattr(self, "lines", None)
                if lines is not None:
                    lines.to_console()  # the goal's file closed: this node's lines are the console's again

    def _show_end(self, said, failed):
        """How the action ended, on the UI's panel for END_SHOW_S, before the action ends and the orchestrator
        puts its own panel back. Only for a goal with the UI whose orchestrator's clock is kept alive
        (self._kept): its progress goes on being sent the while. No button: there is nothing left to ask.
        Nothing here changes how the action ends: a send that fails is logged and the end goes on."""
        ctx = self._kept
        if ctx is None or not said:
            return
        panel = {"title": PANEL_TITLE, "message": said, "buttons": []}
        if failed:
            panel["style"] = "error"  # the bridge's schema: a panel's own style, absent where it is normal
        try:
            began, sent_at = time.monotonic(), None
            while time.monotonic() - began < END_SHOW_S:
                if self._shutting_down or not self._ui_live():
                    return  # the node is being stopped, or the UI is not there: nobody to say it to
                if time.monotonic() - began >= END_UNSHOWN_S and not self._shown(ctx):
                    # the bridge does not hold it (it refuses this goal's token: restarted, or the
                    # orchestrator has the screen again): the action is not held up for a panel nobody sees
                    return
                if ctx.poll_ui_event() is not None:
                    sent_at = None  # a press on the panel before this one took the bridge's panel down
                now = time.monotonic()
                if sent_at is None or now - sent_at >= PANEL_REFRESH_S:
                    ctx.update_user_panel(panel_json=panel)
                    sent_at = now
                time.sleep(POLL_S)
        except Exception as e:  # noqa: BLE001 - the end is the run's own, whatever the panel does
            self.get_logger().warning(f"the end was not shown on the panel: {e!r}")

    def _lines_to_file(self, runs):
        """This goal's full lines go to runs/console-<stamp>.log from here until it ends (NodeLogHandler); the
        file's place as the flight view says it, or None where they stay on the console (full_console on, a
        test's engine with no node log, a file that cannot be opened)."""
        lines = getattr(self, "lines", None)
        if lines is None or self._param("full_console"):
            return None
        name = time.strftime("console-%Y%m%d-%H%M%S.log", time.gmtime())
        return "runs/%s" % name if lines.to_file(Path(runs) / name) else None

    def _execute(self, action_id, params, ctx):
        if self._cancelled_early():
            raise Cancelled()  # nothing ran
        check_params(params)
        ui = self._ui_live()  # asked once for this goal: what the log says is what is decided on
        orchestrated = self._timed(ui)  # ... and whether it is the orchestrator's (its token came with it)
        # ... and, where it is, whether the orchestrator's 120 s bound it whole (`timed`) or its clock is
        # kept alive and its line-ups wait for the person (wait_s > 0: _ui_wait_s)
        # ... and only for a goal that has the UI as it comes: one whose bridge is not there asks on the
        # console of a node nobody sits at, and keeps the 35 s it always had (review of 2026-11-19)
        wait_s = self._ui_wait_s() if (orchestrated and ui) else 0.0
        timed = orchestrated and wait_s <= 0.0
        if orchestrated and not timed:
            ctx = self._kept = KeptAlive(ctx)
            self._hear_bridge()  # whether the screen still shows what is asked on it (_shown)
        # the settings that decide how this goal flies, as they are now: a `ros2 param set` lives only in the
        # running node and a restart puts the deployment's values back, so every goal's are in the log
        self.get_logger().info(
            "open_box goal: line_up=%s ui=%s orchestrated=%s line_up_waits=%s line_up_far_m=%s air_control=%s "
            "wrist_first=%s wrist_alone=%s"
            % (
                str(bool(self._param("line_up"))).lower(),
                "live" if ui else "none",
                str(orchestrated).lower(),
                # how long a line-up of this goal may take: what the orchestrator's 120 s leave, or the wait
                ("%.0fs-of-the-goal" % LINE_UP_S)
                if timed
                else "%.0fs" % (wait_s if orchestrated else CONSOLE_LINE_UP_S),
                float(self._param("line_up_far_m") or 0.0),
                self._param("air_control") or "impedance",
                str(bool(self._param("wrist_first"))).lower(),
                str(bool(self._param("wrist_alone"))).lower(),
            )
        )
        self._air_control()  # a value it cannot be: refused here, before anything runs
        if not self.mock and not self._allowed():  # the mock moves nothing: it is always allowed
            raise EngineError("not_allowed", NOT_ALLOWED + ": nothing ran", recoverable=False)
        # whether this goal lines up before its first move, decided once and before anything is started; where
        # it is asked (the UI's panel, or with no UI this node's console and ~/answer) and how long a line-up
        # may take (the orchestrator's goal: what its 120 s leave; one sent by hand: CONSOLE_LINE_UP_S)
        lines_up, carried, refused = self._lines_up()
        t0 = time.monotonic()
        ask = Asking(ui, timed, t0, carried=carried, refused=refused, wait_s=wait_s if orchestrated else None)
        self._hear_answers()  # the person's y or n, from here to the action's end
        if self.mock:
            return self._mock(action_id, ctx, lines_up, ask)
        ctx.checkpoint()  # cancelled before it started: nothing runs
        env = self._env()  # this goal's params, read once (REVIEW_RULES: none is a start-up copy)
        # the selection as the goal came: read once, used for the line-up, every write and the run's argv
        instance_id = self.instance_id
        runs = state_dir(env) / "runs"
        log = RunLog(runs, RunLog.existing(runs))
        track = Tracker(action_id, instance_id)
        # the flight view (flight_view): the run's lines read as they are logged, one short line a step
        # on this node's stdout. It only reads, and nothing it meets stops the run or its reader. A line-up's
        # lines are the engine's own on it (Asking.view)
        view = ask.view = FlightView()
        view.log = self._lines_to_file(runs)
        lined_up_s, run_, went_at = 0.0, None, None
        if lines_up:
            view.start(action_id, instance_id, find=False)  # the find comes after the line-up's yes
        if lines_up and self._param("wrist_first"):
            # the wrist looks while the user lines up (owner 2026-11-16: "keep the wrist look"): the run's own
            # process is the line-up's eyes, started now and held at its line-up until the yes, so that one
            # process alone ever commands the arm. Its budget runs from here, the line-up inside it (for a
            # goal sent by hand, from its go: went_at)
            run_ = self._lined_up_run(ctx, env, action_id, instance_id, ask)
            went_at = time.monotonic()
        elif lines_up:
            # nothing moves, and nothing runs but on the line-up's yes
            try:
                lined_up_s = self._line_up(
                    ctx, lambda: self._box_source(env, instance_id, refused=[list(p) for p in ask.refused]), ask
                )
                if self._cancelled_before_go(ctx):
                    raise Cancelled()  # nothing ran
            except BaseException as e:
                view.arm = "at rest (nothing moved)"
                view.finish(None, cancelled=isinstance(e, Cancelled), timed_out=False, held=False)
                raise
        first = MOCK_PHASES[action_id][0][0]  # the first phase's name
        try:
            ctx.progress(first, 0.0, "starting", pausable=False)
            panel = Panel(ctx, self._ui_live)  # on the UI from now until the action ends
            panel.say(first, 0.0, "starting")
            panel.tick()
            world = WorldWriter(self.get_logger())
        except BaseException:
            if run_ is not None:
                # a run that lined up has its go and flies already, and the loop that follows it (its cancel,
                # its budget, its stop where following it raises) has not begun: its own stop here, or
                # nothing would stop it once this goal has ended
                try:
                    run_.stop(CANCEL_ESCALATE_S)
                finally:
                    self._run = None
                view.finish(run_.finish(), cancelled=False, timed_out=False, held=False)
                self.get_logger().error("the run was stopped (Ctrl+C): the engine failed before it could follow it")
            raise
        if run_ is None:
            if not lines_up:
                view.start(action_id, track.instance_id)
            # --guide: where it finds no press from where the chair stands it asks to be guided and waits on
            # its stdin, as a run that lines up first does (_follow)
            run_ = Run(self._argv(action_id, track.instance_id) + ["--guide"], env, view.relay(LOGGER.log), talk=True)
        self._run = run_
        # The run's own time. Under the orchestrator's 120 s: RUN_BUDGET_S from the goal's start, the line-up
        # inside it. A goal sent by hand has no such bound, and the time its person took is none of the run's:
        # RUN_BUDGET_S from the line-up's go, so that a slow line-up never cuts the run it lets go
        budget_s = RUN_BUDGET_S - lined_up_s if timed else RUN_BUDGET_S
        since = run_.started if (timed or went_at is None) else went_at
        code = None
        try:
            code = self._follow(run_, log, track, ctx, world, panel, budget_s, since=since, guide=(ask, env, action_id))
        finally:
            self._run = None
            if run_.proc.poll() is None:
                # _follow raised with the run still going: nothing else would stop it, and it
                # would go on moving the arm after its goal ended. Its own stop first (Ctrl+C:
                # the motion in flight cancelled), a kill if it has not ended in time.
                self.get_logger().error(
                    f"the run outlived its goal: its own stop (Ctrl+C), a kill after {CANCEL_ESCALATE_S:.0f} s"
                )
                run_.stop(CANCEL_ESCALATE_S)
            self._progress(log, track, ctx, panel)  # the last rows, written as it ended
            if not (ctx.cancelled() or panel.pressed):
                self._sync_world(track, ctx, world)  # the last rows' word: a release, a put-back
            view.finish(
                code, cancelled=ctx.cancelled() or panel.pressed, timed_out=run_.timed_out, held=bool(track.held)
            )
            if code is None:
                # following it raised (a line-up in the run's middle that ended it; a failure of the engine's):
                # what was queued for the world state is written, and its writer's thread ends with the goal
                world.wait(WORLD_WAIT_S)
        if ctx.cancelled() or panel.pressed:
            world.wait(WORLD_WAIT_S)  # those queued before the cancel
            self._ends_holding = bool(track.held)  # said on the panel as the action ends (end_says)
            raise Cancelled()
        result = {
            "executed": True,
            "lid_off": track.lid_off(code),
            "lid_xyz": track.lid_spot if track.lid_off(code) else None,
            "presses": track.presses,
            "grips": track.grips,
            "duration_s": round(time.monotonic() - t0, 1),
            "video": run_.video,
            "run_log": str(log.path) if log.path else None,
        }
        error = None
        if run_.timed_out:
            error = EngineError(
                "timeout",
                f"the run took longer than {budget_s:.0f} s and was stopped its own way (Ctrl+C)",
                recoverable=True,
            )
        elif code != 0:
            error = error_for(action_id, code, run_.lines)
        if error is not None:
            if track.held:
                self._ends_holding = True
                # the next run would start holding it (availability): nothing to retry until it is emptied
                error = EngineError(error.error_code, f"{error} -- the lid is still in the fingers", recoverable=False)
            if not error.recoverable and self.orchestrated:  # standalone: no world state to tell
                self._report_arm_error(error, ctx, world)
            world.wait(WORLD_WAIT_S)
            raise error
        world.wait(WORLD_WAIT_S)  # every write answered before the result goes out
        if result["lid_off"] and track.instance_id:
            # `opened` is the object record's own field (the orchestrator refuses any other); where
            # the lid went is the result's lid_xyz. A refused write leaves open_box offered on an
            # open container: said, in the log and in the result (REVIEW_RULES O6)
            result["world_state"] = "opened" if track.opened else "not updated"
            if result["world_state"] != "opened":
                self.get_logger().error(
                    f"the lid is off, but the world state was not updated: {track.instance_id} is not marked opened"
                )
        return result

    def _follow(self, run_, log, track, ctx, world, panel, budget_s=RUN_BUDGET_S, since=None, guide=None):
        """Progress from the run's rows until it ends, on `panel` too; a cancel (the orchestrator's, the
        panel's Cancel, which is made the orchestrator's, or the person's n on ~/answer) is its Ctrl+C. Its
        exit code. The world-state writes the rows call for are `world`'s, never this loop's to wait on.
        `budget_s`: the run's own time, counted from `since` (None: from its start): what a line-up before
        it left of RUN_BUDGET_S.

        `guide` ((Asking, the run's env, its action)): where the run asks to be guided (no press plans from
        where the chair stands: press_demo.guide_chair, run 2 of 2026-11-17), it is taken through a line-up
        in the middle of itself (_run_lines_up) and followed on from its go. A line-up that ends any other
        way has ended the run, safely, before it raises out of here. Under the orchestrator's 120 s the
        run's budget is not moved by it (the line-up is given only where a long run still fits after it:
        Asking.left_s); for a goal sent by hand the budget starts again at the go, RUN_BUDGET_S whole: the
        time a person takes to move a chair is none of the run's, and no budget may cut a motion short."""
        sent = escalated = None
        since = run_.started if since is None else since
        while run_.proc.poll() is None:
            self._progress(log, track, ctx, panel)
            if panel.tick():
                self._cancel_from_ui()
            self._said_no(panel)
            cancelled = ctx.cancelled() or panel.pressed  # the press itself, whatever the base class made of it
            if not cancelled:
                self._sync_world(track, ctx, world)
            asked = run_.take_guide() if guide is not None else None
            if asked is not None and sent is None and not cancelled:
                ask, env, action_id = guide
                asked_at = time.time()  # (the run's rows carry the same clock's time: Tracker.went)
                # every row the run wrote before it asked is read before its line-up begins: it writes them
                # first (its refusal, the home to rest), and one read only after the yes would say on the
                # panel what was true before it
                self._progress(log, track, ctx, panel)
                why = guide_reason(asked)
                # (on the console the flight view says it, from the run's own line: this is the goal's file's)
                self._logged(
                    "%s (%s): the run is not ended, the chair is guided"
                    % (ASKED_FOR.get(asked.get("what"), "no press plans from where the chair stands"), why),
                    ask.view,
                    logging.WARNING,
                )
                # why, to whoever follows the action's progress: a person at the terminal program has no
                # console to read it on. The run asks only once its arm is at rest (press_demo.guide_chair),
                # so this is the first word that the chair is to move
                ctx.progress(line_up.PHASE, 0.0, why[:1].upper() + why[1:] + GUIDE_SAYS, pausable=False)
                self._run_lines_up(ctx, env, run_, action_id, track.instance_id, ask, asked=asked)
                # the person's yes, and the run has its go. The line-up had the screen: the run's panel is
                # sent again at once, and what it and the progress say is where the run is NOW (Tracker.went),
                # never its last word from before the line-up, which was that no press planned
                again = track.went(asked_at)
                ctx.progress(*again, pausable=False)
                panel.say(*again)
                panel.again()
                if not ask.timed:
                    since, budget_s = time.monotonic(), RUN_BUDGET_S
                continue
            now = time.monotonic()
            if cancelled and sent is None:
                self.get_logger().warning("cancelled: the run's own stop (Ctrl+C) sent")
                panel.stopping = PANEL_CANCELLED
                run_.interrupt()
                sent = now
            elif sent is None and now - since > budget_s:
                self.get_logger().error(
                    f"the run is still going after {budget_s:.0f} s: its own stop (Ctrl+C), before the "
                    "orchestrator's 120 s cancel"
                )
                run_.timed_out = True
                panel.stopping = PANEL_TIMED_OUT
                run_.interrupt()
                sent = now
            elif sent is not None and escalated is None and now - sent > CANCEL_ESCALATE_S:
                self.get_logger().error(f"the run did not stop in {CANCEL_ESCALATE_S:.0f} s: a second Ctrl+C")
                run_.interrupt()
                escalated = now
            elif escalated is not None and now - escalated > CANCEL_KILL_S:
                self.get_logger().error("the run did not stop: killed (a motion may still be in flight)")
                run_.proc.kill()
            time.sleep(POLL_S)
        return run_.finish()

    @staticmethod
    def _sync_held(track, ctx, world):
        """The world state's gripper as the run's rows say it (REVIEW_RULES C7): holding the
        selected container's lid from the grip that held it to its release or put-back. A cancel,
        or a run stopped between the two, then leaves it said: the base class takes no world-state
        write once an action is cancelled, so it is written as it happens, never after. Queued on
        `world` (WorldWriter), in order."""
        if not track.instance_id or track.held == track.held_said:
            return
        track.held_said = track.held
        holding = f"{track.instance_id}:lid" if track.held else None
        world.put(f"the gripper holds {holding}", lambda: ctx.write_world_state(gripper={"holding": holding}))

    @staticmethod
    def _report_arm_error(error, ctx, world):
        """A run that ended where a person must clear it (an e-stop, an arm held part way, the lid
        still in the fingers) says so in the world state's system_status, which the orchestrator
        honours: no object or action flow until the arm's domain is normal again."""
        status = {"arm": {"state": "error", "error_code": error.error_code, "message": str(error)}}
        world.put(f"the arm is in error ({error.error_code})", lambda: ctx.write_world_state(system_status=status))

    def _sync_world(self, track, ctx, world):
        """What the run's rows so far tell the world state: the gripper's `holding`, and the
        container `opened` once the lid is down on the table and let go."""
        self._sync_held(track, ctx, world)
        if track.lid_down():
            self._mark_opened(track, ctx, world)

    @staticmethod
    def _mark_opened(track, ctx, world):
        """The container's object record `opened`, once, as the lid is let go on the table
        (REVIEW_RULES C6): a run that fails or is cancelled after that, on its way home, leaves
        the container open, and open_box must not be offered on it again. track.opened is the
        orchestrator's answer."""
        if not track.instance_id or track.opened is not None:
            return
        track.opened = False  # asked; True once the orchestrator takes it
        world.put(
            f"{track.instance_id} is opened",
            lambda: ctx.update_object(track.instance_id, opened=True),
            lambda ok: setattr(track, "opened", ok),
        )

    @staticmethod
    def _progress(log, track, ctx, panel):
        """The progress each new row of the run log says (a run cannot pause once started), to the
        orchestrator and on the run's panel."""
        rows = log.rows()  # finds the run's log once it is there
        said = [track.up()] if log.path is not None else []
        said += [track.add(row) for row in rows]
        for progress in said:
            if progress is not None:
                ctx.progress(*progress, pausable=False)
                panel.say(*progress)

    def _argv(self, action_id, instance_id=""):
        argv = [executable(EXECUTABLES[action_id])]
        argv += ["--container", config_path(self._param("container"), "containers")]
        argv += ["--bench-world", config_path(self._param("world"))]
        argv += ["--speed-scale", str(float(self._param("speed_scale")))]
        layers = self._param("body_layers")  # only an explicit "" turns the check off; unset is the default
        argv += ["--body-layers", PARAMS["body_layers"] if layers is None else str(layers)]
        if instance_id:  # the container the orchestrator selected; standalone: any of its class
            argv += ["--instance-id", instance_id]
        if self._param("wrist_survey"):  # the run surveys only without a selection (tasks/survey)
            argv += ["--wrist-survey"]
        if self._param("wrist_first"):
            argv += ["--wrist-first"]
        if self._param("wrist_alone"):  # read by the wrist look alone (--wrist-first), and only with a selection
            argv += ["--wrist-alone"]
        return argv + ["--execute"]  # only ever run where allow_execute is on (execute())

    def _env(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["BOX_OPENING_VIDEO"] = "1" if self._param("video") else "0"
        env["BOX_OPENING_VIZ"] = "1" if self._param("viz") else "0"
        env["BOX_OPENING_MISSION_FRAMES"] = "1" if self._param("mission_frames") else "0"
        if self._param("state_dir"):
            env["BOX_OPENING_STATE"] = self._param("state_dir")
        env[AIR_CONTROL_ENV] = self._air_control()
        for name, var in (("planner_worlds_as", PLANNER_WORLDS_AS_ENV), ("planner_worlds_dir", PLANNER_WORLDS_DIR_ENV)):
            if self._param(name):  # unset: whatever the node's own environment says, as BOX_OPENING_BODY_LAYERS
                env[var] = str(self._param(name))
        return env

    def _mock(self, action_id, ctx, lines_up=False, ask=None):
        """A successful real run at the interface, nothing moved (review round 2): its phases in
        its words (each `mock_phase_s` long), the gripper's `holding` from the grip to the release and
        the container `opened` at the release, written as the real run writes them, its panel on the UI
        and its result. Like the real run it cannot pause; a cancel ends it. `lines_up`: execute()'s
        decision for this goal (_lines_up), as the real run takes it; `ask`: where its line-up is said and
        asked (Asking): with no UI on this node's console and ~/answer, as the real run's is, so that the
        console's side and the terminal program can be rehearsed on the mock too."""
        track = Tracker(action_id, self.instance_id)
        phase_s = max(0.0, float(self._param("mock_phase_s")))
        t0 = time.monotonic()
        ask = Asking(self._ui_live(), self._timed(self._ui_live()), t0) if ask is None else ask
        if lines_up:
            # the real run's line-up, of a box that comes into reach over a few of the mock's phases. Its lines
            # are the flight view's (the mock's run itself has none)
            ask.view, ask.find = FlightView(), False
            ask.view.start(action_id, track.instance_id, find=False)
            try:
                self._line_up(
                    ctx, lambda: line_up.MockApproach(MOCK_APPROACH_PHASES * phase_s, far_m=self._far_m()), ask
                )
            except BaseException as e:
                ask.view.arm = "the mock moves nothing"
                ask.view.finish(None, cancelled=isinstance(e, Cancelled), timed_out=False, held=False)
                raise
        world = WorldWriter(self.get_logger())
        panel = Panel(ctx, self._ui_live)  # the real run's, Cancel included

        def cancel_point():
            """A cancel (the orchestrator's, the panel's Cancel, or the person's n) ends the mock here, as
            the real run."""
            if panel.tick():
                self._cancel_from_ui()
            self._said_no(panel)
            if ctx.cancelled() or panel.pressed:
                world.wait(WORLD_WAIT_S)
                raise Cancelled()

        for phase, pct, doing in MOCK_PHASES[action_id]:
            cancel_point()
            ctx.progress(phase, pct, doing, pausable=False)
            panel.say(phase, pct, doing)
            if phase == "grip":
                track.held = True
            elif phase == "release":
                track.held, track.set_down, track.released = False, True, True
                track.lid_spot = list(MOCK_LID_XYZ)
            self._sync_world(track, ctx, world)
            until = time.monotonic() + phase_s  # the phase's length, a cancel read all through it
            while time.monotonic() < until:
                cancel_point()
                time.sleep(POLL_S)
        world.wait(WORLD_WAIT_S)
        result = {
            "executed": True,
            "mock": True,
            "lid_off": True,
            "lid_xyz": track.lid_spot,
            "presses": 1,
            "grips": 1,
            "duration_s": round(time.monotonic() - t0, 1),
            "video": None,
            "run_log": None,
        }
        if track.instance_id:
            result["world_state"] = "opened" if track.opened else "not updated"
        return result

    def _on_shutdown(self):
        """SIGTERM (docker stop) or Ctrl+C: a run under way stops its own way first."""
        self._shutting_down = True  # nothing more is shown or waited for (_show_end)
        run_ = self._run
        if run_ is not None:
            run_.stop(SHUTDOWN_WAIT_S)


class NodeLogHandler(logging.Handler):
    """Every box_opening log record as a line of the node's log, at its own level; while a goal's file is open
    (to_file), in that file instead, an error in both.

    Owner 2026-11-17, after three runs watched through `sheppy logs`: "i changed the way my box opening terminal
    outputs, no? why is it still the old outputs?" The flight view was there, on stdout, with every full line
    of the run between its lines on stderr: sheppy shows both streams as one. So a run's full lines are kept
    in a file beside its run log, and the console is the flight view's (and this node's own few lines)."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self._file = None

    def to_file(self, path):
        """The lines from now on go to `path` (appended); False where it cannot be opened: the console as before."""
        self.to_console()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115 -- closed by to_console
        except OSError:
            self._file = None
        return self._file is not None

    def to_console(self):
        """The lines go to the node's log again."""
        f, self._file = self._file, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass

    def keep(self, text, levelname="INFO"):
        """`text` as a line of the goal's file, and nowhere else: what the console is told another way (the
        flight view's line for it: Capability._logged). False where no file is open, or it failed: the line
        is then not kept."""
        self.acquire()  # the handler's own lock (emit holds it already): the run's reader writes here too
        try:
            f = self._file
            if f is None:
                return False
            now = time.time()
            stamp = time.strftime("%H:%M:%S", time.gmtime(now)) + ".%03d" % (1000 * (now % 1))
            f.write("%s %-7s %s\n" % (stamp, levelname, text))
            f.flush()
        except (OSError, ValueError):
            return False
        finally:
            self.release()
        return True

    def emit(self, record):
        logger = self.node.get_logger()
        msg = record.getMessage()
        if self.keep(msg, record.levelname) and record.levelno < logging.ERROR:
            return  # (a file that fails is not the run's problem: the console)
        if record.levelno >= logging.ERROR:
            logger.error(msg)
        elif record.levelno >= logging.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)


def use_node_logger(node):
    """The box_opening logger writes to `node`'s log instead of the console."""
    LOGGER.handlers.clear()
    node.lines = NodeLogHandler(node)  # a goal's full lines go to its file through it (Capability._lines_to_file)
    LOGGER.addHandler(node.lines)


def main():
    run(Capability)


if __name__ == "__main__":
    main()

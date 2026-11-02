"""Autonomous microwave button press for the Kinova Gen3, as a RAMMP capability engine.

Port of ``feeding_deployment.button_press`` (branch ``button-task``) to RAMMP's arm driver, run by
``capability_engine.CapabilityEngine``. Interface: ``docs/interface.md``; design and history:
``README.md``.

Modules:
  node                     the capability: the press_button action
  press_button             one press run: detect -> plan and gate every leg -> go -> press -> return
  dome_pattern             main button detector: the 5 chrome domes, wrist camera
  scene_finder             first rough look from the fixed scene camera, staging only
  panel_frame, geometry    pure maths: the button-centred panel frame, plane fit, pixel rays
  kinematics               PyBullet IK and every motion safety gate (no ROS)
  press_arm                the arm, through the engine's self.arm: state, trajectories, gripper, stream
  perception               the action's camera subscriptions and tf

``dome_pattern``, ``panel_frame`` and ``geometry`` were copied from ``feeding_deployment``.
"""

import sys

_output = None


class Abort(Exception):  # noqa: N818 -- the run's name for it since feeding_deployment
    """Stop the run and HOLD the arm where it is.

    Raised by the run's own checks: a gate that refuses a leg before it is sent, a detection that
    fails, a press its torque checks stopped. The node fails the action with its message."""


class MotionFailed(Abort):
    """A motion that was sent failed (the driver's error, or it ended off target). The arm holds
    where it is and nothing moves it again: no back-off, no retry."""


def say(*parts, sep=" ", end="\n", flush=False):
    """A run's progress line, with print()'s signature. Goes to stdout, or wherever set_output()
    points it (the node: its log)."""
    out = _output or sys.stdout
    out.write(sep.join(map(str, parts)) + end)
    if flush:
        out.flush()


def set_output(stream):
    """Send say() to `stream` (anything with write/flush) instead of stdout; None restores stdout."""
    global _output
    _output = stream

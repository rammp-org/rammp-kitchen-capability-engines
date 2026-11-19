# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""The microwave open case every benchmark here runs: flatbot's arm model (robot_state_publisher on snorlax,
2026-11-18), door_common configured for the microwave, the door and start joints of
tests/arm/test_door_planning.py::test_microwave_open_plans."""

import tempfile
from pathlib import Path

import numpy as np
from microwave import open_door as microwave_open

from capability_engine import sim

HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])
DOOR = {
    "closed_normal": [-0.9998242162892161, 0.01874930723133269, 0.0],
    "closed_handle": [0.8433989796495204, -0.19333636203672608, 0.26055629510592393],
    "door_z": [0.13539826826136991, 0.40185965793508877],
    "hinge": [0.879997694735208, 0.13497556172597747, 0.2593880762812125],
    "hinge_pivot": [0.8983214551036536, 0.15554629533022274, 0.2593880762812125],
}
GEOM = {
    "h_raw": np.array(DOOR["closed_handle"]),
    "orient": HANDLE_QUAT,
    "n": np.array(DOOR["closed_normal"]),
    "hinge": np.array(DOOR["hinge_pivot"]),
    "hinge_seen": np.array(DOOR["hinge"]),
    "door_z": np.array(DOOR["door_z"]),
}
Q0 = np.radians((0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0))
MAX_SWING, MIN_SWING = 50, 45  # the microwave's swing before the push-open


def make(urdf):
    sim.reset_for_tests()
    scene, rb = sim.make_sim(urdf)
    microwave_open.configure(Path(tempfile.mkdtemp()))
    return scene, rb


def after_swing():
    return microwave_open.PushOpen(
        None
    )  # no planner: the push-open is cuRobo's and isn't checked here


def summary(plan):
    """What a plan is, for comparing planners: the candidate's identity and its joint paths."""
    if plan is None:
        return None
    return {
        "pitch": plan["pitch"],
        "side": plan["side"],
        "swing": plan["swing"],
        "standoff": plan["standoff"],
        "margin": plan["margin"],
        "q_g": np.asarray(plan["q_g"]).tolist(),
        "q_pre": np.asarray(plan["q_pre"]).tolist(),
        "leg": np.asarray(plan["leg"]).tolist(),
        "arc": np.asarray(plan["arc"]).tolist(),
        "wps": np.asarray(plan["wps"]).tolist(),
        "grasp": [list(plan["grasp"].position), list(plan["grasp"].orientation)],
        "pre_p": np.asarray(plan["pre_p"]).tolist(),
    }

"""Exercise the wrist re-look path without moving hardware."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


class Refused(Exception):
    pass


class NoHandleProposal(Refused):
    pass


def _relook(detect):
    path = Path(__file__).resolve().parents[1] / "fridge" / "open_door.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "detect_with_wrist_relook"
    )
    scope = dict(
        detect=detect,
        timing=SimpleNamespace(timed=lambda _name: lambda fn: fn),
        shared=SimpleNamespace(NoHandleProposal=NoHandleProposal),
        Refused=Refused,
        np=np,
        R=Rotation,
        pl=SimpleNamespace(free=lambda pos, quat, label, world, **_: (pos, quat, world)),
        say=lambda _message: None,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    return scope[fn.name]


def _inputs():
    arm = SimpleNamespace(get_state=lambda: dict(position=np.zeros(7), ee_pos=[0, 0, 0, 0, 0, 0, 1]))
    camera = Mock()
    camera.grab.return_value = {"base_T_cam": np.eye(4)}
    boxes = [object()]
    model = SimpleNamespace(df=SimpleNamespace(n=[-1, 0, 0]), door={"closed_handle": [0.6, 0, 0]})
    model.full = Mock(return_value=boxes)
    return arm, camera, model, boxes


def test_blocked_first_wrist_view_is_skipped_and_second_view_is_fresh():
    arm, camera, model, boxes = _inputs()
    detect = Mock(side_effect=[NoHandleProposal("empty"), "fit"])
    planner = Mock()
    motion = object()
    planner.motion.side_effect = [(None, "blocked"), (motion, None)]
    planner.run.return_value = True

    assert _relook(detect)(arm, planner, camera, lambda: model) == "fit"
    model.full.assert_called_once_with(0.0)
    goals = [call.args[0][0] for call in planner.motion.call_args_list]
    np.testing.assert_allclose(goals[0][0], [-0.03, -0.04, 0])
    np.testing.assert_allclose(goals[1][0], [-0.03, 0.04, 0])
    assert all(goal[2] is boxes for goal in goals)
    assert planner.join.call_args.args[:2] == (motion, boxes)
    planner.run.assert_called_once()
    camera.require_frames_after_now.assert_called_once()


def test_fit_rejection_does_not_trigger_a_motion():
    arm, camera, model, _ = _inputs()
    detect = Mock(side_effect=Refused("bad fit"))
    planner, scene_model = Mock(), Mock(return_value=model)
    with pytest.raises(Refused, match="bad fit"):
        _relook(detect)(arm, planner, camera, scene_model)
    scene_model.assert_not_called()
    planner.motion.assert_not_called()


def test_no_scene_model_refuses_before_planning():
    arm, camera, _, _ = _inputs()
    detect = Mock(side_effect=NoHandleProposal("empty"))
    planner = Mock()
    scene_model = Mock(side_effect=Refused("bad scene fit"))
    with pytest.raises(NoHandleProposal, match="bad scene fit"):
        _relook(detect)(arm, planner, camera, scene_model)
    planner.motion.assert_not_called()

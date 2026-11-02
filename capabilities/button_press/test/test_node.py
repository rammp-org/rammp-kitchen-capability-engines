"""The engine: the action's params are validated, the mock moves nothing, and a failed run is an
EngineError named after its step. Needs ROS 2 and the RAMMP interfaces (make test-arm's image)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
node_mod = pytest.importorskip("button_press.node")
import rclpy  # noqa: E402
from capability_engine import EngineError, StreamClosed  # noqa: E402

from button_press import Abort, MotionFailed  # noqa: E402


@pytest.fixture
def engine(request):
    mock = getattr(request, "param", False)
    rclpy.init(args=["--ros-args", "-p", "use_orchestrator:=false", "-p", f"mock:={str(mock).lower()}"])
    cap = node_mod.Capability()
    yield cap
    cap.destroy_node()
    rclpy.shutdown()


class Ctx:
    def __init__(self):
        self.progress_calls = []

    def checkpoint(self):
        pass

    def progress(self, phase, pct, message="", pausable=True):
        self.progress_calls.append((phase, pct, pausable))


@pytest.mark.parametrize(
    "params",
    [{"target": "popcorn"}, {"presses": -1}, {"presses": 11}, {"presses": True}, {"presses": 1.5}, {"no_return": 1}],
)
def test_bad_params_are_refused_before_anything_runs(engine, params):
    with pytest.raises(EngineError) as e:
        engine.execute("press_button", params, Ctx())
    assert e.value.error_code == "bad_params"


@pytest.mark.parametrize("engine", [True], indirect=True)
def test_the_mock_is_a_dry_run_that_detects_and_moves_nothing(engine):
    ctx = Ctx()
    assert engine.execute("press_button", {"target": "timer_clock"}, ctx) == {
        "dry_run": True,
        "pressed": 0,
        "returned": False,
        "moved": False,
        "log_file": "",
    }  # the real result's keys


def test_the_arm_does_not_move_unless_allowed(engine):
    assert engine._args({}).execute is False  # allow_execute defaults to false


@pytest.mark.parametrize(
    "step,error,recoverable",
    [
        ("redetect", Abort("dome detector: nothing in 7 frames"), True),
        ("redetect", MotionFailed("back out to re-detect: trajectory ended HALTED -- HOLDING HERE"), False),
        ("redetect", StreamClosed("the driver closed the stream"), False),
        # at the panel: the fingertip may be at the button, and a retry would press it again
        ("approach", Abort("to pre-press: pre-check failed at point 3/12"), False),
        ("press", Abort("press 2 (contact): no touch within 30 mm -- session closed, HOLDING HERE"), False),
        ("return", Abort("back out from the panel: pre-check failed at point 1/15"), False),
    ],
)
def test_a_failed_run_is_named_after_its_step_and_only_retried_away_from_the_panel(
    engine, monkeypatch, step, error, recoverable
):
    closed = []

    class FakeRun:
        def __init__(self, args, eng, kin, checkpoint, step):
            self.step, self.log_path = step, ""
            self.arm = SimpleNamespace(sent_goals=0)

        def close(self):
            closed.append(True)

    def press_button(run, args, pub):
        run.step("preflight", 0.0)
        run.step(step, 0.4)
        raise error

    monkeypatch.setattr(node_mod, "Run", FakeRun)
    monkeypatch.setattr(node_mod, "press_button", press_button)
    with pytest.raises(EngineError) as e:
        engine.execute("press_button", {}, Ctx())
    assert (e.value.error_code, e.value.recoverable) == (f"{step}_failed", recoverable)
    assert str(error) in str(e.value)
    assert closed == [True]  # the scene camera and the run log are closed however it ends


def test_the_steps_at_the_panel_are_not_pausable(engine, monkeypatch):
    ctx = Ctx()

    class FakeRun:
        def __init__(self, args, eng, kin, checkpoint, step):
            self.step, self.log_path = step, ""
            self.arm = SimpleNamespace(sent_goals=3)

        def close(self):
            pass

    def press_button(run, args, pub):
        for name, frac, pausable in (("plan", 0.5, True), ("approach", 0.6, False), ("press", 0.7, False)):
            run.step(name, frac, pausable=pausable)
        return {"dry_run": False, "pressed": 1, "returned": True}

    monkeypatch.setattr(node_mod, "Run", FakeRun)
    monkeypatch.setattr(node_mod, "press_button", press_button)
    out = engine.execute("press_button", {}, ctx)
    assert out["moved"] is True and out["pressed"] == 1
    assert ("approach", 0.6, False) in ctx.progress_calls and ("press", 0.7, False) in ctx.progress_calls
    assert ctx.progress_calls[-1][:2] == ("done", 1.0)


def test_the_run_settings_cannot_change_while_the_node_runs(engine):
    """They are read at startup: an accepted `ros2 param set allow_execute false` would not stop the arm."""
    from rclpy.parameter import Parameter

    for name, value in (("allow_execute", True), ("mount_rpy_deg", [0.0, 0.0, 0.0]), ("far", "wrist")):
        (res,) = engine.set_parameters([Parameter(name, value=value)])
        assert not res.successful and "restart" in res.reason

"""No press plans from where the chair stands (reach_press.no_way_in): with the engine's guidance it is not
the run's end. Run 2 of 2026-11-17 (flatbot 16:43:13) ended there, "arm: where the look left it"; the owner:
"it should NOT just fail EVER". The reach-out's own harness (test_reach_press) flies the run."""

from types import SimpleNamespace

import pytest
from test_reach_press import _Chain, _depth_press_flow, _flow, _refusals, _way_down

from box_opening.tasks import reach_press

pytestmark = pytest.mark.usefixtures("home_20261021")


def _guided(monkeypatch):
    """The run as the engine starts it (press_demo --guide): whatever args the harness builds, guided."""
    real = reach_press._reach_press

    def guided(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
        args.guide = True
        return real(node, ctx, cfg, runner, watcher, args, scene_fix, locator)

    monkeypatch.setattr(reach_press, "_reach_press", guided)


def test_with_the_engines_guidance_a_box_no_press_plans_for_is_no_stop(ctx, monkeypatch, tmp_path, capsys):
    """Before anything moves toward the box: the line says NO WAY IN and why at every tilt, the row says the
    run is guided, nothing moved, and the answer is NO_WAY_IN (press_demo then has the chair moved and looks
    again). No STOP line: the engine reads a run's last STOP as how it ended."""
    _guided(monkeypatch)
    chain = _Chain(refuse=[lambda xyz, quat: True])  # no straight line anywhere: no tilt slides in
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, chain=chain)
    assert code == reach_press.NO_WAY_IN == 7 and rec.events == [] and ctx.client.executed == []
    (note,) = _refusals(tmp_path)
    assert note["guided"] is True
    assert note["why"].startswith("no reach-out press plans for the scene camera's box (30 deg: the slide in (")
    out = capsys.readouterr().out
    assert "[press_demo] NO WAY IN: %s; nothing moved" % note["why"] in out and "STOP" not in out
    assert ctx.no_way_in == note["why"]  # the reason press_demo hands the engine with its request


def test_a_run_by_hand_stops_there_as_it_always_has(ctx, monkeypatch, tmp_path, capsys):
    chain = _Chain(refuse=[lambda xyz, quat: True])
    args = SimpleNamespace(execute=True, press_only=True, guide=False)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, chain=chain, run_args=args)
    assert code == 1 and rec.events == []
    (note,) = _refusals(tmp_path)
    assert "guided" not in note
    out = capsys.readouterr().out
    assert "[press_demo] STOP: %s; nothing moved" % note["why"] in out and "NO WAY IN" not in out


def test_guided_the_aimed_button_no_press_reaches_sends_the_arm_home_and_is_no_stop_either(
    ctx, monkeypatch, tmp_path, capsys
):
    """After the look and the aim: the arm goes home from the look pose, as it always has there, and only
    then is the chair to move (the arm at rest whenever the chair may be moving)."""
    _guided(monkeypatch)
    asked = []
    scene_only = lambda tilt, tip: len(asked) == 1  # noqa: E731 - plan_reach's, for the scene camera's button
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose, not the slide
    code, rec, _c, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, scene_only), chain=chain)
    assert code == reach_press.NO_WAY_IN and got == {}
    assert rec.events == ["reach:look", "aim:reach-aim", "home"]
    out = capsys.readouterr().out
    assert "NO WAY IN: no press reaches the aimed button (as straight lines: " in out
    assert "— nothing was pressed — returning home" in out and "STOP" not in out
    (note,) = _refusals(tmp_path)
    assert note["guided"] is True and ctx.no_way_in == note["why"]
    import json

    rows = [json.loads(line) for f in tmp_path.glob("run-*.jsonl") for line in open(f)]
    legs = [r["leg"] for r in rows if r["leg"] in ("reach_refused", "home")]
    assert legs == ["reach_refused", "home"]  # said before the home: the engine reads that home as going to rest


def test_guided_an_arm_that_does_not_get_home_from_the_look_holds_and_that_is_the_stop(
    ctx, monkeypatch, tmp_path, capsys
):
    """The chair is never told to move with the arm standing out at the look pose: a home that does not get
    there ends the run, the arm holding, as a run's own stop there does (the engine: arm_holds)."""
    _guided(monkeypatch)
    monkeypatch.setattr(reach_press, "_go_home", lambda ctx, cfg, runner, why, **k: False)
    asked = []
    scene_only = lambda tilt, tip: len(asked) == 1  # noqa: E731
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])
    code, _rec, _c, _h, _got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, scene_only), chain=chain)
    assert code == 1
    out = capsys.readouterr().out
    assert "STOP: no press reaches the aimed button" in out and "the arm holds" in out

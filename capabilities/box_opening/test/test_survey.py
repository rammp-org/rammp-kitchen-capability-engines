"""The wrist survey (tasks/survey): where the scene camera finds no container, one or two poses for the wrist camera
over the arm's side, chosen from the arm's kinematics and the chair model, the finder on their frames, a look again
where it asks, and the run's end where nothing (or two things) is found."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeClient, flatbot_urdf

from box_opening.constants import HOME, PLANNER_LIMIT_MARGIN_RAD
from box_opening.models.container import ContainerModel
from box_opening.perception import wrist_finder as wf
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import mat_to_quat_xyzw
from box_opening.tasks import press_demo
from box_opening.tasks import survey as sv

CONTAINER = "capabilities/box_opening/config/containers/ankou_pink.yaml"
DOWN = mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))  # the hand straight down
# flatbot's robot file's arm spheres (rammp-deployments dev 63e7563 robot.yaml; the base and the shoulder are exempt),
# the gripper's own few: the body model adds the hand's (config/robot_body.yaml)
ARM = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_half_arm_1_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.062}
      - {center: [0.0, -0.06, 0.0], radius: 0.062}
      - {center: [0.0, -0.12, 0.0], radius: 0.062}
      - {center: [0.0, -0.17, 0.0], radius: 0.06}
      arm_half_arm_2_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.058}
      - {center: [0.0, 0.0, -0.07], radius: 0.056}
      - {center: [0.0, 0.0, -0.15], radius: 0.056}
      - {center: [0.0, 0.0, -0.21], radius: 0.056}
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      - {center: [0.0, -0.06, 0.0], radius: 0.058}
      - {center: [0.0, -0.12, 0.0], radius: 0.058}
      - {center: [0.0, -0.17, 0.0], radius: 0.055}
      arm_spherical_wrist_1_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      - {center: [0.0, 0.0, -0.085], radius: 0.06}
      arm_spherical_wrist_2_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.055}
      - {center: [0.0, -0.085, 0.0], radius: 0.055}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      - {center: [0.0, -0.05, -0.045], radius: 0.045}
      - {center: [0.045, 0.0, -0.05], radius: 0.036}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.04}
      - {center: [0.0, 0.0, 0.078], radius: 0.035}
"""
# flatbot's chair and user as its generated layers have them (unpadded, arm_mount), the boxes near the arm
SEAT = [
    ("armrest_arm_0", (-0.1735, -0.013, 0.3555), (0.339, 0.102, 0.051)),
    ("armrest_arm_1", (0.016, -0.013, 0.3675), (0.040, 0.102, 0.027)),
    ("backrest", (-0.4435, -0.3505, 0.284), (0.423, 0.483, 0.380)),
    ("base", (-0.075, -0.3635, -0.0315), (0.542, 0.457, 0.253)),
]
PERSON = [
    ("knees", (0.1505, -0.3505, 0.197), (0.765, 0.483, 0.080)),
    ("lap", (0.100, -0.3505, 0.2775), (0.664, 0.483, 0.081)),
    ("legs", (0.2015, -0.3505, -0.182), (0.867, 0.483, 0.678)),
    ("torso", (-0.116, -0.3505, 0.7045), (0.232, 0.483, 0.775)),
]


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _layer(tmp, name, boxes):
    lines = ["layer: %s" % name, "meta: {frame: arm_mount}", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, list(c), list(d)) for n, c, d in boxes]
    (tmp / ("layer_%s.yaml" % name)).write_text("\n".join(lines) + "\n")


def flatbot_chair(tmp):
    """flatbot's chain with its chair and user attached as the body model (the arm's spheres above, the hand's
    from config/robot_body.yaml): (chain, Body)."""
    _layer(tmp, "seat", SEAT)
    _layer(tmp, "person", PERSON)
    (tmp / "robot.yaml").write_text(ARM)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_seat.yaml, layer_person.yaml]\n")
    approach.set_body(body.load(str(tmp / "planner.yaml")))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    chain = approach._chain()
    return chain, chain.body


def _grid(lids):
    lids = np.asarray(lids, float)
    return sv.Grid(lids=lids, counter_z=lids[:, 2] - 0.121, weight=np.ones(len(lids)), lid_r=0.0525)


def test_the_lens_stands_where_asked_looking_where_asked_its_long_side_across_the_view():
    cam = sv.camera()
    d = np.array([0.0, 0.5, -1.0]) / np.linalg.norm([0.0, 0.5, -1.0])
    R_tool, t_tool = cam.tool_for((0.3, 0.2, 0.7), d, (1.0, 0.0, 0.0))
    R_opt, lens = cam.view(R_tool, t_tool)
    assert np.allclose(lens, (0.3, 0.2, 0.7)) and np.allclose(R_opt[:, 2], d)
    assert R_opt[:, 0] @ [1.0, 0.0, 0.0] > 0.999  # the image's long side along x, across the view
    assert np.allclose(R_opt.T @ R_opt, np.eye(3)) and np.linalg.det(R_opt) > 0.999
    # the lens 65 mm off the tool's axis toward its +y, 105 mm back from tool_frame: the camera config's
    assert np.allclose(R_tool.T @ (lens - t_tool), sv.LENS_IN_TOOL_M)
    assert cam.tool_for((0.3, 0.2, 0.7), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)) is None  # no long side across that


def test_a_lid_counts_seen_whole_in_view_with_depth_and_checked_within_the_finders_bounds():
    cam = sv.camera()
    R_opt = np.diag([1.0, -1.0, -1.0])  # straight down from [0.4, 0.4, 0.8]
    lids = [
        (0.40, 0.40, 0.40),  # 0.40 m straight below: whole, its button checkable
        (0.40, 0.40, 0.05),  # 0.75 m below: whole, past the check's range
        (1.40, 0.40, 0.40),  # out of the view
        (0.40, 0.40, 0.70),  # 0.10 m: no stereo depth that near
        (0.16, 0.40, 0.40),  # in the colour image's left columns, where the OAK gives no depth
    ]
    checks, sees = sv.cover(cam, R_opt, (0.4, 0.4, 0.8), _grid(lids))
    assert sees.tolist() == [True, True, False, False, False]
    assert checks.tolist() == [True, False, False, False, False]
    # seen 50 deg off straight down: whole, but past the finder's incidence for the button
    R_tool, t_tool = cam.tool_for((0.4, 0.0, 0.8), (0.0, np.sin(np.radians(50)), -np.cos(np.radians(50))), (1, 0, 0))
    R_opt, lens = cam.view(R_tool, t_tool)
    lid = lens + 0.45 * R_opt[:, 2]
    checks, sees = sv.cover(cam, R_opt, lens, _grid([lid]))
    assert sees.tolist() == [True] and checks.tolist() == [False]


@pytest.mark.usefixtures("home_20261021")  # the side counter's band was measured against that rest pose
def test_the_band_keeps_the_arm_off_a_counter_beside_the_chair_no_nearer_than_the_rest_pose_has_it(tmp_path):
    chain, check = flatbot_chair(tmp_path)
    band = sv.Band(check, HOME)
    assert band.y_edge == pytest.approx(0.038 + sv.EDGE_GAP_M)  # the armrest's edge on the arm's side
    # the rest pose is no survey pose: its forearm stands inside the band (the upper arm alone keeps no nearer)
    assert band.refusal(HOME).startswith("arm_forearm_link 28 mm inside")
    low = chain.nearest_solution(list(HOME), [0.45, 0.30, 0.30], DOWN)  # the hand down over the counter
    assert low is not None and "where a counter up to 0.45 m may stand beside the chair" in band.refusal(low)
    assert "of the way" in band.path_refusal([HOME, low])
    # a look again's way there never rises above the survey's own poses (an upper cabinet may hang there)
    band.ceiling_z = sv.sphere_top_z(check, HOME) - 0.01
    assert "higher than the survey's own poses reached, 0% of the way" in band.path_refusal([HOME, HOME])
    band.ceiling_z = None
    # a band built for lower counters only: the same hand stands inside it too, and one over it stands clear
    assert (
        "inside" in sv.Band(check, HOME, top_z=0.30).refusal(low)
        and sv.Band(check, HOME, top_z=0.10).refusal(low) is None
    )


# the first survey view chosen while only the wrist and the hand were judged on the way (fix/survey's first sim,
# 11-06): its straight way out of HOME took the forearm 10 mm deeper into the band, and 10 mm into the 0.91 m
# counter 20 mm from the chair
FOREARM_DEEPER = [-0.0765, -0.6593, -4.5669, -0.7185, 0.7008, -1.9475, 4.0164]
# ... and the first chosen once the band held every sphere, the user not yet: its straight way out of HOME took the
# upper arm inside the planner's padding of the user and the forearm 20 mm into the user's torso box
TORSO_CROSSING = [-0.2314, -0.6068, -4.6356, -0.9731, -2.3351, 1.8956, 0.6719]


@pytest.mark.usefixtures("home_20261021")
def test_only_the_upper_arm_keeps_no_nearer_than_the_rest_pose_and_every_link_is_judged_on_the_way(tmp_path):
    chain, check = flatbot_chair(tmp_path)
    band = sv.Band(check, HOME)
    upper = np.array(["half_arm" in lk for lk in band.links])
    assert upper.any() and (~upper).any()
    # HOME holds the hand 5-50 mm over the band and the forearm inside it: those keep KEEP_ABOVE_M all the same
    assert (band.rest_m[~upper] < sv.KEEP_ABOVE_M).any() and np.all(band.required[~upper] == sv.KEEP_ABOVE_M)
    assert np.all(band.required[upper] <= sv.KEEP_ABOVE_M) and (band.required[upper] < 0.0).any()
    # a way out of HOME that sweeps the forearm deeper into where a counter may stand is refused, naming it
    why = band.path_refusal([list(HOME), FOREARM_DEEPER])
    assert why is not None and why.startswith("arm_forearm_link") and "where it started" in why, why
    # ... and one that takes it through the user, padded as the planner pads them (the chair's own boxes are not
    # judged on the way: HOME stands inside the armrest's)
    assert check.pose_refusal(TORSO_CROSSING) is None and band.refusal(TORSO_CROSSING) is None
    why = band.path_refusal([list(HOME), TORSO_CROSSING])
    assert why is not None and "mm from the user (the planner keeps 20 mm)" in why, why
    d, _links = band.body.sphere_distances(np.array([list(HOME), TORSO_CROSSING]), layers=("person",))
    assert (d.min(axis=1) > band.user_pad_m).all()  # both ends clear of the user: the way between is not


def test_a_sphere_outside_the_band_where_a_way_starts_never_enters_it_one_inside_never_goes_deeper():
    """body.first_hit's rule: outside, not a millimetre in, however near the start; inside, DEEPER_M deeper at most."""
    band = object.__new__(sv.Band)
    band.ceiling_z, band.top_z = None, 0.45
    # the stub's one sphere stands as far from the band as the first joint says (m, < 0 inside)
    band.body = SimpleNamespace(
        sphere_distances=lambda Q, layers=None: (np.asarray(Q, float)[:, :1], ["arm_bracelet_link"])
    )

    def way(d0, d1):
        return band.path_refusal([[d0] + [0.0] * 6, [d1] + [0.0] * 6])

    assert "arm_bracelet_link 1 mm into" in way(0.003, -0.001)  # 3 mm out at the start: never in
    assert way(0.003, 0.0) is None and way(0.08, 0.001) is None
    assert way(-0.010, -0.014) is None  # 10 mm in at the start: may move in it, 4 mm deeper
    assert "(10 mm where it started)" in way(-0.010, -0.016)  # ... never DEEPER_M (5 mm) deeper


def test_the_survey_views_hold_the_arm_clear_and_see_the_side_counter_at_every_height(tmp_path):
    chain, check = flatbot_chair(tmp_path)
    model = ContainerModel.load(CONTAINER)
    cam = sv.camera()
    views, why, stats = sv.survey_views(chain, check, list(HOME), cam, model)
    assert why is None and 1 <= len(views) <= sv.MAX_VIEWS, why
    band = sv.Band(check, HOME)
    upper = np.array(["half_arm" in lk for lk in band.links])
    prev = list(HOME)
    for v in views:
        assert check.pose_refusal(v.joints) is None and band.refusal(v.joints) is None
        assert chain.within_limits(v.joints, margin=PLANNER_LIMIT_MARGIN_RAD)
        R, t = chain.fk(v.joints)
        R_opt, lens = cam.view(R, t)
        assert np.allclose(lens, v.cam_xyz, atol=1e-4) and np.allclose(R_opt, v.rot_cam, atol=1e-4)
        assert lens[2] > sv.COUNTER_Z_M[1] + sv.KEEP_ABOVE_M  # high over the highest counter of the band
        # the forearm, the wrist and the hand KEEP_ABOVE_M off the band, whatever HOME holds them at; the upper
        # arm, mounted 58 mm from the band's face, nearer
        d, _links = band.body.sphere_distances(np.atleast_2d(v.joints), layers=(sv.BAND_LAYER,))
        assert d[0][~upper].min() >= sv.KEEP_ABOVE_M - 1e-9 and d[0][upper].min() < sv.KEEP_ABOVE_M
        assert band.path_refusal([prev, v.joints]) is None  # its way there, every link, as straight joint lines
        prev = v.joints
    # the views see most of the cells the arm presses, over every counter height of the band
    assert stats["sees"] >= 0.5 and stats["checks"] > 0.0 and stats["scored_cells"] > 0
    assert v.name == "survey:look-%d" % len(views)


def test_a_look_again_shows_the_place_whole_and_its_button_checkable_or_says_why_not(tmp_path):
    chain, check = flatbot_chair(tmp_path)
    model = ContainerModel.load(CONTAINER)
    cam = sv.camera()
    view, why = sv.look_again_view(chain, check, list(HOME), cam, model, (0.45, 0.40), 0.239 + 0.10)
    assert why is None and view is not None and view.checks.tolist() == [True]
    assert view.tilt_deg <= max(sv.LOOK_AGAIN_TILT_DEG) + 0.5 and view.name == "survey:look-again-1"
    assert check.pose_refusal(view.joints) is None and sv.Band(check, HOME).refusal(view.joints) is None
    lens = np.asarray(view.cam_xyz)
    assert (
        min(sv.LOOK_AGAIN_RANGE_M) - 1e-6
        <= np.linalg.norm(lens - [0.45, 0.40, 0.339])
        <= max(sv.LOOK_AGAIN_RANGE_M) + 1e-6
    )
    view, why = sv.look_again_view(chain, check, list(HOME), cam, model, (1.20, 0.60), 0.36)  # past the arm's reach
    assert view is None and why.startswith("no look again over [1.200, 0.600] the arm holds clear")
    # never higher than the survey's own poses reached: over the box an upper cabinet may hang
    views, _why, _stats = sv.survey_views(chain, check, list(HOME), cam, model)
    ceiling = max(sv.sphere_top_z(check, v.joints) for v in views)
    view, why = sv.look_again_view(chain, check, list(HOME), cam, model, (0.45, 0.40), 0.339, ceiling_z=ceiling)
    assert view is not None and sv.sphere_top_z(check, view.joints) <= ceiling + 1e-3
    view, why = sv.look_again_view(chain, check, list(HOME), cam, model, (0.45, 0.40), 0.339, ceiling_z=0.60)
    assert view is None and why.startswith("no look again over [0.450, 0.400] the arm holds clear")


def test_the_places_looked_at_again_are_the_unchecked_and_the_cut_strongest_first():
    P = SimpleNamespace
    find = P(
        places=(
            P(xy=(0.40, 0.40), top_z=0.36, button="unchecked", patches=(P(n_slab=300),)),
            P(xy=(0.60, 0.30), top_z=0.36, button="none", patches=(P(n_slab=500),)),  # checked: no button
            P(xy=(0.70, 0.50), top_z=0.36, button="seen", patches=(P(n_slab=400),)),
        ),
        patches=(
            P(xy=(0.41, 0.41), top_z=0.36, why="its top 60x105 mm, cut off by the view", n_slab=100),  # that place
            P(xy=(0.20, 0.50), top_z=0.35, why="its top 60x105 mm, cut off by the view", n_slab=200),
            P(xy=(0.30, 0.60), top_z=0.35, why="its top a ring, 40% filled — an open cup's rim?", n_slab=900),
        ),
    )
    assert sv.look_again_places(find, 0.105) == [((0.40, 0.40), 0.36), ((0.20, 0.50), 0.35)]


# ----------------------------------------------------------------------------------------- the flow


def _frame(tag="survey"):
    return wf.WristFrame(
        color=np.zeros((4, 4, 3), np.uint8),
        depth=np.zeros((4, 4), np.float32),
        k=np.eye(3),
        rot_cam=np.eye(3),
        trans_cam=np.zeros(3),
        tag=tag,
    )


def _view(name):
    return sv.SurveyView(
        name, [0.0] * 7, (0.2, 0.2, 0.7), np.eye(3), np.ones(1, bool), np.ones(1, bool), 35.0, what=name
    )


class _Runner:
    def __init__(self):
        self.notes = []

    def note(self, kind, **fields):
        self.notes.append((kind, fields))

    def wait_still(self, before):
        return 0.0, False


FIX = SimpleNamespace(top_xyz=(0.45, 0.40, 0.34), pose=None, table_z_scene=0.24)


def _found(fix=FIX):
    return wf.WristFind(fix, "the container at [0.450, 0.400]", 1.0, places=())


def _unchecked(xy=(0.45, 0.40)):
    q = wf.Place(xy=xy, top_z=0.34, table_z=0.24, patches=(SimpleNamespace(n_slab=300),), button="unchecked")
    return wf.WristFind(
        None, "wrist camera: a lid-sized top at [0.450, 0.400], but no view could check its button", places=(q,)
    )


def _flow(monkeypatch, finds, views=2, states=None, again=True):
    """run_survey with its motions, frames and finder scripted: the views flown, in order."""
    flown = []
    monkeypatch.setattr(approach, "body_check", lambda: SimpleNamespace(layers=["person"]))
    monkeypatch.setattr(approach, "_chain", lambda: None)
    monkeypatch.setattr(sv, "Band", lambda check, rest: SimpleNamespace(ceiling_z=None))
    monkeypatch.setattr(
        sv,
        "survey_views",
        lambda *a, **k: (
            [_view("survey:look-%d" % (i + 1)) for i in range(views)],
            None,
            {"checks": 0.3, "sees": 0.7, "choose_s": 0.1},
        ),
    )
    monkeypatch.setattr(
        sv,
        "look_again_view",
        lambda chain, check, q, cam, model, xy, z, rest=None, n=1, ceiling_z=None: (
            (_view("survey:look-again-%d" % n), None) if again else (None, "no look again over there")
        ),
    )
    monkeypatch.setattr(sv, "sphere_top_z", lambda check, q: 0.9)
    states = list(states or [])

    def fly(ctx, runner, view, band, execute):
        flown.append(view.name)
        return states.pop(0) if states else ("flown", None)

    monkeypatch.setattr(sv, "fly_view", fly)
    monkeypatch.setattr(sv, "grab_frame", lambda node, watcher, runner, tag, **k: (_frame(tag), None, None, None))
    monkeypatch.setattr(sv, "save_frame", lambda *a, **k: None)
    finds = list(finds)
    monkeypatch.setattr(wf, "find_container", lambda frames, model, **k: finds.pop(0))
    ctx = SimpleNamespace(client=FakeClient(), model=ContainerModel.load(CONTAINER))
    runner = _Runner()
    s = sv.run_survey(None, ctx, runner, SimpleNamespace(grab=None), True, rest=list(HOME))
    return s, flown, runner


def test_the_survey_stops_at_the_first_view_that_finds_the_box(monkeypatch):
    s, flown, runner = _flow(monkeypatch, [_found()])
    assert flown == ["survey:look-1"] and s.fix is FIX and not s.stopped and len(s.clouds) == 1
    assert [f["stage"] for k, f in runner.notes if k == "wrist_survey"] == ["views", "survey:look-1"]


def test_a_place_no_view_could_check_is_looked_at_again_before_it_is_a_fix(monkeypatch):
    s, flown, _r = _flow(monkeypatch, [_unchecked(), _unchecked(), _found()])
    assert flown == ["survey:look-1", "survey:look-2", "survey:look-again-1"] and s.fix is FIX
    assert len(s.clouds) == 3  # every frame's points kept, for what else needs the counter


def test_nothing_found_is_the_finders_why_and_a_refused_view_is_passed_over(monkeypatch):
    nothing = wf.WristFind(None, "wrist camera: nothing on the counter is the container (2 things at lid height)")
    s, flown, _r = _flow(monkeypatch, [nothing], states=[("refused", "planning failed for survey:look-1: IK_FAIL")])
    assert flown == ["survey:look-1", "survey:look-2"] and s.fix is None and not s.stopped
    assert s.why == nothing.why
    s, flown, _r = _flow(
        monkeypatch, [], views=1, states=[("refused", "its way there: arm_bracelet_link 12 mm into ...")]
    )
    assert s.fix is None and "survey:look-1" in s.why and "not flown: its way there" in s.why


def test_a_place_the_arm_cannot_look_again_at_is_refused_with_the_finders_why(monkeypatch):
    s, flown, runner = _flow(monkeypatch, [_unchecked(), _unchecked()], again=False)
    assert flown == ["survey:look-1", "survey:look-2"] and s.fix is None and "no view could check its button" in s.why
    assert (
        "wrist_survey",
        {"stage": "look-again", "xy": [0.45, 0.4], "flown": False, "why": "no look again over there"},
    ) in runner.notes


def test_a_survey_motion_stopped_part_way_holds_and_says_so(monkeypatch):
    s, flown, _r = _flow(monkeypatch, [], states=[("stopped", "failed (aborted)")])
    assert flown == ["survey:look-1"] and s.stopped and s.fix is None and "stopped part way" in s.why


def test_a_view_the_runner_refuses_before_it_moves_is_not_flown_one_that_fails_in_flight_stopped_part_way(
    monkeypatch,
):
    """fly_view: the runner's own gates (the sanity gate, an unarmed client) refuse a leg before anything moves: the
    arm stands where it was, the next view is tried; a leg that failed in flight left the arm part way."""
    from box_opening.primitives import core
    from box_opening.runtime.runner import LegResult

    leg = SimpleNamespace(traj=SimpleNamespace(points=[SimpleNamespace(positions=list(HOME))] * 2))
    monkeypatch.setattr(core, "_plan_motion", lambda *a, **k: (leg, None))
    monkeypatch.setattr(core, "_full_world", lambda ctx: ("full", "world_empty.yaml"))
    ctx = SimpleNamespace(client=FakeClient())
    band = SimpleNamespace(path_refusal=lambda traj: None)
    for outcome, state in (("refused", "refused"), ("failed", "stopped")):
        runner = SimpleNamespace(run=lambda legs, execute, o=outcome: [LegResult("survey:look-1", o, False, "why")])
        assert sv.fly_view(ctx, runner, _view("survey:look-1"), band, True) == (state, "%s (why)" % outcome)
    runner = SimpleNamespace(run=lambda legs, execute: [LegResult("survey:look-1", "arrived", True)])
    assert sv.fly_view(ctx, runner, _view("survey:look-1"), band, True) == ("flown", None)
    band = SimpleNamespace(path_refusal=lambda traj: "arm_forearm_link 3 mm into ...")
    assert sv.fly_view(ctx, runner, _view("survey:look-1"), band, True) == (
        "refused",
        "its way there: arm_forearm_link 3 mm into ...",
    )


def test_a_view_whose_plan_the_client_refuses_is_asked_for_again(monkeypatch):
    """The planner's joint plan at times ends on another turn of the wrist and the client refuses it (private rig,
    2026-11-12: 2 of 31 wrist-turn plans); asked again it plans. A planner that cannot plan it is not asked twice."""
    from box_opening.primitives import core
    from box_opening.runtime.runner import LegResult

    leg = SimpleNamespace(traj=SimpleNamespace(points=[SimpleNamespace(positions=list(HOME))] * 2))
    monkeypatch.setattr(core, "_full_world", lambda ctx: ("full", "world_empty.yaml"))
    ctx = SimpleNamespace(client=FakeClient())
    band = SimpleNamespace(path_refusal=lambda traj: None)
    runner = SimpleNamespace(run=lambda legs, execute: [LegResult("survey:angle-1", "arrived", True)])
    asked = []

    def plan(fails, error):
        def _plan(*a, **k):
            asked.append(a[2])
            if len(asked) <= fails:
                raise error("planning failed for survey:angle-1: the plan ends 0.1376 rad from its joint goal")
            return leg, None

        return _plan

    monkeypatch.setattr(core, "_plan_motion", plan(2, core.PlanRefused))
    assert sv.fly_view(ctx, runner, _view("survey:angle-1"), band, True) == ("flown", None) and len(asked) == 3
    asked.clear()
    monkeypatch.setattr(core, "_plan_motion", plan(9, core.PlanRefused))
    state, why = sv.fly_view(ctx, runner, _view("survey:angle-1"), band, True)
    assert state == "refused" and "ends 0.1376 rad" in why and len(asked) == sv.SURVEY_PLAN_TRIES
    asked.clear()
    monkeypatch.setattr(core, "_plan_motion", plan(9, RuntimeError))
    assert sv.fly_view(ctx, runner, _view("survey:angle-1"), band, True)[0] == "refused" and len(asked) == 1


def test_without_the_body_model_there_is_no_survey(monkeypatch):
    monkeypatch.setattr(approach, "body_check", lambda: None)
    s = sv.run_survey(None, SimpleNamespace(client=FakeClient(), model=None), _Runner(), None, True)
    assert s.fix is None and "body model" in s.why and not s.views


# ----------------------------------------------------------------------------------------- press_demo


def _no_box(monkeypatch, survey=None, wrist_survey=True, instance_id=None, execute=True):
    """press_demo.wrist_survey_or_exit after a scene look that found nothing: (what it returned or its exit
    code, the homes it asked for, the run's notes)."""
    homes, runner = [], _Runner()
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why) or True)
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: survey)
    locator = SimpleNamespace(
        last_why="the segmenter boxed nothing of its classes in 70 frames", look_record=lambda: {"timing": {}}
    )
    ctx = SimpleNamespace(worlds=SimpleNamespace(wrist_camera_config="camera_oak_wrist.yaml"), survey_clouds=None)
    args = Namespace(wrist_survey=wrist_survey, instance_id=instance_id, execute=execute)
    cfg = SimpleNamespace(park_tool_down=False)
    try:
        got = press_demo.wrist_survey_or_exit(None, ctx, cfg, runner, None, args, locator)
    except SystemExit as e:
        got = e.code
    return got, homes, runner.notes, ctx


def test_without_the_survey_no_box_is_the_runs_end_as_before(monkeypatch):
    got, homes, notes, _ctx = _no_box(monkeypatch, wrist_survey=False)
    assert got == 2 and homes == ["NO BOX — scene camera: the segmenter boxed nothing of its classes in 70 frames"]
    assert notes[0][0] == "reach_refused" and notes[0][1]["why"].startswith("NO BOX — scene camera:")


def test_the_orchestrators_selection_is_never_surveyed_for(monkeypatch):
    got, homes, notes, _ctx = _no_box(monkeypatch, survey=SimpleNamespace(fix=FIX), instance_id="food_container_2")
    assert got == 2 and "no wrist survey for the selected container food_container_2" in homes[0]


def test_the_surveys_fix_goes_on_to_the_reach_out_and_its_clouds_on_the_ctx(monkeypatch):
    survey = sv.Survey(
        fix=FIX, clouds=[{"points": np.zeros((3, 3), np.float32), "cam_xyz": (0, 0, 0), "tag": "survey:look-1"}]
    )
    got, homes, notes, ctx = _no_box(monkeypatch, survey=survey)
    assert got is FIX and not homes and not notes and ctx.survey_clouds is survey.clouds


def test_the_survey_finding_nothing_is_no_box_home_and_exit_2(monkeypatch):
    survey = sv.Survey(
        why="wrist camera: two places pass as this container, [0.400, 0.400] and [0.600, 0.300]: refusing to guess"
    )
    got, homes, notes, _ctx = _no_box(monkeypatch, survey=survey)
    assert got == 2 and homes[0].endswith("; wrist survey: " + survey.why)
    assert notes[0][1]["why"].startswith("NO BOX — scene camera: ") and "refusing to guess" in notes[0][1]["why"]


def test_a_survey_motion_stopped_part_way_ends_the_run_with_the_arm_holding(monkeypatch, capsys):
    got, homes, _n, _ctx = _no_box(
        monkeypatch, survey=sv.Survey(stopped=True, why="the move to survey:look-1 stopped part way")
    )
    assert got == 1 and not homes


def test_a_preview_ends_at_the_survey_pose(monkeypatch):
    got, homes, _n, _ctx = _no_box(
        monkeypatch, survey=sv.Survey(preview=True, why="a preview: no frames"), execute=False
    )
    assert got == 0 and not homes


# ------------------------------------------------------------------------- the front: the wrist turned

# flatbot's user with the arm 8 in forward on the chair (flatbot_fwd8's layers, the boxes near the arm)
PERSON_FWD8 = [
    ("knees", (-0.0527, -0.3505, 0.197), (0.765, 0.483, 0.080)),
    ("lap", (-0.1032, -0.3505, 0.2775), (0.664, 0.483, 0.081)),
    ("legs", (-0.0017, -0.3505, -0.182), (0.867, 0.483, 0.678)),
    ("torso", (-0.3192, -0.3505, 0.7045), (0.232, 0.483, 0.775)),
]


def _front_chair(tmp):
    _layer(tmp, "person", PERSON_FWD8)
    (tmp / "robot.yaml").write_text(ARM)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_person.yaml]\n")
    approach.set_body(body.load(str(tmp / "planner.yaml")))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    chain = approach._chain()
    return chain, chain.body


def test_the_wrist_turns_where_the_arm_rests_until_its_camera_sees_the_table_in_front(tmp_path):
    """Owner 2026-11-11: "don't do a 'look down' pose. the arm would just need to angle itself so it can see the
    scene". At HOME the wrist camera looks forward 8 deg under level and the box at [0.79, -0.24] lies just under its
    view; the three wrist joints alone bring the front of the arm into it."""
    chain, check = _front_chair(tmp_path)
    cam, model = sv.camera(), ContainerModel.load(CONTAINER)
    views, why, stats = sv.angled_views(chain, check, list(HOME), cam, model)
    assert views, why
    R0, lens0 = sv._lens(chain, cam, HOME)
    band = sv.FrontBand(check, list(HOME))
    assert band.x_edge == pytest.approx(-0.0017 + 0.867 / 2 + sv.EDGE_GAP_M, abs=1e-6)  # past the user's legs
    # the 11-11 box, where the scene camera had it and where the wrist had it the day before, and one on the
    # arm's other side: each whole in some view
    lids = [(0.787, -0.240, 0.369), (0.71, -0.29, 0.31), (0.65, 0.10, 0.34)]
    shown = np.zeros(len(lids), bool)
    q_prev = list(HOME)
    for v in views:
        turn = np.subtract(v.joints, HOME)
        assert np.allclose(turn[:4], 0.0) and 0.1 < np.abs(turn[4:]).max() <= sv.FRONT_MAX_RAD  # the wrist alone
        assert np.linalg.norm(np.subtract(v.cam_xyz, lens0)) < 0.15  # the lens stays where the arm rests
        assert abs(float(v.rot_cam[2, 0])) < 0.1 and v.person_m > 0.02  # the image level, off the user
        shown |= sv.cover(cam, v.rot_cam, v.cam_xyz, _grid(lids))[1]
        # clear of the user and of where a table up to 0.35 m may stand in front, there and on the way
        assert check.pose_refusal(v.joints) is None and band.refusal(v.joints) is None
        assert band.path_refusal([q_prev, v.joints]) is None
        # the hand stays over the user's lap, behind where a table may stand, and no lower than 0.35 m
        C, r, links = check.spheres_at(v.joints)
        hand = np.array([("robotiq" in lk or "bracelet" in lk) for lk in links])
        assert (C[hand, 0] + r[hand]).max() < band.x_edge + 0.10 and (C[hand, 2] - r[hand]).min() > band.top_z
        q_prev = v.joints
    assert shown.all()
    _c0, sees0 = sv.cover(cam, R0, lens0, _grid(lids))
    assert not sees0.all()  # from HOME itself the wrist's place lies under the view


def test_the_front_views_are_chosen_by_what_each_adds_until_the_front_is_shown(tmp_path):
    """One view does not show the front (study/demo/wristlook, 2026-11-12: the one view flown until then showed
    42 of the 86 places the arm opens the box at): the views are chosen by the cells each adds, the one that
    shows the most first."""
    chain, check = _front_chair(tmp_path)
    cam, model = sv.camera(), ContainerModel.load(CONTAINER)
    views, why, stats = sv.angled_views(chain, check, list(HOME), cam, model)
    assert 2 <= len(views) <= sv.FRONT_MAX_VIEWS and stats["sees"] > 0.9, (why, stats)
    assert [v.name for v in views] == ["survey:angle-%d" % (i + 1) for i in range(len(views))]
    assert all(a.score >= b.score for a, b in zip(views, views[1:], strict=False))
    one, _why, alone = sv.angled_views(chain, check, list(HOME), cam, model, max_views=1)
    assert len(one) == 1 and one[0].joints == views[0].joints and alone["sees"] < stats["sees"] - 0.2
    # the places the first view does not show are on the arm's far side and under it: later views show them
    lids = [(0.45, 0.25, 0.315), (0.50, -0.30, 0.315), (0.60, 0.40, 0.315), (0.75, 0.0, 0.315)]
    shown = np.zeros(len(lids), bool)
    for v in views:
        shown |= sv.cover(cam, v.rot_cam, v.cam_xyz, _grid(lids))[1]
    assert shown.all()


def test_the_wrist_looks_first_where_the_scene_camera_has_the_box(tmp_path):
    chain, check = _front_chair(tmp_path)
    cam, model = sv.camera(), ContainerModel.load(CONTAINER)
    plain, _why, _stats = sv.angled_views(chain, check, list(HOME), cam, model)
    views, why, stats = sv.angled_views(chain, check, list(HOME), cam, model, hint=(0.55, 0.28))
    assert views[0].name == "survey:angle-hint" and "[0.55, 0.28]" in views[0].what
    assert sv.cover(cam, views[0].rot_cam, views[0].cam_xyz, _grid([(0.55, 0.28, 0.315)]))[1].all()
    assert [v.name for v in views[1:]] == ["survey:angle-%d" % (i + 1) for i in range(len(views) - 1)]
    assert stats["sees"] > 0.9  # the chosen views still follow, for what the hint's does not show
    # a hint no turn of the wrist reaches is passed over, and says so: the views as without one
    views, why, stats = sv.angled_views(chain, check, list(HOME), cam, model, hint=(0.30, -1.50))
    assert [v.joints for v in views] == [v.joints for v in plain]
    assert any("where the box may stand, at [0.30, -1.50]" in w for w in stats["refused"])


def test_a_lid_a_view_cut_is_the_place_to_look_again_at_the_passing_one_first():
    P = SimpleNamespace
    cut = "its top 45x100 mm, cut off by the view"
    find = P(
        places=(
            P(xy=(0.60, 0.20), top_z=0.32, button="unchecked", patches=(P(n_slab=150, cut=True),)),
            P(xy=(0.70, -0.10), top_z=0.32, button="unchecked", patches=(P(n_slab=900, cut=False),)),  # whole
        ),
        patches=(
            P(xy=(0.50, 0.30), top_z=0.31, why=cut, n_slab=400),
            P(xy=(0.80, 0.00), top_z=0.31, why="its top a ring, 40% filled — an open cup's rim?", n_slab=900),
        ),
    )
    assert sv.again_place(find, 0.105) == ((0.60, 0.20), 0.32)  # the place that passes, cut, before a refused top
    assert sv.again_place(find, 0.105, looked=[(0.61, 0.21)]) == ((0.50, 0.30), 0.31)
    assert sv.again_place(find, 0.105, looked=[(0.60, 0.20), (0.50, 0.30)]) is None
    assert sv.again_place(P(places=find.places[1:], patches=find.patches[1:]), 0.105) is None  # nothing cut


def test_the_look_again_turns_the_wrist_to_put_the_cut_lid_in_the_middle_or_says_why_not(tmp_path):
    chain, check = _front_chair(tmp_path)
    cam, model = sv.camera(), ContainerModel.load(CONTAINER)
    view, why = sv.again_view(chain, check, list(HOME), cam, model, (0.55, 0.30), 0.315, n=2)
    assert view is not None and why is None and view.name == "survey:angle-again-2"
    assert np.allclose(np.subtract(view.joints, HOME)[:4], 0.0)  # the wrist alone, as every front view
    pc = (np.array([0.55, 0.30, 0.315]) - np.asarray(view.cam_xyz)) @ view.rot_cam
    u, v = cam.k[0, 0] * pc[0] / pc[2] + cam.k[0, 2], cam.k[1, 1] * pc[1] / pc[2] + cam.k[1, 2]
    assert abs(u - cam.size[1] / 2) < 40 and abs(v - cam.size[0] / 2) < 40  # the lid in the middle of the image
    view, why = sv.again_view(chain, check, list(HOME), cam, model, (0.30, -1.50), 0.315)
    assert view is None and why.startswith("no look again at [0.30, -1.50]: ")


def test_the_turn_puts_the_aim_in_the_middle_of_the_image_or_says_how_far_it_misses(tmp_path):
    chain, _check = _front_chair(tmp_path)
    cam = sv.camera()
    q, miss = sv.turn_to(chain, cam, list(HOME), (0.65, -0.05, 0.37))
    assert miss < 0.05 and np.allclose(np.subtract(q, HOME)[:4], 0.0)
    # a wrist held to a tenth of a radian does not get there, and says by how much
    q, miss = sv.turn_to(chain, cam, list(HOME), (0.65, -0.05, 0.37), max_rad=0.1)
    assert miss > sv.FRONT_AIM_TOL and np.abs(np.subtract(q, HOME)).max() <= 0.1 + 1e-9


def test_the_front_band_stands_past_the_users_knees_and_says_where(tmp_path):
    chain, check = _front_chair(tmp_path)
    band = sv.FrontBand(check, list(HOME), top_z=0.60)
    # the arm leaned forward from its rest, over where a table may stand: refused, and it says in front
    whys = []
    for lean in np.linspace(0.2, 1.4, 13):
        q = list(HOME)
        q[1] += lean
        whys.append(band.refusal(q) or "")
    assert any("may stand in front of the chair" in w for w in whys), whys
    assert band.refusal(list(HOME)) is None  # the rest pose is as near as it may be


def test_the_front_look_goes_by_the_lids_geometry_and_never_looks_again_from_above(monkeypatch):
    asked = []
    s_views = [_view("survey:angle-far")]
    monkeypatch.setattr(
        sv, "angled_views", lambda *a, **k: (s_views, None, {"checks": 0.0, "sees": 0.5, "choose_s": 0.1})
    )
    monkeypatch.setattr(sv, "FrontBand", lambda check, rest: SimpleNamespace(ceiling_z=None))
    monkeypatch.setattr(approach, "body_check", lambda: SimpleNamespace(layers=["person"]))
    monkeypatch.setattr(approach, "_chain", lambda: None)
    monkeypatch.setattr(sv, "sphere_top_z", lambda check, q: 0.9)
    monkeypatch.setattr(sv, "fly_view", lambda ctx, runner, view, band, execute: ("flown", None))
    monkeypatch.setattr(sv, "grab_frame", lambda node, watcher, runner, tag, **k: (_frame(tag), None, None, None))
    monkeypatch.setattr(sv, "save_frame", lambda *a, **k: None)
    monkeypatch.setattr(sv, "look_again_view", lambda *a, **k: pytest.fail("a look again from above"))

    def find(frames, model, check_buttons=True):
        asked.append(check_buttons)
        return _unchecked()

    monkeypatch.setattr(wf, "find_container", find)
    ctx = SimpleNamespace(client=FakeClient(), model=ContainerModel.load(CONTAINER))
    runner = _Runner()
    s = sv.run_survey(None, ctx, runner, SimpleNamespace(grab=None), True, rest=list(HOME), front=True)
    assert asked == [False] and s.fix is None and "no view could check" in s.why
    assert [f["stage"] for k, f in runner.notes if k == "wrist_survey"] == ["views", "survey:angle-far"]


def test_the_hand_keeps_its_floor_from_the_user_at_every_front_view_and_on_its_way(tmp_path, monkeypatch):
    """FRONT_HAND_USER_M: the hand (every link past joint 5) no nearer the user's boxes than the floor at a view, a
    hint's view or a look again, nor on the way there; one that would bring it nearer is not flown and says so.
    0.0 (the default): no floor of the look's own, the views as they were first chosen."""
    chain, check = _front_chair(tmp_path)
    cam, model = sv.camera(), ContainerModel.load(CONTAINER)
    assert sv.FRONT_HAND_USER_M == 0.0
    free = sv.FrontBand(check, list(HOME))
    views0, _why, stats0 = sv.angled_views(chain, check, list(HOME), cam, model)
    assert free.hand_user_m == 0.0 and stats0["hand_refused"] == 0 and stats0["hand_kept_mm"] == 0.0
    hand = [float(free._hand_m(v.joints)[0].min()) for v in views0]
    rest_m = float(free._hand_m(HOME)[0].min())
    near = views0[int(np.argmin(hand))]  # the view turned toward the user's side
    assert min(hand) < rest_m - 0.05 and min(hand) < 0.15
    again0, _w = sv.again_view(chain, check, list(HOME), cam, model, (0.45, -0.25), 0.315)
    assert again0 is not None and float(free._hand_m(again0.joints)[0].min()) < min(hand)  # a look again goes nearer
    assert free.refusal(near.joints) is None and free.path_refusal([list(HOME), near.joints]) is None

    floor = min(hand) + 0.02
    monkeypatch.setattr(sv, "FRONT_HAND_USER_M", floor)
    band = sv.FrontBand(check, list(HOME))
    assert band.hand_user_m == pytest.approx(floor)
    said = "mm from the user, nearer than the look keeps it (%.0f mm)" % (1000 * floor)
    assert said in band.refusal(near.joints) and band.refusal(list(HOME)) is None
    assert said + ", " in band.path_refusal([list(HOME), near.joints])  # ... and where on the way
    views, _why, stats = sv.angled_views(chain, check, list(HOME), cam, model)
    assert views and all(float(band._hand_m(v.joints)[0].min()) >= floor for v in views)
    assert near.joints not in [v.joints for v in views] and stats["hand_refused"] >= 1
    assert stats["hand_kept_mm"] == pytest.approx(1000 * floor, abs=0.1) and stats["sees"] <= stats0["sees"]
    # the hint's view and a look again are held to it too, and say so
    aim = [float(x) for x in near.what.split("[")[1].rstrip("]").split(",")]
    hinted, _why, hstats = sv.angled_views(chain, check, list(HOME), cam, model, hint=aim)
    assert all(v.name != "survey:angle-hint" for v in hinted)
    assert any("where the box may stand" in w and said in w for w in hstats["refused"])
    again, why = sv.again_view(chain, check, list(HOME), cam, model, (0.45, -0.25), 0.315)
    assert again is None and why.startswith("no look again at [0.45, -0.25]: the hand (") and said in why
    # never asked to stand farther from the user than it rests
    assert sv.FrontBand(check, list(HOME), hand_user_m=1.0).hand_user_m < rest_m
    assert sv.FrontBand(check, list(HOME), hand_user_m=1.0).refusal(list(HOME)) is None


def _cut_lid(xy=(0.60, 0.20), passing=True, frames=(0,)):
    """What the finder says of a lid a view cuts at `xy`: a fix every sighting of which is cut, or (passing
    False) no fix and its top refused as cut off by the view."""
    why = "" if passing else "its top 45x100 mm, cut off by the view"
    patches = tuple(SimpleNamespace(frame=i, xy=xy, top_z=0.34, n_slab=200, cut=True, why=why) for i in frames)
    if not passing:
        return wf.WristFind(None, "wrist camera: the container may stand at ..., cut off by the view", patches=patches)
    place = wf.Place(xy=xy, top_z=0.34, table_z=0.24, patches=patches, button="unchecked")
    fix = SimpleNamespace(
        top_xyz=(xy[0], xy[1], 0.34), table_z_scene=0.24, partial="cut off by the wrist camera's view"
    )
    return wf.WristFind(fix, "the container at ..., cut", 0.4, places=(place,), patches=patches)


def _whole_lid(xy=(0.60, 0.20), frame=1):
    patch = SimpleNamespace(frame=frame, xy=xy, top_z=0.34, n_slab=900, cut=False, why="")
    place = wf.Place(xy=xy, top_z=0.34, table_z=0.24, patches=(patch,), button="unchecked")
    fix = SimpleNamespace(top_xyz=(xy[0], xy[1], 0.34), table_z_scene=0.24, partial="")
    return wf.WristFind(fix, "the container at ..., whole", 0.6, places=(place,), patches=(patch,))


NOTHING = wf.WristFind(None, "wrist camera: nothing on the counter is the container (0 things at lid height)")


def _front_flow(monkeypatch, finds, views=3, again=True, hint=None):
    """run_survey(front=True) with its motions, frames and finder scripted: (the Survey, the views flown in order,
    the runner, how the views were asked for)."""
    flown, asked = [], {}
    monkeypatch.setattr(approach, "body_check", lambda: SimpleNamespace(layers=["person"]))
    monkeypatch.setattr(approach, "_chain", lambda: None)
    monkeypatch.setattr(sv, "FrontBand", lambda check, rest: SimpleNamespace(ceiling_z=None))
    monkeypatch.setattr(sv, "sphere_top_z", lambda check, q: 0.9)

    def chosen(*a, **k):
        asked.update(k)
        names = ["survey:angle-%d" % (i + 1) for i in range(views)]
        return [_view(n) for n in names], None, {"checks": 0.0, "sees": 0.96, "choose_s": 0.2}

    monkeypatch.setattr(sv, "angled_views", chosen)
    monkeypatch.setattr(
        sv,
        "again_view",
        lambda chain, check, q, cam, model, xy, z, rest=None, n=1: (
            (_view("survey:angle-again-%d" % n), None) if again else (None, "no look again at [%.2f, %.2f]" % xy)
        ),
    )
    monkeypatch.setattr(sv, "look_again_view", lambda *a, **k: pytest.fail("a look again from above"))
    monkeypatch.setattr(
        sv, "fly_view", lambda ctx, runner, view, band, execute: flown.append(view.name) or ("flown", None)
    )
    monkeypatch.setattr(sv, "grab_frame", lambda node, watcher, runner, tag, **k: (_frame(tag), None, None, None))
    monkeypatch.setattr(sv, "save_frame", lambda *a, **k: None)
    finds = list(finds)
    monkeypatch.setattr(wf, "find_container", lambda frames, model, **k: finds.pop(0))
    ctx = SimpleNamespace(client=FakeClient(), model=ContainerModel.load(CONTAINER))
    runner = _Runner()
    s = sv.run_survey(None, ctx, runner, SimpleNamespace(grab=None), True, rest=list(HOME), front=True, hint=hint)
    return s, flown, runner, asked


def test_the_front_look_goes_from_view_to_view_until_one_shows_the_lid_whole(monkeypatch):
    whole = _whole_lid(frame=1)
    s, flown, _r, asked = _front_flow(monkeypatch, [NOTHING, whole], hint=(0.70, -0.20))
    assert flown == ["survey:angle-1", "survey:angle-2"] and s.fix is whole.fix
    assert asked["hint"] == (0.70, -0.20)  # where the scene camera has the box: the first view's aim
    s, flown, _r, _a = _front_flow(monkeypatch, [NOTHING, NOTHING, NOTHING])
    assert flown == ["survey:angle-1", "survey:angle-2", "survey:angle-3"] and s.fix is None
    assert "nothing on the counter is the container" in s.why


def test_a_lid_the_view_cut_is_looked_at_again_and_taken_where_that_look_shows_it_whole(monkeypatch):
    whole = _whole_lid(frame=1)
    for passing in (True, False):  # a fix every sighting of which is cut; a top refused only as cut
        s, flown, _r, _a = _front_flow(monkeypatch, [_cut_lid(passing=passing), whole])
        assert flown == ["survey:angle-1", "survey:angle-again-1"] and s.fix is whole.fix


def test_a_cut_lid_the_look_again_does_not_show_whole_is_no_fix(monkeypatch):
    """A wrong fix is worse than NO BOX: the turned wrist put the place in the middle of the image and no lid
    showed whole there, so what the first view cut was not the lid."""
    still_cut = _cut_lid(passing=True, frames=(0,))  # the look again's frame (1) holds no lid there
    s, flown, _r, _a = _front_flow(monkeypatch, [_cut_lid(), still_cut, still_cut, still_cut])
    assert flown == ["survey:angle-1", "survey:angle-again-1", "survey:angle-2", "survey:angle-3"]
    assert s.fix is None and "the look again at it showed no lid whole there" in s.why


def test_a_cut_lid_no_look_again_can_be_flown_at_is_taken_as_it_is_or_passed_over(monkeypatch):
    cut = _cut_lid(passing=True)
    s, flown, runner, _a = _front_flow(monkeypatch, [cut], again=False)
    assert flown == ["survey:angle-1"] and s.fix is cut.fix  # as before the look again: a cut lid's fix, said so
    assert [f for k, f in runner.notes if f.get("stage") == "look-again"][0]["flown"] is False
    whole = _whole_lid(frame=1)
    s, flown, _r, _a = _front_flow(monkeypatch, [_cut_lid(passing=False), whole], again=False)
    assert flown == ["survey:angle-1", "survey:angle-2"] and s.fix is whole.fix


def test_the_front_look_looks_again_no_more_than_twice(monkeypatch):
    finds = [_cut_lid(xy=(0.50 + 0.1 * i, 0.2), passing=False) for i in range(6)]
    s, flown, _r, _a = _front_flow(monkeypatch, finds)
    assert [n for n in flown if "again" in n] == ["survey:angle-again-1", "survey:angle-again-2"]
    assert len(flown) == 3 + sv.FRONT_AGAIN_MAX and s.fix is None


def _first(monkeypatch, survey, scene_fix=None, execute=True):
    """press_demo.wrist_first with its look scripted: (what it returned or its exit code, how the look was asked
    for, the run's notes, the ctx)."""
    asked, runner = {}, _Runner()
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: asked.update(k) or survey)
    ctx = SimpleNamespace(worlds=SimpleNamespace(wrist_camera_config="camera_oak_wrist.yaml"), survey_clouds=None)
    args = Namespace(wrist_first=True, instance_id=None, execute=execute)
    try:
        got = press_demo.wrist_first(None, ctx, SimpleNamespace(park_tool_down=False), runner, None, args, scene_fix)
    except SystemExit as e:
        got = e.code
    return got, asked, runner.notes, ctx


def test_the_wrists_box_is_taken_over_the_scene_cameras(monkeypatch):
    survey = sv.Survey(fix=FIX, clouds=[{"points": np.zeros((3, 3), np.float32), "cam_xyz": (0, 0, 0), "tag": "a"}])
    scene = SimpleNamespace(top_xyz=(0.787, -0.240, 0.369))
    got, asked, notes, ctx = _first(monkeypatch, survey, scene_fix=scene)
    assert got is FIX and asked["front"] is True and ctx.survey_clouds is survey.clouds and not notes


def test_where_the_wrist_finds_none_the_run_goes_on_with_the_scene_cameras_or_none(monkeypatch):
    survey = sv.Survey(why="wrist camera: nothing on the counter is the container")
    got, _asked, notes, ctx = _first(monkeypatch, survey, scene_fix=SimpleNamespace(top_xyz=(0.6, 0.0, 0.3)))
    assert got is None and ctx.survey_clouds is None
    assert notes == [("wrist_first", {"found": False, "why": survey.why, "scene": True})]
    got, _asked, notes, _ctx = _first(monkeypatch, survey)
    assert got is None and notes[0][1]["scene"] is False


def test_the_wrist_looks_first_at_the_scene_cameras_box_and_its_word_goes_into_the_no_box(monkeypatch):
    survey = sv.Survey(why="wrist camera: nothing on the counter is the container (0 things at lid height)")
    rough = SimpleNamespace(top_xyz=(0.787, -0.240, 0.369), rough=True, partial="cut off by the frame toward +y")
    asked, runner, said = {}, _Runner(), []
    monkeypatch.setattr(sv, "run_survey", lambda *a, **k: asked.update(k) or survey)
    ctx = SimpleNamespace(worlds=SimpleNamespace(wrist_camera_config="camera_oak_wrist.yaml"), survey_clouds=None)
    args = Namespace(wrist_first=True, instance_id=None, execute=True)
    cfg = SimpleNamespace(park_tool_down=False)
    assert press_demo.wrist_first(None, ctx, cfg, runner, None, args, rough, said=said) is None
    assert asked["hint"] == (0.787, -0.240) and said == [survey.why]
    assert press_demo.wrist_first(None, ctx, cfg, runner, None, args, None, said=said) is None
    assert asked["hint"] is None and len(said) == 2
    # neither camera has it: the run's NO BOX says what each said, the robot's own words
    homes = []
    monkeypatch.setattr(press_demo, "try_home", lambda ctx, runner, execute, why: homes.append(why) or True)
    locator = SimpleNamespace(last_why="the segmenter boxed nothing of its classes", look_record=lambda: {})
    args = Namespace(wrist_survey=False, instance_id=None, execute=True)
    with pytest.raises(SystemExit) as e:
        press_demo.wrist_survey_or_exit(None, ctx, cfg, runner, None, args, locator, wrist_look=said[0])
    assert e.value.code == 2
    assert homes == [
        "NO BOX — scene camera: the segmenter boxed nothing of its classes; the wrist's look in front: " + survey.why
    ]


def test_a_wrist_turn_stopped_part_way_holds_and_a_preview_ends_turned(monkeypatch):
    got, _a, _n, _c = _first(monkeypatch, sv.Survey(stopped=True, why="the move to survey:angle-far stopped part way"))
    assert got == 1
    got, _a, _n, _c = _first(monkeypatch, sv.Survey(preview=True, why="a preview: no frames"), execute=False)
    assert got == 0

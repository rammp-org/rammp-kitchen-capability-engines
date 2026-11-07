"""The wrist survey's poses (tasks/survey): one or two for the wrist camera over the arm's side, chosen from the
arm's kinematics and the chair model for any counter height, clear of the chair, its user and where a counter may
stand; and the look again nearer above a place the finder could not check."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import flatbot_urdf

from box_opening.constants import HOME, PLANNER_LIMIT_MARGIN_RAD
from box_opening.models.container import ContainerModel
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import mat_to_quat_xyzw
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


def test_the_band_keeps_the_arm_off_a_counter_beside_the_chair_no_nearer_than_the_rest_pose_has_it(tmp_path):
    chain, check = flatbot_chair(tmp_path)
    band = sv.Band(check, HOME)
    assert band.y_edge == pytest.approx(0.038 + sv.EDGE_GAP_M)  # the armrest's edge on the arm's side
    # the rest pose is no survey pose: its forearm stands inside the band (the upper arm alone keeps no nearer)
    assert band.refusal(HOME).startswith("arm_forearm_link 28 mm inside")
    low = chain.nearest_solution(list(HOME), [0.45, 0.30, 0.30], DOWN)  # the hand down over the counter
    assert low is not None and "where a counter up to 0.45 m may stand beside the chair" in band.refusal(low)
    assert "of the way" in band.path_refusal([HOME, low])
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
    band.top_z = 0.45
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

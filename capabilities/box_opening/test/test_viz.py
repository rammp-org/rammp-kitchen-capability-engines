"""Box opening's perception drawn for Foxglove (perception/viz): the
annotated scene and wrist frames, the markers and the status, built from
what the look, the aim and the press computed; the worker that draws them
only for a subscriber; and that nothing it does reaches the run. Nothing
here spins a node: the publishers are stand-ins."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.perception import viz

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
DATA = Path(__file__).parent / "data"
F_162417 = "scene_upright_flatbot_20261027_162417.npz"  # the 10-27 16:24 frame the OWL missed and RF-DETR boxed
F_1623 = "wrist_reach_aim_flatbot_20261027_1623.npz"  # the 10-27 16:23 reach-out aim from the look pose


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


@pytest.fixture(scope="module")
def look_1624(pink):
    """The 16:24 look as locate() leaves it: the segmenter's band through the
    real gate (the fix), and the holder box as an OWL box would give it
    (refused: its top 55 mm below lid height), on the frame's depth drawn
    grey — the fixture keeps no colour."""
    import cv2

    from box_opening.perception.scene import color_cloud
    from box_opening.perception.scene_source import SceneFix, scene_fix_from_boxes, table_from_scene_points
    from box_opening.perception.segmenter_source import segmenter_candidates

    z = np.load(DATA / F_162417)
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    tz = table_from_scene_points(pts, uv, 400)
    kw = dict(cam_xyz=T[:3, 3], side_table_z=None, image_wh=(640, 400), cam_R=T[:3, :3], rivals=True)
    holder = tuple(float(v) for v in z["seg_objects"][0][1:])
    cands = segmenter_candidates([("food container", 0.88, holder)], ("food container",), (640, 400))
    got = scene_fix_from_boxes(cands, pts, uv, None if np.isnan(tz) else float(tz), pink, src="segmenter", **kw)
    owl = [(0.31, holder)]
    got_owl = scene_fix_from_boxes(owl, pts, uv, None if np.isnan(tz) else float(tz), pink, src="OWL", **kw)
    assert got.pose is not None and got_owl.pose is None
    frame = cv2.cvtColor(np.clip(z["depth_mm"] / 6, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fix = SceneFix(
        pose=got.pose,
        top_xyz=got.top,
        n_points=got.n,
        score=0.88,
        table_z_scene=got.table_z,
        source="segmenter",
        n_sightings=2,
        sources=("segmenter", "segmenter"),
    )
    judged = {"segmenter": (cands, got, frame, T), "owl": (owl, got_owl, frame, T)}
    return viz.SceneLook(
        t=1.0,
        fix=fix,
        why=None,
        judged=judged,
        boxes={"segmenter": [["food container", 0.88, *holder]]},
        color=frame,
        k=z["k"],
        dist=z["dist"],
    )


@pytest.fixture(scope="module")
def reach_1623(pink):
    """(the 16:23 frame's colour, k, dist, camera pose, its lid height and
    prior, the tilted aim's sighting on it)."""
    import cv2

    from box_opening.perception.button_aim import find_button

    f = np.load(DATA / F_1623)
    color = cv2.imdecode(f["color_png"], cv2.IMREAD_COLOR)
    lid_z, prior = float(f["lid_z"]), tuple(float(v) for v in f["prior_xy"])
    s, why = find_button(
        color,
        f["depth_mm"].astype(np.float32) / 1000.0,
        f["k"],
        f["rot_cam"],
        f["trans_cam"],
        float(f["table_z"]),
        pink,
        dist=f["dist"],
        lid_z=lid_z,
        prior_xy=prior,
    )
    assert s is not None and s.oblique, why
    return color, f["k"], f["dist"], (f["rot_cam"], f["trans_cam"]), lid_z, prior, s


def _tick(reach, **kw):
    color, k, dist, cam, lid_z, prior, s = reach
    base = dict(
        t=2.0,
        color=color,
        k=k,
        dist=dist,
        cam=cam,
        sighting=s,
        hit=True,
        why=None,
        hits=1,
        frames=3,
        need=2,
        t_aim=0.4,
        lid_z=lid_z,
        prior_xy=prior,
        points=[s.xyz],
    )
    base.update(kw)
    return viz.AimTick(**base)


def _stroke(s, bent=False, q_end=None):
    """A compliant stroke over the sighting's button, as compliant_stroke plans it: the pads 60 mm over its top,
    the springs' reference 30 mm past its bottom — straight on down, or `bent` to the pads at q_end."""
    return SimpleNamespace(
        pad0=(s.xyz[0], s.xyz[1], s.xyz[2] + 0.06),
        end_z=s.xyz[2] - 0.012 - 0.03,
        over=0.03,
        k=470.0,
        bent=bent,
        bend_m=0.004 if bent else 0.0,
        q_end=q_end,
    )


class _Pub:
    def __init__(self, subscribers=0, broken=False):
        self.subscribers, self.broken, self.sent = subscribers, broken, []

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, msg):
        if self.broken:
            raise RuntimeError("the publisher's handle is gone")
        self.sent.append(msg)


def _viz(pink, **subs):
    pubs = {k: _Pub(subs.get(k, 0)) for k in ("scene", "wrist", "markers", "status")}
    return viz.Viz(None, pink, "arm_mount", pubs=pubs, thread=False, ran=None), pubs


# --- what is drawn ----------------------------------------------------------


def test_the_status_mirrors_the_console_lines_that_say_a_phase():
    assert viz.said("[press_demo] SCENE: box at [0.661, -0.265] yaw 0.0 deg") == (
        "SCENE",
        "SCENE: box at [0.661, -0.265] yaw 0.0 deg",
    )
    assert viz.said("[press_demo]   SCENE: box at [0.661, -0.265]")[0] == "SCENE"  # detect-only's report
    assert viz.said("[press_demo] AIM: button circle at pixel (292, 239)")[0] == "AIM"
    assert viz.said("[press_demo] ONE MOTION from the look pose to below the button: reach:re-aim")[0] == "ONE MOTION"
    assert viz.said("[press_demo] PRESS (compliant): straight down 72 mm")[0] == "PRESS"
    assert viz.said("[press_demo] STOP: no button from the look pose (0 of 0 hits) — returning home")[0] == "STOP"
    assert viz.said("[press_demo] NO BOX — scene camera: none of its 3 boxes — returning home")[0] == "NO BOX"
    assert viz.said("[press_demo]   no fix: scene segmenter: saw no food container")[0] == "LOOK"
    # the lid off's (reach_lid_off): the knob's verdict, the press once more, the grip, where the lid goes
    assert (
        viz.said(
            "[press_demo] AFTER PRESS 1: knob UP 9.0 mm — the box is open (1 look); the button 3 mm from "
            "where the press was aimed"
        )[0]
        == "AFTER PRESS"
    )
    assert viz.said(
        "[press_demo] the knob is not up (0.9 mm above the lid): pressing once more — the same press, "
        "from the look pose"
    ) == (
        "AGAIN",
        "the knob is not up (0.9 mm above the lid): pressing once more " "— the same press, from the look pose",
    )
    assert (
        viz.said(
            "[press_demo] GRIP: the fingers open on the way down; ONE MOTION from the look pose to the knob: "
            "grip:open"
        )[0]
        == "GRIP"
    )
    assert viz.said("[press_demo] lid goes to [0.52, -0.31, 0.213] — 180 mm from the box, beside the box")[0] == "SPOT"
    assert viz.said("[press_demo] LID GRIPPED — 0.390 in the grip band")[0] == "LID GRIPPED"
    for quiet in (
        "[press_demo] the button stands 11 mm above its lid",
        "[runner] SLOW MODE: every motion dilated",
        "[viz] (the markers failed: x)",
        "fix: top [0.4, 0.1, 0.2] yaw 0.0",
        "[press_demo] backing out the way the hand came down: grip:up, reach:slide-out",
    ):
        assert viz.said(quiet) is None, quiet


def test_every_box_the_gate_judged_gets_its_verdict_and_the_gates_own_reason(look_1624):
    owl_cands, owl_got = look_1624.judged["owl"][:2]
    assert owl_got.tried_boxes == (owl_cands[0][1],) and owl_got.tried_boxes[0] is owl_cands[0][1]  # the gate's own box
    cands = viz.judged_cands(look_1624.judged, look_1624.boxes, look_1624.fix)
    assert [(c.source, c.verdict) for c in cands] == [("segmenter", "fix"), ("owl", "refused")]
    band, owl = cands
    assert band.kind == "band" and band.cls == "food container"
    assert band.det_box != band.box  # the holder under the band
    assert band.tag == "SEG 0.88 FIX" and "side lift, 299 lid points" in band.why
    assert owl.tag == "OWL 0.31 X"
    assert owl.why == "from the side, its top is -55 mm from lid height (the table at 0.213 + 121 mm)"
    assert "(OWL 0.31)" not in owl.line()


def test_a_box_that_passed_is_the_fix_only_where_the_look_made_its_fix_from_it(look_1624, pink):
    """A pass is one sighting: with no fix (it passed in one frame, a fix
    needs two that agree) it is "passed", yellow, not the green FIX under a
    NO BOX caption; so it is where the fix is the other detector's."""
    from dataclasses import replace

    why = "scene segmenter: one of its boxes passed in 1 frames, and a fix needs two that agree; scene OWL: not asked"
    none = look_1624.replace(fix=None, why=why)
    band = viz.judged_cands(none.judged, none.boxes, none.fix)[0]
    assert (band.verdict, band.tag) == ("passed", "SEG 0.88 PASSED")
    assert band.why.startswith("it passed (side lift, 299 lid points")
    assert band.why.endswith("but no fix came of it: see the caption")
    owls = replace(look_1624.fix, source="owl", sources=("owl", "owl"))
    assert viz.judged_cands(look_1624.judged, look_1624.boxes, owls)[0].why.endswith("but the fix is the OWL's")
    mixed = replace(look_1624.fix, source="owl", sources=("segmenter", "owl"))
    assert viz.judged_cands(look_1624.judged, look_1624.boxes, mixed)[0].verdict == "fix"  # one of its two sightings
    # drawn: the box yellow, no green anywhere in the frame (no fix's lid, no FIX), the caption NO BOX
    img = viz.scene_image(none, pink)
    bx0, by0, bx1, by1 = (int(round(v)) for v in band.box)
    side = img[by0 + 4 : by1 - 4, bx1].astype(int)
    assert ((side[:, 1] > 180) & (side[:, 2] > 200) & (side[:, 0] < 80)).all()
    frame = img[:400].reshape(-1, 3).astype(int)
    assert not ((frame[:, 1] > 150) & (frame[:, 1] > frame[:, 2] + 60) & (frame[:, 1] > frame[:, 0] + 60)).any()
    assert viz.scene_caption(none) == "NO BOX - " + why


def test_two_places_and_boxes_after_the_winner_are_told_apart():
    from box_opening.perception.scene_source import SceneBoxFix

    a, b, c = (10.0, 10.0, 50.0, 40.0), (100.0, 10.0, 140.0, 40.0), (200.0, 10.0, 240.0, 40.0)
    pose = SimpleNamespace(xyz=(0.6, -0.2, 0.19), yaw=0.0)
    got = SceneBoxFix(
        pose,
        (0.6, -0.2, 0.31),
        300,
        b,
        ("its top is +40 mm from lid height (segmenter 0.70)",),
        0.19,
        "side",
        (),
        (),
        ((0.75, -0.2, 0.31),),
        tried_boxes=(a,),
    )
    cands = viz.judged_cands({"segmenter": ([(0.70, a), (0.66, b), (0.40, c)], got, None, None)})
    assert [c.verdict for c in cands] == ["refused", "two places", "not judged"]
    assert "150 mm from it" in cands[1].why and cands[1].tag == "SEG 0.66 2 PLACES"


def test_a_reason_is_the_boxs_the_gate_gave_it_not_one_whose_score_reads_the_same():
    """OWL A 0.314 refused, B 0.312 passed, C 0.311 never judged: every
    reason ends "(OWL 0.31)", yet only A's is A's — C is not judged, and
    a box equal to B's is not B."""
    from box_opening.perception.scene_source import SceneBoxFix

    a, b, c = [10.0, 10.0, 50.0, 40.0], [100.0, 10.0, 140.0, 40.0], [200.0, 10.0, 240.0, 40.0]
    pose = SimpleNamespace(xyz=(0.6, -0.2, 0.19), yaw=0.0)
    got = SceneBoxFix(
        pose,
        (0.6, -0.2, 0.31),
        300,
        b,
        ("its top is +40 mm from lid height (OWL 0.31)",),
        0.19,
        "top",
        tried_boxes=(a,),
    )
    fix = SimpleNamespace(source="owl", sources=("owl", "owl"))
    cands = viz.judged_cands({"owl": ([(0.314, a), (0.312, b), (0.311, c), (0.30, list(b))], got, None, None)}, fix=fix)
    assert [x.verdict for x in cands] == ["refused", "fix", "not judged", "not judged"]
    assert cands[0].why == "its top is +40 mm from lid height"
    assert cands[2].why == cands[3].why == "a box ahead of it passed"
    # reasons that do not pair with boxes one to one (a gate that kept one without the other) tell no box anything
    from dataclasses import replace

    for boxes in ((), (a, c)):
        unpaired = viz.judged_cands(
            {"owl": ([(0.314, a), (0.312, b), (0.311, c)], replace(got, tried_boxes=boxes), None, None)}, fix=fix
        )
        assert [(x.verdict, x.why) for x in unpaired] == [
            ("not judged", "the gate's reasons do not say which box each was"),
            ("fix", unpaired[1].why),
            ("not judged", "the gate's reasons do not say which box each was"),
        ], boxes


def test_the_scene_image_draws_the_fix_where_the_camera_sees_it_and_says_every_verdict(look_1624, pink):
    img = viz.scene_image(look_1624, pink)
    assert img.shape[1] == 640 and img.shape[0] > 400 + 3 * viz.LINE_PX  # the frame, then the caption and two verdicts
    T = look_1624.judged["segmenter"][3]
    u, v = viz.project([look_1624.fix.top_xyz], T[:3, :3], T[:3, 3], look_1624.k, look_1624.dist)[0]
    # the lid's top, a green cross at the fix's own pixel
    near = img[int(v) - 3 : int(v) + 4, int(u) - 3 : int(u) + 4].reshape(-1, 3).astype(int)
    assert any(g > 150 and g > r + 60 and g > b + 60 for b, g, r in near)
    # the OWL's refused box in red down both its sides, and the band above it green
    x0, y0, x1, y1 = (int(round(v)) for v in look_1624.judged["owl"][0][0][1])
    for side in (img[y0 + 5 : y1 - 5, x0], img[y0 + 5 : y1 - 5, x1]):
        assert ((side[:, 2] > 200) & (side[:, 1] < 120)).all()
    bx0, by0, bx1, by1 = (int(round(v)) for v in look_1624.judged["segmenter"][0][0].box)
    assert ((img[by0 + 4 : by1 - 4, bx1, 1] > 180) & (img[by0 + 4 : by1 - 4, bx1, 2] < 120)).all()
    # a SCENE line said after the look is its caption; before it, its own words
    assert viz.scene_caption(look_1624).startswith("SCENE: box at [0.661, -0.265], its lid's top at 0.310")
    assert "SEG 0.88, 2 sightings (segmenter+segmenter)" in viz.scene_caption(look_1624)


def test_a_look_with_no_fix_still_shows_its_frame_and_why(look_1624, pink):
    none = look_1624.replace(fix=None, why="scene segmenter: saw no food container in 75 frames", judged={})
    assert viz.scene_caption(none) == "NO BOX - scene segmenter: saw no food container in 75 frames"
    img = viz.scene_image(none, pink)
    assert img is not None and img.shape[0] > 400
    assert viz.scene_image(none.replace(color=None), pink) is None
    with pytest.raises(TypeError):
        none.replace(colour=None)


def test_the_wrist_image_draws_the_button_the_tilted_aim_found_its_lid_and_the_scene_cameras_prior(reach_1623, pink):
    _color, k, dist, cam, lid_z, prior, s = reach_1623
    img = viz.wrist_image(_tick(reach_1623), pink)
    assert img.shape[1] == 640 and img.shape[0] > 400
    u, v = s.uv
    centre = img[int(v) - 2 : int(v) + 3, int(u) - 2 : int(u) + 3].reshape(-1, 3).astype(int)
    assert any(r > 200 and g < 110 for b, g, r in centre)  # the press point's red cross
    pu, pv = viz.project([(prior[0], prior[1], lid_z)], cam[0], cam[1], k, dist)[0]
    mark = img[int(pv) - 2 : int(pv) + 3, int(pu) - 2 : int(pu) + 3].reshape(-1, 3).astype(int)
    assert any(g > 180 and r > 180 and b < 80 for b, g, r in mark)  # where the scene camera put the button, yellow
    # a refused frame says why; a pose-less one still circles the hit in the image
    assert viz.wrist_image(_tick(reach_1623, hit=False, why="camera moving", cam=None), pink).shape[0] > 400
    # a frame refused after the hit carries no sighting (aim_tick): no button, no cross at the hit's pixel —
    # that pixel is the hit's frame's, not this one's; the hits stay, through this frame's own pose
    arms = [(int(round(v)), int(round(u)) + d) for d in (-9, 9)] + [(int(round(v)) + d, int(round(u))) for d in (-9, 9)]
    hit_img = viz.wrist_image(_tick(reach_1623), pink)
    assert all(hit_img[p][2] > 200 and hit_img[p][1] < 110 for p in arms)  # the cross's arms, red
    stale = viz.wrist_image(_tick(reach_1623, hit=False, sighting=None, why="camera moving"), pink)
    assert all((stale[p] == _color[p]).all() for p in arms)


def test_a_knob_looks_prior_is_where_the_press_was_aimed_not_the_scene_cameras_button(reach_1623, pink, monkeypatch):
    """The knob look after a press (reach_press._aim's "reach-after") is
    given the button the press was aimed at, its top the press's: its frame
    and its markers name them so, not as the scene camera's. The first aim
    ("reach-aim") is given the scene camera's."""
    from builtin_interfaces.msg import Time

    color, k, dist, cam, lid_z, prior, s = reach_1623
    v, _pubs = _viz(pink)
    aimer = SimpleNamespace(
        frames=1,
        hits=1,
        last_sighting=s,
        last_why=None,
        last_frame=(SimpleNamespace(color=color, k=k, dist=dist), cam),
        t0=0.0,
        lid_z=lid_z,
        prior_xy=prior,
        points=[s.xyz],
        _need=lambda: 2,
        fix_hits=lambda: [0],
        fix=lambda measured=False: s.xyz,
    )
    said = []
    monkeypatch.setattr(viz, "_label", lambda img, text, org, bgr, taken=None: said.append(text))
    monkeypatch.setattr(
        viz, "_strip", lambda img, lines: said.extend(ln[0] if isinstance(ln, tuple) else ln for ln in lines) or img
    )
    for tag, button, lid in (
        ("reach-aim", "the scene camera's button", "the scene camera's lid"),
        ("reach-after", "where the press was aimed", "the press's top"),
    ):
        v.aim_tick(aimer, True, 0, tag)
        tick = v._slots.pop("wrist")
        assert tick.prior_said == (button, lid)
        said.clear()
        viz.wrist_image(tick, pink)
        assert button in said and any(ln.startswith("the lid ring at z") and (", %s " % lid) in ln for ln in said)
        names = {m.ns for m in viz.markers({"aim": tick}, pink, "arm_mount", Time()).markers}
        (other,) = {"the scene camera's button", "where the press was aimed"} - {button}
        assert "aim: " + button in names and "aim: " + other not in names


def test_the_aim_hands_over_every_hit_and_its_fix_and_other_frames_at_most_every_period(reach_1623, pink, monkeypatch):
    """aim() is called by aim_button_at_staging after every tick: a tick that
    took no frame hands nothing over, a hit and the fix always, a refused
    frame at most every WRIST_PERIOD_S. Only references and small copies go
    over."""
    color, k, dist, cam, lid_z, prior, s = reach_1623
    now = [10.0]
    monkeypatch.setattr(viz, "_clock", lambda: now[0])
    v, _pubs = _viz(pink)
    judged = SimpleNamespace(color=color, k=k, dist=dist)  # the aimer's snapshot of the frame it judged
    aimer = SimpleNamespace(
        frames=0,
        hits=0,
        last_sighting=None,
        last_why=None,
        last_frame=(judged, cam),
        t0=9.0,
        lid_z=lid_z,
        prior_xy=prior,
        points=[],
        _need=lambda: 2,
    )
    taken = []

    def tick(dt, frame=True, hit=False, fixed=False, why="no circle"):
        now[0] += dt
        n0 = aimer.frames
        aimer.frames += int(frame)
        if hit:
            aimer.hits += 1
            aimer.last_sighting = s
            aimer.points.append(s.xyz)
        else:
            aimer.last_why = why
        v.aim_tick(aimer, fixed, n0)
        item = v._slots.pop("wrist", None)
        taken.append(None if item is None else (item.hit, item.done is not None, item.why))
        # a hit's own sighting; a refused frame none (the hit before it is another frame's)
        assert item is None or item.sighting is (s if hit else None)

    tick(0.07)  # a refused frame: the first, handed over
    tick(0.07)  # refused again, 0.07 s after: not
    tick(0.02, frame=False)  # no new frame: nothing
    tick(0.07, hit=True)  # a hit: at once
    tick(0.07)  # refused 0.07 s after the hit: not
    tick(0.15)  # 0.22 s after: handed over
    tick(0.07, hit=True, fixed=True)  # the fix
    assert taken == [
        (False, False, "no circle"),
        None,
        None,
        (True, False, None),
        None,
        (False, False, "no circle"),
        (True, True, None),
    ]
    v.aim_tick(aimer, True, aimer.frames)
    item = v._slots["wrist"]
    assert item.done is aimer and item.points == [s.xyz, s.xyz] and item.points is not aimer.points
    assert item.color is color and item.cam is cam  # the judged frame and its pose, by reference
    assert item.t_aim == pytest.approx(1.52)  # 0.52 s of ticks, the aimer made 1 s before


def test_the_markers_hold_the_box_the_table_the_aimed_button_and_the_stroke(look_1624, reach_1623, pink):
    from builtin_interfaces.msg import Time

    s = reach_1623[-1]
    shown = {"scene": look_1624, "aim": _tick(reach_1623), "press": viz.PressPlan(3.0, _stroke(s), s.xyz[2], 0.012)}
    ma = viz.markers(shown, pink, "arm_mount", Time())
    first, rest = ma.markers[0], ma.markers[1:]
    assert first.action == first.DELETEALL and first.header.frame_id == "arm_mount"
    assert len({m.id for m in rest}) == len(rest) and all(m.header.frame_id == "arm_mount" for m in rest)
    by = {}
    for m in rest:
        by.setdefault(m.ns, []).append(m)
    fix = look_1624.fix
    lid = by["box: lid"][0]
    assert (lid.pose.position.x, lid.pose.position.y) == pytest.approx(fix.top_xyz[:2])
    assert lid.pose.position.z + 0.5 * lid.scale.z == pytest.approx(fix.top_xyz[2])  # its top is the lid's measured top
    holder = by["box: holder"][0]
    assert holder.pose.position.z - 0.5 * holder.scale.z == pytest.approx(fix.pose.xyz[2])  # it stands on the table
    assert by["table (measured)"][0].pose.position.z == pytest.approx(fix.table_z_scene)
    assert by["aim: button"][0].pose.position.x == pytest.approx(s.xyz[0])
    from box_opening.tasks.depth_press import TOUCH_SLOW_M

    down = by["press: down to the bottom"][0].points
    assert down[0].z == pytest.approx(s.xyz[2] + TOUCH_SLOW_M) and down[1].z == pytest.approx(s.xyz[2] - 0.012)
    springs = by["press: the springs' reference"][0].points
    assert (springs[1].x, springs[1].y, springs[1].z) == pytest.approx((s.xyz[0], s.xyz[1], s.xyz[2] - 0.042))
    texts = [m.text for m in rest if m.text]
    assert "SCENE: SEG 0.88, 2 sightings (segmenter+segmenter): box at [0.661, -0.265], lid top 0.310" in texts
    assert any(t.startswith("PRESS: 72 mm down") for t in texts)
    assert all(t.isascii() for t in texts)
    # a new look is a new picture: the markers of the last aim and press are gone
    only = viz.markers({"scene": look_1624}, pink, "arm_mount", Time())
    assert not [m for m in only.markers if m.ns.startswith(("aim", "press"))]


def test_a_bent_strokes_springs_reference_runs_to_where_its_joints_end_or_is_not_drawn(reach_1623, pink, monkeypatch):
    """A bent stroke's reference ends bend_m aside (StrokePlan.q_end): the
    line runs there, the pads at q_end by the run's own chain — and with no
    chain built, stops at the button's bottom, the label saying the bend."""
    from builtin_interfaces.msg import Time

    from box_opening.runtime import approach
    from box_opening.tasks.depth_press import DOWN_CONTACT_TOOL

    s = reach_1623[-1]
    end = (s.xyz[0] + 0.004, s.xyz[1], s.xyz[2] - 0.042)  # the pads' reference at the bend's end

    class Chain:  # fk: the tool at q's first three values, straight down
        def fk(self, q):
            return np.eye(3), np.asarray(q[:3], float)

    q_end = [end[0], end[1], end[2] - DOWN_CONTACT_TOOL[2], 0.0, 0.0, 0.0, 0.0]
    plan = viz.PressPlan(3.0, _stroke(s, bent=True, q_end=q_end), s.xyz[2], 0.012)

    def springs():
        ma = viz.markers({"press": plan}, pink, "arm_mount", Time())
        label = [m.text for m in ma.markers if m.ns == "press: label"][0]
        return [m.points for m in ma.markers if m.ns == "press: the springs' reference"], label

    monkeypatch.setattr(approach, "_CHAIN", Chain())
    lines, label = springs()
    assert [(p.x, p.y, p.z) for p in lines[0]] == [
        pytest.approx((s.xyz[0], s.xyz[1], s.xyz[2] - 0.012)),
        pytest.approx(end),
    ]
    assert "4 mm aside;" in label
    monkeypatch.setattr(approach, "_CHAIN", None)  # none built: none is built here
    lines, label = springs()
    assert lines == [] and "4 mm aside (bent: its end not drawn)" in label


def test_the_grip_is_drawn_down_to_under_the_lid_and_its_lift_straight_up_or_not_without_a_chain(pink, monkeypatch):
    """The lid off's grip (reach_lid_off.grip_descent): the closed pads
    from where the hand stands turned down straight down — the approach to
    GRIP_SLOW_M over where they end, then down to it, GRIP_BELOW_LID_M under
    the lid — and the lift straight up lift_off_m (plan_lift). Its x, y are
    the pads' at q0 by the run's own chain; with none built it is not drawn."""
    from builtin_interfaces.msg import Time

    from box_opening.runtime import approach
    from box_opening.tasks import reach_lid_off as rlo
    from box_opening.tasks.depth_press import DOWN_CONTACT_TOOL

    class Chain:  # fk: the tool at q's first three values, straight down
        def fk(self, q):
            return np.eye(3), np.asarray(q[:3], float)

    lid_z = 0.312
    pads0 = (0.66, -0.31, lid_z + 0.09)  # the closed pads where the hand turned down
    q0 = [pads0[0], pads0[1], pads0[2] - DOWN_CONTACT_TOOL[2], 0.0, 0.0, 0.0, 0.0]
    stroke = rlo.GripDescent(legs=[], quat=[1.0, 0.0, 0.0, 0.0], q0=q0, pad_z=lid_z - rlo.GRIP_BELOW_LID_M)
    plan = viz.GripPlan(4.0, stroke, lid_z)

    def grip():
        ma = viz.markers({"grip": plan}, pink, "arm_mount", Time())
        return {m.ns: m for m in ma.markers[1:] if m.ns.startswith("grip")}

    monkeypatch.setattr(approach, "_CHAIN", Chain())
    by = grip()
    end = lid_z - 0.002
    lines = {
        ns: [(p.x, p.y, p.z) for p in by[ns].points]
        for ns in ("grip: approach", "grip: down to the knob", "grip: the lift")
    }
    assert lines["grip: approach"] == [pytest.approx(pads0), pytest.approx((pads0[0], pads0[1], end + 0.035))]
    assert lines["grip: down to the knob"] == [
        pytest.approx((pads0[0], pads0[1], end + 0.035)),
        pytest.approx((pads0[0], pads0[1], end)),
    ]
    assert lines["grip: the lift"][1] == pytest.approx((pads0[0], pads0[1], end + rlo.lift_off_m(pink)))  # 40 mm up
    pads = by["grip: the closed pads"].pose.position
    assert (pads.x, pads.y, pads.z) == pytest.approx((pads0[0], pads0[1], end))
    assert by["grip: label"].text == (
        "GRIP: the closed pads down to 0.310, 2 mm under the lid at 0.312; then 40 mm "
        "straight up, and up and back toward the arm"
    )
    monkeypatch.setattr(approach, "_CHAIN", None)  # none built: none is built here
    assert grip() == {}


def test_the_lid_spot_is_drawn_where_it_goes_down_over_the_tops_of_what_the_cameras_saw_standing(pink, monkeypatch):
    """Where the lid goes (reach_lid_off.choose_and_carry): the lid on the
    table at the spot, its tier and why, over the map it was chosen on —
    every square something stands in a tile at its top (TableMap.standing),
    made MAP_STEP to a checkpoint. No map (no camera showed the table): the
    spot alone."""
    from builtin_interfaces.msg import Time

    from box_opening.models.container import ContainerPose
    from box_opening.perception.clear_table import TableMap

    tz = 0.213
    tall = np.full((20, 20), np.nan)
    tall[3, 4], tall[10:15, 0:10] = 0.05, 0.12  # a stray cup, and a 10 x 5 square block
    tmap = TableMap(
        x0=0.40,
        y0=-0.50,
        cell=0.02,
        tall=tall,
        seen=np.ones((20, 20), bool),
        off=np.zeros((20, 20), bool),
        box_xy=(0.66, -0.31),
    )
    drop = ContainerPose(xyz=(0.52, -0.31, tz), yaw=0.0)
    says = "where the cameras see room for it and the hand (35 mm to spare)"
    steps = []
    monkeypatch.setattr(viz, "_checkpoint", lambda: steps.append(1))

    def spot(tmap):
        steps.clear()
        ma = viz.markers({"spot": viz.SpotPick(5.0, tmap, drop, "clear", says, tz)}, pink, "arm_mount", Time())
        return {m.ns: m for m in ma.markers[1:]}, len(steps)

    by, n_steps = spot(tmap)
    lid = by["lid spot"]
    assert lid.type == lid.CYLINDER and (lid.scale.x, lid.scale.z) == pytest.approx((0.105, 0.025))
    assert (lid.pose.position.x, lid.pose.position.y, lid.pose.position.z) == pytest.approx((0.52, -0.31, tz + 0.0125))
    assert by["lid spot: label"].text == "LID SPOT (clear): [0.52, -0.31] - " + says
    tiles = by["lid spot: what the cameras saw standing"]
    assert tiles.type == tiles.CUBE_LIST and len(tiles.points) == 51  # every square that stands, once
    tops = sorted((round(p.x, 3), round(p.y, 3), round(p.z, 3)) for p in tiles.points)
    assert (0.47, -0.41, round(tz + 0.05, 3)) in tops  # the cup's square: its centre, at its top
    assert {z for _x, _y, z in tops} == {round(tz + 0.05, 3), round(tz + 0.12, 3)}
    alone, n_alone = spot(None)
    assert set(alone) == {"lid spot", "lid spot: label"}  # no map: the spot alone
    assert n_steps - n_alone == 1 + 1  # the tiles' marker, and one checkpoint per MAP_STEP squares (51: one)
    tall[:, 10:] = 0.01  # 200 squares more: made in three steps
    assert spot(tmap)[1] - n_alone == 1 + 3


# --- the worker: drawn only for a subscriber, never in the run's way ---------


def test_a_run_with_the_viz_off_never_imports_it_and_the_module_defines_no_dataclass():
    """No hook site imports viz: press_demo.viz_hook looks it up, and only
    the viz's own thread imports it, with the viz on, so a run with it off —
    the default: no $BOX_OPENING_VIZ — never loads it. Loaded, it
    defines no dataclass: four were ~3.4 ms of its import, at every launch
    (review 2026-10-28). A fresh interpreter: this one has it."""
    import ast
    import os
    import subprocess

    from box_opening.constants import VIZ_ENV

    code = "\n".join(
        (
            "import sys, threading",
            "from box_opening.tasks import depth_press, press_demo, reach_lid_off, reach_press",
            "from box_opening.perception import button_aim, scene_source",
            "assert press_demo.VIZ_MODULE not in sys.modules, 'imported at a module level'",
            "class Node:",
            "    def create_publisher(self, *a, **k):",
            "        raise AssertionError('a publisher made with the viz off')",
            "assert press_demo.start_viz(Node(), None, 'arm_mount') is None",  # the run's own call: off by default
            "assert press_demo.start_viz(Node(), None, 'arm_mount', env={'BOX_OPENING_VIZ': '0'}) is None",
            "for name in ('scene', 'aim', 'press', 'grip', 'spot'):",
            "    press_demo.viz_hook(name, None, None)",
            "press_demo.stop_viz(True)",
            "assert press_demo.VIZ_MODULE not in sys.modules, 'imported with it off'",
            "assert threading.active_count() == 1, 'a thread started with it off'",
            "assert press_demo.start_viz(object(), None, 'arm_mount', env={'BOX_OPENING_VIZ': '1'}) is None",  # on,
            "assert press_demo.VIZ_MODULE not in sys.modules, 'imported'",  # ... but nothing to publish on
            "assert threading.active_count() == 1, 'a thread started'",
        )
    )
    env = {k: v for k, v in os.environ.items() if k != VIZ_ENV}  # the robot's: no switch set
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert got.returncode == 0, got.stderr[-2000:]
    assert got.stdout == "[viz] (no Foxglove view of this run: object publishes nothing)\n"
    tree = ast.parse(Path(viz.__file__).read_text())
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        and "dataclasses" in ([getattr(n, "module", None)] + [a.name for a in n.names])
    ]


def test_the_missions_thread_only_starts_the_viz_thread_which_loads_it_once_the_mission_waits():
    """start_viz, on the mission's thread, starts the viz's thread and
    nothing more (its import and its Viz were ~1.8 ms of every launch's
    mission thread: review 2026-10-28). That thread loads viz at the first
    VIZ_BOOT_S the mission's thread — its native id, read from its
    schedstat — waits through, then runs it there; viz_hook calls in only
    once the viz runs (viz._VIZ), never into a module still loading. A
    stop_viz before that has the thread load and start nothing. A fresh
    interpreter, the mission's run time driven."""
    import subprocess

    code = "\n".join(
        (
            "import sys, threading, time",
            "from box_opening.tasks import press_demo",
            "busy, asked = [True], []",
            "def runtime(tid):",
            "    asked.append(tid)",
            "    return lambda: time.monotonic_ns() if busy[0] else 0",
            "press_demo.thread_runtime = runtime",
            "class Node:",
            "    def create_publisher(self, *a, **k):",
            "        raise AssertionError('made before the mission waited a second')",
            "press_demo.VIZ_BOOT_S = 0.02",
            "stopped = press_demo.start_viz(Node(), None, 'arm_mount', env={'BOX_OPENING_VIZ': '1'})",
            "assert stopped is not None and press_demo.VIZ_MODULE not in sys.modules",
            "time.sleep(0.2)",  # the mission's thread computes (its run time all of the time): nothing is loaded
            "assert asked == [threading.get_native_id()], asked",
            "assert press_demo.VIZ_MODULE not in sys.modules, 'loaded while the mission computed'",
            "press_demo.viz_hook('scene', None, None)",  # no viz yet: nothing
            "busy[0] = False",  # it waits
            "t0 = time.monotonic()",
            "while getattr(sys.modules.get(press_demo.VIZ_MODULE), '_VIZ', None) is None"
            " and time.monotonic() - t0 < 10:",
            "    time.sleep(0.01)",
            "viz = sys.modules[press_demo.VIZ_MODULE]",
            "v = viz._VIZ",
            "assert v is not None and v.thread.name == 'viz' and v.thread is not threading.current_thread()",
            "assert v._ran() == 0 and v.pubs is None",  # its gate reads the mission's thread; no publisher yet
            "press_demo.stop_viz()",
            "assert viz._VIZ is None and v.done.is_set() and press_demo._VIZ_STOPPED is None",
            # stopped before its thread loaded the viz: it never does
            "del sys.modules[press_demo.VIZ_MODULE], sys.modules['box_opening.perception'].viz",  # as if never loaded
            "busy[0] = True",
            "press_demo.start_viz(Node(), None, 'arm_mount', env={'BOX_OPENING_VIZ': 'on'})",
            "press_demo.stop_viz()",
            "busy[0] = False",
            "time.sleep(0.2)",
            "assert press_demo.VIZ_MODULE not in sys.modules, 'loaded after its stop'",
            "assert threading.active_count() == 1, threading.enumerate()",
        )
    )
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr[-2000:]
    assert got.stdout == ""
    from box_opening.tasks import press_demo

    assert (press_demo.VIZ_BOOT_S, press_demo.VIZ_BOOT_RUN_FRAC) == (viz.PERIOD_S, viz.IDLE_RUN_FRAC)  # viz's own gate


def test_a_hook_never_calls_into_a_viz_still_loading_or_not_running(monkeypatch):
    from types import ModuleType

    from box_opening.tasks import press_demo

    called = []
    loading = ModuleType(press_demo.VIZ_MODULE)  # in sys.modules as its import runs: nothing defined yet
    monkeypatch.setitem(sys.modules, press_demo.VIZ_MODULE, loading)
    press_demo.viz_hook("scene", 1, 2)
    loading._VIZ = None  # defined, not running
    loading.scene = lambda *a: called.append(a)
    press_demo.viz_hook("scene", 1, 2)
    assert called == []
    loading._VIZ = object()  # running: set last
    press_demo.viz_hook("scene", 1, 2)
    assert called == [(1, 2)]


def test_nothing_is_drawn_or_encoded_for_an_image_nobody_subscribes_to(look_1624, reach_1623, pink, monkeypatch):
    """The latched markers and status go out whoever listens: they are the
    late Foxglove's picture. An image is drawn and encoded only once someone
    subscribes, then once — again only when its caption comes."""
    calls = []
    for name in ("scene_image", "wrist_image", "jpeg"):
        real = getattr(viz, name)
        monkeypatch.setattr(viz, name, lambda *a, _real=real, _name=name: calls.append(_name) or _real(*a))
    look = look_1624.replace()  # its own: the worker captions it
    now = [5.0]
    monkeypatch.setattr(viz, "_clock", lambda: now[0])
    v, pubs = _viz(pink)
    v.put("scene", look)
    v.put("wrist", _tick(reach_1623, t=5.0))
    v.step()
    assert calls == [] and pubs["scene"].sent == [] and pubs["wrist"].sent == []
    assert len(pubs["markers"].sent) == 1  # latched: the box and the aim's hit, for whoever connects
    v.step()
    assert calls == []  # kept, not drawn
    pubs["scene"].subscribers = 1  # Foxglove opens the annotated scene
    v.step()
    assert calls == ["scene_image", "jpeg"] and len(pubs["scene"].sent) == 1
    msg = pubs["scene"].sent[0]
    assert msg.format == "jpeg" and bytes(msg.data[:2]) == b"\xff\xd8"
    v.step()
    assert len(calls) == 2  # drawn once
    now[0] = 6.0
    v.said_line("[press_demo] SCENE: box at [0.661, -0.265] yaw 0.0 deg from 299 lid points — reaching out")
    v.said_line("[press_demo] the driver enters its joint impedance now, while the reach is planned")  # not a phase
    v.step()
    assert look.caption.startswith("SCENE: box at [0.661, -0.265] yaw 0.0 deg")
    assert calls[2:] == ["scene_image", "jpeg"] and len(pubs["scene"].sent) == 2  # drawn again, captioned
    assert [m.data for m in pubs["status"].sent] == [look.caption]
    assert len(pubs["markers"].sent) == 1  # what they show did not change


def test_after_a_press_the_knob_looks_verdict_captions_its_wrist_frame(reach_1623, pink, monkeypatch):
    """The knob look after a press is the aim again (reach_press._aim): its
    AIM line captions its frame, then the AFTER PRESS line its verdict —
    said after the frame, it replaces the AIM line there — as ALREADY OPEN
    does the first aim's frame of a box found open. The lid off's
    lower-case decisions, the press once more and where the lid goes, reach
    the status."""
    now = [5.0]
    monkeypatch.setattr(viz, "_clock", lambda: now[0])
    v, pubs = _viz(pink, wrist=1)
    aim = "AIM: button circle at pixel (292, 239) r 31 px, 50 mm across, +9.0 mm above its own lid; pressing its centre"
    already = (
        "ALREADY OPEN: the knob stands 9 mm above its lid — not pressing it, which would shut the box; taking "
        "the lid off"
    )
    after = (
        "AFTER PRESS 1: knob UP 9.0 mm — the box is open (1 look); the button 3 mm from where the press was " "aimed"
    )
    for verdict in (already, after):  # the first aim's frame, then a knob look's
        tick = _tick(reach_1623, t=now[0])
        v.put("wrist", tick)
        for line in (aim, verdict):
            now[0] += 0.5
            v.said_line("[press_demo] " + line)
            v.step()
            assert tick.caption == line and pubs["status"].sent[-1].data == line
    assert len(pubs["wrist"].sent) == 4  # each frame drawn with each line
    for line in (
        "the knob is not up (0.9 mm above the lid): pressing once more — the same press, from the look pose",
        "lid goes to [0.52, -0.31, 0.213] — 180 mm from the box, beside the box",
    ):
        v.said_line("[press_demo] " + line)
        v.step()
        assert pubs["status"].sent[-1].data == line and tick.caption == after  # no wrist caption: no new frame
    assert len(pubs["wrist"].sent) == 4


def test_a_viz_failure_never_reaches_the_mission_and_is_said_once(pink, capsys, monkeypatch):
    v, pubs = _viz(pink, scene=1)
    monkeypatch.setattr(viz, "_VIZ", v)
    viz.scene(object(), None)  # no grabber: the hand-over fails
    viz.scene(object(), None)
    viz.aim(object(), True)
    viz.press(None, "not a height", None)
    viz.grip(None, "not a height")
    viz.spot(None, None, "clear", "room", "not a table")
    viz.spot(None, None, "clear", "room", "not a table")
    out = capsys.readouterr().out
    assert out.count("[viz] (the annotated scene failed") == 1
    assert "[viz] (the annotated wrist frame failed" in out and "[viz] (the stroke's markers failed" in out
    assert "[viz] (the grip's markers failed" in out and out.count("[viz] (the lid spot's markers failed") == 1
    # on the worker's side: a look it cannot draw, a publisher that raises
    pubs["markers"].broken = True
    v.put(
        "scene",
        viz.SceneLook(
            t=0.0, fix=None, why=None, judged={"owl": "not a judged message"}, boxes={}, color=None, k=None, dist=None
        ),
    )
    v.step()
    v.step()
    out = capsys.readouterr().out
    assert out.count("[viz] (the markers failed") == 1
    v.close()
    viz.scene(object(), None)  # closed: quiet
    assert capsys.readouterr().out == ""


def test_the_console_goes_on_as_it_was_and_its_phase_lines_reach_the_status(pink, capsys):
    v, pubs = _viz(pink)
    v.mirror()
    try:
        print("[press_demo] REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm")
        sys.stdout.write("[press_demo] AIMED from the look pose: ")
        sys.stdout.write("button at [0.720, -0.297, 0.310], 12 mm from the scene camera's\n[runner] a")
        print("nother line", flush=True)
    finally:
        v.close()
    assert sys.stdout is not v._mirror
    assert capsys.readouterr().out == (
        "[press_demo] REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm\n"
        "[press_demo] AIMED from the look pose: button at [0.720, -0.297, 0.310], 12 mm from the scene camera's\n"
        "[runner] another line\n"
    )
    v.closed = False
    v.step()
    assert [m.data for m in pubs["status"].sent] == [
        "AIMED from the look pose: button at [0.720, -0.297, 0.310], 12 mm from the scene camera's"
    ]


def test_stop_publishes_what_the_run_handed_over_last_though_the_mission_never_waited(look_1624, pink, monkeypatch):
    """A NO BOX run ends right after its look: try_home, sys.exit, stop().
    The gate never opened (the mission's thread ran all the time), yet the
    look, its markers and the NO BOX line go out before the worker ends —
    ungated, the run being over — and well inside CLOSE_WAIT_S."""
    import time

    monkeypatch.setattr(viz, "PERIOD_S", 0.02)
    pubs = {k: _Pub(1) for k in ("scene", "wrist", "markers", "status")}
    v = viz.Viz(None, pink, "arm_mount", pubs=pubs, ran=lambda: time.monotonic_ns())  # it ran 100 % of the time
    v.put("scene", look_1624.replace(fix=None, why="scene segmenter: one of its boxes passed in 1 frames"))
    v.said_line("[press_demo] NO BOX — scene camera: scene segmenter: one of its boxes passed — returning home")
    time.sleep(0.1)  # five wakes, every one refused by the gate
    assert not any(p.sent for p in pubs.values())
    t0 = time.monotonic()
    v.close()
    assert time.monotonic() - t0 < viz.CLOSE_WAIT_S and v.done.is_set()
    assert [m.data for m in pubs["status"].sent] == [
        "NO BOX — scene camera: scene segmenter: one of its boxes passed — returning home"
    ]
    assert len(pubs["markers"].sent) == 1 and len(pubs["scene"].sent) == 1 and viz._checkpoint is None
    # with no publishers made yet (no idle wake), nothing is made at the end: they would go with the node
    made = []
    w = viz.Viz(
        SimpleNamespace(create_publisher=lambda *a, **k: made.append(a)),
        pink,
        "arm_mount",
        ran=lambda: time.monotonic_ns(),
    )
    w.put("scene", look_1624)
    w.close()
    assert made == [] and w.done.is_set()


def test_the_publishers_add_no_waitable_to_the_executor_that_spins_their_node(pink):
    """press_demo's perception executor spins the node the publishers live
    on: rclpy's default QoS event handler would be one waitable each,
    waited on at every wake of the run."""
    from rclpy.qos import DurabilityPolicy

    made = []

    class Node:
        def create_publisher(self, msg_type, topic, qos, **kw):
            made.append((topic, qos, kw))
            return _Pub()

    v = viz.Viz(Node(), pink, "arm_mount", thread=False, ran=None)
    for _step in v._making():
        pass
    pubs = v.pubs
    assert set(pubs) == {"scene", "wrist", "markers", "status"}
    assert sorted(t for t, _q, _k in made) == sorted(
        [viz.SCENE_TOPIC, viz.WRIST_TOPIC, viz.MARKER_TOPIC, viz.STATUS_TOPIC]
    )
    for _t, qos, kw in made:
        ev = kw["event_callbacks"]
        assert ev.use_default_callbacks is False and not (ev.deadline or ev.liveliness or ev.incompatible_qos)
        assert qos.depth == 1 and qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_the_publishers_are_made_a_step_at_a_time_each_behind_a_checkpoint_while_the_mission_waits(pink, monkeypatch):
    """The marker type's import (~27 ms: /opt/ros's modules compile from
    source), the marker builder's module, the markers' C side, then each
    publisher: its own step each, a checkpoint between, as drawing has; the
    mission's thread computing again (Busy) leaves the rest for its next
    wait — and the worker makes them only at a wake the gate opens."""
    import time

    log = []

    class Node:
        def create_publisher(self, msg_type, topic, qos, **kw):
            log.append(topic)
            return _Pub()

    v = viz.Viz(Node(), pink, "arm_mount", thread=False, ran=None)
    busy_at = [2]

    def check():
        log.append("check")
        if log.count("check") == busy_at[0]:
            raise viz.Busy()

    monkeypatch.setattr(v, "_check", check)
    making = v._making()
    assert v._make(making) and v.pubs is None and log == ["check", "check"]  # two steps, then Busy at the second
    busy_at[0] = 5
    assert v._make(making) and v.pubs is None
    assert log == ["check", "check", "check", viz.SCENE_TOPIC, "check", viz.WRIST_TOPIC, "check"]
    busy_at[0] = 0
    assert v._make(making) and viz._checkpoint is None
    assert log[7:] == [viz.MARKER_TOPIC, "check", viz.STATUS_TOPIC, "check"]
    assert set(v.pubs) == {"scene", "wrist", "markers", "status"}
    # a step that fails is said once, and the worker goes on without a viz
    broken = viz.Viz(SimpleNamespace(create_publisher=lambda *a, **k: 1 / 0), pink, "arm_mount", thread=False, ran=None)
    assert broken._make(broken._making()) is False and broken.pubs is None
    # on its thread, only at a wake the gate opens: the mission's thread running all the time, none is made
    monkeypatch.setattr(viz, "PERIOD_S", 0.01)
    made = []
    w = viz.Viz(
        SimpleNamespace(create_publisher=lambda *a, **k: made.append(a)),
        pink,
        "arm_mount",
        ran=lambda: time.monotonic_ns(),
    )
    time.sleep(0.08)
    w.close()
    assert made == [] and w.done.is_set()


def test_the_marker_types_import_waits_for_the_mission_to_have_waited_a_whole_second(pink, monkeypatch):
    """The import is one step no checkpoint breaks into (each module's
    compile 1-4.5 ms of C holding the GIL): it waits for every wake of
    IMPORT_QUIET_S to have found the mission's thread all but still (under
    IMPORT_RUN_FRAC) — a compute between starts the count again, a short
    one the gate lets by (20 %) included — not just one wake."""
    import threading
    import time

    now = {"t": 0.0, "ran": 0}
    monkeypatch.setattr(viz, "_clock", lambda: now["t"])
    v = viz.Viz(None, pink, "arm_mount", thread=False, ran=lambda: now["ran"])
    v._idle()  # the first window
    for i in (1, 2, 3):
        now["t"] = 0.25 * i
        assert v._idle() and not v._quiet()  # 0.75 s of waits
    now["t"] = 1.0
    assert v._idle() and v._quiet()  # a whole second of them
    now["t"], now["ran"] = 1.25, int(0.2e9)  # it computed: 80 %
    assert not v._idle() and not v._quiet()
    for i in (1, 2, 3):
        now["t"] = 1.25 + 0.25 * i
        assert v._idle() and not v._quiet()  # the count starts again
    now["t"], now["ran"] = 2.25, now["ran"] + int(0.03e9)  # a 30 ms compute in the window: 12 %
    assert v._idle() and not v._quiet()  # the gate draws here, but the import waits again
    for i in (1, 2, 3):
        now["t"] = 2.25 + 0.25 * i
        assert v._idle() and not v._quiet()
    now["t"] = 3.25
    assert v._idle() and v._quiet()  # a whole second since
    monkeypatch.undo()
    # on its thread: the import begins only then, though the gate opens at every other wake meanwhile
    monkeypatch.setattr(viz, "PERIOD_S", 0.01)
    monkeypatch.setattr(viz, "IMPORT_QUIET_S", 0.1)
    for mission, begins in (("waits", True), ("computes every other wake", False)):
        reads = [0]

        def ran(mission=mission, reads=reads):
            reads[0] += 1
            return 0 if mission == "waits" else (reads[0] // 2) * 10**9  # 1 s of running at every other wake

        w = viz.Viz(SimpleNamespace(create_publisher=lambda *a, **k: _Pub()), pink, "arm_mount", thread=False, ran=ran)
        began, making = [], w._making
        w._making = lambda: began.append(time.monotonic()) or making()  # noqa: B023 - this iteration's
        w.thread = threading.Thread(target=w._run, daemon=True)
        t0 = time.monotonic()
        w.thread.start()
        while not (began if begins else reads[0] > 30) and time.monotonic() - t0 < 10.0:
            time.sleep(0.005)
        w.close()
        assert w.done.is_set()
        if begins:
            assert began and began[0] - t0 >= 0.1, (began, t0)
        else:
            assert began == [] and reads[0] > 30, reads  # 15 wakes the gate opened at: none began it


def test_it_is_off_unless_turned_on_and_off_it_hands_nothing_over(pink, monkeypatch):
    """Off by default (constants.viz_enabled): a run nobody watches in
    Foxglove starts no viz. 1, on, true or yes turns it on; anything else,
    a typo included, leaves it off."""
    from box_opening.constants import VIZ_ENV

    assert not viz.enabled({}) and not viz.enabled({VIZ_ENV: ""})
    assert all(viz.enabled({VIZ_ENV: w}) for w in ("1", "on", "True", " yes "))
    assert not any(viz.enabled({VIZ_ENV: w}) for w in ("0", "off", "False", " no ", "2", "onn"))
    monkeypatch.setattr(viz, "_VIZ", None)
    for env in ({}, {VIZ_ENV: "0"}):
        assert viz.start(object(), pink, "arm_mount", env=env) is None and viz._VIZ is None
    viz.scene(object(), None)  # every entry point a no-op
    viz.aim(object(), True)
    viz.press(object(), 0.3, object())
    viz.grip(object(), 0.3)
    viz.spot(None, object(), "clear", "room", 0.2)
    viz.stop()


def test_a_node_that_cannot_publish_starts_no_viz_and_says_so_once(pink, monkeypatch, capsys):
    monkeypatch.setattr(viz, "_VIZ", None)
    stdout = sys.stdout
    assert viz.start(object(), pink, "arm_mount", env={"BOX_OPENING_VIZ": "1"}) is None
    assert sys.stdout is stdout and viz._VIZ is None
    assert capsys.readouterr().out == "[viz] (no Foxglove view of this run: object publishes nothing)\n"


def test_the_worker_works_only_while_the_mission_waits_and_leaves_the_rest_when_it_computes(
    look_1624, pink, monkeypatch
):
    """The worker shares the mission's GIL: it works when the mission's
    thread ran under IDLE_RUN_FRAC since its last wake — waiting on the
    driver, a frame, a plan — and stops at its next step aside once that
    thread computes again, what is left waiting for the next wait."""
    now = {"t": 0.0, "ran": 0, "busy": None}

    def clock():
        if now["busy"] is not None:  # inside a batch: time goes on, and the mission's thread runs that share of it
            now["t"] += 0.002
            now["ran"] += int(now["busy"] * 0.002e9)
        return now["t"]

    monkeypatch.setattr(viz, "_clock", clock)
    pubs = {k: _Pub(1) for k in ("scene", "wrist", "markers", "status")}
    v = viz.Viz(None, pink, "arm_mount", pubs=pubs, thread=False, ran=lambda: now["ran"])
    assert v._idle() is False  # no window yet
    now["t"], now["ran"] = 0.25, int(0.20e9)  # it aimed through the window: 80 %
    assert v._idle() is False
    now["t"], now["ran"] = 0.50, int(0.21e9)  # a motion: 4 %, its arm's state spun
    assert v._idle() is True
    v.put("scene", look_1624.replace())
    now["busy"] = 0.9  # ... and it computes again as the batch begins
    v.step()
    assert pubs["markers"].sent == [] and pubs["scene"].sent == [] and v._markers_due
    assert viz._checkpoint is None
    now["busy"] = 0.03
    v.step()  # the next wait: all of it
    assert len(pubs["markers"].sent) == 1 and len(pubs["scene"].sent) == 1 and not v._markers_due


def test_a_second_ctrl_c_in_stops_wait_leaves_the_worker_to_end_and_waits_for_it_through_a_third(
    look_1624, pink, monkeypatch, capsys
):
    """stop() waits for the worker's last batch; a second Ctrl+C there
    (abort.py: KeyboardInterrupt regardless), real SIGINTs while the worker
    is inside OpenCV: it must not be left there for the interpreter's
    teardown (a daemon thread coming back out of OpenCV then aborts the
    process, exit 134), nor reach press_demo's finally. stop has the worker
    quit at its next step and waits for its end through a third Ctrl+C,
    nothing more published, then returns — the perception executor's
    shutdown comes next. Its end is the worker's own (done): a join that a
    KeyboardInterrupt cut short marks the live thread stopped on 3.10
    (bpo-45274), so is_alive() and later joins lie."""
    import os
    import signal
    import threading
    import time

    import cv2

    from box_opening.runtime.abort import AbortFlag, sigint_handler

    monkeypatch.setattr(viz, "PERIOD_S", 0.01)
    monkeypatch.setattr(viz, "QUIT_WAIT_S", 10.0)  # however slow this machine's blur: the wait outlasts it
    entered, spans = threading.Event(), []
    big = np.zeros((2000, 2000, 3), np.uint8)
    real = viz.scene_image

    def slow(look, model):  # the last batch, inside OpenCV (the GIL let go) when the Ctrl+Cs come
        t0 = time.monotonic()
        entered.set()
        cv2.GaussianBlur(big, (51, 51), 0)
        spans.append((t0, time.monotonic()))
        viz._yield()
        return real(look, model)

    monkeypatch.setattr(viz, "scene_image", slow)
    pubs = {k: _Pub(1) for k in ("scene", "wrist", "markers", "status")}
    v = viz.Viz(None, pink, "arm_mount", pubs=pubs, ran=lambda: time.monotonic_ns())  # the gate never opens
    v.mirror()
    monkeypatch.setattr(viz, "_VIZ", v)
    v.put("scene", look_1624.replace())
    flag, sigs, armed = AbortFlag(), [], [True]
    flag.requested = True  # after a first Ctrl+C: the next raises regardless
    handler = sigint_handler(flag)

    def on_sigint(signum, frame):
        sigs.append(time.monotonic())
        if armed[0]:
            handler(signum, frame)

    def ctrl_c():
        assert entered.wait(10.0)
        os.kill(os.getpid(), signal.SIGINT)
        t0 = time.monotonic()
        while not v.ctrl_c and time.monotonic() - t0 < 5.0:  # the second came into the last batch's wait
            time.sleep(0.0005)
        os.kill(os.getpid(), signal.SIGINT)  # ... and a third into the wait for the worker's end

    old = signal.signal(signal.SIGINT, on_sigint)
    kill = threading.Thread(target=ctrl_c, daemon=True)
    try:
        kill.start()
        viz.stop()  # returns: no Ctrl+C is raised into press_demo's finally
        returned = time.monotonic()
        armed[0] = False
        kill.join(10.0)
    finally:
        armed[0] = False
        signal.signal(signal.SIGINT, old)
    assert v.done.is_set() and v.quit and v.ctrl_c  # the worker's own end
    assert len(sigs) == 2 and spans and spans[0][0] < sigs[0] < sigs[1] < spans[0][1] <= returned, (sigs, spans)
    assert pubs["scene"].sent == [] and viz._checkpoint is None and viz._VIZ is None
    assert sys.stdout is not v._mirror
    assert "[viz] (a Ctrl+C: what was left is not published)" in capsys.readouterr().out


def test_a_run_unwinding_from_a_ctrl_c_publishes_nothing_more(look_1624, pink, monkeypatch):
    import time

    monkeypatch.setattr(viz, "PERIOD_S", 0.01)
    pubs = {k: _Pub(1) for k in ("scene", "wrist", "markers", "status")}
    v = viz.Viz(None, pink, "arm_mount", pubs=pubs, ran=lambda: time.monotonic_ns())  # the gate never opens
    v.put("scene", look_1624.replace())
    v.said_line("[press_demo] STOP: interrupted")
    t0 = time.monotonic()
    v.close(last=False)
    assert time.monotonic() - t0 < viz.CLOSE_WAIT_S and v.done.is_set()  # not the last batch's wait
    assert not any(p.sent for p in pubs.values())


def test_a_threads_run_time_is_read_from_the_kernel():
    import threading
    import time

    from box_opening.tasks.press_demo import thread_runtime

    ran = thread_runtime(threading.get_native_id())
    if ran is None:
        pytest.skip("this kernel keeps no schedstat")
    a = ran()
    sum(i * i for i in range(200000))
    assert ran() - a > 1e6  # over a millisecond of this thread's own running
    # ... read from another thread, as the viz's thread reads the mission's: this one, in a join, barely runs
    got = []
    other = threading.Thread(target=lambda: (got.append(ran()), time.sleep(0.05), got.append(ran())))
    other.start()
    other.join()
    assert got[1] - got[0] < 5e6
    assert thread_runtime(2**30) is None  # no such thread


def test_the_foxglove_layout_parses_and_shows_every_viz_topic_beside_the_cameras():
    import json

    root = Path(__file__).resolve().parents[1]  # the capability
    doc = json.loads((root / "docs" / "foxglove" / "box_opening_perception.json").read_text())
    assert {"configById", "layout", "globalVariables", "userNodes", "playbackConfig"} <= set(doc)

    def leaves(n):
        return [n] if isinstance(n, str) else leaves(n["first"]) + leaves(n["second"])

    assert sorted(leaves(doc["layout"])) == sorted(doc["configById"])  # every panel laid out, nothing laid out twice
    panels = doc["configById"].values()
    images = {p["imageMode"]["imageTopic"] for p in panels if p.get("imageMode", {}).get("imageTopic")}
    assert images == {
        viz.SCENE_TOPIC,
        viz.WRIST_TOPIC,
        "/scene_camera/color/image_raw/compressed",
        "/perception/scene_segmenter/overlay/compressed",
    }
    # the wrist's view is box opening's own: flatbot's wrist driver publishes no image_raw/compressed (live graph,
    # 2026-10-28), and a panel on its raw ~23 MB/s would make the bridge read it on the mission's Jetson
    assert "/wrist_camera/color/image_raw" not in json.dumps(doc)
    three_d = doc["configById"]["3D!boxopening"]
    assert three_d["topics"][viz.MARKER_TOPIC]["visible"] and three_d["followTf"] == "arm_mount"
    urdf = [layer for layer in three_d["layers"].values() if layer["layerId"] == "foxglove.Urdf"]
    assert [(u["sourceType"], u["topic"]) for u in urdf] == [("topic", "/arm/robot_description")]
    assert any(p.get("topicPath") == viz.STATUS_TOPIC + ".data" for p in panels)


def test_the_tilted_aim_hands_its_frames_and_its_fix_over_as_press_demo_runs_it(reach_1623, pink, monkeypatch, capsys):
    """press_demo.aim_button_at_staging with the real ButtonAimer on the
    16:23 frame, a viz started: the aim and the console as they were, the
    fix's tick handed over with its aimer, and drawn — its button in the
    markers where the aim put the press."""
    from box_opening.perception import depth_source
    from box_opening.tasks import press_demo

    color, k, dist, cam, lid_z, prior, _s = reach_1623
    f = np.load(DATA / F_1623)

    class Grab:  # a new still frame at every read, as the wrist OAK gives one every 67 ms
        def __init__(self):
            self.color, self.k, self.dist, self.n = color, k, dist, 0
            self.depth = f["depth_mm"].astype(np.float32) / 1000.0

        @property
        def color_stamp(self):
            self.n += 1
            return SimpleNamespace(sec=self.n, nanosec=0)

    monkeypatch.setattr(depth_source, "camera_pose_at", lambda g, offset_s=None, stamp=None: cam)
    monkeypatch.setattr(press_demo, "_spin_detect", lambda node: None)
    monkeypatch.setattr(press_demo, "save_aim_frame", lambda grab, s, tag: None)
    v, pubs = _viz(pink, wrist=1)
    monkeypatch.setattr(viz, "_VIZ", v)
    w = SimpleNamespace(grab=Grab(), model=pink, table_z=float(f["table_z"]))
    out = {}
    got, status = press_demo.aim_button_at_staging(
        None, w, 0.0, timeout_s=10.0, tag="reach-aim", out=out, lid_z=lid_z, prior_xy=prior, min_hits=2
    )
    assert got is not None, status
    said = capsys.readouterr().out
    assert "AIM: button circle at pixel" in said and "[viz]" not in said
    assert set(out) == {"button_up_mm", "measured_xyz"}
    tick = v._slots["wrist"]
    assert tick.done is not None and tick.hit and tick.hits == 2 and tick.color is color
    assert tick.prior_said == viz.SCENE_PRIOR  # the first aim's: the scene camera's button
    v.step()
    assert len(pubs["wrist"].sent) == 1 and pubs["wrist"].sent[0].format == "jpeg"
    button = [m for m in pubs["markers"].sent[-1].markers if m.ns == "aim: button" and m.type == m.CYLINDER][0]
    assert (button.pose.position.x, button.pose.position.y, button.pose.position.z) == pytest.approx(got[0])
    # the knob look after a press is the same aim, its tag saying whose button it was given (reach_press._aim)
    again, status = press_demo.aim_button_at_staging(
        None, w, 0.0, timeout_s=10.0, tag="reach-after", lid_z=lid_z, prior_xy=prior, min_hits=2
    )
    assert again is not None, status
    assert v._slots["wrist"].prior_said == viz.PRIOR_WORDS["reach-after"]


def test_every_look_of_the_detect_only_report_and_the_stroke_are_handed_over(
    look_1624, reach_1623, pink, monkeypatch, capsys
):
    import time

    from box_opening.tasks import press_demo

    v, _pubs = _viz(pink)
    monkeypatch.setattr(viz, "_VIZ", v)
    looks = []

    def locate(timeout):
        time.sleep(0.004)
        looks.append(timeout)
        return look_1624.fix

    loc = SimpleNamespace(
        last_why=None,
        last_judged=look_1624.judged,
        last_boxes=look_1624.boxes,
        locate=locate,
        grab=SimpleNamespace(color=look_1624.color, k=look_1624.k, dist=look_1624.dist),
    )
    assert press_demo.scene_detect_report(loc, duration_s=0.01) == len(looks) > 0
    assert "[press_demo]   SCENE: box at [0.661, -0.265]" in capsys.readouterr().out
    look = v._slots["scene"]
    assert look.fix is look_1624.fix and look.judged == look_1624.judged and look.judged is not loc.last_judged
    stroke = _stroke(reach_1623[-1])
    viz.press(stroke, 0.3101, SimpleNamespace(button_travel_m=0.012))
    plan = v._slots["press"]
    assert plan.stroke is stroke and (plan.top_z, plan.travel_m) == (0.3101, 0.012)


def test_the_full_run_hands_over_each_press_the_grip_and_the_lid_spot_as_reach_lid_off_flies_them(
    ctx, pink, monkeypatch, tmp_path, capsys
):
    """run_reach_press as the full run (test_reach_lid_off's harness), a viz
    running: the press and the press once more (reach_lid_off's own PRESS
    lines), the grip's way down and the lid's spot are handed over as each
    is said, the very objects the run flies — and drawn: the stroke, the
    grip and the lid where it went down."""
    import test_reach_lid_off as trlo

    from box_opening.runtime import approach
    from box_opening.tasks import reach_lid_off as rlo

    v, pubs = _viz(pink)
    monkeypatch.setattr(viz, "_VIZ", v)
    handed = []
    put = v.put
    monkeypatch.setattr(v, "put", lambda kind, item: (handed.append((kind, item)), put(kind, item)))
    aims = [(trlo.BUTTON, 0.4), (trlo.BUTTON, 2.0), (trlo.BUTTON, 1.5), (trlo.UP, 8.0)]  # not up, pressed again, up
    code, _rec, chain, _rows = trlo._full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    assert [kind for kind, _item in handed] == ["press", "press", "grip", "spot"]
    (_k1, first), (_k2, again), (_k3, grip), (_k4, spot) = handed
    for press in (first, again):
        assert press.top_z == pytest.approx(trlo.BUTTON[2]) and press.travel_m == ctx.press_cfg.button_travel_m
    assert again.stroke is not first.stroke  # the press once more's own way in, solved again
    # the grip goes by the lid GRIP_LID_ABOVE_HELD_M above where the hold held the pads, 6 mm under
    # the button's top (test_reach_lid_off's full flow)
    lid = trlo.BUTTON[2] - 0.006 + rlo.GRIP_LID_ABOVE_HELD_M
    assert isinstance(grip.stroke, rlo.GripDescent) and grip.lid_z == pytest.approx(lid)
    assert grip.stroke.pad_z == pytest.approx(grip.lid_z - rlo.GRIP_BELOW_LID_M)
    assert spot.drop is ctx.lid_drop and spot.tier == "beside the box" and spot.tmap is None  # no camera data
    assert "[viz]" not in capsys.readouterr().out
    monkeypatch.setattr(approach, "_CHAIN", chain)  # the run's chain, as approach._chain() built it
    v.step()
    shown = {m.ns: m for m in pubs["markers"].sent[-1].markers[1:]}
    assert {"press: label", "grip: down to the knob", "grip: the lift", "lid spot", "lid spot: label"} <= set(shown)
    assert shown["grip: the closed pads"].pose.position.z == pytest.approx(grip.stroke.pad_z)
    lid = shown["lid spot"].pose.position
    assert (lid.x, lid.y) == pytest.approx(ctx.lid_drop.xyz[:2])
    assert shown["lid spot: label"].text.startswith("LID SPOT (beside the box): ")

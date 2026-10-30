"""Preflight's camera and detector checks: the streams the mission reads,
judged before the arm moves rather than discovered mid-run."""


def test_preflight_checks_every_stream_the_mission_reads():
    """Bench 2026-09-22/23: preflight passed, and the run then failed at the
    wrist with "ALIGNED depth (align_depth.enable:=true?)" — a manifest
    update had dropped the flag and nothing checked the camera streams."""
    from box_opening.perception.scene import COLOR_TOPIC, DEPTH_TOPIC
    from box_opening.tasks.preflight import camera_topics

    topics = {t: (required, hint) for t, required, hint in camera_topics()}
    aligned = "/wrist_camera/aligned_depth_to_color/image_raw"
    assert aligned in topics and topics[aligned][0]  # the mission cannot aim without it
    assert "align_depth.enable:=true" in topics[aligned][1]
    for t in ("/wrist_camera/color/image_raw", "/wrist_camera/color/camera_info"):
        assert topics[t][0]
    for t in (COLOR_TOPIC, DEPTH_TOPIC):  # the scene camera: the mission falls back to the wrist search
        assert t in topics and not topics[t][0]


def test_stream_verdicts_fail_a_required_stream_and_warn_an_optional_one():
    from box_opening.tasks.preflight import stream_verdicts

    topics = [("/a", True, "fix a"), ("/b", False, "fix b"), ("/c", True, "fix c")]
    got = stream_verdicts(topics, {"/a": 0, "/b": 0, "/c": 12}, window_s=3.0)
    levels = {t: (lvl, text) for t, lvl, text in got}
    assert levels["/a"][0] == "FAIL" and "fix a" in levels["/a"][1]
    assert levels["/b"][0] == "WARN" and "fix b" in levels["/b"][1]
    assert levels["/c"][0] == "PASS" and "4.0 Hz" in levels["/c"][1]


def test_a_deaf_detector_is_told_apart_from_a_missing_one():
    """The wrist owl_detector ran 400 s without one frame (2026-09-22) and
    the mission called it "not running". Preflight reads the heartbeats:
    none -> not running; only BLIND -> up but deaf (restart box_opening);
    anything else -> fine."""
    from box_opening.perception.owl_source import BLIND, HEARTBEAT
    from box_opening.tasks.preflight import owl_verdict

    lvl, text = owl_verdict([], "wrist")
    assert lvl == "WARN" and "not running" in text
    lvl, text = owl_verdict([BLIND] * 6, "wrist")
    assert lvl == "WARN" and "no wrist frame" in text and "sheppy restart box_opening" in text
    assert owl_verdict([BLIND, BLIND, HEARTBEAT], "scene")[0] == "PASS"  # frames came back
    assert owl_verdict([HEARTBEAT] * 6, "scene")[0] == "PASS"
    assert owl_verdict([0.57], "scene")[0] == "PASS"  # a box: very much alive


FLATBOT = "capabilities/box_opening/config/world_flatbot.yaml"
BENCH = "capabilities/box_opening/config/world_bench.yaml"


def test_preflight_checks_the_wrist_camera_tf_press_demo_refuses_on(tmp_path):
    """Preflight passed every check while press_demo --execute refused with
    'no TF from arm_mount to wrist_camera_color_optical_frame': it looked
    up only the finger tips. On flatbot TF also places the scene camera."""
    from box_opening.tasks.preflight import tf_checks
    from box_opening.worlds import WorldStore

    got = {(t, s): req for t, s, req, _fix in tf_checks(WorldStore(FLATBOT, out_dir=tmp_path))}
    assert got[("arm_mount", "wrist_camera_color_optical_frame")] is True
    assert got[("arm_mount", "scene_camera_color_optical_frame")] is False  # camera_oak_scene.yaml link_frame
    assert len(got) == 2


def test_the_bench_checks_only_the_wrist_tf(tmp_path):
    """The bench's scene camera is placed by its calibration, not TF."""
    from box_opening.tasks.preflight import tf_checks
    from box_opening.worlds import WorldStore

    got = [(t, s, req) for t, s, req, _fix in tf_checks(WorldStore(BENCH, out_dir=tmp_path))]
    assert got == [("base_link", "end_effector_link", True)]


def test_the_required_tf_is_the_one_press_demo_refuses_on(tmp_path):
    """Parity with press_demo.readiness_refusal's lookup, so neither drifts."""
    from box_opening.perception.d405 import CAMERA_CONFIG, camera_config
    from box_opening.tasks.preflight import tf_checks
    from box_opening.worlds import WorldStore

    for world in (FLATBOT, BENCH):
        store = WorldStore(world, out_dir=tmp_path)
        required = [(t, s) for t, s, req, _f in tf_checks(store) if req]
        assert required == [
            (store.base_frame, camera_config(store.wrist_camera_config or CAMERA_CONFIG)["parent_frame"])
        ]


def test_tf_and_other_client_verdicts():
    from box_opening.tasks.preflight import other_clients_verdict, tf_verdicts

    checks = [("arm_mount", "wrist", True, "fix w"), ("arm_mount", "scene", False, "fix s")]
    assert [lvl for lvl, _t in tf_verdicts(checks, [True, True])] == ["PASS", "PASS"]
    (lw, tw), (ls, ts) = tf_verdicts(checks, [False, False])
    assert lw == "FAIL" and "fix w" in tw
    assert ls == "WARN" and "fix s" in ts
    lvl, text = other_clients_verdict(["rammp_adl_runtime"])
    assert lvl == "FAIL" and "rammp_adl_runtime" in text
    assert other_clients_verdict([])[0] == "PASS"

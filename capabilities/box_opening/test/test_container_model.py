import math

import pytest
import yaml

from box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    attitude_quat,
    from_container,
)

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"


def test_load_and_validate():
    m = ContainerModel.load(CFG)
    assert m.measure_me is False  # owner accepted values 2026-08-25
    assert len(m.dims) == 3 and len(m.lid_dims) == 3
    assert m.button_offset[2] == pytest.approx(m.dims[2])  # button = top
    assert m.touch_nm > 0 and m.hover_standoff > 0


def test_from_container_rotates_by_yaw():
    cpose = ContainerPose(xyz=(1.0, 2.0, 0.0), yaw=math.pi / 2)
    # +x offset in container frame maps to +y in base at yaw 90 deg
    assert from_container(cpose, (0.1, 0.0, 0.05)) == pytest.approx([1.0, 2.1, 0.05])


def test_attitude_quat_top_down_points_tool_down():
    import numpy as np

    q = attitude_quat([180.0, 0.0, 0.0], yaw=0.7)
    x, y, z, w = q
    # z-column of the rotation matrix (tool z axis in world frame)
    tool_z = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    assert tool_z[2] == pytest.approx(-1.0, abs=1e-6)  # tool z straight down


def test_load_press_demo_cfg():
    from box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    assert cfg.detect_source in ("depth", "vlm")
    assert cfg.grip_band[0] < cfg.grip_band[1] < 0.8
    assert 0.0 <= cfg.grip_clear_m <= 0.02
    assert all(abs(v) <= 0.02 for v in cfg.grip_offset_xy)
    assert cfg.lift_m > 0 and 0 < cfg.lift_speed <= 1.0
    assert cfg.press_speed == pytest.approx(0.35)  # owner decision 2026-08-25
    assert cfg.travel_m > 0
    assert cfg.min_hits >= 2 and cfg.timeout_s > 0


def _cfg_variant(tmp_path, old, new):
    p = tmp_path / "variant.yaml"
    p.write_text(open(CFG).read().replace(old, new))
    return str(p)


def test_loader_validates_staging_against_real_container_geometry(tmp_path):
    from box_opening.models.container import load_press_demo

    # recessed button: dims.z 0.112, button_offset.z 0.05 -> staging
    # must clear 0.112 - 0.05 + 0.10 = 0.162; the shipped 0.12 fails
    with pytest.raises(ValueError, match="staging_m"):
        load_press_demo(
            _cfg_variant(
                tmp_path,
                "button_offset: [0.0, 0.0, 0.112]",
                "button_offset: [0.0, 0.0, 0.05]",
            )
        )


def test_loader_refuses_the_retired_tag_source(tmp_path):
    from box_opening.models.container import load_press_demo

    with pytest.raises(ValueError, match="detect.source"):
        load_press_demo(_cfg_variant(tmp_path, "source: vlm", "source: tag"))


def test_a_guarded_descent_never_flies_faster_than_a_transit(tmp_path):
    """warp_fast_speed is the free-air part of a stroke that ends in
    contact. It may run as fast as the arm ever moves in the open and no
    faster: past that the arm would be travelling quicker toward something
    it is about to touch than it does anywhere else. (With TRANSIT_SPEED at
    1.0 that bound coincides with the warp's own hard cap.)"""
    from box_opening.constants import TRANSIT_SPEED
    from box_opening.models.container import load_press_demo

    at_the_limit = load_press_demo(
        _cfg_variant(tmp_path, "warp_fast_speed: 0.75", "warp_fast_speed: %g" % TRANSIT_SPEED)
    )
    assert at_the_limit.warp_fast_speed == pytest.approx(TRANSIT_SPEED)
    with pytest.raises(ValueError, match="warp_fast_speed"):
        load_press_demo(_cfg_variant(tmp_path, "warp_fast_speed: 0.75", "warp_fast_speed: 1.1"))


def test_shipped_press_attitude_is_fixed_and_faces_ahead_looking_down():
    """The wrist keeps ONE orientation from LOOK through the press: the
    shipped attitude is the LOOK pose's own (tool z down, tool x along
    world +y, the camera side toward +x) and does not turn with the box's
    bearing (owner 2026-09-16: no more flipping)."""
    import numpy as np

    from box_opening.models.container import ContainerModel
    from box_opening.perception.d405 import quat_to_mat

    m = ContainerModel.load("capabilities/box_opening/config/containers/oxo_pop.yaml")
    assert m.press_yaw_steer is False
    R = quat_to_mat(*m.press_quat([0.45, -0.15, 0.08]))
    assert np.allclose(R[:, 2], [0.0, 0.0, -1.0], atol=1e-6)  # looking down
    assert np.allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-6)  # facing ahead
    for xyz in ([0.3, 0.3, 0.1], [0.6, -0.3, 0.1]):
        assert np.allclose(quat_to_mat(*m.press_quat(xyz)), R, atol=1e-9)


def test_bearing_steering_still_turns_with_the_box_when_enabled():
    import math
    from dataclasses import replace

    import numpy as np

    from box_opening.models.container import ContainerModel, attitude_quat
    from box_opening.perception.d405 import quat_to_mat

    m = ContainerModel.load("capabilities/box_opening/config/containers/oxo_pop.yaml")
    steered = replace(m, press_yaw_steer=True, press_attitude_rpy_deg=(180.0, 0.0, 0.0))
    xyz = [0.4, -0.2, 0.1]
    want = quat_to_mat(*attitude_quat([180.0, 0.0, 0.0], math.atan2(-0.2, 0.4)))
    assert np.allclose(quat_to_mat(*steered.press_quat(xyz)), want, atol=1e-9)


def test_every_shipped_container_config_loads_and_validates():
    """A typo in a container YAML must fail here, not at the bench. The
    new round container (ankou_pink.yaml, 2026-09-22) ships with photo-
    estimated geometry the owner accepted flying on; its detector prompts are the ones the
    phrasing sweep chose (a pink lid is the anchor; 'clear', 'jar',
    'canister' alone all scored 0.00)."""
    from pathlib import Path

    from box_opening.models.container import ContainerModel, load_press_demo

    folder = Path("capabilities/box_opening/config/containers")
    files = sorted(folder.glob("*.yaml"))
    assert {f.name for f in files} >= {"oxo_pop.yaml", "oxo_pop_small.yaml", "ankou_pink.yaml"}
    for f in files:
        model = ContainerModel.load(str(f))
        cfg = load_press_demo(str(f))
        # nothing in the environment is configured (owner 2026-10-27): the
        # lid's spot is the cameras' every run, never a pose in the yaml
        assert "open_container" not in yaml.safe_load(f.read_text()), f.name
        assert len(model.dims) == 3 and all(v > 0 for v in model.dims), f.name
        assert cfg.owl_queries and cfg.owl_min_score > 0, f.name
    pink = ContainerModel.load(str(folder / "ankou_pink.yaml"))
    # estimated from photos; the owner accepted flying on the estimates
    # (2026-09-22) — every one of them fails safe, and the first press
    # measures the button's height and diameter itself
    assert not pink.measure_me
    assert pink.dims[0] == pink.dims[1]  # round: the lid's diameter both ways
    cfg = load_press_demo(str(folder / "ankou_pink.yaml"))
    assert all("pink lid" in q for q in cfg.owl_queries)
    assert not any("button" in q for q in cfg.owl_queries)  # a button-only box fails the footprint floor
    oxo = load_press_demo(str(folder / "oxo_pop.yaml"))
    assert oxo.owl_queries != cfg.owl_queries


def test_the_default_container_is_the_pink_canister(monkeypatch):
    """Owner, 2026-09-22: "this box is the new default". Every CLI that
    takes --container, and both OWL detector nodes, fall back to this one.
    The same function serves them all, so they cannot disagree. (The
    installed share dir comes from the ament index, which the unit-test
    shell does not carry: pointed at the source tree here.)"""
    from pathlib import Path

    import ament_index_python.packages as pk

    from box_opening.tasks.cli_common import default_container_yaml

    share = Path("src/box_opening").resolve()
    monkeypatch.setattr(pk, "get_package_share_directory", lambda name: str(share))
    got = default_container_yaml()
    assert got.endswith("/config/containers/ankou_pink.yaml")
    assert Path(got).exists()


def test_an_unmeasured_container_may_be_observed_but_not_flown_at():
    """measure_me: true refuses --execute — except for the observation that
    does the measuring (--detect-only), which never goes near the box."""
    import pytest

    from box_opening.tasks.cli_common import refuse_unmeasured

    class M:
        measure_me = True

    refuse_unmeasured(M(), execute=False)  # a dry run is always fine
    refuse_unmeasured(M(), execute=True, measuring=True)  # the look, the report, home
    with pytest.raises(SystemExit):
        refuse_unmeasured(M(), execute=True)
    M.measure_me = False
    refuse_unmeasured(M(), execute=True)


PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"


def test_the_canister_is_found_by_the_segmenter_as_the_food_container_it_stands_in():
    """ankou_pink names the scene segmenter's class (RF-DETR boxes the blue
    holder as a COCO bowl, "food container"); the OXOs name none and keep
    the scene OWL alone. Its OWL is the fallback, at the lower floor."""
    from box_opening.models.container import load_press_demo

    pink = load_press_demo(PINK)
    assert pink.segmenter_classes == ("food container",) and pink.segmenter_box == "holder"
    assert pink.owl_min_score_scene == pytest.approx(0.05)
    for name in ("oxo_pop.yaml", "oxo_pop_small.yaml"):
        oxo = load_press_demo("capabilities/box_opening/config/containers/" + name)
        assert oxo.segmenter_classes == () and oxo.segmenter_box == "holder"


def _pink_variant(tmp_path, old, new):
    text = open(PINK).read()
    assert old in text
    p = tmp_path / "variant.yaml"
    p.write_text(text.replace(old, new))
    return str(p)


def test_a_bad_scene_detect_block_is_refused_at_load(tmp_path):
    from box_opening.models.container import load_press_demo

    with pytest.raises(ValueError, match="segmenter_box must be 'holder' or 'as_is'"):
        load_press_demo(_pink_variant(tmp_path, "segmenter_box: holder", "segmenter_box: lid"))
    with pytest.raises(ValueError, match="segmenter_classes must be a list"):
        load_press_demo(
            _pink_variant(tmp_path, "segmenter_classes: [food container]", "segmenter_classes: food container")
        )
    with pytest.raises(ValueError, match="segmenter_classes must be a list"):
        load_press_demo(
            _pink_variant(tmp_path, "segmenter_classes: [food container]", "segmenter_classes: [food container, '']")
        )
    with pytest.raises(ValueError, match="unknown key"):
        load_press_demo(
            _pink_variant(tmp_path, "segmenter_classes: [food container]", "segmenter_class: [food container]")
        )
    raw = yaml.safe_load(open(PINK))
    raw["scene_detect"] = ["food container"]
    listed = tmp_path / "listed.yaml"
    listed.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_press_demo(str(listed))
    none = load_press_demo(_pink_variant(tmp_path, "segmenter_classes: [food container]", "segmenter_classes: []"))
    assert none.segmenter_classes == ()  # an empty list: no segmenter, the OWL alone

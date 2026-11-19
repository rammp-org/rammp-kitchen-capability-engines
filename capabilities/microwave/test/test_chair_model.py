"""Offline checks for loading the active planner's keep-out layers, no ROS or arm."""

import importlib.util
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "microwave_chair_model", Path(__file__).parents[1] / "microwave/chair_model.py"
)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


@pytest.fixture
def config(tmp_path):
    for name in ("platform", "seat", "person"):
        data = dict(
            layer=name,
            meta=dict(frame="arm_mount", padding_assumed_m=0.02),
            obstacles=[dict(name="box", position=[1, 2, 3], dims=[0.2, 0.4, 0.6])],
        )
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data))
    path = tmp_path / "planner.yaml"
    path.write_text(yaml.safe_dump(dict(layers=["seat.yaml", "person.yaml"])))
    return path


def update(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data))


def test_uses_only_the_planners_selected_layers_and_preserves_padding(config):
    # A bundled/unselected platform may be present, but must never be loaded.
    (config.parent / "platform.yaml").write_text("malformed and unused")
    boxes, layers = model.load_planner_chair(config)
    assert layers == ["seat", "person"]
    assert [b[0] for b in boxes] == ["seat/box", "person/box"]
    assert boxes[0][1] == (1, 2, 3)
    assert boxes[0][2] == pytest.approx([0.12, 0.22, 0.32])


def test_resolves_relative_and_absolute_layer_paths(config):
    update(config, lambda data: data.update(layers=[str(config.parent / "person.yaml")]))
    assert model.load_planner_chair(config)[1] == ["person"]


def test_uses_custom_world_padding(config):
    update(config, lambda data: data.update(planner=dict(world_padding=0.04)))
    assert model.load_planner_chair(config)[0][0][2] == pytest.approx([0.14, 0.24, 0.34])


def test_honors_explicit_no_pad_names(config):
    update(config, lambda data: data.update(planner=dict(no_pad_names=["box"])))
    assert model.load_planner_chair(config)[0][0][2] == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize("layers", [[], None, "seat.yaml", ["missing.yaml"], ["seat.yaml", "seat.yaml"]])
def test_invalid_or_missing_layers_never_fall_back(config, layers):
    update(config, lambda data: data.update(layers=layers))
    with pytest.raises((ValueError, OSError)):
        model.load_planner_chair(config)


def test_missing_config_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.load_planner_chair(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["meta"].update(frame="arm_base_link"),
        lambda d: d.update(obstacles=[]),
        lambda d: d["obstacles"][0].update(dims=[0, 1, 1]),
        lambda d: d["obstacles"][0].update(position=[float("nan"), 0, 0]),
        lambda d: d["obstacles"][0].update(rpy_deg=[0, 0, 20]),
        lambda d: d["meta"].update(padding_assumed_m=0.03),
    ],
)
def test_unusable_geometry_is_refused(config, change):
    update(config.parent / "seat.yaml", change)
    with pytest.raises(ValueError):
        model.load_planner_chair(config)


def test_new_file_contents_are_loaded_on_restart(config):
    first = model.load_planner_chair(config)[0][0][1]
    update(config.parent / "seat.yaml", lambda data: data["obstacles"][0].update(position=[4, 5, 6]))
    second = model.load_planner_chair(config)[0][0][1]
    assert first == (1, 2, 3) and second == (4, 5, 6)

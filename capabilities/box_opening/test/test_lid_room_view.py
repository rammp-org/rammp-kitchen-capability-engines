"""The live view of where the lid would go (tasks/lid_room_view): the room
map and spots press_demo makes from the scene camera alone, drawn as
markers. Nothing here spins a node."""

import math

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.tasks import lid_room_view as lrv

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
DATA = "capabilities/box_opening/test/data/"
TABLE_Z = 0.187
BOX = (0.39, -0.20)


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


def _flat(x0, x1, y0, y1, z, step=0.006):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, float(z))]


def _open_table_with_box():
    """A bare table with the box standing on it, and the floor past its far edge."""
    x0, x1, y0, y1 = BOX[0] - 0.06, BOX[0] + 0.06, BOX[1] - 0.06, BOX[1] + 0.06
    t = _flat(0.25, 0.75, -0.70, 0.20, TABLE_Z)
    t = t[~((t[:, 0] > x0) & (t[:, 0] < x1) & (t[:, 1] > y0) & (t[:, 1] < y1))]
    pts = [t, _flat(x0, x1, y0, y1, TABLE_Z + 0.121)]
    for z in np.arange(TABLE_Z + 0.02, TABLE_Z + 0.121, 0.006):
        pts += [
            _flat(x0, x1, y0, y0 + 0.006, z),
            _flat(x0, x1, y1 - 0.006, y1, z),
            _flat(x0, x0 + 0.006, y0, y1, z),
            _flat(x1 - 0.006, x1, y0, y1, z),
        ]
    return np.concatenate(pts)


def _by_ns(markers, ns):
    return [m for m in markers.markers if m.ns == ns]


def test_on_an_open_table_it_draws_the_lid_where_press_demo_puts_it_and_the_hand_turned_as_it_comes_down(pink):
    from builtin_interfaces.msg import Time

    from box_opening.perception import clear_table as ct
    from box_opening.tasks.press_demo import clear_lid_drops, place_quat, room_map

    fix, why = lrv.fix_around(_open_table_with_box(), BOX, pink)
    assert why is None and fix.table_z_scene == pytest.approx(TABLE_Z, abs=0.003)
    view = lrv.room_view(pink, fix, "clicked")
    assert view.drops and view.why is None
    # press_demo's own answer, from the same points
    drops, spots, _why = clear_lid_drops(pink, fix.pose, room_map(None, fix, fix.pose, False), fix.table_z_scene)
    assert [d.xyz for d in view.drops] == [d.xyz for d in drops]
    # every spot judged is kept, the ones picked among them
    assert view.spots[0] in view.judged and len(view.judged) > len(view.spots)
    assert "lid goes to [%.2f, %.2f]" % view.drops[0].xyz[:2] in lrv.status_line(view)

    ma = lrv.view_markers(view, pink, "arm_mount", Time())
    assert ma.markers[0].action == ma.markers[0].DELETEALL
    lid = _by_ns(ma, "lid")[0]
    assert (lid.pose.position.x, lid.pose.position.y) == pytest.approx(view.drops[0].xyz[:2])
    ax, _ay = ct._hand_axes(place_quat(pink, view.drops[0]))
    for part in _by_ns(ma, "hand"):
        yaw = 2 * math.atan2(part.pose.orientation.z, part.pose.orientation.w)
        assert math.cos(yaw - math.atan2(ax[1], ax[0])) == pytest.approx(1.0)
    standing = _by_ns(ma, "standing")
    assert standing and all(abs(m.pose.position.x - BOX[0]) < 0.1 for m in standing)  # only the box stands
    assert max(m.scale.z for m in standing) == pytest.approx(0.121, abs=0.02)


def test_the_1757_table_from_the_scene_camera_alone_has_no_room_and_it_says_why(pink):
    """press_demo with the wrist's view found room at 17:57; the scene camera
    alone, which is all this view has, does not, and the status says what
    was short."""
    from builtin_interfaces.msg import Time

    f = np.load(DATA + "table_room_flatbot_20261023_1757.npz")
    scene = f["scene_mm"] / 1000.0 + np.r_[f["wrist_box"] - f["scene_box"], 0.0]
    fix, why = lrv.fix_around(scene, tuple(f["wrist_box"]), pink)
    assert why is None and fix.table_z_scene == pytest.approx(float(f["table_z"]), abs=0.01)
    view = lrv.room_view(pink, fix, "scene OWL")
    assert not view.drops and "short of the room it needs" in view.why
    line = lrv.status_line(view)
    assert line.startswith(
        "box at [0.39, -0.21] (scene OWL), %d planner cuboid(s): nowhere to set the lid down" % len(view.boxes)
    )
    ma = lrv.view_markers(view, pink, "arm_mount", Time())
    assert not _by_ns(ma, "lid")
    tall = [m for m in _by_ns(ma, "standing") if m.color.r > 0.8 and m.color.g < 0.3]
    assert any(m.pose.position.x > 0.5 and m.pose.position.y < -0.25 for m in tall)  # the microwave reaches the camera
    assert _by_ns(ma, "unseen") and _by_ns(ma, "room at each spot")
    assert len(_by_ns(ma, "planner cuboids")) == len(view.boxes) > 0


def test_its_point_cloud_carries_each_points_height_above_the_table(pink):
    from builtin_interfaces.msg import Time
    from sensor_msgs_py import point_cloud2

    fix, _why = lrv.fix_around(_open_table_with_box(), BOX, pink)
    cloud = lrv.view_points(lrv.room_view(pink, fix, "--box"), "arm_mount", Time())
    assert [f.name for f in cloud.fields] == ["x", "y", "z", "h"]
    h = np.array([p[0] for p in point_cloud2.read_points(cloud, field_names=("h",))])
    assert len(h) == len(fix.table_points)
    assert np.median(h) == pytest.approx(0.0, abs=0.002) and h.max() == pytest.approx(0.121, abs=0.003)


def test_a_spot_said_where_the_camera_sees_nothing_is_refused_with_why(pink):
    fix, why = lrv.fix_around(_open_table_with_box(), (2.0, 2.0), pink)
    assert fix is None and "sees nothing within 60 cm of [2.00, 2.00]" in why

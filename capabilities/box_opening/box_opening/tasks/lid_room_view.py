"""Watch where the lid would go, live: the scene camera's room map around
the box (perception/clear_table) and the spot press_demo would set the lid
down on (clear_lid_drops), rebuilt every few seconds and drawn for Foxglove
(or RViz) beside the robot. NOTHING MOVES: no arm client is ever made.

    scripts/sheppy_task.sh lid_room_view

then add to Foxglove's 3D panel:

    /box_opening/lid_room/markers   the map, the spots, the lid and the hand, and the
                                          cuboids the planner's worlds get from the map
    /box_opening/lid_room/points    the scene camera's points, field h = height above its table
    /box_opening/lid_room/status    one line: where the lid goes, or why nowhere

The box is where the mission's own SceneLocator finds it: from sheppy's
scene_segmenter, or the scene OWL while press_demo.launch.py's
owl_detector runs. Without either, click the box,
or any spot to ask "what if the box stood here", with the 3D panel's
publish-point tool (/clicked_point; in the 2D top-down view, so the click
lands where it looks), or pass --box X Y. A click stays the box until the
next one, and outranks the cameras' detectors.

Only the scene camera, as the mission itself since 2026-10-28
(reach_lid_off.resolve_lid_drop; the scrapped staging run added the wrist's
look from staging, which needed the arm to fly there). The scene camera,
14 cm above the table, cannot vouch for the table it sees edge-on beside
the box (flatbot 2026-10-23 17:57: no spot from the scene camera alone).
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from box_opening import LOGGER as log
from box_opening.perception import clear_table as ct

PERIOD_S = 3.0  # a map and a spot search a period: ~1 s of one core, niced
OWL_RETRY_S = 30.0  # absent detectors (no segmenter, no owl_detector) are asked again this often
DETECTOR_NAMES = {"segmenter": "scene segmenter", "owl": "scene OWL"}  # SceneFix.source, as the status names it
STATUS_TOPIC = "/box_opening/lid_room/status"
MARKER_TOPIC = "/box_opening/lid_room/markers"
POINTS_TOPIC = "/box_opening/lid_room/points"
CLICK_TOPIC = "/clicked_point"

# what each part of the hand looks like here (clear_table.HAND_PARTS says
# how far it reaches; these say only how tall to draw it)
DRAW_H_M = {"the fingers": 0.06, "the wrist camera": 0.03}
TALL_RGB = (0.86, 0.20, 0.20)  # reaches the wrist camera at the set-down
SHORT_RGB = (0.95, 0.60, 0.10)
CLEAR_RGBA = (0.20, 0.75, 0.35, 0.35)
EDGE_RGBA = (0.55, 0.35, 0.85, 0.6)
UNSEEN_RGBA = (0.5, 0.5, 0.5, 0.15)
LID_RGBA = (0.95, 0.40, 0.70, 0.9)
PLANNER_RGBA = (0.30, 0.55, 1.00, 0.25)


@dataclass
class RoomView:
    fix: object  # the SceneFix the map was made from
    source: str  # where the box came from: "scene segmenter", "scene OWL", "clicked", "--box"
    tmap: object = None
    drops: list = None
    spots: list = None
    judged: list = None
    why: str = None
    boxes: list = None  # the standing cuboids clear_table.obstacle_boxes makes of the same map (no planner gets them)


def fix_around(pts_base, box_xy, model):
    """(SceneFix, None) for a box said to stand at `box_xy` — the scene's
    points around it and the table height under it, as SceneLocator.locate
    would have kept them — or (None, why)."""
    from box_opening.models.container import ContainerPose
    from box_opening.perception.scene_source import SceneFix, table_around

    pts = np.asarray(pts_base, float)
    bx, by = float(box_xy[0]), float(box_xy[1])
    near = np.hypot(pts[:, 0] - bx, pts[:, 1] - by) < ct.MAP_R_M
    if not near.any():
        return None, "the scene camera sees nothing within %.0f cm of [%.2f, %.2f]" % (100 * ct.MAP_R_M, bx, by)
    # the commonest height in a ring round the spot is the table (the
    # mission's side lift reads it so); everything in view may sit in it
    tz = table_around(pts, (bx, by), below_z=float(pts[near, 2].max()) + 0.01)
    if math.isnan(tz):
        return None, "too little of the table shows around [%.2f, %.2f]" % (bx, by)
    fix = SceneFix(
        pose=ContainerPose(xyz=(bx, by, tz), yaw=0.0),
        top_xyz=(bx, by, tz + float(model.dims[2])),
        n_points=0,
        score=0.0,
        table_z_scene=tz,
        table_points=pts[near].astype(np.float32),
    )
    return fix, None


def room_view(model, fix, source):
    """The room map round `fix` and the spots on it, exactly as press_demo
    makes them from the scene camera alone (room_map, clear_lid_drops)."""
    from box_opening.tasks.press_demo import clear_lid_drops, container_half, room_map

    view = RoomView(fix=fix, source=source)
    view.tmap = room_map(None, fix, fix.pose, at_staging=False)
    if view.tmap is None:
        view.why = "the scene camera's points show no table plane around the box"
        return view
    view.judged = []
    view.drops, view.spots, view.why = clear_lid_drops(
        model, fix.pose, view.tmap, fix.table_z_scene, judged_out=view.judged
    )
    # the arm is left out of nothing here: this view does not know where it is
    view.boxes, _note = ct.obstacle_boxes(
        view.tmap,
        fix.table_z_scene,
        fix.pose.xyz[:2],
        container_half(model),
        keep_clear=[d.xyz[:2] for d in view.drops],
    )
    return view


def status_line(view):
    """One line saying what the view found, in press_demo's words."""
    bx, by = view.fix.pose.xyz[0], view.fix.pose.xyz[1]
    head = "box at [%.2f, %.2f] (%s)" % (bx, by, view.source)
    if view.boxes:
        head += ", %d planner cuboid(s)" % len(view.boxes)
    if not view.drops:
        return "%s: nowhere to set the lid down: %s" % (head, view.why)
    best, lid = view.spots[0], view.drops[0]
    return "%s: lid goes to [%.2f, %.2f] — %.0f mm from the box, %.0f mm to spare%s%s; %d spot(s) with room" % (
        head,
        lid.xyz[0],
        lid.xyz[1],
        1000 * math.hypot(lid.xyz[0] - bx, lid.xyz[1] - by),
        1000 * best.room_m,
        "" if best.what == "nothing" else " (%s nearest anything)" % best.what,
        "; the hand turns %+.0f deg" % math.degrees(best.turn) if best.turn else "",
        len(view.spots),
    )


def _room_rgba(room_m):
    """Red at 3 cm short of room, yellow at none to spare, green at 3 cm."""
    t = max(-1.0, min(1.0, room_m / 0.03))
    if t < 0:
        return (1.0, 1.0 + t, 0.0, 0.9)
    return (1.0 - t, 1.0, 0.0, 0.9)


class MarkerList:
    """A visualization_msgs/MarkerArray in `frame` at `stamp`, made one
    marker at a time (add), each numbered in turn. Its first marker deletes
    every marker the last one drew. perception/viz draws the mission's
    perception with it too."""

    def __init__(self, frame, stamp):
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray

        self._Marker, self._ColorRGBA = Marker, ColorRGBA
        self.frame, self.stamp, self.n = frame, stamp, 0
        self.array = MarkerArray()
        clear_all = Marker()
        clear_all.header.frame_id = frame
        clear_all.action = Marker.DELETEALL
        self.array.markers.append(clear_all)

    def add(self, ns, kind, rgba, scale=(1.0, 1.0, 1.0)):
        m = self._Marker()
        m.header.frame_id, m.header.stamp = self.frame, self.stamp
        m.ns, m.id, m.type, m.action = ns, self.n, kind, self._Marker.ADD
        self.n += 1
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color = self._ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
        self.array.markers.append(m)
        return m


def yawed(m, yaw):
    """Marker `m` turned `yaw` about the vertical."""
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)


def view_markers(view, model, frame, stamp):
    """The view as a visualization_msgs/MarkerArray in `frame` (the map's
    base frame): what stands (each square its own height, red where it
    reaches the wrist camera), the clear table, the table's edge, what no
    camera saw, the room at every spot judged, and at the chosen spot the
    lid, the turned hand with each part's room drawn round it, and the
    carry. The first marker deletes the last period's."""
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker

    markers = MarkerList(frame, stamp)
    out, marker = markers.array, markers.add

    fix, tz = view.fix, float(view.fix.table_z_scene)
    lid_r, lid_h = 0.5 * max(float(model.lid_dims[0]), float(model.lid_dims[1])), float(model.lid_dims[2])
    bx, by = fix.pose.xyz[0], fix.pose.xyz[1]
    box = marker("box", Marker.CYLINDER, (0.25, 0.45, 0.95, 0.35), (model.dims[0], model.dims[1], model.dims[2]))
    box.pose.position.x, box.pose.position.y, box.pose.position.z = bx, by, tz + 0.5 * float(model.dims[2])
    text = marker("status", Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.025))
    text.pose.position.x, text.pose.position.y, text.pose.position.z = bx, by, tz + 0.40
    text.text = status_line(view)
    if view.tmap is None:
        return out

    tm = view.tmap
    X, Y = tm.centres()
    near = np.hypot(X - tm.box_xy[0], Y - tm.box_xy[1]) < ct.MAP_R_M
    stands = np.isfinite(tm.tall)
    # taller than this blocks the wrist camera at the set-down (clearance)
    reach = lid_h + next(p[3] for p in ct.HAND_PARTS if p[0] == "the wrist camera") - ct.UP_SLACK_M
    c = tm.cell
    for i, j in zip(*np.nonzero(stands), strict=False):
        h = float(tm.tall[i, j])
        m = marker("standing", Marker.CUBE, (*(TALL_RGB if h > reach else SHORT_RGB), 0.85), (0.9 * c, 0.9 * c, h))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(X[i, j]), float(Y[i, j]), tz + 0.5 * h

    def tiles(ns, mask, rgba):
        if not mask.any():
            return
        m = marker(ns, Marker.CUBE_LIST, rgba, (0.9 * c, 0.9 * c, 0.002))
        m.points = [Point(x=float(x), y=float(y), z=tz) for x, y in zip(X[mask], Y[mask], strict=False)]

    for b in view.boxes or ():  # what the planner keeps the arm clear of
        m = marker("planner cuboids", Marker.CUBE, PLANNER_RGBA, b["dims"])
        m.pose.position.x, m.pose.position.y, m.pose.position.z = b["position"]
    tiles("clear table", near & tm.seen & ~stands & ~tm.off, CLEAR_RGBA)
    tiles("past the edge", near & tm.off, EDGE_RGBA)
    tiles("unseen", near & ~tm.seen, UNSEEN_RGBA)

    if view.judged:
        m = marker("room at each spot", Marker.SPHERE_LIST, (1.0, 1.0, 1.0, 1.0), (0.008, 0.008, 0.008))
        for s in view.judged:
            m.points.append(Point(x=s.xy[0], y=s.xy[1], z=tz + 0.004))
            m.colors.append(ColorRGBA(**dict(zip("rgba", _room_rgba(s.room_m), strict=False))))
    if not view.drops:
        return out

    from box_opening.tasks.press_demo import place_quat

    for d in view.drops[1:]:  # the planner falls through to these
        m = marker("other spots", Marker.CYLINDER, (*LID_RGBA[:3], 0.3), (2 * lid_r, 2 * lid_r, 0.003))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = d.xyz[0], d.xyz[1], tz + 0.002
    lid = view.drops[0]
    m = marker("lid", Marker.CYLINDER, LID_RGBA, (2 * lid_r, 2 * lid_r, lid_h))
    m.pose.position.x, m.pose.position.y, m.pose.position.z = lid.xyz[0], lid.xyz[1], tz + 0.5 * lid_h
    room = marker(
        "room each part needs",
        Marker.CYLINDER,
        (*LID_RGBA[:3], 0.15),
        (2 * (lid_r + ct.LID_CLEAR_M), 2 * (lid_r + ct.LID_CLEAR_M), 0.002),
    )
    room.pose.position.x, room.pose.position.y, room.pose.position.z = lid.xyz[0], lid.xyz[1], tz + 0.001
    ax, ay = ct._hand_axes(place_quat(model, lid))
    yaw = math.atan2(ax[1], ax[0])
    for name, (xa, xb), (ya, yb), bottom, need in ct.HAND_PARTS:
        hx, hy = 0.5 * (xa + xb), 0.5 * (ya + yb)
        z0 = tz + lid_h + bottom  # the fingertips are the lid's height up at the release (clear_lid_drops)
        part = marker("hand", Marker.CUBE, (0.75, 0.80, 0.90, 0.7), (xb - xa, yb - ya, DRAW_H_M.get(name, 0.03)))
        part.pose.position.x = lid.xyz[0] + hx * ax[0] + hy * ay[0]
        part.pose.position.y = lid.xyz[1] + hx * ax[1] + hy * ay[1]
        part.pose.position.z = z0 + 0.5 * part.scale.z
        yawed(part, yaw)
        edge = marker("room each part needs", Marker.LINE_STRIP, (0.75, 0.80, 0.90, 0.9), (0.002, 0.0, 0.0))
        for u, v in (
            (xa - need, ya - need),
            (xb + need, ya - need),
            (xb + need, yb + need),
            (xa - need, yb + need),
            (xa - need, ya - need),
        ):
            edge.points.append(Point(x=lid.xyz[0] + u * ax[0] + v * ay[0], y=lid.xyz[1] + u * ax[1] + v * ay[1], z=z0))
    from box_opening.primitives.core import CARRY_CLEAR_M

    carry_z = tz + float(model.dims[2]) + lid_h + CARRY_CLEAR_M
    m = marker("carry", Marker.LINE_STRIP, (1.0, 1.0, 1.0, 0.8), (0.003, 0.0, 0.0))
    m.points = [
        Point(x=bx, y=by, z=carry_z),
        Point(x=lid.xyz[0], y=lid.xyz[1], z=carry_z),
        Point(x=lid.xyz[0], y=lid.xyz[1], z=tz + lid_h),
    ]
    return out


def view_points(view, frame, stamp):
    """The scene camera's points the map was made from, as a PointCloud2
    with h, each point's height above the table plane they show (colour by
    it in Foxglove), or None."""
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    pts = getattr(view.fix, "table_points", None)
    got = None if pts is None else ct._heights(pts, view.fix.table_z_scene, view.fix.pose.xyz[:2], ct.MAP_R_M)
    if got is None:
        return None
    p, h = got
    fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, k in enumerate("xyzh")]
    return point_cloud2.create_cloud(Header(frame_id=frame, stamp=stamp), fields, np.c_[p, h].astype(np.float32))


def scene_points(locator):
    """(the scene camera's newest points in the base frame, None), as
    SceneLocator.locate lifts them (the median of its kept depth frames,
    every second pixel), or (None, why)."""
    g = locator.grab
    if not locator.ready():
        missing = g.missing()
        return None, ("the scene camera's %s not in yet" % ", ".join(missing)) if missing else locator._unplaced_why()
    cloud = g.cloud_in_color_frame(depth=g.depth_median(), stride=2)
    T = locator.base_to_color()
    if cloud is None or T is None:
        return None, "no scene depth cloud"
    pc, _uv = cloud
    return pc @ T[:3, :3].T + T[:3, 3], None


def main():
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    from box_opening.models.container import ContainerModel, load_press_demo
    from box_opening.perception.owl_source import announce_container
    from box_opening.perception.scene import load_scene_calibration
    from box_opening.perception.scene_source import SceneLocator
    from box_opening.tasks import cli_common
    from box_opening.tasks.press_demo import SCENE_LOCATE_S
    from box_opening.worlds import WorldStore

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    ap.add_argument("--container", default=None, help="container config YAML (default: the installed ankou_pink.yaml)")
    ap.add_argument("--bench-world", default=None, help="bench world YAML (sheppy_task.sh passes world_flatbot.yaml)")
    ap.add_argument("--scene-calib", default=None, help="scene camera calibration yaml, where TF does not place it")
    ap.add_argument(
        "--box",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="the box stands here (base frame, m) instead of where the OWL or a click says",
    )
    ap.add_argument("--period", type=float, default=PERIOD_S, help="seconds between maps (default %(default)s)")
    ap.add_argument("--no-owl", action="store_true", help="never ask the scene OWL; the box is clicked or --box")
    args = ap.parse_args()
    why = cli_common.rmw_refusal(os.environ)
    if why:
        sys.exit("[lid_room_view] not starting: %s" % why)
    cfg_path = args.container or cli_common.default_container_yaml()
    model = ContainerModel.load(cfg_path)
    cfg = load_press_demo(cfg_path)
    worlds = WorldStore(args.bench_world or cli_common.default_bench_yaml())
    T_base_scene_link = None if worlds.external_static_tf else load_scene_calibration(args.scene_calib)
    if not worlds.external_static_tf and T_base_scene_link is None:
        sys.exit("[lid_room_view] not starting: the scene camera is not calibrated (scripts/calibrate_scene_camera.py)")
    frame = worlds.base_frame

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl+C is ours: the OWL's gate is shut on the way out
    node = rclpy.create_node("box_opening_lid_room_view")
    locator = SceneLocator(
        node,
        cfg,
        model,
        worlds.table_top_z,
        T_base_scene_link,
        spins=False,
        scene_cfg=worlds.scene_camera_cfg,
        base_frame=frame,
    )
    keep = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub_status = node.create_publisher(String, STATUS_TOPIC, keep)
    pub_markers = node.create_publisher(MarkerArray, MARKER_TOPIC, keep)
    pub_points = node.create_publisher(PointCloud2, POINTS_TOPIC, keep)
    manual = {"xy": tuple(args.box) if args.box else None, "source": "--box"}

    def clicked(msg):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        if msg.header.frame_id and msg.header.frame_id != frame:
            T = locator.grab.lookup(frame, msg.header.frame_id)
            if T is None:
                log.info("[lid_room_view] click in %s ignored: no TF to %s" % (msg.header.frame_id, frame))
                return
            p = T[:3, :3] @ p + T[:3, 3]
        manual["xy"], manual["source"] = (float(p[0]), float(p[1])), "clicked"
        log.info("[lid_room_view] the box is now the click at [%.2f, %.2f]" % manual["xy"])

    node.create_subscription(PointStamped, CLICK_TOPIC, clicked, 10)
    if not args.no_owl:
        owl_announcer = announce_container(node, cfg_path)  # noqa: F841 (it keeps the latched message alive)
    pexec = SingleThreadedExecutor()
    pexec.add_node(node)
    spinner = threading.Thread(target=pexec.spin, name="perception", daemon=True)
    spinner.start()
    log.info(
        "[lid_room_view] nothing moves. Every %.0f s: the scene camera's room map and the lid's spot on %s, %s"
        % (args.period, MARKER_TOPIC, POINTS_TOPIC)
    )
    said, owl_next = None, 0.0
    try:
        while True:
            t0 = time.monotonic()
            fix, why, source = None, None, None
            if manual["xy"] is not None:
                source = manual["source"]
                pts, why = scene_points(locator)
                if pts is not None:
                    fix, why = fix_around(pts, manual["xy"], model)
            elif not args.no_owl and t0 >= owl_next:
                fix = locator.locate(SCENE_LOCATE_S)
                if fix is None:
                    why = locator.last_why or "the scene camera found no box"
                    if locator.owl.latest is None and locator.last_seg_state != "live":
                        why = (
                            "neither sheppy's scene_segmenter nor an owl_detector is answering "
                            "(press_demo.launch.py starts the OWL)"
                        )
                        owl_next = time.monotonic() + OWL_RETRY_S
                else:
                    source = DETECTOR_NAMES.get(getattr(fix, "source", "owl"), "scene camera")
            else:
                why = (
                    "--no-owl, and no box clicked"
                    if args.no_owl
                    else "neither sheppy's scene_segmenter nor an owl_detector is answering"
                )
            stamp = node.get_clock().now().to_msg()
            if fix is None:
                line = "no map — %s" % why
                if manual["xy"] is None:
                    line += ". Click the box in Foxglove's 3D panel (publish point), or pass --box X Y"
                pub_status.publish(String(data=line))
            else:
                view = room_view(model, fix, source)
                line = status_line(view)
                pub_status.publish(String(data=line))
                pub_markers.publish(view_markers(view, model, frame, stamp))
                cloud = view_points(view, frame, stamp)
                if cloud is not None:
                    pub_points.publish(cloud)
            if line != said:
                log.info("[lid_room_view] %s" % line)
                said = line
            time.sleep(max(0.2, args.period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_owl:
            locator.owl.disable()
            time.sleep(0.2)  # the latched False must leave before the node goes
        # the spinning thread stops BEFORE rclpy is torn down: one still in
        # spin() at exit ends in "terminate called without an active
        # exception" (press_demo's finally; this view's first live run)
        pexec.shutdown()
        spinner.join(timeout=3.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

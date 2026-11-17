"""Follows the container while the user lines up: the scene camera's sightings, one line each on stdout.

    box_watch --container <config> --bench-world <world> [--instance-id ID]

The engine's line-up (line_up.BoxSource) runs it as a child while its panel is up. It reads the scene
camera and the scene segmenter exactly as a run's own look does (perception/scene_source.SceneLocator: the
same boxes, the same gate), and tells of each message's sighting as it is made, before any vote
(SceneLocator.on_sighting): while the chair is driven no two sightings agree, and the user is told which
way to move from each. It moves nothing, and cannot: it holds no client for the arm or the gripper.

The lines that begin with MARK are for the engine (every other line here is the perception's own log):

    @@ {"robot": {"urdf": path or null, "root": frame}}     once, first: the arm's URDF as the robot publishes
                                                            it (/arm/robot_description), for reach_probe
    @@ {"fix": {"xyz": [x, y, z], "table_z": z, "age_s": s, "rough": bool, "score": s}}
    @@ {"none": "why"}                                      a look that ended with no fix, and why

`xyz` is the lid's top in the planning frame. A rough sighting (a box the frame cuts, placed from the part
in view) says which way to move and is never asked on.

What it cannot see is the line-up's limit today: flatbot's scene camera does not see the lid whole in front
of the arm (the reach map of 2026-11-11), and nothing here turns the wrist to look.
"""

import argparse
import json
import math
import os
import signal
import sys
import threading

MARK = "@@ "
LOOK_S = 1.0  # each look's length: how soon a stop is heard, and how often "none" is said


def fix_line(sighting, age_s):
    """A sighting as the engine reads it, or None where it has no place (a top that is not a number)."""
    top = [float(v) for v in sighting.top]
    if not all(math.isfinite(v) for v in top):
        return None
    table_z = float(sighting.table_z)
    return {
        "fix": {
            "xyz": [round(v, 4) for v in top],
            "table_z": round(table_z, 4) if math.isfinite(table_z) else None,
            "age_s": round(max(0.0, float(age_s)), 3),
            "rough": sighting.lift == "rough",
            "score": round(float(sighting.score), 3),
        }
    }


def watch(locator, say, stopped, look_s=LOOK_S):
    """Look after look until `stopped()`: every sighting said as it is made, and a look with no fix's why."""

    def told(sighting, age_s):
        line = fix_line(sighting, age_s)
        if line is not None:
            say(line)

    locator.on_sighting = told
    while not stopped():
        if locator.locate(look_s) is None and locator.last_why:
            say({"none": str(locator.last_why)[:300]})


def main(argv=None):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    from box_opening.models.container import ContainerModel, load_press_demo
    from box_opening.perception.scene import load_scene_calibration, scene_calib_path
    from box_opening.perception.scene_source import SceneLocator
    from box_opening.runtime.driver import rmw_refusal
    from box_opening.tasks import cli_common
    from box_opening.worlds import WorldStore

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--container", default=None, help="the container's config")
    ap.add_argument("--bench-world", default=None, help="the robot's planning frame and cameras")
    ap.add_argument("--scene-calib", default=None, help="the scene camera's calibration, where TF does not carry it")
    ap.add_argument("--instance-id", default=None, help="the container the user selected")
    args = ap.parse_args(argv)
    why = rmw_refusal(os.environ)
    if why:
        sys.exit(why)

    def say(msg):
        sys.stdout.write(MARK + json.dumps(msg) + "\n")
        sys.stdout.flush()

    stop = threading.Event()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cfg_path = args.container or cli_common.default_container_yaml()
    model, cfg = ContainerModel.load(cfg_path), load_press_demo(cfg_path)
    worlds = WorldStore(args.bench_world or cli_common.default_bench_yaml())
    node = rclpy.create_node("box_opening_line_up")
    try:
        # the arm's URDF first (it spins this node itself, up to 5 s), then the cameras on a thread of their own
        say({"robot": {"urdf": cli_common._arm_urdf_path(node, worlds.base_frame), "root": worlds.base_frame}})
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, name="perception", daemon=True).start()
        by_tf = worlds.external_static_tf
        T_base_link = None if by_tf else load_scene_calibration(args.scene_calib)
        if not by_tf and T_base_link is None:
            say({"none": "no scene camera calibration, and the robot's TF does not carry its mount"})
            return 1
        tag_px = None
        if not by_tf:
            from box_opening.perception.scene_calib import scene_tag_px

            tag_px = scene_tag_px(args.scene_calib or scene_calib_path())
        locator = SceneLocator(
            node,
            cfg,
            model,
            worlds.table_top_z,
            T_base_link,
            spins=False,
            tag_px=tag_px,
            scene_cfg=worlds.scene_camera_cfg,
            base_frame=worlds.base_frame,
            instance_id=args.instance_id,
        )
        watch(locator, say, stop.is_set)
    finally:
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001 - already down: it is ending either way
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

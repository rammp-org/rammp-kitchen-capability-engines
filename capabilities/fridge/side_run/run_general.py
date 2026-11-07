"""RUN GENERAL (fridge image + cuRobo interfaces; side_docker.sh) -- MOVES THE ARM unless --dry: the side grasp from
wherever the arm is. One look (SAM 3 on the scene camera, as run_curve.py), side_grasp.plan_side_general (the slide's
arm configuration by search; cuRobo, with this detection's fridge loaded as its world -- replacing the planner's
world -- from the current joints to just out of the door; the merge and the straight slide), then ONE move_joints
of the joined route, timed by fast_timing (per-joint limits, tool-speed caps). Prints the time from start to the
first motion command. --dry: everything but the motion (no RammpArm). --hinge-right: the door is hinged on the
handle's right (the hand comes in from the right, toward the free edge on the left).
"""
import json, os, sys, threading, time

T_START = time.time()
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

import rammp_arm  # noqa: E402
from capability_engine import sim  # noqa: E402

DRY = "--dry" in sys.argv
EST_FILE = "/work/handle_estimate.json"
# --pull=<m>: PART 2 right after the grasp, in this process: joint impedance (HARDEST), <m> straight back along the
# measured normal turning pull_lib.TURN_DEG. Its knots are solved before part 1 moves (none after the arrival)
_PULL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pull=")), None)  # "OUT" or "OUT,TOWARD_HINGE"
PULL_M = float(_PULL.split(",")[0]) if _PULL else None
SWING_DEG = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--swing=")), None)  # the door's
# swing after the grasp (Cailyn's arc about the hinge pin, the hand turning with the door), under HARDEST impedance
HAND_TURNS_DEG = lambda swing: [swing] + [d for d in range(int(swing // 5) * 5, int(swing) - 31, -5) if d < swing and d > 0]
# the hand's turn over the swing: the door's own, else the largest of these (5 deg steps, at most 30 less) that solves
PUSH = "--push" in sys.argv  # part 3 (push_lib) right after the swing, planned before any motion
PULL_SIDE_M = float(_PULL.split(",")[1]) if _PULL and "," in _PULL else 0.0  # > 0: along the door's swing (arc)
ARRIVE_TOL_M = 0.01  # each motion must end this close to its planned end, or the run stops there
EST_WAIT_S = 6.0  # wait this long for an ok door estimate, then refuse
EST_MAX_AGE_S = 2.0  # the estimate file must be this fresh
HINGE_SIGN = -1 if "--hinge-right" in sys.argv else 1  # the door's hinge on the handle's left (default) or right


def stamp(msg):
    print(f"[{time.time() - T_START:6.2f} s] {msg}", flush=True)


rclpy.init()
node = rclpy.create_node("fridge_side_general")
ex = MultiThreadedExecutor()
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
got = {}
node.create_subscription(JointState, "/joint_states", lambda m: got.__setitem__("js", m), qos_profile_sensor_data)
node.create_subscription(String, "/arm/robot_description", lambda m: got.setdefault("urdf", m.data),
                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
t0 = time.time()
while not ("urdf" in got and "js" in got) and time.time() - t0 < 10:
    time.sleep(0.01)
assert "urdf" in got and "js" in got, f"no {'/arm/robot_description' if 'urdf' not in got else '/joint_states'} in 10 s"
sim.set_model_provider(lambda: got["urdf"])

from door_common import common as dc  # noqa: E402
from door_common import open_door as shared  # noqa: E402
from fridge import constants as F  # noqa: E402
from fridge import open_door, side_grasp  # noqa: E402
from fridge.perception import WristCamera  # noqa: E402
from rammp_perception_interfaces.srv import Detect  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the helpers beside this script; /work holds the data
import depth_rule  # noqa: E402
import fast_timing  # noqa: E402
import pull_lib  # noqa: E402
import push_lib  # noqa: E402
import route_smooth  # noqa: E402
from pull_back_gains import HARDEST  # noqa: E402

open_door.configure("/tmp/side_general", 30, True, False)
sim.set_keep_out(None)  # as run_side.py
shared.configure(HANDLE_SOURCE="sam3")
dc.set_output(print)
scene, rb = dc.make_sim()
_JT_DEFAULT = rammp_arm.joint_trajectory  # rammp_arm's own timing, for the pull



_PULL_CACHE = {}  # grasp joints -> solve_pull's result: the configuration check solves it, the run reuses it


def solve_pull(q_g, hinge, hinge_sign, n):
    """The pull from grasp joints q_g (--swing / --pull): (knots or None, cart lengths, out_n, what, why). Cached."""
    key = (tuple(np.round(np.asarray(q_g, float), 6)), tuple(np.round(np.asarray(hinge, float), 6)), int(hinge_sign))
    if key not in _PULL_CACHE:
        _PULL_CACHE[key] = _solve_pull(q_g, hinge, hinge_sign, n)
    return _PULL_CACHE[key]


def _solve_pull(q_g, hinge, hinge_sign, n):
    out_n = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
    ee_g = np.r_[sim.tool_pose(scene, rb, q_g)[0], sim.tool_pose(scene, rb, q_g)[1]]
    if SWING_DEG:
        tried = []
        for hand_deg in HAND_TURNS_DEG(SWING_DEG):  # the largest hand turn the wrist allows; the impedance takes the rest
            poses, rad = pull_lib.swing_hand_turn(ee_g, hinge, hinge_sign, SWING_DEG, hand_deg)
            pk, worst, why_ns = pull_lib.swing_knots_ns(scene, rb, q_g, poses)  # the spare joint keeps J6 off its guard
            if pk is not None and PUSH:
                # the release that follows without a stop (RELEASE_ARC_DEG further along the arc, then the slide) must
                # solve from this swing's end too: a larger hand turn can leave the wrist no room for it (11-06)
                ee_end = np.r_[sim.tool_pose(scene, rb, np.asarray(pk[-1]))[0], sim.tool_pose(scene, rb, np.asarray(pk[-1]))[1]]
                ex_try, why_ex = push_lib.plan_exit(scene, rb, {"hinge": list(hinge), "hinge_sign": int(hinge_sign),
                                                                 "normal": list(n), "quat": list(ee_g[3:7])},
                                                    ee_end, pk[-1], SWING_DEG)
                if ex_try is None:
                    tried.append(f"hand {hand_deg:g} deg: the release after it: {why_ex}")
                    pk = None
                    continue
            if pk is not None:
                break
            tried.append(f"hand {hand_deg:g} deg: {why_ns}")
        why_ns = " | ".join(tried)
        pl = [float(np.linalg.norm(b_[0] - a_[0])) for a_, b_ in zip([(ee_g[:3], None)] + poses[:-1], poses)]
        msgs = [why_ns] if pk is None else []
        dd = poses[-1][0] - ee_g[:3]
        what = (f"door swing {SWING_DEG:g} deg about the hinge pin {np.round(hinge, 3)} (radius {rad * 100:.1f} cm), "
                f"the hand turning {hand_deg:g} deg: tool {np.dot(dd, out_n) * 100:.1f} cm out, "
                f"{np.linalg.norm(dd - np.dot(dd, out_n) * out_n) * 100:.1f} cm sideways; worst J6 {worst['J6']:.0f}, "
                f"J4 {worst['J4']:.0f} deg")
        return pk, pl, out_n, what, (msgs[-1].strip() if msgs else "")
    pk, pl, why = pull_lib.knots((scene, rb), ee_g, q_g, out_n, PULL_M)
    return pk, pl, out_n, f"{PULL_M * 100:.0f} cm back along {np.round(out_n, 3)} turning {pull_lib.TURN_DEG:g} deg", why



def reversal_deg(q_start, cfg):
    """How much the arm has to reverse joints on the way in with slide configuration cfg: per joint, where getting from
    q_start to the merge's start and then along the merge + slide go opposite ways, the smaller of the two moves (deg),
    summed. 0: every joint moves one way all through (no joint has to stop and come back)."""
    k = np.asarray(cfg["knots"], float)
    a = np.asarray(sim.short_delta(np.asarray(q_start, float), k[0]), float)
    b = k[-1] - k[0]
    return float(np.degrees(np.sum(np.where(a * b < 0, np.minimum(np.abs(a), np.abs(b)), 0.0))))

def depth_gate(q_send, ch, plan, n, tail, mono):
    """The depth rule (11-06): the route exactly as it will be timed and sent from q_send -- no part of the
    arm (any link, gripper open) deeper into the door, along the MEASURED normal n, than the arm's deepest point at the
    grasp. Raises SystemExit (nothing moves) otherwise."""
    tool_fn = lambda q: sim.tool_pose(scene, rb, q)[0]  # noqa: E731
    t, q_timed, _ = fast_timing.samples([q_send] + rammp_arm._unwrap_from(q_send, ch["route"]), rammp_arm.DT,
                                        tool_fn=tool_fn, protect_tail=tail, monotone_tail=mono)
    exc, i, link, _ = depth_rule.check(rb, q_timed, -np.asarray(n, float), np.asarray(plan["grasp"], float), q_timed[-1])
    if exc > depth_rule.DEPTH_TOL_M:
        raise SystemExit(f"not moving: depth rule -- {link} {exc * 100:.1f} cm deeper into the door than the grasp's "
                         f"deepest point, {t[i]:.2f} s into the {t[-1]:.2f} s route")
    stamp(f"depth rule ok: nothing deeper than the grasp's deepest point (worst {exc * 1000:+.1f} mm, {link}, "
          f"{len(q_timed)} samples)")


def tail_knots(route):
    """The route's knots within 11 cm of its end (fast_timing keeps them exact)."""
    end = sim.tool_pose(scene, rb, np.asarray(route[-1]))[0]
    k = 0
    for q in reversed(route):
        if np.linalg.norm(sim.tool_pose(scene, rb, np.asarray(q))[0] - end) > 0.11:
            break
        k += 1
    return k
# the per-joint time law with the tool-speed caps, in this process only
fast_timing.install(rammp_arm, tool_fn=lambda q: sim.tool_pose(scene, rb, q)[0])
cu = side_grasp.CuroboClient(node)
sam3 = node.create_client(Detect, "/perception/sam3/detect", callback_group=ReentrantCallbackGroup())
dc.set_camera(WristCamera(node, "/scene_camera/color/image_raw", "/scene_camera/color/camera_info",
                          "/scene_camera/depth/image_rect_raw", rammp_arm.FRAME, "", sam3=sam3,
                          sam3_prompts=["handle"], sam3_threshold=0.3))
ai = None
try:
    if not DRY:
        ai = rammp_arm.RammpArm("fridge_side_general_arm")  # waits for the arm's state and trajectory server
        stamp("arm interface up")
    h, orient, n, *_ = shared.detect_handle()
    # the door's direction MUST be measured: wait up to EST_WAIT_S for a fresh ok estimate, otherwise do not move
    # (11-06: a run on the assumed square door, the real door ~34 deg off, drove through the door)
    t_wait, est, why = time.time(), None, "no estimate file"
    while time.time() - t_wait < EST_WAIT_S:
        try:
            e = json.load(open(EST_FILE))
            age = time.time() - os.path.getmtime(EST_FILE)
        except (OSError, ValueError) as ex:
            e, age, why = None, None, f"cannot read {EST_FILE}: {ex}"
        if e is not None:
            if age > EST_MAX_AGE_S:
                why = f"{EST_FILE} is {age:.1f} s old (estimator not running?)"
            elif not e.get("ok"):
                why = f"estimator not ok: {e.get('why')}"
            else:
                est = e
                break
        time.sleep(0.2)
    if est is None:
        raise SystemExit(f"not moving: the door direction is not measured after {EST_WAIT_S:.0f} s -- {why}")
    n = np.asarray(est["normal"], float)
    stamp(f"handle {np.round(h, 3)} (this look); normal {np.round(n, 3)} (yaw {est.get('yaw_deg')}) measured, from {EST_FILE}")
    js = got["js"]
    q_now = np.array([js.position[js.name.index(f"arm_joint_{i}")] for i in range(1, 8)])
    _start = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--start=")), None)
    if _start:  # DRY only: plan as if the arm stood at this saved pose
        if not DRY:
            raise SystemExit("not moving: --start is for --dry")
        q_now = np.asarray(json.load(open(_start))["joints_rad"], float)
    stamp(f"arm joints {np.round(np.degrees(q_now), 1)}")
    stamp(f"hinge on the handle's {'left' if HINGE_SIGN > 0 else 'right'} (--hinge-right for a right-hinged door)")
    try:
        plan = side_grasp.plan_side_general(scene, rb, h, orient, n,
                                            side_grasp.hinge_from_handle(h, n, F.MEASURED, HINGE_SIGN),
                                            q_now, cu, retime=fast_timing.samples,
                                            after=(lambda q: (lambda r: None if r[0] is not None else r[4] or "no IK")(
                                                solve_pull(q, side_grasp.hinge_from_handle(h, n, F.MEASURED, HINGE_SIGN),
                                                           HINGE_SIGN, n))) if (PULL_M or SWING_DEG) else None)
    except dc.Refused as e:
        raise SystemExit(f"not moving: {e}")
    plan.pop("_timed", None)
    ch = plan["chosen"]
    # the timing for this route: smoothing keeps its last ~10 cm exact, one slow-down over the merge + slide
    fast_timing.install(rammp_arm, tool_fn=lambda q: sim.tool_pose(scene, rb, q)[0], **ch["timing_args"])
    stamp(f"planned in {plan['times']['total_s']:.2f} s: {len(ch['route'])} knots ({ch['n_free']} cuRobo), "
          f"{ch['timing']['duration_s']:.2f} s with fast_timing, slide config {ch['reconfig_deg']:.0f} deg from now, "
          f"{ch['limit_margin_deg']:.0f} deg inside the limits ({ch['limit_margin_joint']}); last 10 cm within "
          f"{ch['timing']['last_10cm_dev_max_m'] * 1000:.1f} mm of the slide line; cuRobo world {plan['world']}")
    json.dump(plan, open("/work/side_plan_general.json", "w"))
    pull = None
    if PULL_M or SWING_DEG:
        pk, pl, out_n, what, why_p = solve_pull(np.asarray(ch["route"][-1], float), plan["hinge"], plan["hinge_sign"], n)
        if pk is None:
            raise SystemExit(f"not moving: the pull ({what}) doesn't solve from the grasp: {why_p}")
        pull = (pk, pl, out_n, what)
        stamp(f"pull solved: {what}, {len(pk)} knots")
    push = None
    if PUSH:
        if not SWING_DEG:
            raise SystemExit("not moving: --push follows a --swing")
        ee_sw = np.r_[sim.tool_pose(scene, rb, np.asarray(pk[-1]))[0], sim.tool_pose(scene, rb, np.asarray(pk[-1]))[1]]
        ex, why_x = push_lib.plan_exit(scene, rb, plan, ee_sw, pk[-1], SWING_DEG)
        if ex is None:
            raise SystemExit(f"not moving: part 3: {why_x}")
        door_sw = dict(plan["door"]); door_sw["door_open_deg"] = float(SWING_DEG)
        door_sw["door_open_deg"] = float(ex["deg_after"])
        # the slide off the bar + out from the door after the release (11-06: room for rounding the edge)
        st_ex = {"position": list(ex["slide"][-1]), "ee_pos": list(np.r_[ex["slide_poses"][-1][0], ex["slide_poses"][-1][1]])}
        pp, why_p3 = push_lib.plan_push(scene, rb, door_sw, st_ex)
        if pp is None:
            raise SystemExit(f"not moving: part 3 (round the edge + swing to 90) doesn't plan: {why_p3}")
        push = (ex, pp)
        stamp(f"part 3 solved: let go, round the edge ({len(pp['round'])} knots), swing to 90 from placement "
              f"{pp['placement']}; closest the arm to itself {pp['self_min_m'] * 100:.1f} cm, to the door {pp['door_min_m'] * 100:.1f} cm, to the scene camera {pp['camera_min_m'] * 100:.1f} cm")
    if DRY:
        depth_gate(q_now, ch, plan, n, tail_knots(ch["route"]), 0)
        stamp(f"DRY: would send the route to {np.round(plan['grasp'], 3)}; nothing sent")
        raise SystemExit
    st = ai.get_state()
    moved = float(np.degrees(np.max(np.abs(sim.short_delta(np.asarray(st["position"]), q_now)))))
    if moved > 2.0:
        raise SystemExit(f"not moving: the arm moved {moved:.1f} deg since the plan's start joints")
    # smoothing settings from THIS route (nothing per spot): the knots within 11 cm of the end stay exact; from 3 knots
    # before the first non-free knot on, the speed only decreases (one slow-down into the handle)
    tool_fn = lambda q: sim.tool_pose(scene, rb, q)[0]  # noqa: E731
    end = tool_fn(np.asarray(ch["route"][-1]))
    tail = 0
    for k in reversed(ch["route"]):
        if np.linalg.norm(tool_fn(np.asarray(k)) - end) > 0.11:
            break
        tail += 1
    # only slow down over the SLIDE (the straight insertion into the handle): one even speed in, no speed-up-then-brake
    # (11-06: without it 0.18 m/s at 15-10 cm, then 0.12 inside the last 10 cm -- a bump). Not over the merge, so a
    # slow spot before the slide can't drag the insertion (11-06: a reach-edge dip once held 2 cm/s over 13 cm)
    mono = 0  # 11-06: full speed into the handle -- with no near-end cap the slide speeds up evenly and stops once
    fast_timing.install(rammp_arm, tool_fn=tool_fn, protect_tail=tail, monotone_tail=mono)
    depth_gate(np.asarray(st["position"], float), ch, plan, n, tail, mono)
    stamp(f"MOVING: first motion command now ({len(ch['route'])} knots, smoothed timing: last {tail} knots exact, "
          f"one slow-down over the last {mono})")
    res_in = ai.move_joints(ch["route"], speed_scale=1.0)
    st = ai.get_state()
    e = float(np.linalg.norm(np.asarray(st["ee_pos"][:3]) - plan["grasp"]))
    stamp(f"DONE: tool {np.round(st['ee_pos'][:3], 3)}, {e * 100:.1f} cm from the planned end; gripper "
          f"{st['gripper_pos']:.2f} (open, not closed)")
    if not res_in or e > ARRIVE_TOL_M:  # 11-06: an aborted reach-in (arm fault) went on to the pull -- never
        raise SystemExit(f"stopped: the reach-in did not arrive ({'failed' if not res_in else 'ok'}, {e * 100:.1f} cm from "
                         f"the planned end) -- nothing after it is sent")
    if pull is not None:
        pk, pl, out_n, what = pull
        # 11-06 ("bumpy, sudden movements"): every motion after the reach-in timed like the reach-in (fast_timing:
        # velocity / acceleration / bend limited, eased in and out), its PATH kept exactly (no shape smoothing: the knots
        # are the checked ones) -- not rammp_arm's straight joint segments between knots
        rammp_arm.joint_trajectory = _JT_DEFAULT  # 11-06: back to the arm's own timing for these
        ee0 = np.asarray(st["ee_pos"], float)
        # the gripper closes on the bar WHILE the arm swings (11-06): /setpoint/gripper is independent of the
        # arm's control mode; the 2F-85 takes ~1 s, so the bar is held before the hand's turn lags the door's much
        grip_t = (threading.Thread(target=ai.gripper, args=(1.0,), kwargs={"force": F.GRASP_FORCE}, daemon=True)
                  if SWING_DEG and "--no-grip" not in sys.argv else None)
        knots, lens, t_open = list(pk), list(pl), None
        if push is not None:
            # 11-06: NO stop between the pull and the release -- the pull runs straight on RELEASE_ARC_DEG further
            # along the door's arc in the same trajectory, the gripper opening slowly while the arm still moves (a stopped
            # hand opening pushed the door back); the arc stretched so it lasts >= OPEN_DURING_S past the swing's end
            ex, pp = push
            q_send = np.asarray(ai.get_state()["position"], float)
            stretch = 1.0
            for _ in range(10):
                arc_l = [x * stretch for x in ex["arc_lens"]]
                t_, q_, _ = _JT_DEFAULT(q_send, list(pk) + [list(x) for x in ex["arc"]], list(pl) + arc_l, 1.0)
                i45 = int(np.argmin([np.max(np.abs(sim.short_delta(np.asarray(x, float), np.asarray(pk[-1], float)))) for x in q_]))
                if t_[-1] - t_[i45] >= push_lib.OPEN_DURING_S:
                    break
                stretch *= 1.5
            knots, lens = list(pk) + [list(x) for x in ex["arc"]], list(pl) + arc_l
            t_open = max(0.0, float(t_[i45]) - push_lib.OPEN_LEAD_S)
        stamp(f"MOVING: pull, joint impedance (hardest): {what}" + ("; gripper closing on the bar meanwhile" if grip_t else "")
              + (f"; on {push_lib.RELEASE_ARC_DEG:g} deg further, the gripper opening from {t_open:.1f} s while it moves"
                 if t_open is not None else ""))
        if grip_t is not None:
            grip_t.start()
        open_t = None
        if t_open is not None:
            def _open_later(t0=time.time()):
                if grip_t is not None:
                    grip_t.join(timeout=5.0)
                time.sleep(max(0.0, t0 + t_open - time.time()))
                ai.gripper(0.0, speed=push_lib.GRIP_OPEN_SPEED)
            open_t = threading.Thread(target=_open_later, daemon=True)
            open_t.start()
        res = ai.move_joints(knots, cart_lengths=lens, impedance=HARDEST, speed_scale=1.0)
        if grip_t is not None:
            grip_t.join(timeout=5.0)
        if not res:
            raise SystemExit("stopped: the pull did not complete (the arm refused or aborted it) -- nothing after it is sent")
        st2 = ai.get_state(); d = np.asarray(st2["ee_pos"][:3], float) - ee0[:3]
        from scipy.spatial.transform import Rotation as _R
        turned = np.degrees((_R.from_quat(st2["ee_pos"][3:7]) * _R.from_quat(ee0[3:7]).inv()).magnitude())
        planned_end = sim.tool_pose(scene, rb, np.asarray(pk[-1]))[0]
        if push is not None:
            open_t.join(timeout=10.0)  # the slide only once the gripper is fully open
            deg_hand = push_lib.door_deg_from_hand(plan, st2["ee_pos"])
            stamp(f"pulled on to {deg_hand:.1f} deg (planned {ex['deg_after']:g}) while the gripper opened")
            # 11-06 (a jerk after the slide-off; stop between forward and push): the slide-off, rounding the edge,
            # the forward step and the swing as ONE trajectory under the hardest impedance -- no stop, and no switch to position
            # control (the driver switches the actuators one by one, which jerked the arm)
            seq = [list(x) for x in ex["slide"]] + [list(x) for x in pp["round"]] + [list(x) for x in pp["swing"]]
            seq_l = list(ex["slide_lens"]) + list(pp["round_lens"]) + list(pp["swing_lens"])
            stamp(f"gripper open ({float(ai.get_state()['gripper_pos']):.2f}); MOVING: off the bar, round the edge, forward, swing to "
                  f"{F.PUSH_OPEN_TO_DEG} -- one motion (impedance)")
            # 11-06 ("super slow and discontinued", then a jerk the impedance aborted on): after the release the
            # arm is free -- regular speed, timed smoothly (fast_timing: velocity / acceleration / bend limited, the checked
            # path kept exactly), not rammp_arm's straight joint segments between knots
            # under impedance the arm lags its path in proportion to joint speed: capped so the lag stays well inside the
            # tracking limit (11-06: at ~1 rad/s it passed 0.15 rad and the driver aborted)
            fast_timing.install(rammp_arm, speed_fraction=push_lib.AFTER_OPEN_FRACTION,
                                tool_fn=lambda q: sim.tool_pose(scene, rb, q)[0], protect_tail=10 ** 6, monotone_tail=0)
            acc0 = fast_timing.ACCEL.copy()
            fast_timing.ACCEL = np.array([push_lib.AFTER_OPEN_ACCEL] * 7)
            try:
                res_s = ai.move_joints(seq, cart_lengths=seq_l, impedance={**HARDEST, "path_tol": push_lib.AFTER_OPEN_PATH_TOL},
                                       speed_scale=1.0)
            finally:
                rammp_arm.joint_trajectory = _JT_DEFAULT
                fast_timing.ACCEL = acc0
            if not res_s:
                stamp("the motion after the release did not complete (the arm refused or aborted it)")
            st4 = ai.get_state()
            stamp(f"PUSH DONE: tool {np.round(st4['ee_pos'][:3], 3)}, "
                  f"{np.linalg.norm(np.asarray(st4['ee_pos'][:3]) - pp['pts'][-1][0]) * 100:.1f} cm from the swing's end")
        stamp(f"PULL DONE ({res}): tool {np.round(st2['ee_pos'][:3], 3)}: {np.dot(d, out_n) * 100:.1f} cm out along the "
              f"normal, {np.linalg.norm(d - np.dot(d, out_n) * out_n) * 100:.1f} cm sideways, hand turned {turned:.1f} deg; "
              f"{np.linalg.norm(np.asarray(st2['ee_pos'][:3]) - planned_end) * 100:.1f} cm from the planned end; "
              f"gripper {st2['gripper_pos']:.2f}")
    json.dump({"name": "fridge_side_grasp", "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
               "handle_detected": plan["handle"], "door_normal": plan["normal"], "grasp_tool_planned": plan["grasp"],
               "grasp_tool_measured": [float(v) for v in st["ee_pos"][:3]],
               "grasp_quat": [float(v) for v in st["ee_pos"][3:7]],
               "grasp_joints_rad": [float(v) for v in st["position"]], "start_joints_rad": plan["q0"],
               "grasp_offset_from_handle_m": [float(a - b) for a, b in zip(st["ee_pos"][:3], plan["handle"])]},
              open("/work/fridge_side_grasp_pose.json", "w"), indent=1)
finally:
    if ai is not None:
        ai.close()
    rclpy.shutdown()

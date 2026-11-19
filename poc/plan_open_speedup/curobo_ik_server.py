# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""PoC 2's IK side: cuRobo's IKSolver on the GPU, serving batched IK over a Unix socket -- a stand-in for the
batched IK service the planner would grow. Runs in the rammp-curobo image (see run_curobo.sh).

Requests (multiprocessing.connection, pickled numpy), all in the URDF root frame (arm_mount, the sim's frame too),
joints in the sim's arm order (arm_joint_1..7), quaternions xyzw:
    ("ik", pos (B,3), quat (B,4), seeds (B,7))  -> dict(q (B,7), ok (B,), pos_err (B,), rot_err (B,), ms)
    ("fk", q (B,7))                             -> dict(pos (B,3), quat (B,4))
    ("self", q (B,7))                           -> dict(d (B,)): cuRobo's self-collision cost, 0 = clear
usage: curobo_ik_server.py <urdf> <socket> [num_seeds] [regularization 0/1] [particle 0/1] [grad_iters] [batch]
`batch` > 0: every call padded (or split) to that many problems, so cuRobo can replay CUDA graphs.
"""

import os
import sys
import tempfile
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from rammp_curobo.robot_config import load_robot_config, load_robot_spec
from rammp_curobo.urdf_model import prepare_urdf, write_urdf

ARM = [f"arm_joint_{i}" for i in range(1, 8)]

urdf_src, sock = sys.argv[1], sys.argv[2]
num_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
regularization = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
particle = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
grad_iters = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
batch = int(sys.argv[7]) if len(sys.argv) > 7 else 0

t = TensorDeviceType()
# flatbot's own robot file (rammp-deployments config/robots/flatbot/curobo)
robot_file = os.environ.get("ROBOT_FILE", "/data/flatbot_curobo/robot.yaml")
urdf = write_urdf(
    prepare_urdf(open(urdf_src).read(), load_robot_spec(robot_file)), tempfile.mkdtemp()
)  # as the node
robot = RobotConfig.from_dict(load_robot_config(robot_file, urdf_path=urdf), t)
ik = IKSolver(
    IKSolverConfig.load_from_robot_config(
        robot,
        None,
        tensor_args=t,
        num_seeds=num_seeds,
        use_cuda_graph=batch > 0,  # needs one batch size: every call padded to `batch`
        self_collision_check=True,
        self_collision_opt=True,
        regularization=regularization,
        use_particle_opt=particle,
        grad_iters=grad_iters,
    )
)
world = RobotWorld(RobotWorldConfig.load_from_config(robot, None, tensor_args=t))
names = list(ik.kinematics.joint_names)
to_cu = [ARM.index(n) for n in names]  # sim order -> cuRobo order
to_sim = [names.index(n) for n in ARM]


def tens(a):
    return torch.as_tensor(
        np.ascontiguousarray(a, dtype=np.float32), device=t.device
    )  # cuRobo's kernels need C order


def wxyz(q):
    return np.asarray(q)[:, [3, 0, 1, 2]]


def xyzw(q):
    return np.asarray(q)[:, [1, 2, 3, 0]]


def _solve(pos, quat, seeds):
    goal = Pose(tens(pos), tens(wxyz(quat)))
    r = ik.solve_batch(goal, seed_config=tens(np.asarray(seeds)[:, to_cu])[None])
    m = len(pos)
    return (
        r.solution.reshape(m, -1, 7)[:, 0].cpu().numpy()[:, to_sim].astype(float),
        r.success.reshape(m, -1)[:, 0].cpu().numpy(),
        r.position_error.reshape(m, -1)[:, 0].cpu().numpy(),
        r.rotation_error.reshape(m, -1)[:, 0].cpu().numpy(),
    )


def solve(pos, quat, seeds):
    pos, quat, seeds = (np.asarray(x, float) for x in (pos, quat, seeds))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if batch <= 0:
        parts = [_solve(pos, quat, seeds)]
    else:  # chunks of exactly `batch`, the last padded with copies of its first problem
        parts = []
        for i in range(0, len(pos), batch):
            chunk = [a[i : i + batch] for a in (pos, quat, seeds)]
            k = len(chunk[0])
            padded = [np.concatenate([a, np.repeat(a[:1], batch - k, axis=0)]) for a in chunk]
            parts.append(tuple(x[:k] for x in _solve(*padded)))
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    q, ok, pe, re = (np.concatenate(x) for x in zip(*parts, strict=True))
    return dict(q=q, ok=ok, pos_err=pe, rot_err=re, ms=ms)


def fk(q):
    s = ik.kinematics.get_state(tens(np.asarray(q)[:, to_cu]))
    return dict(pos=s.ee_position.cpu().numpy(), quat=xyzw(s.ee_quaternion.cpu().numpy()))


def self_coll(q):
    s = world.get_kinematics(tens(np.asarray(q)[:, to_cu]))
    return dict(
        d=world.get_self_collision_distance(s.link_spheres_tensor.unsqueeze(1))
        .reshape(-1)
        .cpu()
        .numpy()
    )


# warm up: the first calls compile kernels
for b in (batch,) if batch > 0 else (1, 64, 1024):
    solve(
        np.tile([[0.5, 0.0, 0.4]], (b, 1)),
        np.tile([[0, 0.7071, 0, 0.7071]], (b, 1)),
        np.zeros((b, 7)),
    )
print(
    f"READY joints {names} seeds {num_seeds} regularization {regularization} particle {particle} grad_iters {grad_iters} batch {batch}",
    flush=True,
)
with Listener(sock, family="AF_UNIX") as listener:
    while True:
        with listener.accept() as conn:
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                op, args = req[0], req[1:]
                if op == "quit":
                    sys.exit(0)
                conn.send({"ik": solve, "fk": fk, "self": self_coll}[op](*args))

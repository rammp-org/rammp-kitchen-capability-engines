# ruff: noqa: E501, T201  -- PoC benchmark code, not capability code
"""The planner side of the IK socket (curobo_ik_server.py)."""

import time
from multiprocessing.connection import Client

import numpy as np


class CuroboIK:
    def __init__(self, sock="/data/ik.sock", wait_s=300):
        t0 = time.time()
        while True:
            try:
                self.conn = Client(sock, family="AF_UNIX")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.time() - t0 > wait_s:
                    raise
                time.sleep(1)
        self.calls, self.gpu_ms, self.wall_ms, self.problems = 0, 0.0, 0.0, 0

    def _ask(self, *req):
        self.conn.send(req)
        return self.conn.recv()

    def ik(self, pos, quat, seeds):
        t0 = time.perf_counter()
        r = self._ask(
            "ik", np.asarray(pos, float), np.asarray(quat, float), np.asarray(seeds, float)
        )
        self.calls += 1
        self.problems += len(pos)
        self.gpu_ms += r["ms"]
        self.wall_ms += (time.perf_counter() - t0) * 1e3
        return r

    def fk(self, q):
        return self._ask("fk", np.asarray(q, float))

    def self_collision(self, q):
        return self._ask("self", np.asarray(q, float))["d"]

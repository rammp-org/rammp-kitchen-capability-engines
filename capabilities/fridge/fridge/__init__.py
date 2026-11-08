"""The fridge capability.

The planning's local IK solves thousands of 6x6 systems per search. Multi-threaded OpenBLAS wakes its
threads for each one, and on a busy Jetson (cuRobo, SAM 3 and perception share it) each solve then took
~26 ms instead of ~10 us: 70 of a refused search's 147 s (2026-11-08). One thread is plenty for these
sizes. Set here, before numpy is first imported by the package, and in the Dockerfile; an explicit
setting in the environment wins.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

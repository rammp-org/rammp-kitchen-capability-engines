"""Reading and writing a trajectory point's time_from_start.

One place, because the write has an edge every hand-rolled copy but one got
wrong: a time whose fraction rounds up to a whole second (t = 1.9999999996)
must carry into `sec` — written as sec=1, nanosec=1000000000 it is not a
valid Duration, and what a consumer makes of it is its own business.
"""


def secs(duration):
    """A builtin_interfaces Duration as float seconds."""
    return duration.sec + duration.nanosec * 1e-9


def set_stamp(duration, t):
    """Write float seconds `t` into a Duration, carrying a rounded-up fraction."""
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1000000000:
        sec, nanosec = sec + 1, nanosec - 1000000000
    duration.sec = sec
    duration.nanosec = nanosec

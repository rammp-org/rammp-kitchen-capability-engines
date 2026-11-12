"""The run's console.log: every say() line also goes to it while a run has one (set_tee)."""

import io

import button_press
from button_press import say, set_tee


def test_say_also_writes_to_the_tee_until_it_is_cleared():
    out, tee = io.StringIO(), io.StringIO()
    button_press.set_output(out)
    try:
        set_tee(tee)
        say("  run ended OK")
        set_tee(None)
        say("after the run")
    finally:
        button_press.set_output(None)
        set_tee(None)
    assert out.getvalue() == "  run ended OK\nafter the run\n"
    assert tee.getvalue() == "  run ended OK\n"


def test_a_closed_tee_never_stops_the_run():
    tee = io.StringIO()
    tee.close()
    button_press.set_output(io.StringIO())
    try:
        set_tee(tee)
        say("still fine")
    finally:
        button_press.set_output(None)
        set_tee(None)

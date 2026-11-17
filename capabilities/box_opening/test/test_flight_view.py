"""The flight view (box_opening/flight_view.py): a run's console read into one short line a step.

The lines fed here are the runs' own, cut to what the view reads: flatbot 2026-11-12, the evening
the owner asked for it — the two runs that took the lid off (20:53) and the ones that did not.
"""

import pytest

from box_opening.flight_view import FlightView

P = "[press_demo] "


def _view(lines, code, **how):
    """The view's lines for a run whose console said `lines` [(seconds, text)] and ended `code`."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    for t, text in lines:
        now[0] = float(t)
        view.feed(text)
    now[0] += 5.0
    view.finish(code, **how)
    assert out[0].startswith("BOX OPENING   ")
    return out[1:]


FOUND = [
    (0, P + "body check: 30 boxes (/config/robot/curobo/planner.yaml), 58 spheres on 12 links"),
    (16, "survey:angle-1     motion       1.00 full                   0     0.75    0.25    0.24"),
    (16, "EXECUTING (human on the physical e-stop; Ctrl+C stops everything)"),
    (
        18,
        P + "WRIST SURVEY survey:angle-1 (the wrist turned to look at [0.60, 0.00]): the container at [0.645, -0.156], "
        "its top 89 mm above the counter, in 1 of 1 frame; button unchecked",
    ),
    (
        18,
        P + "WRIST SURVEY: box at [0.645, -0.156] yaw 0.0 deg from 2982 lid points (WRIST 0.60, 1 sighting (wrist); "
        "17.7 s to locate [streams 0.02 seg 0.17], 18.8 s since start) — reaching out",
    ),
    (
        19,
        P
        + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm; gravity alone moves joint_4 0.10 Nm",
    ),
    (
        22,
        P
        + "AIM: button circle at pixel (315, 192) r 31 px, 43 mm across, +1.3 mm above its own lid, depth on 98% of it",
    ),
    (22, P + "AIMED from the look pose: button at [0.663, -0.154, 0.310], 18 mm from the scene camera's"),
]
PRESSED = [
    (
        22,
        P + "ONE MOTION from the look pose to below the button: reach:re-aim, reach:slide-in, "
        "depth:approach, depth:press",
    ),
    (22, P + "PRESS (compliant): straight down 72 mm to the button's bottom, its reference on 15 mm below it"),
    (
        26,
        P + "CONTACT: the pads met the box at 0.351, +41 mm from the top the cameras measured; the stroke "
        "cut 41 mm short",
    ),
    (26, P + "PRESSED with ~21 N, 2 N sideways: the pads held 21 mm above the stroke's cut end"),
    (
        26,
        P + "GRIP STRAIGHT FROM THE PRESS: up 45 mm, the fingers opening round the knob once the pads are 37.5 mm over",
    ),
]


def test_a_run_that_takes_the_lid_off_reads_as_eight_steps_and_done():
    """20:53:57: found by the wrist, aimed, pressed, gripped straight from the press, and the lid lifted,
    carried and set down in one motion. A step's lower-case line as it starts, its numbered line as it
    ends, with how long it took; the lift, the carry and the set-down end together, as they flew."""
    out = _view(
        FOUND
        + PRESSED
        + [
            (29, P + "lid goes to [0.65, 0.05, 0.206] — 200 mm from the box (45 mm outside its no-land radius)"),
            (
                29,
                P
                + "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch — 40 mm straight up",
            ),
            (34, P + "LID GRIPPED — grip 0.492 vs band [0.3, 0.55]"),
            (34, P + "SET-DOWN CONTACT: the lid met the table +62 mm from where the plan put it"),
            (34, P + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance: home"),
            (39, P + "DONE — box open, lid set down at [0.65, 0.05] (clear), arm home"),
            (40, P + "VIDEO: what the wrist and scene cameras saw from the grip on: /state/captures/lid-video.mp4"),
        ],
        0,
    )
    assert out == [
        "   find      ...   looking for the box",
        " 1 FIND      ok    box 0.66 m out (wrist camera, 1 view)      18.0 s",
        "   look      ...   moving to look at the button",
        " 2 LOOK      ok    button found, 43 mm across, 0.68 m out      3.0 s",
        "   press     ...   pressing the button",
        " 3 PRESS     ok    21 N                                        4.0 s",
        "             note  lid touched 41 mm higher than the camera read it",
        "   grip      ...   rising 45 mm, opening the fingers, coming down round the knob",
        " 4 GRIP      ok    knob held                                   8.0 s",
        " 5 LIFT      ok",
        " 6 CARRY     ok    lid to 20 cm from the box",
        " 7 SET DOWN  ok    let go where the lid met the table",
        "   home      ...   letting go and going home",
        " 8 HOME      ok                                                5.0 s",
        "RESULT  DONE — box open, lid on the table, arm home           45 s",
    ]


def test_a_grip_that_misses_a_second_press_and_a_knob_out_of_reach_each_say_so_and_what_follows():
    """19:12:49: the grip closed on nothing, the look read the knob down, the second press, a knob the
    camera could not find, and no way down to it. A failure says what the run does next; what the run
    could not tell is `??`; the result names the step that ended it and where everything was left."""
    out = _view(
        FOUND[:1]
        + [
            (
                4,
                P + "WRIST LOOK: the box at [0.724, -0.236, 0.297]; the scene camera had it at [0.822, -0.192, 0.382], "
                "107 mm away — the wrist's is taken",
            ),
            (
                4,
                P + "WRIST SURVEY: box at [0.724, -0.236] yaw 0.0 deg from 2033 lid points (WRIST 0.60) — reaching out",
            ),
            (7, P + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm"),
            (9, P + "AIMED from the look pose: button at [0.750, -0.247, 0.319], 28 mm from the scene camera's"),
            (10, P + "TURN: 15 deg short of straight down (no hand nearer straight down flies here)"),
            (10, P + "ONE MOTION from the look pose to below the button: reach:re-aim, reach:slide-in, depth:turn"),
            (16, P + "PRESSED with ~17 N, 2 N sideways: the pads held 24 mm above the stroke's cut end"),
            (16, P + "GRIP STRAIGHT FROM THE PRESS: up 25 mm, the fingers opening to the knob's width"),
            (20, P + "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch — 40 mm"),
            (21, "STOP: leg grip:close -> arrived (grip 0.768 vs band [0.3, 0.55]) — arm holds"),
            (21, P + "the grip straight from the press did not hold the knob (grip 0.768 vs band [0.3, 0.55])"),
            (
                21,
                P + "the grip straight from the press: the fingers closed on air — back to the look pose "
                "to look at the knob",
            ),
            (21, P + "ONE MOTION from the button back to the look pose: depth:rise, depth:turn-back, reach:slide-out"),
            (
                27,
                P
                + "AFTER PRESS 1: knob 1.4 mm above the lid (2 looks); the button 7 mm from where the press was aimed",
            ),
            (
                27,
                P
                + "the knob is not up (1.4 mm above the lid): pressing once more — the same press, from the look pose",
            ),
            (28, P + "PRESS 2 of 2: ONE MOTION from the look pose to below the button: reach:re-aim, reach:slide-in"),
            (33, P + "PRESSED with ~18 N, 3 N sideways: the pads held 25 mm above the stroke's cut end"),
            (33, P + "ONE MOTION from the button back to the look pose: depth:rise, depth:turn-back, reach:slide-out"),
            (
                45,
                P + "AFTER PRESS 2: the wrist does not find the button from the look pose (0 of 0 hits "
                "agree in 43 frames)",
            ),
            (45, P + "the knob is not judged after press 2 (the wrist does not find the button): going on to the grip"),
            (
                46,
                P + "STOP: the hand does not come down to the knob (the way down (walk in: 0 deg short: IK did not "
                "converge 67% of the way)) — the lid is not taken off — returning home",
            ),
        ],
        1,
    )
    assert out == [
        "   find      ...   looking for the box",
        "             note  scene camera puts it 107 mm away; using the wrist's",
        " 1 FIND      ok    box 0.76 m out (wrist camera)               4.0 s",
        "   look      ...   moving to look at the button",
        " 2 LOOK      ok    button found, 0.79 m out                    2.0 s",
        "   press     ...   pressing the button",
        " 3 PRESS     ok    17 N, hand tilted 15°                       6.0 s",
        "   grip      ...   rising 25 mm, opening the fingers, coming down round the knob",
        " 4 GRIP      FAIL  fingers closed on nothing                   5.0 s",
        "             next  going back to look at the knob",
        "   check     ...   going back to look at the knob",
        " 5 CHECK     ok    knob is down (1.4 mm); pressing again       6.0 s",
        "   press 2   ...   pressing the button",
        " 6 PRESS 2   ok    18 N, hand tilted 15°                       5.0 s",
        "   check     ...   going back to look at the knob",
        " 7 CHECK     ??    camera cannot find the knob; trying the grip  12.0 s",
        " 8 GRIP      FAIL  the hand cannot come down to the knob from here",
        "   home      ...   going home",
        " 9 HOME      ok                                                5.0 s",
        "RESULT  FAILED at step 8 (GRIP)                               51 s",
        "        why: the hand cannot come down to the knob from here",
        "        box: pressed 2 times, not confirmed open    lid: on the box    arm: home",
    ]


def test_a_box_no_press_reaches_fails_the_press_with_how_far_out_it_is():
    """19:12:26, the button 0.86 m out; 20:15:04, a thing by the microwave taken for the box 1.16 m out
    and nothing sent home; 18:41, three views of clutter and the scene camera's box."""
    out = _view(
        FOUND[:1]
        + [
            (
                4,
                P + "WRIST SURVEY: box at [0.792, -0.245] yaw 0.0 deg from 1503 lid points (WRIST 0.60) — reaching out",
            ),
            (6, P + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm"),
            (9, P + "AIM: button circle at pixel (322, 171) r 32 px, 45 mm across, +1.0 mm above its own lid"),
            (9, P + "AIMED from the look pose: button at [0.820, -0.254, 0.316], 29 mm from the scene camera's"),
            (
                10,
                P + "STOP: no press reaches the aimed button (the way down (walk in: 0 deg short: a joint jump of 0.21 "
                "rad 52% of the way)) — nothing was pressed — returning home",
            ),
        ],
        1,
    )
    assert out[3:] == [
        " 2 LOOK      ok    button found, 45 mm across, 0.86 m out      3.0 s",
        " 3 PRESS     FAIL  the box is out of the arm's reach (0.86 m out)",
        "   home      ...   going home",
        " 4 HOME      ok                                                5.0 s",
        "RESULT  FAILED at step 3 (PRESS)                              15 s",
        "        why: the box is out of the arm's reach (0.86 m out)",
        "        box: not pressed    lid: on the box    arm: home",
    ]
    out = _view(
        FOUND[:1]
        + [
            (
                3,
                P + "WRIST LOOK: the box at [1.075, -0.426, 0.291]; the scene camera had it at [0.824, -0.234, 0.393], "
                "316 mm away — the wrist's is taken",
            ),
            (3, P + "WRIST SURVEY: box at [1.075, -0.426] yaw 0.0 deg from 380 lid points (WRIST 0.60) — reaching out"),
            (
                4,
                P
                + "STOP: no reach-out press plans for the scene camera's box (30 deg: the look pose (its wrist 111 mm "
                "past the arm's reach)); nothing moved",
            ),
        ],
        1,
    )
    assert out[1:] == [
        "             note  scene camera puts it 316 mm away; using the wrist's",
        " 1 FIND      ok    box 1.16 m out (wrist camera)               3.0 s",
        " 2 PRESS     FAIL  the box is out of the arm's reach (1.16 m out)",
        "RESULT  FAILED at step 2 (PRESS)                               9 s",
        "        why: the box is out of the arm's reach (1.16 m out)",
        "        box: not pressed    lid: on the box    arm: where the look left it (not sent home)",
    ]
    two = "wrist camera: two places pass as this container, [0.755, -0.219] and [0.990, 0.185]: refusing to guess"
    out = _view(
        FOUND[:1]
        + [
            (3, P + "WRIST SURVEY survey:angle-hint (the wrist turned to look where the box may stand): " + two),
            (5, P + "WRIST SURVEY survey:angle-1 (the wrist turned to look at [0.45, 0.15]): " + two),
            (7, P + "WRIST SURVEY survey:angle-2 (the wrist turned to look at [0.45, 0.30]): " + two),
            (10, P + "WRIST LOOK: no container (%s) — the scene camera's box is taken" % two),
            (
                10,
                P + "SCENE: box at [0.788, -0.213] yaw 0.0 deg from 482 lid points (SEG 0.79, 2 "
                "sightings) — reaching out",
            ),
            (
                12,
                P + "STOP: no reach-out press plans for the scene camera's box (30 deg: the way down (walk in)) — "
                "nothing was pressed — returning home",
            ),
        ],
        1,
    )
    assert out[1:4] == [
        "             note  more than one thing looks like the box; looking from another angle",  # said once
        "             note  wrist camera found no box; using the scene camera's",
        " 1 FIND      ok    box 0.82 m out (scene camera, 3 views)     10.0 s",
    ]


def test_a_refused_move_and_a_stop_with_the_lid_in_the_hand_say_where_things_were_left():
    """18:42: the safety gate refused the carry, so no finger closed (a note, and the grip's failure in its
    own words). And a carry that stops part way: the lid in the hand, the arm holding where it stopped."""
    out = _view(
        FOUND
        + PRESSED
        + [
            (
                30,
                P
                + "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch — 40 mm straight up",
            ),
            (30, "REFUSED: lid:carry — trajectory sanity gate: arm_joint_2 excursion 0.444 rad > |Δ| + margin 0.444"),
            (
                30,
                P + "the grip straight from the press did not hold the knob (trajectory sanity gate: arm_joint_2 "
                "excursion 0.444 rad > |Δ| + margin 0.444)",
            ),
        ],
        1,
    )
    assert out[8:10] == [
        "             note  the safety gate refused a planned move (lid:carry)",
        " 4 GRIP      FAIL  a planned move was refused, so the fingers never closed   4.0 s",
    ]
    out = _view(
        FOUND
        + PRESSED
        + [
            (29, P + "lid goes to [0.60, 0.05, 0.2] — 210 mm from the box"),
            (29, P + "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch — 40 mm"),
            (33, P + "LID GRIPPED — grip 0.45 vs band [0.3, 0.55]"),
            (
                33,
                P + "STOP: the lid's way from the box to the table stopped during lid:carry (path tolerance) — the arm "
                "holds the lid",
            ),
        ],
        1,
        held=True,
    )
    assert out[8:] == [
        " 4 GRIP      ok    knob held                                   7.0 s",
        " 5 CARRY     FAIL  stopped part way (path tolerance)",
        "RESULT  FAILED at step 5 (CARRY)                              38 s",
        "        why: stopped part way (path tolerance)",
        "        box: open (knob up)    lid: in the hand    arm: holding where it stopped",
    ]


def test_the_view_never_stops_the_reader_that_feeds_it():
    """It only reads: a line it does not know is passed over, and neither a line it cannot read, a console
    that fails, nor an end it was not told of raises. node.Run._read feeds it from the thread that drains
    the run's pipe; a reader that died would leave the run blocked on a full pipe, mid-motion."""

    def broken(line):
        raise OSError("the console is gone")

    view = FlightView(write=broken)
    view.start()
    for text in (
        None,
        3.5,
        b"\xff\xfe",
        "",
        "STOP: ",
        P + "PRESSED with ~x N",
        P + "AIMED from the look pose: button at [",
    ):
        view.feed(text)
    view.finish(None, cancelled=True)
    out = []
    view = FlightView(write=out.append, clock=lambda: 0.0)
    view.feed(P + "LID GRIPPED — grip 0.4 vs band [0.3, 0.55]")  # fed before start(): still no raise
    view.finish(None)
    assert out[-2].startswith("RESULT  FAILED") and "lid: in the hand" in out[-1]
    # a cancel ends the step in flight, and the result says so
    out = _view(FOUND[:1], None, cancelled=True)
    assert out[1] == " 1 FIND      FAIL  cancelled                                   5.0 s"
    assert out[2].startswith("RESULT  CANCELLED at step 1 (FIND)")
    # relay(): each line logged as before, then read
    seen = []
    view = FlightView(write=out.append, clock=lambda: 0.0)
    both = view.relay(lambda level, text: seen.append((level, text)))
    both(20, P + "DONE — box open, lid set down at [0.65, 0.05] (clear), arm home")
    assert seen == [(20, P + "DONE — box open, lid set down at [0.65, 0.05] (clear), arm home")] and view.done


@pytest.mark.parametrize(
    "stop, step",
    [
        ("the way home stopped during home (path tolerance) — the arm holds", "HOME"),
        ("the way down to the knob stopped during grip:down (the driver aborted) — the arm holds", "GRIP"),
        ("the way in stopped during depth:approach (cancelled) — the arm holds", "PRESS"),
        (
            "the lid's way from the box to the table stopped during place:touch (path tolerance) — the "
            "arm holds the lid",
            "SET DOWN",
        ),
        ("pressed, but the knob did not pop — returning home", "PRESS"),
    ],
)
def test_a_stop_names_the_step_it_ended(stop, step):
    out = _view(
        FOUND[:1]
        + [(3, P + "WRIST SURVEY: box at [0.70, -0.20] yaw 0 (WRIST 0.60) — reaching out"), (5, P + "STOP: " + stop)],
        1,
    )
    assert any(line[3:12].strip() == step and " FAIL " in line for line in out), out

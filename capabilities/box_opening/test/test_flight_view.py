"""The flight view (box_opening/flight_view.py): a run's console read into one short line a step.

The lines fed here are the runs' own, cut to what the view reads: flatbot 2026-11-12, the evening
the owner asked for it — the two runs that took the lid off (20:53) and the ones that did not.
"""

import pytest

from box_opening.flight_view import FlightView, no_press_reason, not_seen_reason

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


def test_a_box_no_press_reaches_fails_the_press_with_why_and_how_far_out_it_is():
    """19:12:26, the button 0.86 m out; 20:15:04, a thing by the microwave taken for the box 1.16 m out
    and nothing sent home; 18:41, three views of clutter and the scene camera's box. Only the second is
    "out of the arm's reach" (its look's wrist stands past it): the others say what refused the way in."""
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
        " 3 PRESS     FAIL  no press plans from here (0.86 m out): the way down is not a line the arm can follow",
        "   home      ...   going home",
        " 4 HOME      ok                                                5.0 s",
        "RESULT  FAILED at step 3 (PRESS)                              15 s",
        "        why: no press plans from here (0.86 m out): the way down is not a line the arm can follow",
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


# flatbot 2026-11-17 16:43:13 (run 2), the run's own lines
RUN2_FOUND = (
    P + "WRIST SURVEY: box at [0.712, -0.331] yaw 0.0 deg from 1978 lid points (WRIST 0.60, 1 sighting (wrist); "
    "the wrist camera reads the table at 0.206 seen over x 0.53..1.16, y -0.83..-0.02) — reaching out"
)
RUN2_WHY = (
    "no reach-out press plans for the scene camera's box (30 deg: the way down (the slide in: trajectory sanity "
    "gate: arm_joint_1 excursion 2.012 rad > |Δ| + margin 1.632; arm_joint_3 excursion 1.525 rad > |Δ| + margin "
    "1.120); 35 deg: the way down (the slide in: trajectory sanity gate: arm_joint_1 excursion 2.188 rad > |Δ| + "
    "margin 2.049; arm_joint_3 excursion 1.668 rad > |Δ| + margin 1.567); 40 deg: the way down (the press stroke "
    "is not a straight line the arm can fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded "
    "20 mm + 5 mm), 46% of the way)); 45 deg: the way down (the press stroke is not a straight line the arm can "
    "fly (the chair and user: arm_half_arm_1_link 0 mm into person/lap (padded 20 mm + 5 mm), 92% of the way)))"
)
RUN2_REASON = (
    "no press plans from here (0.79 m out): the slide in swings the arm wide (30, 35 deg); the press stroke "
    "would pass too close to the user (40, 45 deg)"
)


def test_a_press_that_does_not_plan_says_the_real_reason_not_that_the_box_is_out_of_reach():
    """Run 2 of 2026-11-17: the box 0.79 m out, well within the arm's reach. The slide in swung the arm wide
    at 30 and 35 deg (the sanity gate) and the press stroke crossed the user's padding at 40 and 45. The
    console said "the box is out of the arm's reach (0.79 m out)": the one thing that was not so."""
    out = _view(FOUND[:1] + [(8, RUN2_FOUND), (16, P + "STOP: " + RUN2_WHY + "; nothing moved")], 1)
    assert out[1:] == [
        " 1 FIND      ok    box 0.79 m out (wrist camera)               8.0 s",
        " 2 PRESS     FAIL  " + RUN2_REASON,
        "RESULT  FAILED at step 2 (PRESS)                              21 s",
        "        why: " + RUN2_REASON,
        "        box: not pressed    lid: on the box    arm: where the look left it (not sent home)",
    ]
    # each part of the way in, and what refused it there, in the run's own words for them
    for why, said in (
        ("30 deg: the look pose (its wrist 22 mm past the arm's reach)", "the box is out of the arm's reach"),
        (
            "30 deg: the look pose (its wrist 60 mm past the arm's reach); 35 deg: the way down (the slide in: a "
            "joint limit 40% of the way)",
            "no press plans from here: out of the arm's reach (30 deg); the slide in runs a joint to its limit (35 "
            "deg)",
        ),
        ("30 deg: the slide in (IK did not converge 52% of the way)", "the slide in leaves what the arm reaches"),
        ("45 deg: the look pose (START_IN_LIMIT_MARGIN (joint_4))", "the look pose runs a joint to its limit"),
        (
            "40 deg: the way down (walk in: the chair and user: arm_forearm_link 3 mm into seat/armrest_arm)",
            "the way down would pass too close to the chair (40 deg)",
        ),
        (
            "30 deg: the slide in (the cabinet/edge0: arm_bracelet_link 4 mm into it)",
            "the slide in would meet the table",
        ),
        ("30 deg: the way down (walk in)", "the way down does not plan (30 deg)"),
    ):
        assert said in no_press_reason("no reach-out press plans for the scene camera's box (%s)" % why), why
    assert no_press_reason("no press reaches the aimed button (x)", 0.8).startswith("no press plans from here (0.80 m")


def test_a_run_that_finds_no_press_from_where_the_chair_stands_is_guided_and_goes_on():
    """The run does not end on it (owner 2026-11-17: "it should NOT just fail EVER"): the press's failure
    says why, the line-up is a step of its own with each thing the person is told and the question, and
    at their yes the run looks for the box again and reads on as any run."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    for t, text in ((0, FOUND[0][1]), (8, RUN2_FOUND), (16, P + "NO WAY IN: " + RUN2_WHY + "; nothing moved")):
        now[0] = float(t)
        view.feed(text)
    view.guide_start("the arm rests and the wrist looks: move the chair as told")
    how = "answer in open_box_terminal, or publish y or n on /box_opening/answer (std_msgs/String)"
    for t, say, ask in (
        (17, "Looking for the container", None),
        (20, "Move closer, about 15 cm", None),
        (24, "Hold still, checking", None),
        (26, "Ready. Open the container?", "y / n"),
        (27, "Hold still, checking", None),
        (29, "Ready. Open the container?", "y / n"),
    ):
        now[0] = float(t)
        view.guide_say(say, ask=ask, how=how)
    now[0] = 31.0
    view.guide_end(True, "yes", again=True)
    now[0] = 36.0
    view.feed(
        P + "WRIST SURVEY survey:angle-1 (the wrist turned to look at [0.60, 0.00]): the container at [0.640, "
        "-0.200], its top 89 mm above the counter, in 1 of 1 frame; button unchecked"
    )
    view.feed(P + "WRIST SURVEY: box at [0.640, -0.200] yaw 0.0 deg from 2000 lid points (WRIST 0.60) — reaching out")
    now[0] = 60.0
    # (the run that goes on is a run as any: "lid on the table" is said of a set-down it felt meet the table)
    view.feed(P + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped there")
    view.feed(P + "DONE — box open, lid on the table at [0.65, 0.05] (clear), let go where the arm felt it, arm home")
    view.finish(0)
    assert out[1:] == [
        "   find      ...   looking for the box",
        " 1 FIND      ok    box 0.79 m out (wrist camera)               8.0 s",
        " 2 PRESS     FAIL  " + RUN2_REASON,
        "             next  the run does not end here: wait while the arm comes to rest",
        "   line up   ...   the arm rests and the wrist looks: move the chair as told",
        "             say   Looking for the container",
        "             say   Move closer, about 15 cm",
        "             say   Hold still, checking",
        "             ask   Ready. Open the container?   y / n",
        "             how   " + how,  # once
        "             say   Hold still, checking",
        "             ask   Ready. Open the container?   y / n",
        " 3 LINE UP   ok    yes                                        15.0 s",
        "   find      ...   looking for the box again",
        " 4 FIND      ok    box 0.67 m out (wrist camera, 1 view)       5.0 s",
        " 5 HOME      ok",
        "RESULT  DONE — box open, lid on the table, arm home           60 s",
    ]


def test_a_line_up_that_ends_without_its_yes_is_the_runs_end_and_says_where_the_arm_was_left():
    """Every way the guidance can end says what happened: out of time (with what was still to do), a
    cancel, a wrist the look saw out of reach; and the result is written once, by whoever ends the run."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start(find=False)  # line_up on: the line-up comes before anything looks for the box
    view.guide_start("lining the chair up with the box: move it as told")
    view.guide_say("Move closer, about 15 cm")
    now[0] = 35.0
    view.guide_say("Press open again when you are in place")
    view.guide_end(False, "not in place within 35 s (last told: Move closer, about 15 cm)")
    view.guide_say("The wrist is turning back to rest")
    view.arm = "at rest, the wrist turned back"
    now[0] = 41.0
    view.finish(5)
    view.finish(None, cancelled=True)  # the engine's own end of the goal, after the line-up wrote the result
    assert out[1:] == [
        "   line up   ...   lining the chair up with the box: move it as told",
        "             say   Move closer, about 15 cm",
        "             say   Press open again when you are in place",
        " 1 LINE UP   FAIL  not in place within 35 s (last told: Move closer, about 15 cm)  35.0 s",
        "             say   The wrist is turning back to rest",
        "RESULT  FAILED at step 1 (LINE UP)                            41 s",
        "        why: not in place within 35 s (last told: Move closer, about 15 cm)",
        "        box: not pressed    lid: on the box    arm: at rest, the wrist turned back",
    ]
    # an n (the cancel) in the middle of a run's line-up, the arm having gone home from its look first
    out.clear()
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    view.feed(P + "WRIST SURVEY: box at [0.80, -0.25] yaw 0.0 deg (WRIST 0.60) — reaching out")
    view.feed(P + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm")
    view.feed(P + "AIMED from the look pose: button at [0.820, -0.254, 0.316], 29 mm from the scene camera's")
    view.feed(
        P + "NO WAY IN: no press reaches the aimed button (the way down (walk in: 0 deg short: a joint jump of 0.21 "
        "rad 52% of the way)) — nothing was pressed — returning home"
    )
    view.guide_start()
    view.guide_end(False, "cancelled: n")
    view.arm = "stopped where it was (a wrist that looked stands turned)"
    view.finish(130, cancelled=True)
    assert out[4:] == [
        " 2 LOOK      ok    button found, 0.86 m out                    0.0 s",
        " 3 PRESS     FAIL  no press plans from here (0.86 m out): the way down is not a line the arm can follow",
        "             next  the run does not end here: wait while the arm comes to rest",
        "   home      ...   going to rest before the chair is guided",
        " 4 HOME      ok                                                0.0 s",
        "   line up   ...   move the chair as told; the arm rests and the wrist looks",
        " 5 LINE UP   FAIL  cancelled: n                                0.0 s",
        "RESULT  CANCELLED at step 5 (LINE UP)                          0 s",
        "        why: cancelled: n",
        "        box: not pressed    lid: on the box    arm: stopped where it was (a wrist that looked stands turned)",
    ]
    # the wrist sees a lid-sized top out of the arm's reach and the scene camera has none: the find fails,
    # with why, and the chair is moved (no NO BOX, no end)
    out.clear()
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    view.feed(
        P + "OUT OF REACH: the wrist camera sees a lid-sized top at [1.02, -0.20], out of the arm's reach; "
        "nothing was pressed"
    )
    assert out[2:] == [
        " 1 FIND      FAIL  what the wrist camera sees is out of the arm's reach   0.0 s",
        "             next  the run asks for the chair to be guided",
    ]

    # the guide's three never raise either (a console that fails is not the run's problem)
    def broken(line):
        raise OSError("the console is gone")

    view = FlightView(write=broken)
    view.guide_start()
    view.guide_say("Move closer, about 15 cm")
    view.guide_say("Ready. Open the container?", ask="y / n", how="x")
    view.guide_end(True, "yes")
    view.guide_end(False, "cancelled")


NOT_SEEN_LINE = (
    P + "NOT SEEN: no camera sees the container (scene camera: scene segmenter: saw no food container in 14 frames; "
    "the wrist's look in front: wrist camera: no counter in 4 frames (no level plane wider than a container)); "
    "nothing was pressed"
)


def test_a_guided_run_whose_cameras_see_no_container_does_not_end_at_its_find_the_chair_is_guided():
    """Owner 2026-11-19: a box no camera sees at all is guidance too. Until then the view read
    `1 FIND  FAIL  no box found` and `RESULT  FAILED at step 1 (FIND)`. Guided, the find fails with what was
    missing in one short line, nothing says the chair is to move until the line-up's own step, and the
    line-up reads as any other: what the person is told, the question, their yes, and the find again."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    now[0] = 19.4
    view.feed(NOT_SEEN_LINE)
    assert out[1:] == [
        "   find      ...   looking for the box",
        " 1 FIND      FAIL  no camera sees the container               19.4 s",
        "             next  the run asks for the chair to be guided",
    ]
    assert not any("move" in line.lower() for line in out)  # the chair's move is the line-up's word
    view.guide_start("the arm rests and the wrist looks: move the chair as told")
    for t, say, ask in (
        (20, "The arm cannot see the container. Move the chair in front of it", None),
        (31, "Move closer, about 30 cm", None),
        (36, "Hold still, checking", None),
        (38, "Ready. Open the container?", "y / n"),
    ):
        now[0] = float(t)
        view.guide_say(say, ask=ask)
    now[0] = 40.0
    view.guide_end(True, "yes", again=True)
    now[0] = 44.0
    view.feed(P + "WRIST SURVEY: box at [0.640, -0.200] yaw 0.0 deg from 2000 lid points (WRIST 0.60) — reaching out")
    assert out[4:] == [
        "   line up   ...   the arm rests and the wrist looks: move the chair as told",
        "             say   The arm cannot see the container. Move the chair in front of it",
        "             say   Move closer, about 30 cm",
        "             say   Hold still, checking",
        "             ask   Ready. Open the container?   y / n",
        " 2 LINE UP   ok    yes                                        20.6 s",
        "   find      ...   looking for the box again",
        " 3 FIND      ok    box 0.67 m out (wrist camera)               4.0 s",
    ]
    # a container that never comes into view: the same end as one that never comes into reach, and the
    # result names what was missing and where the arm was left
    out.clear()
    view = FlightView(write=out.append, clock=lambda: now[0])
    now[0] = 0.0
    view.start()
    view.feed(NOT_SEEN_LINE)
    view.guide_start("the arm rests and the wrist looks: move the chair as told")
    view.guide_say("The arm cannot see the container. Move the chair in front of it")
    now[0] = 600.0
    view.guide_say("Out of time: the container was never seen. Start open_box again in front of it")
    view.guide_end(False, "the container never came into view within 600 s")
    view.arm = "at rest, the wrist turned back"
    view.finish(5)
    assert out[-4:] == [
        " 2 LINE UP   FAIL  the container never came into view within 600 s 600.0 s",
        "RESULT  FAILED at step 2 (LINE UP)                           600 s",
        "        why: the container never came into view within 600 s",
        "        box: not pressed    lid: on the box    arm: at rest, the wrist turned back",
    ]


def test_a_find_the_chair_is_guided_for_is_one_thing_and_what_ends_a_run_is_the_no_box_it_was():
    """A find the chair is guided for is one thing since 2026-11-19: nothing that could be the container in
    any view. Two things that could be it, a container nothing ties to the selection and a camera that
    delivered nothing end the run NO BOX, as ever, and read as that. As first written the guided find's
    line was searched for words of "two", the scene camera's part included, and a container the wrist did
    not see was called "more than one thing" for what the scene camera made of something further off."""
    assert not_seen_reason(NOT_SEEN_LINE) == "no camera sees the container"
    far_two = (
        "no camera sees the container (scene camera: scene segmenter: two places pass as this container, [0.80, "
        "-0.20] and [1.10, 0.30]: select which container (its instance_id) to open; the wrist's look in front: "
        "wrist camera: no counter in 4 frames (no level plane wider than a container))"
    )
    assert not_seen_reason(far_two) == "no camera sees the container"
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    view.feed(P + "NOT SEEN: " + far_two + "; nothing was pressed")
    assert out[2] == " 1 FIND      FAIL  no camera sees the container                0.0 s"
    # the run's NO BOX, its home and its end, exactly as before: by hand, and for what no chair move cures
    for why in (
        "scene camera: scene segmenter: saw no food container in 14 frames",
        "scene camera: scene segmenter: the selected container (fc_1) was in none of its 70 frames; the wrist's "
        "look in front: a container at [0.620, 0.040], and the selected container fc_1 is in none of the scene "
        "camera's detections: nothing says it is the one selected",
    ):
        lines = [
            (15, P + "NO BOX — " + why + " — returning home"),
            (20, "home               motion       1.00 full                   0     0.75    0.25    0.24"),
        ]
        assert _view(lines, 2) == [
            "   find      ...   looking for the box",
            " 1 FIND      FAIL  no box found                               15.0 s",
            "RESULT  FAILED at step 1 (FIND)                               25 s",
            "        why: no box found",
            "        box: not pressed    lid: on the box    arm: not confirmed home",
        ]


def test_nothing_says_the_chair_is_to_move_before_the_arm_is_at_rest():
    """The review of 2026-11-17: where no press reaches the aimed button the arm homes from the look pose
    (about 5 s on run 1), and the console said `next  the chair is to move: the run does not end here`
    BEFORE `home ... going home`. A person who drives on that word moves the chair under an arm still out
    over the table. Until the line-up's own step begins (the engine's, once the run has asked: its arm at
    rest) no line of the view speaks of moving the chair, and the way to rest is not called the way home."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    view.feed(P + "WRIST SURVEY: box at [0.80, -0.25] yaw 0.0 deg (WRIST 0.60) — reaching out")
    view.feed(P + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm")
    view.feed(P + "AIMED from the look pose: button at [0.820, -0.254, 0.316], 29 mm from the scene camera's")
    view.feed(
        P + "NO WAY IN: no press reaches the aimed button (the way down (walk in: 0 deg short: a joint jump of 0.21 "
        "rad 52% of the way)) — nothing was pressed — returning home"
    )
    now[0] = 5.0  # the home flew
    before = list(out)
    view.guide_start("the arm rests and the wrist looks: move the chair as told")
    assert before[-2:] == [
        "             next  the run does not end here: wait while the arm comes to rest",
        "   home      ...   going to rest before the chair is guided",
    ]
    assert not any("move" in line.lower() or "chair is to" in line for line in before[:-1])
    assert "going home" not in "".join(before)
    # the chair's move is the line-up's word, after the home's end
    assert out[len(before) :] == [
        " 4 HOME      ok                                                5.0 s",
        "   line up   ...   the arm rests and the wrist looks: move the chair as told",
    ]
    # the same where the arm stood at rest already (run 2's case): no word of moving before the line-up
    out.clear()
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    for text in (FOUND[0][1], RUN2_FOUND, P + "NO WAY IN: " + RUN2_WHY + "; nothing moved"):
        view.feed(text)
    assert out[-1] == "             next  the run does not end here: wait while the arm comes to rest"
    assert not any("move" in line.lower() for line in out if "nothing moved" not in line)


def test_the_guidances_own_home_is_a_way_to_rest_not_a_second_press_that_failed():
    """Where a look left the arm away from rest (the side survey's pose), the run homes before it asks to be
    guided (press_demo.guide_chair). Its line for that began "NO WAY IN:", which the view read as a second
    refusal: `2 PRESS FAIL no press plans from here: the way in does not plan` and a second `next`, of a
    run that had tried no press at all (review of 2026-11-17). It is the arm going to rest, once."""
    out, now = [], [0.0]
    view = FlightView(write=out.append, clock=lambda: now[0])
    view.start()
    view.feed(
        P + "OUT OF REACH: the wrist camera sees a lid-sized top at [1.02, -0.20], out of the arm's reach; "
        "nothing was pressed"
    )
    view.feed(P + "TO REST: the arm goes to rest before the chair is guided — returning home")
    now[0] = 6.0
    view.guide_start("the arm rests and the wrist looks: move the chair as told")
    assert out[1:] == [
        "   find      ...   looking for the box",
        " 1 FIND      FAIL  what the wrist camera sees is out of the arm's reach   0.0 s",
        "             next  the run asks for the chair to be guided",
        "   home      ...   going to rest before the chair is guided",
        " 2 HOME      ok                                                6.0 s",
        "   line up   ...   the arm rests and the wrist looks: move the chair as told",
    ]
    assert not any("PRESS" in line for line in out)
    # (that this is the line the run says: test_line_up_run, the arm a look left away from rest)


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


CARRIED = [
    (29, P + "lid goes to [0.65, 0.05, 0.259] — 200 mm from the box (45 mm outside its no-land radius)"),
    (29, P + "CLOSE, then ONE MOTION: lid:lift, lid:clear, lid:carry, place:down, place:touch — 40 mm straight up"),
    (34, P + "LID GRIPPED — grip 0.492 vs band [0.3, 0.55]"),
]


def test_the_result_says_the_lid_is_on_the_table_only_of_a_set_down_that_felt_the_table():
    """Flatbot 2026-11-17 16:40: the lid was let go past the table's edge, over the floor, and the result
    read "DONE — box open, lid on the table, arm home". "On the table" is said only where the run said
    its set-down felt the lid meet the table (its SET-DOWN CONTACT line, which it now writes of a
    credible contact alone). A release with no such line — nothing a run of this build does — is a step
    the view could not vouch for, and the result says what is known."""
    met = [
        (34, P + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped there"),
        (34, P + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance: home"),
        (39, P + "DONE — box open, lid on the table at [0.65, 0.05] (clear), let go where the arm felt it, arm home"),
    ]
    out = _view(FOUND + PRESSED + CARRIED + met, 0)
    assert out[-4] == " 7 SET DOWN  ok    let go where the lid met the table"
    assert out[-1] == "RESULT  DONE — box open, lid on the table, arm home           44 s"
    out = _view(FOUND + PRESSED + CARRIED + met[1:], 0)
    assert out[-4] == " 7 SET DOWN  ??    let go, and no table was felt under the lid"
    assert out[-1].startswith("RESULT  DONE — box open, lid let go (no table was felt under it), arm home")
    assert "on the table" not in out[-1]


def test_a_set_down_that_felt_no_table_is_a_failed_step_and_the_lid_goes_back_on_the_box():
    """The set-down ended and the arm felt no table under the lid: the lid is not let go (the run's
    "SET-DOWN: no table felt" line), and is carried back onto the box. The set-down is the step that
    failed; the put-back that follows ends well, and the result says where things were left: the box
    open, the lid on it, the arm home — never "on the table"."""
    why = "the touch ended 8 mm under where the plan has the lid meet the table and the arm felt no table"
    out = _view(
        FOUND
        + PRESSED
        + CARRIED
        + [
            (
                36,
                P + "SET-DOWN: no table felt — %s; the lid is not let go: back up with it, and back onto the box" % why,
            ),
            (
                37,
                P + "PUT BACK: the set-down felt no table (%s) — the lid back on the box where it was gripped, "
                "then home" % why,
            ),
            (44, P + "home after putting the lid back (the set-down felt no table (%s)) — returning home" % why),
            (
                49,
                P + "STOP: the set-down felt no table (%s) — the lid is back on the box, and the arm is home" % why,
            ),
        ],
        1,
    )
    assert out[8:] == [
        " 4 GRIP      ok    knob held                                   8.0 s",
        " 5 LIFT      ok",
        " 6 CARRY     ok    lid to 20 cm from the box",
        " 7 SET DOWN  FAIL  the arm felt no table under the lid; the lid was not let go",
        "             next  carrying the lid back to the box",
        "             next  putting the lid back on the box",
        " 8 PUT BACK  ok    lid back on the box, arm home",
        "RESULT  FAILED at step 7 (SET DOWN)                           54 s",
        "        why: the arm felt no table under the lid; the lid was not let go",
        "        box: open (knob up)    lid: on the box    arm: home",
    ]


def test_a_lid_put_back_with_no_spot_for_it_leaves_the_arm_home_in_the_result():
    """No spot on the table the cameras measured: the lid goes back on the box and the run's last line
    says the arm is home. The result said "arm: not confirmed home" of that line."""
    why = "no carry plans to a spot on the table the cameras measured (of the spots looked at, 57 not on it)"
    out = _view(
        FOUND
        + PRESSED
        + [
            (29, P + "CLOSE, then ONE MOTION: lid:lift, lid:clear — 40 mm straight up, then 50 mm back toward the arm"),
            (34, P + "LID GRIPPED — grip 0.492 vs band [0.3, 0.55]"),
            (36, P + "PUT BACK: %s — the lid back on the box where it was gripped, then home" % why),
            (49, P + "STOP: %s — the lid is back on the box, and the arm is home" % why),
        ],
        1,
    )
    assert out[-4:] == [
        " 5 CARRY     FAIL  no carry plans to a spot on the table the cameras measured",
        "RESULT  FAILED at step 5 (CARRY)                              54 s",
        "        why: no carry plans to a spot on the table the cameras measured",
        "        box: open (knob up)    lid: on the box    arm: home",
    ]


def test_a_lid_that_slipped_out_on_the_way_is_never_on_the_table_whatever_the_set_down_felt():
    """The knuckle reads shut past the knob at the table: the lid left the fingers on the way (the run's
    "the lid slipped on the way" line), the fingers open, the arm goes home, and the run ends "STOP: the arm
    is home, but the lid slipped out of the fingers". Whatever the set-down's watch felt, the view does not
    say the lid was let go where it met the table: the lid's way to its spot is the step that failed, the
    arm got home, and the result says the lid slipped out and that where it lies is not known. Review
    2026-11-17, fed these lines with the contact line a run then wrote first: " 6 SET DOWN  ok    let go
    where the lid met the table" and "lid: on the table"."""
    slipped = [
        (
            34,
            P + "the fingers read 0.790 at the table, outside the grip band [0.3, 0.55]: the lid slipped on the way "
            "(the set-down felt a contact +12 mm from where the plan put the table: not the lid's, which had left "
            "the fingers)",
        ),
        (34, P + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance: home"),
        (39, P + "STOP: the arm is home, but the lid slipped out of the fingers on the way to [0.65, 0.05]"),
    ]
    want = [
        " 4 GRIP      ok    knob held                                   8.0 s",
        " 5 CARRY     FAIL  the lid slipped out of the fingers on the way",
        "   home      ...   opening the fingers and going home",
        " 6 HOME      ok                                                5.0 s",
        "RESULT  FAILED at step 5 (CARRY)                              44 s",
        "        why: the lid slipped out of the fingers on the way",
        "        box: open (knob up)    lid: slipped out of the fingers on the way (where it lies is not known)    "
        "arm: home",
    ]
    out = _view(FOUND + PRESSED + CARRIED + slipped, 1)
    assert out[-7:] == want
    assert not any("SET DOWN" in line or "on the table" in line for line in out)
    # the same of a console that wrote the contact line first, as a run did until that review
    contact = (34, P + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped")
    out = _view(FOUND + PRESSED + CARRIED + [contact] + slipped, 1)
    assert out[-7:] == want and not any("on the table" in line for line in out)
    # ... and of one whose "slipped on the way" line was not read: the run's last word says it
    out = _view(FOUND + PRESSED + CARRIED + [contact] + slipped[1:], 1)
    assert out[-3:] == [
        "RESULT  FAILED at step 9 (CARRY)                              44 s",
        "        why: the lid slipped out of the fingers on the way",
        "        box: open (knob up)    lid: slipped out of the fingers on the way (where it lies is not known)    "
        "arm: home",
    ]


# ---- spot after spot (owner 2026-11-19: "it should keep trying spots") ---------------------------------------

NEVER = "the touch ended 8 mm under where the plan has the lid meet the table and the arm felt no table"
HELD = (
    "something held the lid up 60 mm over where the plan has it meet the table (more than 35): that is not the "
    "table, and the descent was stopped there"
)
ON = (
    "; the lid is not let go: up with it and on to another spot (this one, and the footprint that was left there, "
    "set aside for this run; 52 s of the run's time left)"
)
SPOT_2 = [
    (36, P + "SET-DOWN: no table felt at spot 1, [0.65, 0.05] — " + NEVER + ON),
    (
        38,
        P + "lid goes to [0.55, -0.20, 0.259] — 310 mm from the box (155 mm outside its no-land radius), where the "
        "cameras see room for it and the hand (20 mm to spare; the hand turns +0 deg); spot 2 of this run, after "
        "spot 1 at [0.65, 0.05]: no table felt by the touch's end",
    ),
    (38, P + "SPOT 2, ONE MOTION: lid:carry, place:down, place:touch — to [0.55, -0.20] and down onto the table"),
]
SPOT_3 = [
    (42, P + "SET-DOWN: no table felt at spot 2, [0.55, -0.20] — " + HELD + ON),
    (
        44,
        P + "lid goes to [0.60, 0.30, 0.259] — 250 mm from the box (95 mm outside its no-land radius), where the "
        "cameras see room for it and the hand (20 mm to spare; the hand turns +0 deg); spot 3 of this run, after "
        "spot 1 at [0.65, 0.05]: no table felt by the touch's end; spot 2 at [0.55, -0.20]: something held the "
        "lid up above where the table can be",
    ),
    (44, P + "SPOT 3, ONE MOTION: lid:carry, place:down, place:touch — to [0.60, 0.30] and down onto the table"),
]


def test_a_lid_taken_on_from_spot_to_spot_is_one_short_line_a_spot_and_the_result_says_where_it_ended():
    """Owner 2026-11-19: a set-down that feels no table does not put the lid back on the box, "it should
    keep trying spots". Each spot the lid is taken to is one numbered line: which spot, how far from the
    box (the first one's carry said that), and what the arm felt there — a failed step where it felt no
    table and kept the lid, never the run's own failure while another spot may take it. The lift and
    the carry are said once. The spot that takes the lid ends as any set-down, and the result says the
    lid is on the table, at which spot, after how many where none was felt."""
    met = [
        (48, P + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped there"),
        (48, P + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance"),
        (
            53,
            P + "DONE — box open, lid on the table at [0.60, 0.30] (clear), spot 3 of this run (no table was felt at "
            "the 2 before it), let go where the arm felt it meet the table, arm home",
        ),
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + SPOT_3 + met, 0)
    assert out[8:] == [
        " 4 GRIP      ok    knob held                                   8.0 s",
        " 5 LIFT      ok",
        " 6 CARRY     ok    lid to 20 cm from the box",
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 SET DOWN  FAIL  spot 2 (31 cm from the box): something held the lid up, lid kept",
        " 9 SET DOWN  ok    spot 3 (25 cm from the box): let go where the lid met the table",
        "   home      ...   letting go and going home",
        "10 HOME      ok                                                5.0 s",
        "RESULT  DONE — box open, lid on the table (spot 3: no table felt at 2 before), arm home  58 s",
    ]
    # the second spot takes it
    met2 = [
        (42, met[0][1]),
        (42, met[1][1]),
        (47, met[2][1].replace("spot 3", "spot 2").replace("the 2 before", "the 1")),
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + met2, 0)
    assert out[11:14] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 SET DOWN  ok    spot 2 (31 cm from the box): let go where the lid met the table",
        "   home      ...   letting go and going home",
    ]
    assert out[-1].startswith("RESULT  DONE — box open, lid on the table (spot 2: no table felt at 1 before), arm home")


def test_where_no_other_spot_is_tried_the_lid_goes_back_on_the_box_and_the_view_says_why():
    """The box is the last resort, and the view says which it was: the run's time leaves none for
    another spot, as many spots as a run tries, a set-down that could not be watched, or no spot left
    on the table. The set-down that ended the tries is the step the run failed at; the put-back ends
    well, and the result says the lid is on the box and the arm home — never "on the table"."""
    back = "; the lid is not let go: back up with it, and back onto the box"
    time_up = (
        " (no other spot is tried: 28 s of the run's time are left: another spot and, should it feel no table "
        "either, the way back to the box take 35 (12 and 23, on record))"
    )
    why_stop = "the set-down felt no table (%s), and no other spot is tried: 28 s of the run's time are left" % NEVER
    ended = [
        (42, P + "SET-DOWN: no table felt — " + NEVER + back + time_up),
        (43, P + "PUT BACK: %s — the lid back on the box where it was gripped, then home" % why_stop),
        (55, P + "STOP: %s — the lid is back on the box, and the arm is home" % why_stop),
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + ended, 1)
    assert out[11:] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 SET DOWN  FAIL  spot 2 (31 cm from the box): no table felt, lid kept",
        "             next  carrying the lid back to the box (no time left for another spot)",
        "             next  putting the lid back on the box",
        " 9 PUT BACK  ok    lid back on the box, arm home",
        "RESULT  FAILED at step 8 (SET DOWN)                           60 s",
        "        why: spot 2 (31 cm from the box): no table felt, lid kept",
        "        box: open (knob up)    lid: on the box    arm: home",
    ]
    # as many spots as a run tries; and a set-down that could not be watched, at the first spot
    spots = " (no other spot is tried: 6 spots is as many as a run tries (SETDOWN_SPOTS))"
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + [(42, P + "SET-DOWN: no table felt — " + NEVER + back + spots)], 1)
    assert "             next  carrying the lid back to the box (as many spots as a run tries)" in out
    blind = (
        " (no other spot is tried: a set-down that could not be watched says nothing of this spot, and the next "
        "would fly as blind)"
    )
    failed = "the set-down's watch failed (boom) and the descent was stopped there"
    out = _view(FOUND + PRESSED + CARRIED + [(36, P + "SET-DOWN: no table felt — " + failed + back + blind)], 1)
    assert out[11:13] == [
        " 7 SET DOWN  FAIL  spot 1: the set-down's watch failed, lid kept",
        "             next  carrying the lid back to the box (a set-down that cannot be watched is tried nowhere else)",
    ]
    # no spot left on the table: put back from over the spot the lid was left at (no line of a carry back)
    # (the reason once: said with the carry back, it is not said again with the put-back, as above)
    none = (
        "no table was felt under the lid (spot 1 at [0.65, 0.05]: no table felt by the touch's end; spot 2 at "
        "[0.55, -0.20]: something held the lid up above where the table can be), and no other spot on the table "
        "the cameras measured has a carry that plans (of the spots looked at, 57 not on the table the cameras measured)"
    )
    ended = [
        (42, P + "SET-DOWN: no table felt at spot 2, [0.55, -0.20] — " + HELD + ON),
        (45, P + "PUT BACK: %s — the lid back on the box where it was gripped, then home" % none),
        (57, P + "STOP: %s — the lid is back on the box, and the arm is home" % none),
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + ended, 1)
    assert out[11:] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 SET DOWN  FAIL  spot 2 (31 cm from the box): something held the lid up, lid kept",
        "             next  putting the lid back on the box (no other spot found on the table)",
        " 9 PUT BACK  ok    lid back on the box, arm home",
        "RESULT  FAILED at step 8 (SET DOWN)                           62 s",
        "        why: spot 2 (31 cm from the box): something held the lid up, lid kept",
        "        box: open (knob up)    lid: on the box    arm: home",
    ]
    # ... the run's time running out while the next spot was looked for, or before its carry was sent
    for why in (
        "no table was felt under the lid (spot 1 at [0.65, 0.05]: no table felt by the touch's end), and no other "
        "spot is tried: the run's time ran out while one was looked for (30 s are left; a spot and the way back to "
        "the box after it take 31)",
        "no table was felt under the lid (spot 1 at [0.65, 0.05]: no table felt by the touch's end), and no other "
        "spot is tried: 45 s of the run's time are left: another spot and, should it feel no table either, the way "
        "back to the box take 46 (23 by this spot's own carry and 23, on record)",
    ):
        ended = [
            (40, P + "PUT BACK: %s — the lid back on the box where it was gripped, then home" % why),
            (52, P + "STOP: %s — the lid is back on the box, and the arm is home" % why),
        ]
        out = _view(FOUND + PRESSED + CARRIED + SPOT_2[:1] + ended, 1)
        assert out[11:14] == [
            " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
            "             next  putting the lid back on the box (no time left for another spot)",
            " 8 PUT BACK  ok    lid back on the box, arm home",
        ]
    # ... and the table itself, by all it looks: something held the lid up at the same height at two spots
    same = (
        " (no other spot is tried: something held the lid up at the same height at two spots, +60 mm from where "
        "the plan has the table at spot 1 and +58 mm at spot 2: that is probably the table itself, standing higher "
        "than the run has it, and it is not pressed on at another spot)"
    )
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + [(42, P + "SET-DOWN: no table felt — " + HELD + back + same)], 1)
    assert out[12:14] == [
        " 8 SET DOWN  FAIL  spot 2 (31 cm from the box): something held the lid up, lid kept",
        "             next  carrying the lid back to the box (held up at the same height twice: probably the table, "
        "higher than planned)",
    ]


def test_a_carry_that_stops_on_the_way_to_another_spot_is_the_runs_failure_not_the_set_down_before_it():
    """A spot left is a failed step, not the run's failure: where what follows stops the run (the carry to
    the next spot stopped part way, the arm holding the lid), the result names that, and where the lid is."""
    stop = "STOP: the lid's way to the table stopped during lid:carry (aborted) — the arm holds the lid"
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + [(40, P + stop)], 1)
    assert out[11:] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 CARRY     FAIL  stopped part way (aborted)",
        "RESULT  FAILED at step 8 (CARRY)                              45 s",
        "        why: stopped part way (aborted)",
        "        box: open (knob up)    lid: in the hand    arm: holding where it stopped",
    ]


def test_a_run_cut_between_two_spots_says_the_cut_and_where_the_lid_was():
    """Between two spots no step of this view is in flight: the last line written is the spot left
    behind. A run the engine's budget cuts there (or a cancel) ended with the lid in the air, and the
    result named the spot left, "no table felt", as where and why it failed (the review of 2026-11-19).
    The view says the cut and the carry in flight — or that the next spot was being chosen — and that
    is where the run failed; the lid is in the hand."""
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2, 130, timed_out=True, held=True)
    assert out[11:] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 CARRY     FAIL  stopped: the run took too long, the lid on its way to spot 2",
        "RESULT  FAILED at step 8 (CARRY)                              43 s",
        "        why: stopped: the run took too long, the lid on its way to spot 2",
        "        box: open (knob up)    lid: in the hand    arm: not confirmed home",
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2[:1], 130, timed_out=True, held=True)
    assert out[11:] == [
        " 7 SET DOWN  FAIL  spot 1: no table felt, lid kept",
        " 8 SET DOWN  FAIL  stopped: the run took too long, the next spot for the lid being chosen",
        "RESULT  FAILED at step 8 (SET DOWN)                           41 s",
        "        why: stopped: the run took too long, the next spot for the lid being chosen",
        "        box: open (knob up)    lid: in the hand    arm: not confirmed home",
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2, 130, cancelled=True, held=True)
    assert out[12] == " 8 CARRY     FAIL  cancelled, the lid on its way to spot 2"
    assert out[13].startswith("RESULT  CANCELLED at step 8 (CARRY)")
    # once that spot's set-down has ended one way or the other, nothing is between two spots any more
    met = [
        (42, P + "SET-DOWN CONTACT: the lid met the table +12 mm from where the plan put it; the touch stopped there"),
        (42, P + "RELEASE as the retreat starts (place:retreat), then HOME, slow, in MEDIUM impedance"),
    ]
    out = _view(FOUND + PRESSED + CARRIED + SPOT_2 + met, 130, timed_out=True, held=False)
    assert not any("on its way to spot" in line or "being chosen" in line for line in out)


def test_a_look_that_goes_again_over_the_lid_the_wrist_sees_is_a_step_of_its_own():
    """RAMMP 1.5, 2026-11-20: the look stood where the scene camera had the box, 19 cm from the lid. The look
    again (tasks/reach_press.look_again) ends that step as what it was, begins its own, and the button found from
    it says where it was found."""
    scene_fix = (
        6,
        P + "SCENE: box at [0.681, -0.035] yaw 0.0 deg from 191 lid points (SEG 0.76, 2 sightings "
        "(segmenter+segmenter); 4.5 s to locate [streams 0.02 seg 0.06], 5.5 s since start) — reaching out",
    )
    look = (7, P + "REACH-OUT: hand tilted 30 deg; looking from 0.15 m back toward the arm; gravity alone moves x")
    again = (
        13,
        P + "LOOK AGAIN 1 of 2: no button where the scene camera had the box; the wrist sees the lid at [0.776, "
        "-0.200, 0.392], 190 mm from there — the look moves over it",
    )
    aim = (18, P + "AIM: button circle at pixel (322, 190) r 31 px, 43 mm across, +0.4 mm above its own lid")
    aimed = (18, P + "AIMED from the look pose: button at [0.789, -0.203, 0.392], 14 mm from the scene camera's")
    out = [" ".join(line.split()) for line in _view([scene_fix, look, again, aim, aimed], 0)]
    assert out[:6] == [
        "find ... looking for the box",
        "1 FIND ok box 0.68 m out (scene camera) 6.0 s",
        "look ... moving to look at the button",
        "2 LOOK ?? no button where the scene camera had the box 6.0 s",
        "look ... moving over the lid the wrist sees, 19 cm from there",
        "3 LOOK ok button found, 43 mm across, 0.81 m out, over the lid the wrist saw 19 cm from where the scene "
        "camera had it 5.0 s",
    ]
    # two looks again and no button: the press's failure as it was, the looks counted in the run's own line
    stop = (
        25,
        P + "STOP: no button from the look pose, nor from 2 looks again (no circle; each look again over the lid "
        "the wrist's frame showed; no look again: a reach looks again no more than 2 times) — nothing was pressed "
        "— returning home",
    )
    twice = (19, again[1].replace("LOOK AGAIN 1", "LOOK AGAIN 2").replace("the scene camera", "the wrist's own look"))
    out = _view([scene_fix, look, again, twice, stop], 2)
    steps = [line for line in out if line[:2].strip().isdigit()]
    assert [line.split()[1:3] for line in steps[:5]] == [
        ["FIND", "ok"],
        ["LOOK", "??"],
        ["LOOK", "??"],
        ["LOOK", "??"],  # the last look again found no button either: not a look that ended well
        ["PRESS", "FAIL"],
    ]
    assert "no button over the lid the wrist saw 19 cm from where the wrist's own look had it either" in steps[3]
    assert "no button from the look pose, nor from 2 looks again" in steps[4]
    # a run with no look again is as it always read: the look ended well, the press has no button
    out = _view([scene_fix, look, (13, P + "STOP: no button from the look pose (no circle) — nothing was pressed")], 2)
    steps = [line for line in out if line[:2].strip().isdigit()]
    assert [line.split()[1:3] for line in steps[:3]] == [["FIND", "ok"], ["LOOK", "ok"], ["PRESS", "FAIL"]]
    # what one round looked at again is not said of the next round's look (a guided run: the chair moved between)
    out = [" ".join(line.split()) for line in _view([scene_fix, look, again, scene_fix, look, aim, aimed], 0)]
    found = [line for line in out if "button found" in line]
    assert len(found) == 1 and "over the lid the wrist saw" not in found[0]


def test_a_selected_id_the_scene_camera_no_longer_has_is_said_with_the_one_taken_for_it():
    """2026-11-20 13:42: the goal's id was in none of the detections. Where the one container in view is read for
    it (perception/segmenter_source), the view says so once, with both ids."""
    said = (
        2,
        "[scene] the selected container (food_container_2) is in none of the segmenter's frames, and they hold one "
        "container of its kind and no other, food_container_5: that one is read for it, for as long as it is the "
        "only one in view",
    )
    out = [" ".join(line.split()) for line in _view([said], 2)]
    assert out[1] == (
        "note food_container_2 is not in the scene camera's list; the only container in view (food_container_5) "
        "is read for it"
    )


def test_the_person_at_the_e_stop_is_shown_how_far_apart_the_two_cameras_are_and_the_mounts_warning():
    """The cameras' check (perception/camera_check) said its findings in the run's full lines only, which the
    flight view does not show: the person at the e-stop was not told that the scene camera's mount is 18 cm off.
    And the view's own "scene camera puts it N mm away" is read from the wrist look's line, whose number is taken
    from the scene camera's fix as the run reads it: in a second round, the correction in force, that is only
    what the correction left, and the view went quiet (review of 2026-11-20). The check's own line gives the
    disagreement as the MOUNT has it: shown, with the warning to tell the team."""
    tie = "(one container: the selection's tie took the wrist's find)"
    cameras = (
        P
        + "CAMERAS: the scene camera has the container at [0.680, -0.034] and the wrist camera at [0.770, -0.192] "
        + tie
        + ": they disagree by -13.9 deg about its lens and 28 mm nearer along it (182 mm apart at 0.75 m), the "
        "wrist's lid 58 mm lower. The scene camera's numbers are turned -13.9 deg about its lens for the rest of "
        "this run"
    )
    warning = (
        P + "WARNING: the scene camera's mount looks 14 deg off (it puts the container 182 mm from where the wrist "
        "camera, which the arm follows, has it): tell the team. Box opening turns its own reading of it for this "
        "run; everything else that uses the scene camera's positions is off by as much"
    )
    heights = (
        P + "WARNING: on this robot the two cameras read heights 50 mm apart (the wrist's lower; more than 30): a "
        "standing difference of their mounts, in height or tilt, not of this run. Which is right only a touch says"
    )
    fix = P + "WRIST SURVEY: box at [0.770, -0.192] yaw 0.0 deg from 250 lid points (WRIST 0.70) — reaching out"

    def look(apart_mm):
        return (
            P + "WRIST LOOK: the box at [0.770, -0.192, 0.385]; the scene camera had it at [0.680, -0.034, 0.443], "
            "%d mm away — the wrist's is taken" % apart_mm
        )

    # a first round: nothing in force, the wrist look's own line has the whole distance, and the check's is the same
    out = _view([(1, look(182)), (1, cameras), (1, warning), (2, heights), (3, fix)], 1)
    assert out[:5] == [
        "   find      ...   looking for the box",
        "             note  scene camera puts it 182 mm away; using the wrist's",
        "             note  WARNING: the scene camera's mount looks 14 deg off (182 mm at the box): tell the team",
        "             note  the two cameras read heights 50 mm apart on this robot (the wrist's lower)",
        " 1 FIND      ok    box 0.79 m out (wrist camera)               3.0 s",
    ]
    # a second round, the correction in force: the wrist look's line has 12 mm (nothing is said of that), and the
    # check's line has the mount's 182 mm: shown, so the view does not go quiet over a mount that is 18 cm off
    out = _view([(1, look(12)), (1, cameras), (3, fix)], 1)
    assert out[:3] == [
        "   find      ...   looking for the box",
        "             note  scene camera's mount puts it 182 mm from the wrist's (-13.9 deg about its lens); its "
        "numbers are turned for this run",
        " 1 FIND      ok    box 0.79 m out (wrist camera)               3.0 s",
    ]
    # a pair nothing establishes is not measured: the full lines say why, and the view says nothing it does not know
    unpaired = (
        P + "CAMERAS: the two cameras' places are not measured against each other: no selection ties the wrist's "
        "find to the scene camera's box, and the scene camera does not hold exactly one container of the kind"
    )
    out = _view([(1, look(182)), (1, unpaired), (3, fix)], 1)
    assert [ln for ln in out[:3] if "note" in ln] == [
        "             note  scene camera puts it 182 mm away; using the wrist's"
    ]

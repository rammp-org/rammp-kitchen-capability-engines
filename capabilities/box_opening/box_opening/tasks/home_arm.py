"""home_arm: return to HOME — the recovery move and the abort-drill mover.

    ros2 run box_opening home_arm --execute
    ros2 run box_opening home_arm --execute --lift-first 0.08   # start refused: lift straight up first

Plans in the pre-detection BENCH world first: no container pose is
known to an isolated home (a failed run may have left the box anywhere),
so the whole placement band is blocked to container height and the move
stays above anything that could be standing there. A recovery home
usually STARTS inside that band, though — the arm is holding at a press,
a grip or a set-down — and then the band makes the start itself invalid.
So a refused start falls back to the bare table world with a printed
caution: the move then ignores where the box may be, and the operator
clears the bench before running it (review 2026-09-02). The attended
abort drill Ctrl+Cs this mid-motion. When even the bare world refuses
the START (the fingers are at the table), --lift-first M flies a plan-free
straight vertical line of M metres first and plans HOME from its end.
A joint too near its limit for the planner to start from
(START_IN_LIMIT_MARGIN, or FINETUNE_TRAJOPT_FAIL just outside the margin)
gets a plan-free step along the joint line toward HOME first, the shortest
that HOME plans from (primitives.ClearMarginFree): refused if the tool
would drop on the way, and obstruction-guarded.

The band stands on the table, so it exists only where a table is known.
A world YAML without one (flatbot: the table changes with every place
the chair goes) and a fresh home_arm process, which measures none, make
the guarded world the floor alone, the same as the bare one. There the
caution prints BEFORE planning, and HOME is planned once in the bare
world: re-planning a refused start in an identical world cannot help.

A home the planner called a success and the client refused, because it
ends elsewhere than HOME (runtime/planner.py: PlanToJoints' FK-pose
fallback, another arm at HOME's tool pose, or another goal's plan), is
not answered with the bare world: the planner did plan in the guarded
world, and the bare world would only plan the same home blind to where
the box may be (review 2026-10-28). It stops, saying what is left.
"""

from box_opening import LOGGER as log
from box_opening.constants import HOME, TRANSIT_SPEED
from box_opening.primitives.core import (
    CLEAR_MARGIN_MAX_STEP_RAD,
    ClearMarginFree,
    LiftFree,
    PlanRefused,
    PlanState,
    _plan_motion,
)
from box_opening.tasks import cli_common

# what is left when the client refuses the planner's home (core.PlanRefused):
# another start, or the operator's hand
REFUSED_HOME_NEXT = "try --lift-first 0.08 (HOME planned from the top of a straight lift), or jog the arm clear by hand"


def _plan_home(ctx, state, say=log.info):
    """HOME from `state`: the guarded bench world first, then the bare
    table world with a caution. Raises the bare world's refusal. With no
    table known the two are the same world: once, in the bare one (the
    caution was printed before planning). A home the client refused
    (core.PlanRefused) raises at once, with what is left to try."""
    if ctx.worlds.table_top_z is None:
        bare = ctx.worlds.push_name("bench", model=None, tag="bare")
        try:
            leg, _ = _plan_motion(ctx, state, "home", ("joints", list(HOME)), bare, TRANSIT_SPEED)
        except PlanRefused as e:
            raise PlanRefused("%s — %s" % (e, REFUSED_HOME_NEXT)) from e
        return leg
    guarded = ctx.worlds.push_name("bench", model=ctx.model)
    try:
        leg, _ = _plan_motion(ctx, state, "home", ("joints", list(HOME)), guarded, TRANSIT_SPEED)
        return leg
    except PlanRefused as e:
        # the planner did plan in this world: the bare one would plan the
        # same home blind to where the box may be
        raise PlanRefused(
            "%s — not re-planned in the bare world, which would not know where the box may be; %s"
            % (e, REFUSED_HOME_NEXT)
        ) from e
    except RuntimeError as e:
        say("[home_arm] home refused in the guarded bench world (%s)" % e)
    # there is no typed confirmation any more (--execute alone arms a run),
    # so this cannot ask for the bench to be cleared first: it says what the
    # move about to fly does NOT know, and the dry-run preview is the place
    # to read it before adding --execute
    say(
        "[home_arm] CAUTION: planning in the bare table world — this move "
        "ignores where the box may be. With --execute it flies NOW; run "
        "without it first if the bench is not clear."
    )
    bare = ctx.worlds.push_name("bench", model=None, tag="bare")
    leg, _ = _plan_motion(ctx, state, "home", ("joints", list(HOME)), bare, TRANSIT_SPEED)
    return leg


def build_legs(ctx, lift_first_m=0.0):
    """HOME from live; with lift_first_m > 0, a plan-free straight lift of
    that much FIRST (primitives.LiftFree), and HOME planned from where the
    lift ends. For the pose the planner refuses as a start — fingers at
    the table after an abort or a jog (2026-09-17: refused in both worlds
    40 mm above the table, 23 deg off vertical). A joint too near its limit
    for the planner gets a plan-free step toward HOME first, the shortest
    one HOME plans from (primitives.ClearMarginFree, 2026-10-23)."""
    state = PlanState(joints=list(ctx.client.joints()), chain=0)
    legs = []
    if lift_first_m > 0:
        lift, state = LiftFree(lift_first_m).plan(ctx, state)
        legs += lift
        log.info(
            "[home_arm] plan-free lift of %.0f mm first (straight up, attitude held, "
            "obstruction-guarded) — HOME is planned from where it ends" % (1000 * lift_first_m)
        )
    if ctx.worlds.table_top_z is None:
        # no table, so no band: the guarded world would be the bare one.
        # Say so before anything is planned, and plan once.
        log.info(
            "[home_arm] CAUTION: no table is known (this bench YAML has none and "
            "home_arm measures none) — this home knows neither the table nor the "
            "box. With --execute it flies NOW; run without it first, or use "
            "--lift-first 0.08 to rise straight up before the planned sweep."
        )
    clear = ClearMarginFree(HOME)
    near = clear.near(ctx, state.joints)
    if not near:
        return legs + [_plan_home(ctx, state)]
    log.info(
        "[home_arm] %s: too near for the planner to start from — trying plan-free steps "
        "along the joint line toward HOME, shortest first"
        % ", ".join("%s is %.3f rad from its limit" % jr for jr in near)
    )
    why = None
    for step, at, info in clear.steps(ctx, state):
        notes = []
        try:
            home = _plan_home(ctx, at, say=notes.append)
        except RuntimeError as e:
            why = e
            continue
        for note in notes:
            log.info(note)
        if step:
            d = info["tool_move"]
            log.info(
                "[home_arm] a plan-free step %.0f%% of the way to HOME first (the largest joint "
                "%.3f rad, the tool %+.0f mm up and %.0f mm across; obstruction-guarded) — HOME "
                "is planned from where it ends"
                % (100 * info["fraction"], info["largest_rad"], 1000 * d[2], 1000 * (d[0] ** 2 + d[1] ** 2) ** 0.5)
            )
        return legs + step + [home]
    raise RuntimeError(
        "no plan-free step toward HOME of up to %.2f rad per joint lets the planner plan HOME "
        "(last: %s) — jog the arm clear by hand" % (CLEAR_MARGIN_MAX_STEP_RAD, why)
    )


def main():
    ap = cli_common.make_parser(__doc__)
    ap.add_argument(
        "--lift-first",
        type=float,
        default=0.0,
        metavar="M",
        help="plan-free straight lift of M metres before planning home (the pose the "
        "planner refuses as a start: fingers at the table after an abort or a jog); "
        "0.08 is usually enough",
    )
    args = ap.parse_args()
    cli_common.run_task(args, lambda ctx: build_legs(ctx, args.lift_first))

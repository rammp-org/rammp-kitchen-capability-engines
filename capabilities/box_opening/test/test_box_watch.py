"""The line-up's eyes (tasks/box_watch): the scene camera's sightings, a line each, for the engine."""

import math
import time
from types import SimpleNamespace

from box_opening.tasks import box_watch


def _sighting(top=(0.652, -0.104, 0.3149), table_z=0.2159, lift="side", score=0.8812):
    return SimpleNamespace(top=top, table_z=table_z, lift=lift, score=score)


def test_a_sighting_is_a_line_with_the_lids_top_its_table_and_its_age():
    assert box_watch.fix_line(_sighting(), 0.0749) == {
        "fix": {"xyz": [0.652, -0.104, 0.3149], "table_z": 0.2159, "age_s": 0.075, "rough": False, "score": 0.881}
    }
    rough = box_watch.fix_line(_sighting(lift="rough", table_z=math.nan), -0.01)["fix"]
    assert rough["rough"] is True and rough["table_z"] is None and rough["age_s"] == 0.0
    assert box_watch.fix_line(_sighting(top=(0.6, math.nan, 0.3)), 0.1) is None  # no place: nothing to say


class Locator:
    """SceneLocator's part: each locate() tells the sightings its script holds for that look, then answers."""

    def __init__(self, looks):
        self.looks, self.on_sighting, self.last_why, self.asked = list(looks), None, None, []

    def locate(self, timeout_s):
        self.asked.append(timeout_s)
        sightings, fix, why = self.looks.pop(0)
        for s in sightings:
            self.on_sighting(s, 0.1)
        self.last_why = why
        return fix


def test_every_sighting_is_said_as_it_is_made_and_a_look_with_none_says_why():
    moving = [_sighting(top=(0.95 - 0.05 * i, 0.0, 0.315)) for i in range(3)]
    loc = Locator(
        [
            (moving, None, "scene segmenter: its fixes did not agree within 15 mm"),  # the chair is moving
            ([], None, "scene segmenter: saw no food container in 5 frames"),
            ([_sighting(), _sighting()], object(), None),  # standing still: a look that commits, said all the same
        ]
    )
    said = []
    box_watch.watch(loc, said.append, stopped=lambda: not loc.looks, look_s=0.02)
    assert [m["fix"]["xyz"][0] for m in said[:3]] == [0.95, 0.9, 0.85]
    assert said[3] == {"none": "scene segmenter: its fixes did not agree within 15 mm"}
    assert said[4] == {"none": "scene segmenter: saw no food container in 5 frames"}
    assert [list(m) for m in said[5:]] == [["fix"], ["fix"]] and loc.asked == [0.02, 0.02, 0.02]


def test_a_look_that_returns_at_once_is_waited_out_so_none_is_said_once_a_period():
    """With no scene segmenter to read (its interface does not import, the container names no classes) locate()
    answers at once: the watch then span a core flat out and wrote a "none" line every pass, thousands a second
    down the engine's pipe. Each pass lasts the look's period, and a stop is still heard inside it."""

    class Blind:
        on_sighting, last_why, looks = None, "the scene segmenter cannot be read: no interface", 0

        def locate(self, timeout_s):
            self.looks += 1
            return None

    loc, said, t0 = Blind(), [], time.monotonic()
    box_watch.watch(loc, said.append, stopped=lambda: time.monotonic() - t0 >= 0.25, look_s=0.1)
    assert 1 <= loc.looks <= 3 and len(said) == loc.looks  # 0.25 s of 0.1 s looks
    # the stop heard while a look was waited out, not a period late (the bound loose: the computer is busy)
    assert time.monotonic() - t0 < 0.25 + 0.1 + 0.4
    # a look that ends early with its fix is followed by the next at once: a wait would only age its sightings
    quick = Locator([([_sighting()], object(), None)] * 3)
    said, t0 = [], time.monotonic()
    box_watch.watch(quick, said.append, stopped=lambda: not quick.looks, look_s=5.0)
    assert len(said) == 3 and time.monotonic() - t0 < 1.0


def test_it_holds_no_client_for_the_arm():
    """What follows the box cannot move anything: the module imports nothing that could."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(box_watch.__file__).read_text())
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not {m for m in names if m and ("runtime.client" in m or "runtime.runner" in m or "primitives" in m)}
    assert "init_runtime" not in Path(box_watch.__file__).read_text()  # cli_common's makes a PlannerClient

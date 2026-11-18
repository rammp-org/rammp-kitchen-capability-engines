"""scripts/kitchen-ui: every place that lists the capabilities agrees.

down.sh must stop every node node.sh can start (a node it misses keeps taking goals with
allow_execute on), and up.sh's default set must be ones node.sh can start.
"""

import re
from pathlib import Path

KUI = Path(__file__).resolve().parent.parent / "scripts/kitchen-ui"


def node_sh_cases():
    """The capabilities node.sh's `case "$C" in` starts (its `name)` labels)."""
    text = (KUI / "node.sh").read_text()
    body = text[text.index('case "$C" in') : text.index("esac")]
    return set(re.findall(r"^  ([a-z_]+)\)$", body, re.M))


def down_caps():
    return set(re.search(r'^CAPS="([^"]*)"$', (KUI / "down.sh").read_text(), re.M).group(1).split())


def up_default_caps():
    return set(
        re.search(r"^CAPS=\$\{KUI_CAPS:-([^}]*)\}$", (KUI / "up.sh").read_text(), re.M)
        .group(1)
        .split()
    )


def test_down_stops_every_capability_node_sh_can_start():
    cases = node_sh_cases()
    assert "grab_container" in cases
    assert cases <= down_caps()


def test_up_starts_by_default_only_capabilities_node_sh_knows():
    assert up_default_caps() <= node_sh_cases()


def test_node_sh_usage_names_every_capability_it_starts():
    usage = re.search(r"node\.sh <([a-z_|]+)>", (KUI / "node.sh").read_text()).group(1)
    assert set(usage.split("|")) == node_sh_cases()

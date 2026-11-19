# ruff: noqa: E501  -- PoC test script
"""The microwave node's ROS params file: sheppy-manifest.yaml's microwave_dev params, standalone (no orchestrator),
allow_execute as asked. usage: make_params.py <sheppy-manifest.yaml> <true|false>  (YAML on stdout)"""

import sys

import yaml

manifest = yaml.safe_load(open(sys.argv[1]))
node = next(n for n in manifest["nodes"] if n["name"] == "microwave")
alt = next(a for a in node["alternatives"] if a["id"] == "microwave_dev")
params = dict(
    alt["params"],
    use_orchestrator=False,
    allow_execute=sys.argv[2] == "true",
    data_dir="/microwave_data",
)
yaml.safe_dump({"microwave": {"ros__parameters": params}}, sys.stdout, sort_keys=False)

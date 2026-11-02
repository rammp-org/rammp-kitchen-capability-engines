import os
from glob import glob

from setuptools import find_packages, setup

package_name = "box_opening"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [f"launch/{package_name}.launch.py"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "containers"), glob("config/containers/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Open the food container: press its button, take its lid off, set it down (CapabilityEngine).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "box_opening_node = box_opening.node:main",
            # the command lines the node runs for its actions, for attended runs by hand
            "press_demo = box_opening.tasks.press_demo:main",
            "home_arm = box_opening.tasks.home_arm:main",
            "preflight = box_opening.tasks.preflight:main",
            # the OWL detector on its own (the node hosts two in its process)
            "owl_detector = box_opening.perception.owl_node:main",
            # a wrist detection set from live frames, and a Foxglove view of where the lid would go
            "record_detection_set = box_opening.tasks.record_detection_set:main",
            "lid_room_view = box_opening.tasks.lid_room_view:main",
        ]
    },
)

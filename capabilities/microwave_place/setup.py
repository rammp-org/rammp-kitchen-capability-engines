from pathlib import Path

from setuptools import setup

package_name = "microwave_place"
robot_body = Path("config/robot_body.yaml")
if not robot_body.is_file():
    robot_body = Path("../box_opening/config/robot_body.yaml")

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [f"launch/{package_name}.launch.py"]),
        (f"share/{package_name}/config", ["config/snorlax_collision_geometry.yaml", str(robot_body)]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Put the held container into the open microwave (CapabilityEngine + self.arm).",
    entry_points={"console_scripts": [f"{package_name}_node = {package_name}.{package_name}_node:main"]},
)

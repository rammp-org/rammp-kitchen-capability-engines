from setuptools import setup

package_name = "place_container"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    # the chair and user boxes cuRobo plans around (level_path.py): copied from the robot's planner config
    package_data={package_name: ["chair/*.yaml", "chair/SOURCE"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [f"launch/{package_name}.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Put the held food container down on the table (CapabilityEngine + self.arm).",
    entry_points={"console_scripts": [f"{package_name}_node = {package_name}.node:main"]},
)

# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=place_container`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws — capability_engine and
# this capability — and runs the same launch file.
#
# Pin an explicit base tag (a release or a dated nightly), never :humble or
# :latest — the tag records which rammp-interfaces-ros2 version this image
# speaks. Use rammp-cuda only if the node needs the GPU, and then drop
# torch/torchvision from your requirements (the base ships the Jetson wheels).
# arm64 only.
#
# 2.2.0-jp6 carries rammp-interfaces-ros2 v2.1.0, all seven packages (arm, base,
# common, orchestrator, perception, peripheral, ui_bridge), with the ExecuteAction
# token and SetEngineOverlayVisible capability_engine uses. Its arm messages are the
# arm driver's (since 73e7ed1716bf, the nightly driver's, only SetContactThresholds
# and CONTACT_STOP were added), so self.arm speaks them exactly.
# rammp_curobo_interfaces it does not carry.
FROM ghcr.io/rammp-org/rammp-base:2.2.0-jp6

WORKDIR /module_ws

# self.arm plans and checks poses on a PyBullet model of the arm. PyPI ships no
# arm64 wheel, so this compiles from source (the base has gcc, g++ and Python.h).
# 3.2.7 is the version the SheppyArm goldens were recorded with. numpy is in the
# base; scipy comes from rosdep (capability_engine's package.xml). Its own layer,
# before the COPYs, so editing code never recompiles it.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit (v2.0.1: self.arm.blend needs its Blend
# action) alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/10ad896883158247039a8bd850bcabcab016a50d.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

COPY capability_engine/ src/capability_engine/
COPY capabilities/place_container/ src/place_container/

# /module_ws is an overlay on the base's /ros2_ws (the RAMMP interfaces); the
# base's entrypoint sources its install/ automatically. The base ships rosdep's
# sources but not its cache, hence the update (as grab_container does).
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && rosdep update --rosdistro humble \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf build log

# `ros2 launch` on Humble has no SIGTERM handler. As PID 1 it ignores the
# SIGTERM `docker stop` (and `sheppy woof`) sends and is SIGKILLed after the
# grace period, leaving the node no chance to stop cleanly. On SIGINT it shuts
# its nodes down properly, so make that the stop signal.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "place_container", "place_container.launch.py"]

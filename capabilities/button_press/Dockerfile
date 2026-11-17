# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=button_press`). The image colcon-builds
# capability_engine and this capability, and runs the same launch file.
#
# 2.2.0-jp6 carries rammp-interfaces-ros2 v2.1.0, all seven packages (arm, base,
# common, orchestrator, perception, peripheral, ui_bridge), with the ExecuteAction
# token and SetEngineOverlayVisible capability_engine uses. Its arm messages are the
# arm driver's (since 73e7ed1716bf, the nightly driver's, only SetContactThresholds
# and CONTACT_STOP were added), so self.arm speaks them exactly.
# rammp_curobo_interfaces it does not carry.
# rammp-base, not rammp-cuda: nothing here uses the GPU (cuRobo runs in the
# planner's own image). arm64 only.
FROM ghcr.io/rammp-org/rammp-base:2.2.0-jp6

WORKDIR /module_ws

# The press gates every leg in PyBullet (kinematics.py), and so does self.arm's
# sim. PyPI ships no arm64 wheel, so this compiles from source (the base has gcc,
# g++ and Python.h). Its own layer, before the COPYs, so editing code never
# recompiles it.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit (v2.0.1: self.arm.blend needs its Blend
# action) alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/10ad896883158247039a8bd850bcabcab016a50d.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

COPY capability_engine/ src/capability_engine/
# The UI panels (Stop, questions, notices) the UI actions show: shared with the other capabilities.
COPY shared/capability_ui/ src/capability_ui/
COPY capabilities/button_press/ src/button_press/

# /module_ws is an overlay on the base's /ros2_ws (the RAMMP interfaces); the
# base's entrypoint sources its install/ automatically. The base ships
# rosdep's sources but no cache, so rosdep update first.
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf build log

# `ros2 launch` on Humble has no SIGTERM handler. As PID 1 it ignores the
# SIGTERM `docker stop` (and `sheppy woof`) sends and is SIGKILLed after the
# grace period, leaving the node no chance to stop cleanly. On SIGINT it shuts
# its nodes down properly, so make that the stop signal.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "button_press", "button_press.launch.py"]

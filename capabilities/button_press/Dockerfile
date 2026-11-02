# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=button_press`). The image colcon-builds
# capability_engine and this capability, and runs the same launch file.
#
# nightly-20261030 is the base the nightly arm driver is built on (interfaces
# 73e7ed1716bf), so self.arm speaks the driver's exact messages. It carries the
# arm, common, orchestrator and perception interfaces; rammp_curobo_interfaces it
# does not. rammp-base, not rammp-cuda: nothing here uses the GPU (cuRobo runs in
# the planner's own image). arm64 only.
FROM ghcr.io/rammp-org/rammp-base:nightly-20261030

WORKDIR /module_ws

# The press gates every leg in PyBullet (kinematics.py), and so does self.arm's
# sim. PyPI ships no arm64 wheel, so this compiles from source (the base has gcc,
# g++ and Python.h). Its own layer, before the COPYs, so editing code never
# recompiles it.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/fd8b89cb18fd854f3add36ab9b1b72933dbabf76.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

COPY capability_engine/ src/capability_engine/
COPY capabilities/button_press/ src/button_press/

# /module_ws is an overlay on the base's /ros2_ws (the RAMMP interfaces); the
# base's entrypoint sources its install/ automatically. The nightly base ships
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

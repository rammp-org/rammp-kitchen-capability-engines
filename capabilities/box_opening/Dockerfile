# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=box_opening`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws, capability_engine and this
# capability, and runs the same node.
#
# nightly-20261030 is the base the nightly arm driver is built on (interfaces
# 73e7ed1716bf, which capability_engine's self.arm needs), so box opening speaks
# the driver's exact messages. It carries the
# arm, common, orchestrator and perception interfaces; rammp_curobo_interfaces it
# does not. Nothing in it uses the GPU since the OWL detectors went (the box comes
# from the scene segmenter); the base is still rammp-cuda's, as built and tested so
# far, and may move to rammp-base. arm64 only.
FROM ghcr.io/rammp-org/rammp-cuda:nightly-20261030

WORKDIR /module_ws

# capability_engine's sim (self.arm) imports PyBullet. PyPI ships no arm64 wheel, so
# this compiles from source; the version and the layer are grab_container's, so a
# machine that built either image builds this one without compiling it again.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/525b7527df7ac75b5446f0a9367e651a0ecdb303.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

COPY capability_engine/ src/capability_engine/
COPY capabilities/box_opening/ src/box_opening/

# /module_ws is an overlay on the base's /ros2_ws (the RAMMP interfaces); the
# base's entrypoint sources its install/ automatically. The base ships rosdep's
# sources but not its cache, hence the update.
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
# its nodes down properly, so make that the stop signal. The node alone (the real
# fragment's command) stops on either.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "box_opening", "box_opening.launch.py"]

# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=box_opening`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws, capability_engine and this
# capability, and runs the same node.
#
# 2.2.0-jp6 carries rammp-interfaces-ros2 v2.1.0, all seven packages (arm, base,
# common, orchestrator, perception, peripheral, ui_bridge), with the ExecuteAction
# token and SetEngineOverlayVisible capability_engine uses. Its arm messages are the
# arm driver's (since 73e7ed1716bf, the nightly driver's, only SetContactThresholds
# and CONTACT_STOP were added), so self.arm speaks them exactly.
# rammp_curobo_interfaces it does not carry.
# Nothing in it uses the GPU since the OWL detectors went (the box comes from the
# scene segmenter); the base is still rammp-cuda's, as built and tested so far, and
# may move to rammp-base. arm64 only.
FROM ghcr.io/rammp-org/rammp-cuda:2.2.0-jp6

WORKDIR /module_ws

# capability_engine's sim (self.arm) imports PyBullet. PyPI ships no arm64 wheel, so
# this compiles from source; the version and the layer are grab_container's, so a
# machine that built either image builds this one without compiling it again.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit (v2.0.1: self.arm.blend needs its Blend
# action) alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/10ad896883158247039a8bd850bcabcab016a50d.tar.gz /tmp/curobo.tar.gz
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

# The run's point-cloud work is numpy on one core at a time; one BLAS thread
# keeps it from spreading over the Jetson's cores, the arm driver's included.
ENV OPENBLAS_NUM_THREADS=1

# `ros2 launch` on Humble has no SIGTERM handler. As PID 1 it ignores the
# SIGTERM `docker stop` (and `sheppy woof`) sends and is SIGKILLed after the
# grace period, leaving the node no chance to stop cleanly. On SIGINT it shuts
# its nodes down properly, so make that the stop signal. The node alone (the real
# fragment's command) stops on either.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "box_opening", "box_opening.launch.py"]

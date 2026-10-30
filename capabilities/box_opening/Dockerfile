# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=box_opening`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws, capability_engine and this
# capability, and runs the same node.
#
# nightly-20261027 is the base the nightly arm driver is built on (interfaces
# 672a329f9036), so box opening speaks the driver's exact messages. It carries the
# arm, common, orchestrator and perception interfaces; rammp_curobo_interfaces it
# does not. rammp-cuda, not rammp-base: the OWL detectors run OWLv2 on the Jetson's
# GPU (the image ships CUDA 12.6 and the Jetson build of torch). arm64 only.
FROM ghcr.io/rammp-org/rammp-cuda:nightly-20261027

WORKDIR /module_ws

# capability_engine's sim (self.arm) imports PyBullet. PyPI ships no arm64 wheel, so
# this compiles from source; the version and the layer are grab_container's, so a
# machine that built either image builds this one without compiling it again.
RUN pip install --no-cache-dir pybullet==3.2.7

# OWLv2 (perception/owl_node): transformers at the version the OWL detectors run with
# on flatbot (2026-10-30). The base's torch, torchvision and numpy are held where they
# are (a constraints file of their installed versions), so nothing swaps the Jetson's
# CUDA torch for a PyPI one.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN pip freeze | grep -iE '^(torch|torchvision|numpy)==' > /tmp/keep.txt \
    && pip install --no-cache-dir -c /tmp/keep.txt \
        transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.7.0 tokenizers==0.22.2 \
    && rm /tmp/keep.txt
# The weights are the host's Hugging Face cache, mounted here (the real fragment's
# volume); nothing is fetched at run time.
ENV HF_HOME=/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

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

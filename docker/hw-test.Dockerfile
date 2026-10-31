# Hardware-test image for self.arm, run ON THE CELL against the real driver (make test-hw).
# The production base (arm64), the same one the capabilities are built FROM, so self.arm speaks
# the driver's exact messages; plus self.arm's dependencies and pytest. Like the x86 test image,
# the repo is mounted at /repo at run time: the code under test is whatever is checked out.
FROM ghcr.io/rammp-org/rammp-base:nightly-20261030

WORKDIR /module_ws

# pybullet as in the capability images (compiled from source: no arm64 wheel).
# hadolint ignore=DL3013
RUN pip install --no-cache-dir pybullet==3.2.7 pytest==8.3.3

# rammp_curobo_interfaces, not in the base, from the RAMMP-CuRobo commit the capabilities pin;
# capability_engine's other dependencies from its package.xml (rosdep).
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/fd8b89cb18fd854f3add36ab9b1b72933dbabf76.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz
COPY capability_engine/package.xml /tmp/deps/capability_engine/package.xml
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src /tmp/deps --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf build log

ENV PYTHONPATH=/repo/capability_engine
RUN printf '%s\n' '#!/bin/bash' '. /opt/ros/humble/setup.bash' '. /ros2_ws/install/setup.bash' \
        '. /module_ws/install/setup.bash' 'exec "$@"' > /hw-entrypoint.sh \
    && chmod +x /hw-entrypoint.sh
ENTRYPOINT ["/hw-entrypoint.sh"]
WORKDIR /repo

#!/bin/bash
# The Unreal touchscreen UI on the robot's own screen (display :0), with the Vulkan loader it needs (without it:
# "Unable to initialize Vulkan", the system loader being too old). The UI bridge connects out to it on 30020.
UE_DIR=${UE_DIR:-$HOME/RammpUI-detection-markers-LinuxArm64}
VULKAN_LOADER=${VULKAN_LOADER:-$HOME/Vulkan-Loader/build/loader}
cd "$UE_DIR" || exit 1
# Always the robot's own screen: an ssh session's DISPLAY (X forwarding) is the laptop's, where Unreal finds no
# video device and exits (11-12).
export LD_LIBRARY_PATH=$VULKAN_LOADER DISPLAY=${UE_DISPLAY:-:0}
exec ./RammpUI-Arm64.sh

# Fridge side grasp: open to 90 deg

Scripts that run the fridge open with the side grasp, outside the capability node:

1. **Detect**: SAM 3 on the scene camera for the handle; the door's direction from the handle estimator
   (`fridge/handle_estimator.py`, fitting `RAMMP.Fridge.Handle5.STL`), which must have a fresh, agreeing estimate.
2. **Plan everything before any motion**: cuRobo to just outside the handle, the slide in from the hinge side
   (`fridge/side_grasp.py`, `SIDE_INSERT_DEG`), the pull along the door's arc about the hinge pin (`pull_lib.py`),
   the release, the route round the door's free edge and the push to 90 deg (`push_lib.py`, built on
   `fridge.push_open`). Checks: the depth rule (`depth_rule.py`), the door model, the arm against itself.
3. **Move**: the reach-in; the pull to `--swing` deg and on `RELEASE_ARC_DEG` while the gripper opens; then one
   motion round the door and the push, under the stiffest joint impedance (`pull_back_gains.py`).

Each motion must arrive, or the run stops there.

## Run

In the fridge image, host network, with a data folder mounted at `/work` holding `robot.urdf` (the arm's
URDF), `fridge_open_home.json` (the start pose) and the estimator's `handle_estimate.json`:

```
python3 /module_ws/src/fridge/side_run/run_general.py --swing=50 --push          # moves the arm
python3 /module_ws/src/fridge/side_run/run_general.py --swing=50 --push --dry    # plans only
python3 /module_ws/src/fridge/side_run/home_closed.py /work/fridge_open_home.json   # home (door closed)
```

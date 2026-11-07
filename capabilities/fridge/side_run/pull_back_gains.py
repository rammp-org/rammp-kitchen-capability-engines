"""The pull's joint impedance: the driver's stiffest allowed gains (kinova-gen3-driver interface/gains.h: kq <= 400
N*m/rad, torque_limit <= the URDF effort limits); zeta 1.0 = critically damped."""
import rammp_arm
HARDEST = {"kq": (400.0,) * 7, "zeta": 1.0, "torque_limit": (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
           "path_tol": rammp_arm.PATH_TOL_RAD}

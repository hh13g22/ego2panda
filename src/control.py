"""Cartesian position control for the Panda via LIBERO's OSC_POSE delta actions.

Action layout: [dx, dy, dz, droll, dpitch, dyaw, gripper]. Each position entry in [-1, 1] is scaled by the
controller to +-5 cm of end-effector motion per control step (20 Hz). Gripper: -1 open, +1 closed.
Orientation is held fixed by always sending zero rotation deltas (gripper pointing down).
"""
import numpy as np

POS_GAIN = 10.0  # action units per metre of position error (10 cm error -> full-scale 5 cm step)
GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0


def position_action(eef_pos: np.ndarray, target_pos: np.ndarray, gripper: float) -> np.ndarray:
    """Proportional step from the current end-effector position toward target_pos (both world frame, metres)."""
    dpos = np.clip(POS_GAIN * (target_pos - eef_pos), -1.0, 1.0)
    return np.concatenate([dpos, np.zeros(3), [gripper]])

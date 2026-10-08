"""Turn "move the gripper to this point" into a LIBERO action.

LIBERO's OSC_POSE controller takes [dx, dy, dz, droll, dpitch, dyaw, gripper]. Position values in [-1, 1]
move the gripper up to 5 cm per control step (20 Hz). Gripper: -1 open, +1 closed. Rotation deltas are
always 0, so the gripper keeps pointing down.
"""
import numpy as np

POS_GAIN = 10.0  # action units per metre of position error (10 cm error -> full-scale 5 cm step)
GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0


def position_action(eef_pos: np.ndarray, target_pos: np.ndarray, gripper: float) -> np.ndarray:
    """Return the action that moves the gripper a step toward target_pos (a bigger error gives a bigger step)."""
    dpos = np.clip(POS_GAIN * (target_pos - eef_pos), -1.0, 1.0)
    return np.concatenate([dpos, np.zeros(3), [gripper]])

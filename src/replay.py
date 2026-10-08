"""Drive the simulated Panda along one retargeted human clip and record what happened.

The recording (camera images, robot state and action at every step) becomes a training demonstration
if LIBERO reports that the task succeeded.
"""
import numpy as np

from src.control import position_action
from src.retarget import retarget
from src.sim import OBJECT, TARGET, policy_observation, settle
from src.trajectory import grasp_interval

STEPS_AFTER_SUCCESS = 10  # keep a few frames of the object resting in the bowl, then end the demo


def replay(env, init_state, traj: np.ndarray) -> dict:
    """Replay one human trajectory on one start state.

    Returns whether the task succeeded, the recorded frames (images, state, action) and the human-to-robot scale.
    """
    env.reset()
    obs = settle(env, env.set_init_state(init_state))
    grasp, release = grasp_interval(traj[:, 4] > 0.5)
    targets, gripper, scale = retarget(traj, grasp, release, obs[f"{OBJECT}_pos"], obs[f"{TARGET}_pos"])

    frames, success_step = [], None
    for k, (target, grip) in enumerate(zip(targets, gripper)):
        action = position_action(obs["robot0_eef_pos"], target, grip)
        frames.append({**policy_observation(obs), "action": action.astype(np.float32)})
        obs, _, done, _ = env.step(action)
        if done and success_step is None:
            success_step = k
        if success_step is not None and k >= success_step + STEPS_AFTER_SUCCESS:
            break
    return {"success": success_step is not None, "frames": frames, "scale": scale}

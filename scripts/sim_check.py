"""Check the robot control before using any human data.

Runs a hand-written pick-and-place (hover, descend, close, lift, carry, lower, open) on LIBERO start
states. Success confirms the coordinate frames, the action scale and the gripper sign are right.

Usage:
    python scripts/sim_check.py --episodes 50 --out-dir results/sim_check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control import GRIPPER_CLOSED, GRIPPER_OPEN, position_action
from src.sim import SUITE, TASK_ID, OBJECT, TARGET, agentview_frame, make_env, settle

HOVER_M = 0.10       # approach height above the object
GRASP_DZ_M = 0.0     # end-effector height relative to the object's centre when closing
CARRY_M = 0.15       # lift / transport height above the bowl
RELEASE_M = 0.08     # release height above the bowl centre


def scripted_waypoints(obj: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, float, int]]:
    """Return the hand-written pick-and-place as (target position, gripper command, number of steps)."""
    up = np.array([0.0, 0.0, 1.0])
    return [
        (obj + HOVER_M * up, GRIPPER_OPEN, 40),
        (obj + GRASP_DZ_M * up, GRIPPER_OPEN, 30),
        (obj + GRASP_DZ_M * up, GRIPPER_CLOSED, 15),
        (obj + CARRY_M * up, GRIPPER_CLOSED, 25),
        (target + CARRY_M * up, GRIPPER_CLOSED, 40),
        (target + RELEASE_M * up, GRIPPER_CLOSED, 20),
        (target + RELEASE_M * up, GRIPPER_OPEN, 20),
    ]


def run_episode(env, init_state, record: bool) -> tuple[bool, list[np.ndarray]]:
    """Run the scripted pick-and-place on one start state; return success and (optionally) the camera frames."""
    env.reset()
    obs = settle(env, env.set_init_state(init_state))
    frames, success = [], False
    for target, gripper, steps in scripted_waypoints(obs[f"{OBJECT}_pos"], obs[f"{TARGET}_pos"]):
        for _ in range(steps):
            obs, _, done, _ = env.step(position_action(obs["robot0_eef_pos"], target, gripper))
            success = success or done
            if record:
                frames.append(agentview_frame(obs))
    return success, frames


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    """Save frames as an animated GIF."""
    images = [Image.fromarray(f) for f in frames[::2]]
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


def main() -> None:
    """Run the scripted episodes, save a GIF of the first one, and save the success rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env, task, init_states = make_env(SUITE, TASK_ID)
    successes = []
    for ep in range(args.episodes):
        success, frames = run_episode(env, init_states[ep % len(init_states)], record=ep == 0)
        successes.append(success)
        if ep == 0:
            save_gif(frames, args.out_dir / "scripted_episode0.gif")
        print(f"episode {ep}: {'success' if success else 'fail'}")
    env.close()

    metrics = {"task": task.language, "episodes": args.episodes, "successes": int(sum(successes)),
               "success_rate": sum(successes) / args.episodes}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()

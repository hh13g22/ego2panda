"""Check that a LIBERO task renders off-screen and save the two camera images.

Usage:
    python scripts/render_check.py --suite libero_object --task-id 0 --out results/render_check.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import make_env

ZERO_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open
SETTLE_STEPS = 10  # let objects settle after reset


def main() -> None:
    """Load the task, render both cameras, fail if the images are blank, and save them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env, task, init_states = make_env(args.suite, args.task_id)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(ZERO_ACTION)

    agent = obs["agentview_image"]
    wrist = obs["robot0_eye_in_hand_image"]
    if agent.std() == 0 or wrist.std() == 0:
        raise RuntimeError("Rendered image is constant — off-screen rendering is not working")

    # MuJoCo renders with the origin at the bottom-left, so flip vertically for display.
    frame = np.concatenate([agent[::-1], wrist[::-1]], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(args.out)

    print(f"task: {task.language}")
    print(f"action dim: {env.env.action_dim}")
    print(f"eef pos (m, world frame): {np.round(obs['robot0_eef_pos'], 3)}")
    print(f"image keys: {[k for k in obs if k.endswith('_image')]}, shape {agent.shape}")
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()

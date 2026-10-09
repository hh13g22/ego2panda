"""Record one policy rollout as a README GIF, continuing past the moment the task succeeds.

lerobot-eval stops recording on the step LIBERO reports success, while the gripper still covers the box, and the
policy itself never lets go (its training demos also end at success). So after success this script takes over with
a scripted release: the gripper opens, then the arm rises straight up, so the box is visible in the bowl. Front
camera, one start state, real time.

Usage:
    python scripts/rollout_gif.py --policy outputs/train/smolvla_ego2panda/checkpoints/last/pretrained_model \
        --start-state 0 --out results/eval/success.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import SUITE, TASK_ID, agentview_frame, make_env, policy_observation, settle

MAX_STEPS = 300    # LIBERO-Goal's episode limit, as in lerobot-eval
OPEN = np.array([0.0] * 6 + [-1.0])               # no motion, gripper open
LIFT = np.array([0.0, 0.0, 0.5] + [0.0] * 3 + [-1.0])  # up 2.5 cm per step command, gripper open
RELEASE = [OPEN] * 10 + [LIFT] * 30                # 0.5 s open, 1.5 s up (20 Hz)
N_ACTION_STEPS = 10
CAMERAS = {"observation.images.image": "observation.images.camera1",
           "observation.images.image2": "observation.images.camera2"}
FRAME_STEP, FRAME_MS = 3, 150  # every 3rd 20 Hz step, played in real time
SEED = 0


def to_batch(obs: dict, task: str) -> dict:
    """Convert one LIBERO observation into the batch the policy's preprocessor expects."""
    policy_obs = policy_observation(obs)
    batch = {CAMERAS[k]: torch.from_numpy(policy_obs[k]).permute(2, 0, 1).float().unsqueeze(0) / 255 for k in CAMERAS}
    batch["observation.state"] = torch.from_numpy(policy_obs["observation.state"]).unsqueeze(0)
    batch["task"] = [task]
    return batch


def main() -> None:
    """Run the policy from one start state until success, then the scripted release; save the front camera as a GIF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--start-state", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.policy).to("cuda").eval()
    policy.config.n_action_steps = N_ACTION_STEPS
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env, task, init_states = make_env(SUITE, TASK_ID)
    env.reset()
    obs = settle(env, env.set_init_state(init_states[args.start_state]))
    policy.reset()
    torch.manual_seed(SEED)

    frames, success_step = [agentview_frame(obs)], None
    with torch.no_grad():
        for step in range(MAX_STEPS):
            action = post(policy.select_action(pre(to_batch(obs, task.language)))).cpu().numpy()[0]
            obs, _, done, _ = env.step(action)
            frames.append(agentview_frame(obs))
            if done:
                success_step = step
                break
    if success_step is None:
        raise RuntimeError(f"No success within {MAX_STEPS} steps from start state {args.start_state}")
    for action in RELEASE:
        obs, _, _, _ = env.step(action)
        frames.append(agentview_frame(obs))
    env.close()

    images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames[::FRAME_STEP]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"success at step {success_step}; wrote {args.out}: {len(images)} frames")


if __name__ == "__main__":
    main()

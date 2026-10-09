"""How much does quantising the vision encoder change the policy's actions? Offline, on held-out frames.

For each mode, the policy predicts an action chunk for the same held-out frames with the same noise; the first
action is compared with the unquantised (bf16) prediction. Also reports the vision encoder's weight memory.

Usage:
    python scripts/quant_action_error.py --policy outputs/distill/smolvla_shallow8/step_010000 \
        --dataset-root data/lerobot/ego2panda --split results/split.json --out results/quant/action_error.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantize import BITS, quantize_vision_encoder

REPO_ID = "ego2panda/libero_goal_cream_cheese_bowl"
N_FRAMES = 200
STEP_CM = 5.0  # an action of 1.0 commands 5 cm of end-effector motion per control step
SEED = 0


def first_actions(path: str, mode: str, frames: list) -> tuple[np.ndarray, dict]:
    """Load the policy, quantise its vision encoder, and predict the first action for every frame."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(path).eval()
    info = quantize_vision_encoder(policy, mode)
    policy = policy.to("cuda")
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    actions = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            torch.manual_seed(SEED + i)  # same noise for every mode, so only the vision encoder differs
            actions.append(post(policy.predict_action_chunk(pre(frame))[:, 0]).cpu().numpy()[0])
    del policy
    torch.cuda.empty_cache()
    return np.array(actions), info


def main() -> None:
    """Compare every mode's actions with bf16 on the same held-out frames and save the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root,
                             episodes=json.loads(args.split.read_text())["val_episodes"], video_backend="pyav")
    indices = np.random.default_rng(SEED).choice(len(dataset), size=N_FRAMES, replace=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=1)
    frames = list(loader)

    reference, _ = first_actions(args.policy, "bf16", frames)
    results = {"frames": N_FRAMES}
    for mode in BITS:
        actions, info = first_actions(args.policy, mode, frames)
        diff = np.abs(actions[:, :3] - reference[:, :3]) * STEP_CM
        results[mode] = {**info,
                         "action_diff_cm_per_step_mean": round(float(diff.mean()), 4),
                         "action_diff_cm_per_step_max": round(float(diff.max()), 4),
                         "gripper_agreement": round(float((np.sign(actions[:, 6]) == np.sign(reference[:, 6])).mean()), 3)}
        print(mode, results[mode], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

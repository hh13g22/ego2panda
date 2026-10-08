"""Compare the policy's predicted motion with unseen human motion (the 3 held-out clips).

For every frame of the held-out episodes, the policy sees the recorded camera images and robot state and
predicts the next action. That prediction is compared with the action the human-derived replay actually took.
The plot shows, for the first episode of each held-out clip, the running sum of each step's commanded movement
(action x 5 cm), once for the human's actions and once for the policy's predictions. These are commands, not
measured positions: the arm moves less than commanded per step, so the sums only compare the shape of the motion.

Usage:
    python scripts/eval_heldout.py --policy outputs/train/smolvla_ego2panda/checkpoints/last/pretrained_model \
        --dataset-root data/lerobot/ego2panda --split results/split.json \
        --replay-metrics results/replay/metrics.json --out-dir results/heldout
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.split_episodes import episodes_per_clip

REPO_ID = "ego2panda/libero_goal_cream_cheese_bowl"
STEP_CM = 5.0  # an action of 1.0 commands 5 cm of end-effector motion per control step
BATCH_SIZE = 16
SEED = 0  # SmolVLA samples noise for every prediction; a fixed seed makes the metrics reproducible
HUMAN_COLOR, POLICY_COLOR = "#e8743b", "#2a6fdb"


def predict_first_actions(policy, preprocessor, postprocessor, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted action, human action, episode index) for every frame of the dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    predicted, human, episode = [], [], []
    with torch.no_grad():
        for batch in loader:
            human.append(batch["action"].numpy())
            episode.append(batch["episode_index"].numpy())
            chunk = policy.predict_action_chunk(preprocessor(batch))
            predicted.append(postprocessor(chunk[:, 0]).cpu().numpy())
    return np.concatenate(predicted), np.concatenate(human), np.concatenate(episode)


def plot_paths(clips: dict[str, int], predicted, human, episode, out_path: Path) -> None:
    """For one episode per clip: summed commanded path (x-y and height) and the gripper command over time."""
    fig, axes = plt.subplots(len(clips), 3, figsize=(12, 3.2 * len(clips)), squeeze=False)
    for row, (clip, ep) in zip(axes, clips.items()):
        mask = episode == ep
        t = np.arange(mask.sum()) / 20
        paths = {name: np.cumsum(a[mask, :3] * STEP_CM, axis=0) for name, a in (("human", human), ("policy", predicted))}
        for name, color in (("human", HUMAN_COLOR), ("policy", POLICY_COLOR)):
            row[0].plot(paths[name][:, 0], paths[name][:, 1], color=color, label=name)
            row[1].plot(t, paths[name][:, 2], color=color)
            row[2].plot(t, (human if name == "human" else predicted)[mask, 6], color=color)
        row[0].set_aspect("equal")
        row[0].set_title(f"clip {clip}: summed x-y command (cm)", loc="left", fontsize=10)
        row[1].set_title("summed height command (cm)", loc="left", fontsize=10)
        row[2].set_title("gripper command (-1 open, +1 closed)", loc="left", fontsize=10)
        row[1].set_xlabel("time (s)")
        row[2].set_xlabel("time (s)")
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Held-out human motion vs policy prediction (running sum of per-step commands, not measured positions)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)


def main() -> None:
    """Predict actions on all held-out episodes, save error metrics and the path plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    split = json.loads(args.split.read_text())
    blocks = episodes_per_clip(json.loads(args.replay_metrics.read_text())["per_clip"])
    first_episode = {clip: blocks[clip][0] for clip in split["val_clips"]}

    policy = SmolVLAPolicy.from_pretrained(args.policy).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.policy, preprocessor_overrides={"device_processor": {"device": "cuda"}})
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root, episodes=split["val_episodes"], video_backend="pyav")

    torch.manual_seed(SEED)
    predicted, human, episode = predict_first_actions(policy, preprocessor, postprocessor, dataset)
    position_error_cm = np.abs(predicted[:, :3] - human[:, :3]) * STEP_CM
    metrics = {
        "val_clips": split["val_clips"],
        "episodes": len(split["val_episodes"]),
        "frames": int(len(human)),
        "mean_abs_error_cm_per_step": {k: round(float(v), 3) for k, v in zip("xyz", position_error_cm.mean(0))},
        "human_mean_abs_command_cm_per_step": {k: round(float(v), 3) for k, v in zip("xyz", (np.abs(human[:, :3]) * STEP_CM).mean(0))},
        "gripper_agreement": round(float((np.sign(predicted[:, 6]) == np.sign(human[:, 6])).mean()), 3),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_paths(first_episode, predicted, human, episode, args.out_dir / "paths.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

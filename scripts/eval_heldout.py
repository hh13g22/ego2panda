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


def release_step(gripper: np.ndarray) -> int:
    """Return the step where the gripper command switches from closed (+1) to open (-1)."""
    opens = np.flatnonzero((gripper[:-1] > 0) & (gripper[1:] <= 0))
    if len(opens) == 0:
        raise ValueError("The human gripper command never switches from closed to open")
    return int(opens[0]) + 1


def plot_paths(clips: dict[str, int], predicted, human, episode, out_path: Path) -> None:
    """Report figure: for one episode per held-out clip, the summed x-y and height commands and the gripper command,
    human (solid) vs policy (dashed), with one shared legend, row labels and a caption."""
    styles = {"human": dict(color=HUMAN_COLOR, linestyle="-", linewidth=1.8, label="Human trajectory"),
              "policy": dict(color=POLICY_COLOR, linestyle="--", linewidth=1.6, label="Policy prediction")}
    fig, axes = plt.subplots(len(clips), 3, figsize=(12, 3.0 * len(clips) + 1.2), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.25, 1.25], "wspace": 0.32, "hspace": 0.45})
    for r, (row, (clip, ep)) in enumerate(zip(axes, clips.items())):
        mask = episode == ep
        t = np.arange(mask.sum()) / 20
        actions = {"human": human[mask], "policy": predicted[mask]}
        for name, a in actions.items():
            path = np.cumsum(a[:, :3] * STEP_CM, axis=0)
            row[0].plot(path[:, 0], path[:, 1], **styles[name])
            row[1].plot(t, path[:, 2], **styles[name])
            row[2].plot(t, a[:, 6], **styles[name])
        start = np.cumsum(actions["human"][:1, :3] * STEP_CM, axis=0)[0]
        row[0].plot(start[0], start[1], "o", color="0.2", markersize=5, label="Start")
        human_path = np.cumsum(actions["human"][:, :3] * STEP_CM, axis=0)
        goal = human_path[release_step(actions["human"][:, 6])]
        row[0].plot(goal[0], goal[1], "*", color="0.2", markersize=11, label="Goal (human release over the bowl)")
        row[0].set_aspect("equal", adjustable="datalim")
        row[0].set_xlabel("x (cm)")
        row[0].set_ylabel(f"Clip {clip}\n\ny (cm)", fontsize=10)
        row[1].set_ylabel("z (cm)")
        row[2].set_yticks([-1, 1], ["open", "closed"])
        row[2].set_ylim(-1.4, 1.4)
        for ax in row[1:]:
            ax.set_xlabel("Time (s)")
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="0.92", linewidth=0.8)
            ax.tick_params(labelsize=9)
        if r == 0:
            for ax, title in zip(row, ("(a) Summed x-y command", "(b) Summed height command", "(c) Gripper command")):
                ax.set_title(title, loc="left", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Policy predictions on held-out human clips", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "At every 20 Hz step the policy sees the recorded camera images and robot state and predicts the next command;\n"
             "curves (a) and (b) add up those commands (action x 5 cm per step), so they compare the shape of the motion, "
             "not measured positions. One episode per held-out clip.",
             ha="center", va="bottom", fontsize=9, color="0.25")
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.98)
    fig.savefig(out_path, dpi=200)


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

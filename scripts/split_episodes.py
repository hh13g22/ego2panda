"""Split the replayed dataset into training and validation episodes, by clip.

Splitting by clip (not by episode) matters: each clip was replayed on 25 start states, so its episodes are
near-copies of each other. One clip per layout (a, b, c) is held out for validation, chosen at random with a
fixed seed. The dataset does not store clip names, but replay_demos.py saved successful episodes in a fixed
order (clips sorted by name, one block per clip), which the per-clip counts in the replay metrics recover.

Usage:
    python scripts/split_episodes.py --replay-metrics results/replay/metrics.json \
        --dataset-root data/lerobot/ego2panda --out results/split.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0
LAYOUTS = ("a", "b", "c")


def episodes_per_clip(per_clip: dict) -> dict[str, list[int]]:
    """Return the dataset episode indices of every clip, in the order replay_demos.py saved them."""
    episodes, next_index = {}, 0
    for clip, stats in per_clip.items():
        episodes[clip] = list(range(next_index, next_index + stats["successes"]))
        next_index += stats["successes"]
    return episodes


def main() -> None:
    """Pick one validation clip per layout and save the train / validation episode lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-metrics", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    per_clip = json.loads(args.replay_metrics.read_text())["per_clip"]
    episodes = episodes_per_clip(per_clip)
    total = json.loads((args.dataset_root / "meta" / "info.json").read_text())["total_episodes"]
    if sum(len(e) for e in episodes.values()) != total:
        raise ValueError(f"Replay metrics count {sum(len(e) for e in episodes.values())} episodes, dataset has {total}")

    rng = np.random.default_rng(SEED)
    val_clips = [str(rng.choice(sorted(c for c in episodes if c[0] == layout and episodes[c])))
                 for layout in LAYOUTS]
    train_clips = sorted(c for c in episodes if c not in val_clips and episodes[c])
    split = {
        "seed": SEED,
        "val_clips": val_clips,
        "train_clips": train_clips,
        "val_episodes": sorted(i for c in val_clips for i in episodes[c]),
        "train_episodes": sorted(i for c in train_clips for i in episodes[c]),
    }
    args.out.write_text(json.dumps(split, indent=2))
    print(f"validation clips {val_clips}: {len(split['val_episodes'])} episodes; "
          f"training clips {len(train_clips)}: {len(split['train_episodes'])} episodes")


if __name__ == "__main__":
    main()

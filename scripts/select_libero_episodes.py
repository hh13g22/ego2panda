"""Select and download the episodes of one LIBERO task in lerobot/libero, for training the robot-data baseline.

Downloads the dataset's metadata and frame tables at a pinned revision, finds the episodes whose task is TASK,
downloads only the video files that contain them, and writes the episode list as {"train_episodes": [...]}
(the same format as results/split.json).

Usage:
    python scripts/select_libero_episodes.py --dataset-root data/lerobot/libero --out results/libero_task_episodes.json
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"  # pinned so the episode list is reproducible
TASK = "put the cream cheese in the bowl"


def video_files(dataset_root: Path, episodes: list[int]) -> list[str]:
    """Return the dataset-relative paths of the video files that hold the given episodes."""
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(dataset_root / "meta" / "episodes" / "**" / "*.parquet"),
                                                                 recursive=True))])
    meta = meta[meta.episode_index.isin(episodes)]
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    paths = set()
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            for chunk, file in zip(meta[f"videos/{key}/chunk_index"], meta[f"videos/{key}/file_index"]):
                paths.add(info["video_path"].format(video_key=key, chunk_index=int(chunk), file_index=int(file)))
    return sorted(paths)


def main() -> None:
    """Find the episodes of TASK and save their indices."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=args.dataset_root,
                      allow_patterns=["meta/**", "data/**"])
    tasks = pd.read_parquet(args.dataset_root / "meta" / "tasks.parquet")
    task_index = int(tasks.loc[TASK, "task_index"])
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"])
                        for f in sorted(glob.glob(str(args.dataset_root / "data" / "**" / "*.parquet"), recursive=True))])
    episodes = sorted(int(e) for e in frames.loc[frames.task_index == task_index, "episode_index"].unique())
    if not episodes:
        raise ValueError(f"No episodes found for task '{TASK}'")
    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=args.dataset_root,
                      allow_patterns=video_files(args.dataset_root, episodes))
    args.out.write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION, "task": TASK,
                                    "task_index": task_index, "train_episodes": episodes}, indent=2))
    print(f"{len(episodes)} episodes of '{TASK}' (task {task_index}), {int((frames.task_index == task_index).sum())} frames")


if __name__ == "__main__":
    main()

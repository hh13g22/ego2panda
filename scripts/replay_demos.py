"""Stage 4: turn the retargeted human clips into a robot training dataset.

Every clip is replayed in LIBERO on start states 25-49 (start states 0-24 are kept for evaluating the
policy). Only replays that succeed are saved, as a LeRobot dataset in the same format LeRobot's LIBERO
evaluation feeds a policy. The replay success rate is saved to --out-dir.

Usage:
    python scripts/replay_demos.py --processed-dir data/processed --dataset-root data/lerobot/ego2panda \
        --out-dir results/replay --first-init 25 --last-init 49 --workers 4
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ID = "ego2panda/libero_goal_cream_cheese_bowl"
IMAGE_SIZE = 256
FPS = 20

_env = _init_states = None


def _worker_init() -> None:
    """Create one simulator per worker process."""
    global _env, _init_states
    from src.sim import SUITE, TASK_ID, make_env
    _env, _, _init_states = make_env(SUITE, TASK_ID, image_size=IMAGE_SIZE)


def _run(job: tuple[str, str, int]) -> dict:
    """Replay one clip on one start state; return the frames only if it succeeded."""
    from src.replay import replay
    clip, traj_path, init_id = job
    result = replay(_env, _init_states[init_id], np.load(traj_path))
    return {"clip": clip, "init": init_id, "success": result["success"], "scale": result["scale"],
            "frames": result["frames"] if result["success"] else None}


def dataset_features() -> dict:
    """Describe the dataset columns: two camera images, the 8-number state and the 7-number action."""
    image = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.image": image,
        "observation.images.image2": image,
        "observation.state": {"dtype": "float32", "shape": (8,),
                              "names": ["x", "y", "z", "ax", "ay", "az", "finger_l", "finger_r"]},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]},
    }


def main() -> None:
    """Replay every clip on every start state, save the successes as a dataset, and save the success rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-init", type=int, default=25)
    parser.add_argument("--last-init", type=int, default=49)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.dataset_root.exists():
        raise FileExistsError(f"{args.dataset_root} exists; delete it to rebuild the dataset")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from src.sim import SUITE, TASK_ID, make_env
    env, task, _ = make_env(SUITE, TASK_ID)
    env.close()

    trajs = sorted(args.processed_dir.glob("*_traj.npy"))
    if not trajs:
        raise FileNotFoundError(f"No *_traj.npy files in {args.processed_dir}")
    jobs = [(p.name.removesuffix("_traj.npy"), str(p), i)
            for p in trajs for i in range(args.first_init, args.last_init + 1)]

    dataset = LeRobotDataset.create(REPO_ID, fps=FPS, features=dataset_features(), root=args.dataset_root,
                                    robot_type="panda")
    per_clip = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
        # One batch of `workers` jobs at a time: at most `workers` episodes are held in RAM while the main
        # process encodes videos (an unbounded imap queue grew to 15 GB and was OOM-killed).
        for start in range(0, len(jobs), args.workers):
            for r in pool.map(_run, jobs[start:start + args.workers]):
                per_clip.setdefault(r["clip"], {"replays": 0, "successes": 0,
                                                "scale_m_per_cm": round(r["scale"], 5)})
                per_clip[r["clip"]]["replays"] += 1
                per_clip[r["clip"]]["successes"] += int(r["success"])
                if r["success"]:
                    for frame in r["frames"]:
                        dataset.add_frame({**frame, "task": task.language})
                    dataset.save_episode()
            print(f"[{min(start + args.workers, len(jobs))}/{len(jobs)}]", flush=True)
    dataset.finalize()

    successes = sum(c["successes"] for c in per_clip.values())
    metrics = {"init_states": [args.first_init, args.last_init], "clips": len(per_clip), "replays": len(jobs),
               "successful_replays": successes, "replay_success_rate": round(successes / len(jobs), 3),
               "clips_with_any_success": sum(c["successes"] > 0 for c in per_clip.values()),
               "per_clip": per_clip}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print({k: v for k, v in metrics.items() if k != "per_clip"})


if __name__ == "__main__":
    main()

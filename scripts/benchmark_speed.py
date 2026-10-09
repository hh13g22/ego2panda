"""Measure inference time and size of SmolVLA policies (teacher vs distilled student).

Times one action-chunk prediction (vision encoder, VLM and all flow-matching steps of the action expert) on a
real training frame at batch size 1, after warm-up, with the GPU synchronised around each call. The unused
text-generation head (lm_head) is removed from every policy after loading; it never runs, so this changes memory,
not speed or outputs.

Usage:
    python scripts/benchmark_speed.py --dataset-root data/lerobot/ego2panda --out results/distill/speed.json \
        --policy teacher=outputs/train/smolvla_ego2panda/checkpoints/last/pretrained_model \
        --policy student=outputs/distill/smolvla_shallow8/step_010000
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill import drop_unused_lm_head

REPO_ID = "ego2panda/libero_goal_cream_cheese_bowl"
WARMUP, REPEATS = 10, 100
SEED = 0


def time_policy(path: str, raw_frame: dict) -> dict:
    """Load one policy and return its parameter count, weight memory and median / p90 chunk-prediction time."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(path).eval()
    drop_unused_lm_head(policy)
    policy = policy.to("cuda")
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": "cuda"}})
    batch = preprocessor(raw_frame)
    torch.manual_seed(SEED)
    times = []
    with torch.no_grad():
        for i in range(WARMUP + REPEATS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append((time.perf_counter() - start) * 1000)
    result = {
        "params_million": round(sum(p.numel() for p in policy.parameters()) / 1e6, 1),
        "vlm_layers": policy.config.num_vlm_layers,
        "chunk_ms_median": round(statistics.median(times), 1),
        "chunk_ms_p90": round(statistics.quantiles(times, n=10)[-1], 1),
        "weights_mb": round(sum(p.numel() * p.element_size() for p in policy.parameters()) / 2**20),
    }
    del policy
    torch.cuda.empty_cache()
    return result


def main() -> None:
    """Time each named policy on the same frame and save the comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True, help="name=path, repeat for each policy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    frame = next(iter(torch.utils.data.DataLoader(
        LeRobotDataset(REPO_ID, root=args.dataset_root, episodes=[0], video_backend="pyav"), batch_size=1)))
    results = {"gpu": torch.cuda.get_device_name(0), "batch_size": 1, "repeats": REPEATS}
    for spec in args.policy:
        name, path = spec.split("=", 1)
        results[name] = time_policy(path, frame)
        print(name, results[name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

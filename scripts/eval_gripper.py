"""Accuracy of the aperture-threshold gripper state against hand labels from label_gripper.py.

Usage: python scripts/eval_gripper.py --labels results/gripper_labels.csv --processed-dir data/processed \
           --out results/gripper_accuracy.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.labels.open() as f:
        rows = [r for r in csv.DictReader(f) if r["label"] != "skip"]
    if not rows:
        raise ValueError(f"No usable labels in {args.labels}")
    truth = np.array([r["label"] == "closed" for r in rows])
    pred = np.array([np.load(args.processed_dir / f"{r['clip']}_traj.npy")[int(r["frame"]), 4] > 0.5 for r in rows])

    result = {
        "labelled_frames": len(rows),
        "accuracy": round(float((pred == truth).mean()), 3),
        "true_closed_predicted_open": int((truth & ~pred).sum()),
        "true_open_predicted_closed": int((~truth & pred).sum()),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

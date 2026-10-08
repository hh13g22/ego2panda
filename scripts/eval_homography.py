"""Homography accuracy vs number of markers used, measured on held-out marker corners.

On every frame where all 4 markers are detected, fit the homography from each subset of k markers and map the
corners of the left-out markers into sheet cm; the error is the distance to their known printed positions.
Usage: python scripts/eval_homography.py --clips data/raw/*.mp4 --out results/homography_accuracy.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aruco_sheet import MARKER_TOP_LEFT_CM, marker_corners_cm
from src.markers import detect_markers, make_detector, to_sheet


def held_out_errors(markers: dict[int, np.ndarray], k: int) -> list[float]:
    """Corner errors (cm) of the markers left out when fitting from every k-marker subset."""
    errors = []
    for used in itertools.combinations(sorted(markers), k):
        H, _ = cv2.findHomography(np.concatenate([markers[i] for i in used]),
                                  np.concatenate([marker_corners_cm(i) for i in used]))
        for j in set(markers) - set(used):
            errors += np.linalg.norm(to_sheet(H, markers[j]) - marker_corners_cm(j), axis=1).tolist()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    detector = make_detector()
    errors = {k: [] for k in (1, 2, 3)}
    n_frames = 0
    for clip in args.clips:
        cap = cv2.VideoCapture(str(clip))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            markers = detect_markers(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(markers) < len(MARKER_TOP_LEFT_CM):
                continue
            n_frames += 1
            for k in errors:
                errors[k] += held_out_errors(markers, k)

    result = {"frames_with_all_4_markers": n_frames}
    for k, e in errors.items():
        e = np.array(e)
        result[f"{k}_markers"] = {"median_cm": round(float(np.median(e)), 3),
                                  "p90_cm": round(float(np.percentile(e, 90)), 3),
                                  "max_cm": round(float(e.max()), 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

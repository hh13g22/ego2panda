"""Hand-label gripper state on randomly sampled frames (ground truth for the aperture threshold).

Shows raw frames one at a time. Keys: c = cube held between thumb and index (gripper should be CLOSED),
o = not holding (OPEN, including a pre-shaped hand on the way to the cube), s = unclear/skip, q = save and quit.
Progress is saved after every key, and re-running resumes where you left off.
Usage: python scripts/label_gripper.py --raw-dir data/raw --processed-dir data/processed \
           --out results/gripper_labels.csv --n 200
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

KEYS = {ord("c"): "closed", ord("o"): "open", ord("s"): "skip"}
DISPLAY_SCALE = 1.5
SEED = 0


def sample_frames(processed_dir: Path, n: int) -> list[tuple[str, int]]:
    """n (clip, frame) pairs drawn uniformly from all frames of the tracked clips."""
    pairs = [(p.name.removesuffix("_track.npy"), i)
             for p in sorted(processed_dir.glob("*_track.npy")) for i in range(len(np.load(p)))]
    if not pairs:
        raise FileNotFoundError(f"No *_track.npy files in {processed_dir}")
    rng = np.random.default_rng(SEED)
    return [pairs[i] for i in sorted(rng.choice(len(pairs), size=n, replace=False))]


def read_frame(raw_dir: Path, clip: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(raw_dir / f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} of {clip}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {(r["clip"], int(r["frame"])) for r in csv.DictReader(f)}
    todo = [p for p in sample_frames(args.processed_dir, args.n) if p not in done]
    print(f"{len(done)} labelled, {len(todo)} to go")

    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["clip", "frame", "label"])
        for k, (clip, index) in enumerate(todo):
            frame = cv2.resize(read_frame(args.raw_dir, clip, index), None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(frame, f"{len(done) + k + 1}/{args.n}  c=held  o=not held  s=skip  q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow("label gripper", frame)
            key = cv2.waitKey(0)
            while key not in KEYS and key != ord("q"):
                key = cv2.waitKey(0)
            if key == ord("q"):
                break
            writer.writerow([clip, index, KEYS[key]])
            f.flush()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Convert an evaluation rollout video (from lerobot-eval) into a small GIF for the README.

Usage:
    python scripts/video_to_gif.py --video outputs/eval/smolvla_ego2panda/videos/libero_goal_6/eval_episode_0.mp4 \
        --out results/eval/success.gif
"""
import argparse
from pathlib import Path

import av
from PIL import Image

FRAME_STEP = 3      # keep every 3rd frame (one video frame per 20 Hz control step)
FRAME_MS = 150      # 3 steps at 20 Hz = 0.15 s, so the GIF plays in real time


def main() -> None:
    """Read the video, keep every FRAME_STEP-th frame, and save it as a looping GIF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with av.open(str(args.video)) as container:
        frames = [Image.fromarray(f.to_ndarray(format="rgb24"))
                  for i, f in enumerate(container.decode(video=0)) if i % FRAME_STEP == 0]
    if not frames:
        raise ValueError(f"No frames decoded from {args.video}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames")


if __name__ == "__main__":
    main()

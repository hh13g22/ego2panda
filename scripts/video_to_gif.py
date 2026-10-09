"""Convert a video into a small real-time GIF for the README.

--fps is the real-time rate of the video's frames: 20 for lerobot-eval rollouts (one frame per 20 Hz control
step, whatever the file's metadata says) and 30 for the phone clips. --height resizes, keeping the aspect ratio.

Usage:
    python scripts/video_to_gif.py --video outputs/eval/smolvla_ego2panda/videos/libero_goal_6/eval_episode_6.mp4 \
        --out results/eval/failure.gif --fps 20
    python scripts/video_to_gif.py --video data/processed/a4_annotated.mp4 --out results/tracking/a4_tracking.gif \
        --fps 30 --height 360
"""
import argparse
from pathlib import Path

import av
from PIL import Image

FRAME_STEP = 3  # keep every 3rd frame


def main() -> None:
    """Read the video, keep every FRAME_STEP-th frame, optionally resize, and save a looping real-time GIF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    with av.open(str(args.video)) as container:
        frames = [Image.fromarray(f.to_ndarray(format="rgb24"))
                  for i, f in enumerate(container.decode(video=0)) if i % FRAME_STEP == 0]
    if not frames:
        raise ValueError(f"No frames decoded from {args.video}")
    if args.height:
        width = round(frames[0].width * args.height / frames[0].height)
        frames = [f.resize((width, args.height), Image.LANCZOS) for f in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 * FRAME_STEP / args.fps),
                   loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {frames[0].width}x{frames[0].height}")


if __name__ == "__main__":
    main()

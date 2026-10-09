"""Save GIFs of one recorded clip replayed by the Panda in several LIBERO layouts (CPU only, no policy).

Each layout puts the cream cheese and bowl somewhere else; retargeting fits the same hand motion onto each, which
is the data augmentation behind the training set. Frames are the front camera, the right way up, in real time.

Usage:
    python scripts/replay_gif.py --traj data/processed/a10_traj.npy --start-states 25 37 49 --out-dir results/replay
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.replay import replay
from src.sim import SUITE, TASK_ID, make_env

FRAME_STEP, FRAME_MS = 3, 150  # every 3rd 20 Hz step, played in real time


def main() -> None:
    """Replay the clip on each start state and save one GIF per replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--start-states", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    env, _, init_states = make_env(SUITE, TASK_ID)
    traj = np.load(args.traj)
    clip = args.traj.stem.removesuffix("_traj")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for start in args.start_states:
        result = replay(env, init_states[start], traj)
        # Policy images are rotated 180 degrees; flipping left-right gives the upright front view.
        images = [Image.fromarray(np.ascontiguousarray(f["observation.images.image"][:, ::-1]))
                  for f in result["frames"][::FRAME_STEP]]
        out = args.out_dir / f"replay_{clip}_layout{start}.gif"
        images[0].save(out, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=True)
        print(f"{clip} layout {start}: success={result['success']}; wrote {out}")
    env.close()


if __name__ == "__main__":
    main()

"""Stage 3: turn per-frame hand tracks into clean trajectories with grasp and release frames.

Reads <clip>_track.npy from track_clips.py. Writes <clip>_traj.npy, one row per frame:
[time s, pinch x cm, pinch y cm, smoothed aperture, closed 0/1]. Also saves the open/closed threshold,
the grasp/release frames of every clip, and a plot of all trajectories.

Usage:
    python scripts/extract_trajectories.py --processed-dir data/processed --out-dir results/trajectories
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trajectory import fill_gaps, grasp_interval, otsu_threshold, smooth

TRAJ_COLUMNS = ("t_s", "pinch_x_cm", "pinch_y_cm", "aperture", "closed")
LAYOUTS = ("a", "b", "c")
GRASP_COLOR, RELEASE_COLOR = "#2a6fdb", "#e8743b"


def plot_trajectories(trajs: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot every clip's pinch path on the sheet, one panel per layout, marking grasp and release."""
    fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, layout in zip(axes, LAYOUTS):
        for name, traj in trajs.items():
            if name[0] != layout:
                continue
            g, r = grasp_interval(traj[:, 4] > 0.5)
            ax.plot(traj[:, 1], traj[:, 2], color="0.75", linewidth=1)
            ax.plot(traj[g:r + 1, 1], traj[g:r + 1, 2], color="0.35", linewidth=1.2)
            ax.scatter(*traj[g, 1:3], s=36, color=GRASP_COLOR, edgecolor="white", linewidth=1.5, zorder=3)
            ax.scatter(*traj[r, 1:3], s=36, color=RELEASE_COLOR, edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(plt.Rectangle((0, 0), 21.0, 29.7, fill=False, edgecolor="0.6", linestyle="--"))
        ax.set_title(f"layout {layout}", loc="left", fontsize=10)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("sheet x (cm)")
    axes[0].set_ylabel("sheet y (cm)")
    axes[0].invert_yaxis()  # y runs down the page, as on the printed sheet
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=GRASP_COLOR, label="grasp"),
               plt.Line2D([], [], marker="o", linestyle="", color=RELEASE_COLOR, label="release"),
               plt.Line2D([], [], color="0.35", label="carrying"),
               plt.Line2D([], [], color="0.6", linestyle="--", label="A4 sheet")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Pinch-point trajectories in the sheet frame", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)


def main() -> None:
    """Smooth every track, choose the threshold, find grasp/release, and save trajectories and plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracks = {p.name.removesuffix("_track.npy"): np.load(p) for p in sorted(args.processed_dir.glob("*_track.npy"))}
    if not tracks:
        raise FileNotFoundError(f"No *_track.npy files in {args.processed_dir}")
    apertures = {name: smooth(fill_gaps(track[:, 3])) for name, track in tracks.items()}
    threshold = otsu_threshold(np.concatenate(list(apertures.values())))

    trajs, events = {}, {}
    for name, track in tracks.items():
        closed = apertures[name] < threshold
        traj = np.column_stack([track[:, 0], smooth(fill_gaps(track[:, 1:3])), apertures[name], closed])
        np.save(args.processed_dir / f"{name}_traj.npy", traj)
        g, r = grasp_interval(closed)
        trajs[name] = traj
        events[name] = {"grasp_frame": g, "release_frame": r,
                        "grasp_cm": traj[g, 1:3].round(1).tolist(), "release_cm": traj[r, 1:3].round(1).tolist(),
                        "frames_missing_position": int(np.isnan(track[:, 1]).sum())}

    (args.out_dir / "gripper_threshold.json").write_text(json.dumps(
        {"method": "Otsu on smoothed aperture, all clips", "threshold": round(threshold, 3)}, indent=2))
    (args.out_dir / "events.json").write_text(json.dumps(events, indent=2))
    plot_trajectories(trajs, args.out_dir / "trajectories.png")
    print(f"threshold {threshold:.3f}")
    for name, e in events.items():
        print(name, e)


if __name__ == "__main__":
    main()

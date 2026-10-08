"""Run ArUco detection + MediaPipe hand tracking on phone clips.

Writes per clip an annotated video (to --video-dir), plus one aperture-over-time plot and a detection-count
JSON for all clips (to --out-dir).
Usage: python scripts/track_clips.py --clips data/raw/1.mp4 data/raw/2.mp4 data/raw/3.mp4 \
           --video-dir data/processed --out-dir results/tracking
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aruco_sheet import SHEET_H_CM, SHEET_W_CM
from src.hands import HandTracker, grip_aperture, pinch_point
from src.markers import detect_markers, make_detector, sheet_homography, to_sheet

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
HAND_EDGES = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
SHEET_OUTLINE_CM = np.array([[0, 0], [SHEET_W_CM, 0], [SHEET_W_CM, SHEET_H_CM], [0, SHEET_H_CM]], dtype=np.float64)
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)  # BGR


def annotate(frame, markers, H, landmarks, index) -> None:
    """Draw markers, projected sheet outline, hand skeleton and pinch point onto frame (in place)."""
    for marker_id, corners in markers.items():
        cv2.polylines(frame, [corners.astype(np.int32)], True, GREEN, 1)
        cv2.putText(frame, str(marker_id), corners[0].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
    if H is not None:
        outline_px = to_sheet(np.linalg.inv(H), SHEET_OUTLINE_CM)  # inverse homography: sheet cm -> pixels
        cv2.polylines(frame, [outline_px.astype(np.int32)], True, YELLOW, 1)
    lines = [f"frame {index}", f"markers {len(markers)}/4"]
    if landmarks is not None:
        for a, b in HAND_EDGES:
            cv2.line(frame, landmarks[a].astype(int), landmarks[b].astype(int), WHITE, 1)
        pinch = pinch_point(landmarks)
        cv2.circle(frame, pinch.astype(int), 4, RED, -1)
        lines.append(f"aperture {grip_aperture(landmarks):.2f}")
        if H is not None:
            x, y = to_sheet(H, pinch)[0]
            lines.append(f"pinch ({x:.1f}, {y:.1f}) cm")
    else:
        lines.append("no hand")
    for row, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 18 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)


def process_clip(clip: Path, out_video: Path) -> dict:
    """Track one clip; returns per-frame marker counts and apertures (NaN where no hand)."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector, tracker = make_detector(), HandTracker(MODEL_PATH)
    n_markers, apertures = [], []

    with av.open(str(out_video), "w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.pix_fmt = "yuv420p"
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stream.width, stream.height = frame.shape[1], frame.shape[0]
            markers = detect_markers(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            H = sheet_homography(markers)
            landmarks = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(index * 1000 / fps))
            n_markers.append(len(markers))
            apertures.append(np.nan if landmarks is None else grip_aperture(landmarks))
            annotate(frame, markers, H, landmarks, index)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
            index += 1
        for packet in stream.encode():
            container.mux(packet)
    cap.release()
    tracker.close()
    return {"fps": fps, "n_markers": np.array(n_markers), "aperture": np.array(apertures)}


def plot_apertures(tracks: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(len(tracks), 1, figsize=(9, 2.4 * len(tracks)), sharex=True, squeeze=False)
    for ax, (name, track) in zip(axes[:, 0], tracks.items()):
        t = np.arange(len(track["aperture"])) / track["fps"]
        missing = track["n_markers"] < 4
        ax.fill_between(t, 0, 1, where=missing, transform=ax.get_xaxis_transform(),
                        color="0.88", step="mid", linewidth=0)
        ax.plot(t, track["aperture"], color="#2a6fdb", linewidth=1.5)
        ax.set_title(f"clip {name}", loc="left", fontsize=10)
        ax.set_ylabel("aperture")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Thumb–index distance / palm length  (grey = fewer than 4 markers; gaps = no hand)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracks, counts = {}, {}
    for clip in args.clips:
        track = process_clip(clip, args.video_dir / f"{clip.stem}_annotated.mp4")
        tracks[clip.stem] = track
        all_markers = track["n_markers"] == 4
        hand = ~np.isnan(track["aperture"])
        counts[clip.stem] = {
            "frames": int(len(hand)),
            "all_4_markers": int(all_markers.sum()),
            "hand": int(hand.sum()),
            "all_4_markers_and_hand": int((all_markers & hand).sum()),
            "marker_count_histogram": {str(k): int((track["n_markers"] == k).sum()) for k in range(5)},
        }
        print(clip.stem, counts[clip.stem])

    plot_apertures(tracks, args.out_dir / "aperture.png")
    (args.out_dir / "detection_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

"""MediaPipe Hand Landmarker wrapper and hand features."""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


class HandTracker:
    """Single-hand landmark tracker for a video stream (MediaPipe VIDEO mode)."""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model missing: {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def track(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """21x2 landmark pixel coordinates, or None if no hand is detected."""
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]])

    def close(self) -> None:
        self._landmarker.close()


def pinch_point(landmarks: np.ndarray) -> np.ndarray:
    """Midpoint of thumb tip and index tip (the virtual gripper centre), in pixels."""
    return (landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) / 2


def grip_aperture(landmarks: np.ndarray) -> float:
    """Thumb-index tip distance divided by palm length (wrist -> middle MCP).

    Normalising by palm length makes the value independent of how far the phone is from the hand.
    """
    pinch = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
    palm = np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])
    return float(pinch / palm)

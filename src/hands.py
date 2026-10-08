"""Find the hand in a video frame with MediaPipe and compute the two hand features the pipeline uses:
the pinch point (virtual gripper centre) and the grip aperture (how open the hand is)."""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


class HandTracker:
    """Tracks one hand through a video with MediaPipe."""

    def __init__(self, model_path: Path):
        """Load the MediaPipe hand model in video mode."""
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model missing: {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def track(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Return the 21 hand landmarks of this frame as pixel coordinates, or None if no hand is found."""
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]])

    def close(self) -> None:
        """Release the MediaPipe model."""
        self._landmarker.close()


def pinch_point(landmarks: np.ndarray) -> np.ndarray:
    """Return the point midway between thumb tip and index tip, used as the gripper centre (pixels)."""
    return (landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) / 2


def grip_aperture(landmarks: np.ndarray) -> float:
    """Return how open the hand is: thumb-to-index distance divided by palm length.

    Dividing by palm length makes the value the same whether the phone is near or far.
    """
    pinch = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
    palm = np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])
    return float(pinch / palm)

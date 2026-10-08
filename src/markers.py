"""ArUco detection and per-frame image -> sheet-plane homography."""
import cv2
import numpy as np

from src.aruco_sheet import DICTIONARY, MARKER_TOP_LEFT_CM, marker_corners_cm


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corners for a tighter homography
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)


def detect_markers(detector: cv2.aruco.ArucoDetector, gray: np.ndarray) -> dict[int, np.ndarray]:
    """Sheet markers visible in the frame: {marker_id: 4x2 pixel corners (TL, TR, BR, BL)}."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c[0] for c, i in zip(corners, ids.ravel()) if int(i) in MARKER_TOP_LEFT_CM}


def sheet_homography(markers: dict[int, np.ndarray]) -> np.ndarray | None:
    """3x3 homography mapping image pixels -> sheet cm, or None unless all 4 markers are visible."""
    if len(markers) < len(MARKER_TOP_LEFT_CM):
        return None
    ids = sorted(markers)
    img_pts = np.concatenate([markers[i] for i in ids])
    sheet_pts = np.concatenate([marker_corners_cm(i) for i in ids])
    H, _ = cv2.findHomography(img_pts, sheet_pts)
    return H


def to_sheet(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Map Nx2 pixel points to Nx2 sheet cm."""
    return cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)

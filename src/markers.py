"""Detect the sheet's ArUco markers in a frame and compute the homography that maps image pixels to
sheet centimetres. The phone moves, so this is recomputed for every frame."""
import cv2
import numpy as np

from src.aruco_sheet import DICTIONARY, MARKER_TOP_LEFT_CM, marker_corners_cm

# With 3 markers the homography error is 0.085 cm median vs 0.37 cm with 1 marker
# (results/homography_accuracy.json), so a frame needs at least 3 markers.
MIN_MARKERS = 3


def make_detector() -> cv2.aruco.ArucoDetector:
    """Create an ArUco detector for the sheet's marker dictionary."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corners for a tighter homography
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)


def detect_markers(detector: cv2.aruco.ArucoDetector, gray: np.ndarray) -> dict[int, np.ndarray]:
    """Return the sheet markers found in a grayscale frame as {marker id: 4 corner pixels}."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c[0] for c, i in zip(corners, ids.ravel()) if int(i) in MARKER_TOP_LEFT_CM}


def sheet_homography(markers: dict[int, np.ndarray]) -> np.ndarray | None:
    """Return the 3x3 matrix that maps image pixels to sheet cm, or None if fewer than 3 markers are seen."""
    if len(markers) < MIN_MARKERS:
        return None
    ids = sorted(markers)
    img_pts = np.concatenate([markers[i] for i in ids])
    sheet_pts = np.concatenate([marker_corners_cm(i) for i in ids])
    H, _ = cv2.findHomography(img_pts, sheet_pts)
    return H


def to_sheet(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Map pixel points to sheet cm with homography H."""
    return cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)

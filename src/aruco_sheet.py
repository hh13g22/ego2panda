"""Physical layout of the printed A4 ArUco sheet (single source of truth).

Sheet frame: origin at the sheet's top-left corner, x to the right, y down the page, units cm.
This is the table-plane frame that every video frame's homography maps into.
"""
import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
SHEET_W_CM, SHEET_H_CM = 21.0, 29.7
MARKER_CM = 5.0
MARGIN_CM = 1.5  # sheet edge to marker edge; keeps markers clear of the printer's unprintable border

# Top-left corner (x, y) of each marker, keyed by marker ID.
MARKER_TOP_LEFT_CM = {
    0: (MARGIN_CM, MARGIN_CM),                                                    # top-left
    1: (SHEET_W_CM - MARGIN_CM - MARKER_CM, MARGIN_CM),                           # top-right
    2: (SHEET_W_CM - MARGIN_CM - MARKER_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),  # bottom-right
    3: (MARGIN_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),                           # bottom-left
}


def marker_corners_cm(marker_id: int) -> np.ndarray:
    """4x2 corners of one marker in sheet cm, in OpenCV's detection order: TL, TR, BR, BL."""
    x, y = MARKER_TOP_LEFT_CM[marker_id]
    s = MARKER_CM
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)

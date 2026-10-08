"""Physical layout of the printed A4 ArUco sheet (single source of truth).

Sheet frame: origin at the sheet's top-left corner, x to the right, y down the page, units cm.
This is the table-plane frame that every video frame's homography maps into.

The PDF is drawn at nominal size; PRINT_SCALE maps it to the paper actually printed. The printer
only offers fit-to-page, which shrinks uniformly, so the printed design is the nominal one times
PRINT_SCALE (the frame origin then sits a few mm off the paper corner, which is harmless because
only relative positions are used).
"""
import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
SHEET_W_CM, SHEET_H_CM = 21.0, 29.7
MARKER_CM = 5.0
MARGIN_CM = 1.5  # sheet edge to marker edge; keeps markers clear of the printer's unprintable border
PRINT_SCALE = 0.98  # measured on Hong's print: 10 cm scale bar -> 9.8 cm, 5 cm marker -> 4.9 cm

# Top-left corner (x, y) of each marker, keyed by marker ID.
MARKER_TOP_LEFT_CM = {
    0: (MARGIN_CM, MARGIN_CM),                                                    # top-left
    1: (SHEET_W_CM - MARGIN_CM - MARKER_CM, MARGIN_CM),                           # top-right
    2: (SHEET_W_CM - MARGIN_CM - MARKER_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),  # bottom-right
    3: (MARGIN_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),                           # bottom-left
}


def marker_corners_cm(marker_id: int) -> np.ndarray:
    """4x2 corners of one printed marker in cm, in OpenCV's detection order: TL, TR, BR, BL."""
    x, y = MARKER_TOP_LEFT_CM[marker_id]
    s = MARKER_CM
    nominal = np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)
    return nominal * PRINT_SCALE

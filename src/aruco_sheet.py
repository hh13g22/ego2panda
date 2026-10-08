"""Where the four ArUco markers are on the printed A4 sheet.

Positions are in the sheet frame: origin at the sheet's top-left corner, x to the right, y down the page,
in cm. Every video frame is mapped into this frame. The printer could only "fit to page", which shrank the
print uniformly to 98%, so all printed positions are the designed ones times PRINT_SCALE.
"""
import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
SHEET_W_CM, SHEET_H_CM = 21.0, 29.7
MARKER_CM = 5.0
MARGIN_CM = 1.5  # sheet edge to marker edge; keeps markers clear of the printer's unprintable border
PRINT_SCALE = 0.98  # measured on the print: 10 cm scale bar -> 9.8 cm, 5 cm marker -> 4.9 cm

# Top-left corner (x, y) of each marker, keyed by marker ID.
MARKER_TOP_LEFT_CM = {
    0: (MARGIN_CM, MARGIN_CM),                                                    # top-left
    1: (SHEET_W_CM - MARGIN_CM - MARKER_CM, MARGIN_CM),                           # top-right
    2: (SHEET_W_CM - MARGIN_CM - MARKER_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),  # bottom-right
    3: (MARGIN_CM, SHEET_H_CM - MARGIN_CM - MARKER_CM),                           # bottom-left
}


def marker_corners_cm(marker_id: int) -> np.ndarray:
    """Return the 4 corners of one printed marker in sheet cm, in the order OpenCV detects them
    (top-left, top-right, bottom-right, bottom-left)."""
    x, y = MARKER_TOP_LEFT_CM[marker_id]
    s = MARKER_CM
    nominal = np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)
    return nominal * PRINT_SCALE

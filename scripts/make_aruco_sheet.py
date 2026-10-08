"""Render the A4 ArUco sheet to a PDF at true physical scale. Print at 100% ("actual size").

Usage: python scripts/make_aruco_sheet.py --out results/aruco_sheet_a4.pdf
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aruco_sheet import DICTIONARY, MARKER_CM, MARKER_TOP_LEFT_CM, SHEET_H_CM, SHEET_W_CM

CM_PER_INCH = 2.54
SCALE_BAR_CM = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig = plt.figure(figsize=(SHEET_W_CM / CM_PER_INCH, SHEET_H_CM / CM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])  # axes fill the page, so data units are cm on paper
    ax.set_xlim(0, SHEET_W_CM)
    ax.set_ylim(SHEET_H_CM, 0)  # y down, matching the sheet frame
    ax.axis("off")

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    for marker_id, (x, y) in MARKER_TOP_LEFT_CM.items():
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(x, x + MARKER_CM, y + MARKER_CM, y))
        ax.text(x + MARKER_CM / 2, y + MARKER_CM + 0.4, f"ID {marker_id}", ha="center", va="top", fontsize=8)

    # Scale bar to check the print came out at 100%.
    x0, y0 = (SHEET_W_CM - SCALE_BAR_CM) / 2, SHEET_H_CM / 2  # sheet centre, clear of all markers
    ax.plot([x0, x0 + SCALE_BAR_CM], [y0, y0], color="black", linewidth=1)
    for xt in (x0, x0 + SCALE_BAR_CM):
        ax.plot([xt, xt], [y0 - 0.2, y0 + 0.2], color="black", linewidth=1)
    ax.text(SHEET_W_CM / 2, y0 + 0.4, "must measure 10.0 cm — print at 100% / actual size",
            ha="center", va="top", fontsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Draw which teacher layers seed the 8-layer student: plain even spacing vs the type-preserving map used here.

SmolVLA's action expert alternates self-attention (even layers) and cross-attention (odd layers), and the two have
different weight shapes, so a student slot can only take a teacher layer of the same type.

Usage:
    python scripts/plot_layer_map.py --out results/distill/layer_map.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill import parity_preserving_layer_map

N_TEACHER, N_STUDENT = 16, 8
SELF_COLOR, CROSS_COLOR, BAD_COLOR = "#cfe0f7", "#fbd9c3", "#d62728"
BOX_W, BOX_H = 0.8, 0.8


def layer_type(i: int) -> str:
    """Self-attention ("S") on even layers, cross-attention ("C") on odd layers."""
    return "S" if i % 2 == 0 else "C"


def draw_row(ax, n: int, y: float, spacing: float) -> list[float]:
    """Draw n labelled layer boxes centred on the figure at height y; return their x centres."""
    xs = [(i - (n - 1) / 2) * spacing for i in range(n)]
    for i, x in enumerate(xs):
        color = SELF_COLOR if layer_type(i) == "S" else CROSS_COLOR
        ax.add_patch(Rectangle((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, facecolor=color, edgecolor="0.3"))
        ax.text(x, y + 0.08, str(i), ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.22, layer_type(i), ha="center", va="center", fontsize=8, color="0.35")
    return xs


def draw_panel(ax, layer_map: list[int], title: str) -> None:
    """Draw teacher and student rows and connect each student slot to the teacher layer it is copied from."""
    teacher_x = draw_row(ax, N_TEACHER, 2.0, 1.0)
    student_x = draw_row(ax, N_STUDENT, 0.0, 2.0)
    for s, t in enumerate(layer_map):
        ok = layer_type(s) == layer_type(t)
        ax.plot([teacher_x[t], student_x[s]], [2.0 - BOX_H / 2, BOX_H / 2], color="0.25" if ok else BAD_COLOR,
                linestyle="-" if ok else "--", linewidth=1.4)
        if not ok:
            ax.text(student_x[s] + 0.5, 0.55, "✗", color=BAD_COLOR, fontsize=12, fontweight="bold", ha="center")
    ax.text(teacher_x[0] - 0.9, 2.0, "Teacher", ha="right", va="center", fontsize=10)
    ax.text(teacher_x[0] - 0.9, 0.0, "Student", ha="right", va="center", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(teacher_x[0] - 2.6, teacher_x[-1] + 0.8)
    ax.set_ylim(-0.7, 2.7)
    ax.axis("off")


def main() -> None:
    """Plot both layer maps and save the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    even = [round(s * (N_TEACHER - 1) / (N_STUDENT - 1)) for s in range(N_STUDENT)]
    ours = parity_preserving_layer_map(N_TEACHER, N_STUDENT)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    draw_panel(axes[0], even, f"(a) Plain even spacing {tuple(even)}: 4 layers land in the wrong slot type")
    draw_panel(axes[1], ours, f"(b) Type-preserving map used here {tuple(ours)}: every slot gets its own type")
    fig.legend(handles=[Patch(facecolor=SELF_COLOR, edgecolor="0.3", label="S: self-attention (actions look at each other)"),
                        Patch(facecolor=CROSS_COLOR, edgecolor="0.3", label="C: cross-attention (actions look at the VLM)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Choosing the student's 8 action-expert layers from the teacher's 16", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}; even={even}; ours={ours}")


if __name__ == "__main__":
    main()

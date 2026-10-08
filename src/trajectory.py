"""Turn per-frame hand tracks into smoothed trajectories with a grasp/release interval."""
import numpy as np
from scipy.signal import savgol_filter

SAVGOL_WINDOW = 9  # frames (0.3 s at 30 fps)
SAVGOL_ORDER = 2


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows of an (N,) or (N, D) array; edges hold the nearest valid value."""
    x = x.reshape(len(x), -1).copy()
    idx = np.arange(len(x))
    for d in range(x.shape[1]):
        valid = ~np.isnan(x[:, d])
        if not valid.any():
            raise ValueError("Column has no valid samples to interpolate from")
        x[:, d] = np.interp(idx, idx[valid], x[valid, d])
    return x.squeeze()


def smooth(x: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing along time."""
    return savgol_filter(x, SAVGOL_WINDOW, SAVGOL_ORDER, axis=0)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold maximising between-class variance of a 1-D sample (Otsu's method)."""
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def grasp_interval(closed: np.ndarray) -> tuple[int, int]:
    """(grasp frame, release frame): first and last frame of the longest run of closed frames."""
    if not closed.any():
        raise ValueError("Gripper never closes in this clip")
    padded = np.concatenate([[0], closed.astype(int), [0]])
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    longest = np.argmax(ends - starts)
    return int(starts[longest]), int(ends[longest] - 1)

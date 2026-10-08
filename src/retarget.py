"""Convert a human pinch trajectory into gripper targets for the simulated Panda.

x, y: a rotation + uniform scale + shift that puts the human grasp point on the sim object and the human
release point on the sim bowl. All other frames use the same mapping, so the shape of the human motion is kept.
z: a phone video gives no reliable height, so height follows the task phase (hover, descend, lift, lower).
Gripper: closed between the human grasp and release.
"""
import numpy as np

from src.control import GRIPPER_CLOSED, GRIPPER_OPEN

SIM_HZ = 20
HOVER_M = 0.10      # approach height above the object
CARRY_M = 0.15      # carry height above the target
RELEASE_M = 0.08    # release height above the target
PHASE_S = 0.5       # duration of each descend / lift / lower ramp
ARRIVE_STEPS = 10   # hold at the grasp / release point so the arm catches up before the gripper switches
GRIP_STEPS = 15     # hold while the gripper closes / opens
POST_RELEASE_S = 0.5


def sheet_to_complex(xy_cm: np.ndarray) -> np.ndarray:
    """Write sheet points as complex numbers x + iy, flipping y so the frame matches the robot's table."""
    return xy_cm[:, 0] - 1j * xy_cm[:, 1]


def similarity_xy(human_xy_cm: np.ndarray, grasp: int, release: int,
                  obj_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Map the human x-y path (cm) into the sim (m) so the grasp lands on the object and the release on
    the target. Returns the mapped path and the scale factor."""
    h = sheet_to_complex(human_xy_cm)
    o, t = complex(*obj_xy), complex(*target_xy)
    if abs(h[release] - h[grasp]) < 1.0:
        raise ValueError("Grasp and release are less than 1 cm apart; the similarity map is degenerate")
    a = (t - o) / (h[release] - h[grasp])
    sim = a * (h - h[grasp]) + o
    return np.column_stack([sim.real, sim.imag]), abs(a)


def ramp(n: int, start: float, end: float) -> np.ndarray:
    """Return n evenly spaced values from start to end."""
    return np.linspace(start, end, n) if n > 0 else np.empty(0)


def phase_z(n: int, grasp: int, release: int, ramp_frames: int,
            obj_z: float, target_z: float) -> np.ndarray:
    """Return a height for every frame: hover, descend to the object, lift to carry height, lower over the target."""
    hover, carry, drop = obj_z + HOVER_M, target_z + CARRY_M, target_z + RELEASE_M
    z = np.empty(n)
    d0 = max(grasp - ramp_frames, 0)
    z[:d0] = hover
    z[d0:grasp] = ramp(grasp - d0, hover, obj_z)
    carry_len = release + 1 - grasp
    up = min(ramp_frames, carry_len // 2)
    down = min(ramp_frames, carry_len - up)
    z[grasp:release + 1] = carry
    z[grasp:grasp + up] = ramp(up, obj_z, carry)
    z[release + 1 - down:release + 1] = ramp(down, carry, drop)
    z[release + 1:] = drop
    return z


def retarget(traj: np.ndarray, grasp: int, release: int, obj_pos: np.ndarray, target_pos: np.ndarray,
             ) -> tuple[np.ndarray, np.ndarray, float]:
    """Turn a human trajectory into one gripper target and gripper command per sim control step.

    Also adds short pauses at grasp and release so the arm arrives before the gripper closes or opens.
    """
    t = traj[:, 0]
    fps = 1.0 / np.median(np.diff(t))
    end = min(len(traj) - 1, release + int(POST_RELEASE_S * fps))
    xy, scale = similarity_xy(traj[:end + 1, 1:3], grasp, release, obj_pos[:2], target_pos[:2])
    z = phase_z(end + 1, grasp, release, int(PHASE_S * fps), obj_pos[2], target_pos[2])
    frames = np.column_stack([xy, z])

    def resample(a: int, b: int) -> np.ndarray:
        """Resample frames a..b from the video frame rate to the sim control rate."""
        ts = np.arange(t[a], t[b], 1.0 / SIM_HZ)
        return np.column_stack([np.interp(ts, t[a:b + 1], frames[a:b + 1, d]) for d in range(3)])

    approach, carry, retreat = resample(0, grasp), resample(grasp, release), resample(release, end)
    hold = lambda p, n: np.repeat(p[None], n, axis=0)  # noqa: E731
    g_pt, r_pt = frames[grasp], frames[release]
    targets = np.concatenate([approach, hold(g_pt, ARRIVE_STEPS + GRIP_STEPS), carry,
                              hold(r_pt, ARRIVE_STEPS + GRIP_STEPS), retreat])
    gripper = np.concatenate([
        np.full(len(approach) + ARRIVE_STEPS, GRIPPER_OPEN),
        np.full(GRIP_STEPS + len(carry) + ARRIVE_STEPS, GRIPPER_CLOSED),
        np.full(GRIP_STEPS + len(retreat), GRIPPER_OPEN),
    ])
    return targets, gripper, scale

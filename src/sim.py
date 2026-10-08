"""LIBERO environment construction (off-screen, EGL)."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_CONFIG_DIR = REPO_ROOT / ".libero"

os.environ.setdefault("MUJOCO_GL", "egl")
# Project-local LIBERO config so ~/.libero (used by other projects) is never read or modified.
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)


def _write_libero_config() -> None:
    """Point LIBERO at the bddl/init files shipped inside the installed package."""
    import importlib.util

    pkg_root = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
    config = {
        "benchmark_root": str(pkg_root),
        "bddl_files": str(pkg_root / "bddl_files"),
        "init_states": str(pkg_root / "init_files"),
        "datasets": str(REPO_ROOT / "data" / "libero_datasets"),
        "assets": str(pkg_root / "assets"),  # unused: assets are fetched from the HF Hub on first use
    }
    LIBERO_CONFIG_DIR.mkdir(exist_ok=True)
    (LIBERO_CONFIG_DIR / "config.yaml").write_text(yaml.safe_dump(config))


_write_libero_config()

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def make_env(suite: str, task_id: int, image_size: int = 256):
    """Build an OSC_POSE LIBERO env for one task. Returns (env, task, init_states)."""
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"BDDL file missing: {bddl_file}")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=image_size,
        camera_widths=image_size,
        controller="OSC_POSE",
    )
    return env, task, task_suite.get_task_init_states(task_id)

"""Measure how far the cream cheese and bowl move across LIBERO's 50 start states of the task.

Usage:
    python scripts/layout_spread.py --out results/layout_spread.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import OBJECT, SUITE, TARGET, TASK_ID, make_env, settle


def main() -> None:
    """Record the settled x-y position of both objects in every start state and save their range."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env, _, init_states = make_env(SUITE, TASK_ID)
    env.reset()
    positions = {OBJECT: [], TARGET: []}
    for state in init_states:
        obs = settle(env, env.set_init_state(state))
        for name in positions:
            positions[name].append(obs[f"{name}_pos"][:2] * 100)
    env.close()
    result = {"start_states": len(init_states)}
    for name, xy in positions.items():
        result[name] = {"x_range_cm": round(float(np.ptp(np.array(xy)[:, 0])), 1),
                        "y_range_cm": round(float(np.ptp(np.array(xy)[:, 1])), 1)}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

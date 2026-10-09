"""Run the standard lerobot-eval rollouts with the policy's vision encoder quantised (see src/quantize.py).

lerobot-eval builds the policy itself, so this wraps its make_policy: the policy is built exactly as usual, then
its vision encoder is quantised. Everything else (environment, start states, seed, success check) is lerobot-eval.

Usage (the first argument is the mode; the rest are lerobot-eval arguments, as in scripts/eval_policy.sh):
    python scripts/eval_quantized.py int8 --policy.path=outputs/distill/smolvla_shallow8/step_010000 \
        --policy.device=cuda --policy.n_action_steps=10 --env.type=libero --env.task=libero_goal --env.task_ids="[6]" \
        --env.observation_height=256 --env.observation_width=256 --eval.n_episodes=25 --eval.batch_size=1 --seed=0 \
        --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
        --output_dir=outputs/eval/quant_int8
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantize import quantize_vision_encoder

from lerobot.scripts import lerobot_eval


def main() -> None:
    """Quantise the policy right after lerobot-eval builds it, then run lerobot-eval unchanged."""
    mode = sys.argv.pop(1)
    build_policy = lerobot_eval.make_policy

    def build_and_quantize(*args, **kwargs):
        policy = build_policy(*args, **kwargs)
        info = quantize_vision_encoder(policy, mode)
        print("quantized vision encoder:", json.dumps(info), flush=True)
        return policy

    lerobot_eval.make_policy = build_and_quantize
    lerobot_eval.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Stage 6: evaluate a policy in LIBERO on libero_goal task 6 ("put the cream cheese in the bowl").
#
# lerobot-eval starts at LIBERO start state 0, so the first 25 episodes use start states 0-24, which the
# training demos never used (they came from start states 25-49). Cameras are rendered at 256x256, like the
# training data. N_ACTION_STEPS is how many actions of each predicted chunk run before the policy looks again.
# RENAME_MAP maps LIBERO's camera names to the policy's (our fine-tuned model uses camera1/camera2; '{}' if none).
#
# Usage:
#     bash scripts/eval_policy.sh POLICY_PATH OUTPUT_DIR N_EPISODES N_ACTION_STEPS RENAME_MAP
#     bash scripts/eval_policy.sh outputs/train/smolvla_ego2panda/checkpoints/last/pretrained_model \
#         outputs/eval/smolvla_ego2panda 25 10 \
#         '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
#     bash scripts/eval_policy.sh HuggingFaceVLA/smolvla_libero outputs/eval/smolvla_libero 25 10 '{}'
set -euo pipefail  # run inside the activated conda env: conda activate ego2panda

POLICY_PATH=$1
OUTPUT_DIR=$2
N_EPISODES=$3
N_ACTION_STEPS=$4
RENAME_MAP=$5

lerobot-eval \
    --policy.path="$POLICY_PATH" \
    --policy.device=cuda \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --env.type=libero \
    --env.task=libero_goal \
    --env.task_ids="[6]" \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --rename_map="$RENAME_MAP" \
    --eval.n_episodes="$N_EPISODES" \
    --eval.batch_size=1 \
    --seed=0 \
    --output_dir="$OUTPUT_DIR"

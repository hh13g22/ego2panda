#!/usr/bin/env bash
# Stage 5: fine-tune SmolVLA (lerobot/smolvla_base) on a LeRobot dataset.
#
# Trains only on the episodes listed under "train_episodes" in EPISODES_JSON (results/split.json holds out the
# validation clips). The dataset's cameras are renamed to the camera names smolvla_base was pretrained with.
# The same command trains our model and the LIBERO robot-data baseline, so both get the same budget.
#
# Usage:
#     bash scripts/train_policy.sh OUTPUT_DIR REPO_ID DATASET_ROOT EPISODES_JSON [STEPS] [BATCH_SIZE]
#     bash scripts/train_policy.sh outputs/train/smolvla_ego2panda ego2panda/libero_goal_cream_cheese_bowl data/lerobot/ego2panda results/split.json 20000 32
#     bash scripts/train_policy.sh outputs/train/smolvla_libero_task lerobot/libero data/lerobot/libero results/libero_task_episodes.json 20000 32
set -euo pipefail  # run inside the activated conda env: conda activate ego2panda

OUTPUT_DIR=$1
REPO_ID=$2
DATASET_ROOT=$3
EPISODES_JSON=$4
STEPS=${5:-20000}
BATCH_SIZE=${6:-32}
TRAIN_EPISODES=$(python -c \
    "import json; print(str(json.load(open('$EPISODES_JSON'))['train_episodes']).replace(' ', ''))")

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --batch_size="$BATCH_SIZE" \
    --num_workers=4 \
    --steps="$STEPS" \
    --save_freq=5000 \
    --log_freq=100 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$(basename "$OUTPUT_DIR")" \
    --wandb.enable=false

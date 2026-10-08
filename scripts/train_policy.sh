#!/usr/bin/env bash
# Stage 5: fine-tune SmolVLA (lerobot/smolvla_base) on the replayed demonstrations.
#
# Trains only on the training clips listed in results/split.json (the validation clips are held out).
# The dataset's cameras are renamed to the camera names smolvla_base was pretrained with.
#
# Usage:
#     bash scripts/train_policy.sh OUTPUT_DIR [STEPS] [BATCH_SIZE]
#     bash scripts/train_policy.sh outputs/train/smolvla_ego2panda 20000 32
set -euo pipefail

OUTPUT_DIR=$1
STEPS=${2:-20000}
BATCH_SIZE=${3:-32}
TRAIN_EPISODES=$(.venv/bin/python -c \
    "import json; print(str(json.load(open('results/split.json'))['train_episodes']).replace(' ', ''))")

.venv/bin/lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=ego2panda/libero_goal_cream_cheese_bowl \
    --dataset.root=data/lerobot/ego2panda \
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

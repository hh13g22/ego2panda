"""Stage 7: distil the fine-tuned SmolVLA into a half-depth student (Shallow-pi style, offline).

The student keeps 8 of the 16 VLM and action-expert layers (see src/distill.py), starts from the teacher's
weights, and trains on the same training episodes against the ground-truth flow target and the frozen
teacher's velocity. Offline: it only sees the recorded dataset, never the simulator.

Usage:
    python scripts/distill_policy.py --teacher outputs/train/smolvla_ego2panda/checkpoints/last/pretrained_model \
        --dataset-root data/lerobot/ego2panda --split results/split.json \
        --out-dir outputs/distill/smolvla_shallow8 --layers 8 --steps 10000 --batch-size 16
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distill import distillation_loss, make_student

REPO_ID = "ego2panda/libero_goal_cream_cheese_bowl"
LR, WARMUP_STEPS, MIN_LR = 1e-4, 1000, 2.5e-6  # SmolVLA's fine-tuning schedule: warmup, then cosine decay
GRAD_CLIP = 10.0
SEED = 0


def learning_rate(step: int, total: int) -> float:
    """Linear warmup, then cosine decay to MIN_LR."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def save_student(student, teacher_path: Path, out: Path) -> None:
    """Save the student weights and config, plus the teacher's pre/post-processors (same data, same stats)."""
    student.save_pretrained(out)
    for f in teacher_path.glob("policy_*processor*"):
        shutil.copy(f, out / f.name)


# Adapted from Shallow-pi: scripts/distill_pytorch.py. Kept: frozen teacher with requires_grad off, student
# seeded from the teacher, teacher forward under no_grad, gradient clipping. Changes: one GPU and a plain loop
# instead of DDP/FSDP and the DualDistillStep wrapper; one dataset instead of routed navigation/manipulation
# batches; the vision encoder stays frozen while the shortened VLM text layers and the expert train; and
# SmolVLA's own fine-tuning optimiser and warmup/cosine schedule replace pi0's.
def main() -> None:
    """Build the student from the teacher, distil it on the training episodes, and save it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=2500)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists; choose a new output directory")

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(SEED)
    teacher = SmolVLAPolicy.from_pretrained(args.teacher).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student, layer_map = make_student(teacher, args.layers)
    student = student.to("cuda").train()
    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"layer map (student <- teacher): {layer_map}")
    print(f"teacher params {sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M, student params "
          f"{sum(p.numel() for p in student.parameters()) / 1e6:.0f}M, trainable {sum(p.numel() for p in trainable) / 1e6:.0f}M",
          flush=True)

    preprocessor, _ = make_pre_post_processors(
        teacher.config, pretrained_path=str(args.teacher), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    meta = LeRobotDatasetMetadata(REPO_ID, root=args.dataset_root)
    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root,
                             episodes=json.loads(args.split.read_text())["train_episodes"],
                             delta_timestamps=resolve_delta_timestamps(student.config, meta), video_backend="pyav")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         drop_last=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)

    step, start = 0, time.time()
    while step < args.steps:
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.steps)
            loss, parts = distillation_loss(student, teacher, preprocessor(batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} task {parts['loss_task']:.4f} teach {parts['loss_teach']:.4f} "
                      f"lr {learning_rate(step, args.steps):.2e} s/step {(time.time() - start) / step:.3f} "
                      f"gpu_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_student(student, args.teacher, args.out_dir / f"step_{step:06d}")
            if step == args.steps:
                break
    print(f"done: {step} steps in {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

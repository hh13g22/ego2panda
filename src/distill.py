"""Shallow-pi style layer distillation for SmolVLA: build a half-depth student and the distillation loss.

Adapted from Shallow-pi (knowledge distillation for flow-based VLAs, originally for pi0). The student keeps
every other transformer layer of both the VLM backbone and the action expert, is initialised from the
teacher's weights, and is trained to match both the ground-truth flow target and the teacher's velocity.

SmolVLA-specific change: its action expert alternates layer types (even layers self-attend, odd layers
cross-attend to the VLM, with differently shaped key/value projections), so the layer map keeps each layer
at a position of the same type instead of the plain evenly spaced map used for pi0.
"""
import copy

import torch

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

LAYER_PREFIXES = ("model.vlm_with_expert.vlm.model.text_model.layers.", "model.vlm_with_expert.lm_expert.layers.")


# Adapted from Shallow-pi: init_from_teachers.build_layer_subsample_map, which picks evenly spaced teacher
# layers with torch.linspace(0, L_teacher - 1, L_student).round(). Change: each pick is restricted to teacher
# layers with the same index modulo `period`. SmolVLA's expert alternates self-attention (even) and
# cross-attention (odd) layers with differently shaped k/v projections, so the plain map (16 -> 8 gives
# [0, 2, 4, 6, 9, 11, 13, 15]) would load a cross-attention layer into a self-attention slot. This map gives
# [0, 3, 4, 7, 8, 11, 12, 15]: still evenly spread, and every layer keeps its type.
def parity_preserving_layer_map(n_teacher: int, n_student: int, period: int = 2) -> list[int]:
    """Return the teacher layer for each student layer: evenly spread, and with the same index modulo `period`."""
    layer_map = []
    for s in range(n_student):
        target = s * (n_teacher - 1) / (n_student - 1)
        same_type = [t for t in range(n_teacher) if t % period == s % period]
        layer_map.append(min(same_type, key=lambda t: abs(t - target)))
    return layer_map


# Adapted from Shallow-pi: init_from_teachers.init_dual_student, which builds a shallow pi0 and renames each
# student layer key to the teacher layer it is seeded from (SEED_LANG_LAYERS / SEED_EXPERT_LAYERS prefixes).
# Changes: one teacher instead of three checkpoints (no navigation/manipulation dual expert); the key prefixes
# are SmolVLA's (VLM text layers and lm_expert layers); the same map is used for both stacks because SmolVLA's
# expert layer i attends to VLM layer i; depth is set through SmolVLA's own num_vlm_layers config instead of
# gemma_depth; and loading is strict, so a wrong key rename fails immediately instead of leaving random weights.
def make_student(teacher: SmolVLAPolicy, n_layers: int) -> tuple[SmolVLAPolicy, list[int]]:
    """Build a SmolVLA with `n_layers` VLM and expert layers, initialised from the teacher's mapped layers."""
    config = copy.deepcopy(teacher.config)
    config.num_vlm_layers = n_layers
    config.num_expert_layers = -1  # expert depth follows the VLM depth, as in the teacher
    config.train_expert_only = False  # the shortened VLM must adapt, so its text layers train too
    config.freeze_vision_encoder = True
    config.load_vlm_weights = False  # every weight is copied from the teacher below
    student = SmolVLAPolicy(config)

    layer_map = parity_preserving_layer_map(teacher.config.num_vlm_layers, n_layers,
                                            teacher.config.self_attn_every_n_layers)
    teacher_state = teacher.state_dict()
    student_state = {}
    for key in student.state_dict():
        source = key
        for prefix in LAYER_PREFIXES:
            if key.startswith(prefix):
                index, rest = key[len(prefix):].split(".", 1)
                source = f"{prefix}{layer_map[int(index)]}.{rest}"
        student_state[key] = teacher_state[source]
    student.load_state_dict(student_state, strict=True)
    return student, layer_map


# Adapted from Shallow-pi: PI0Pytorch.forward / eval_model return the predicted velocity v_t. SmolVLA's
# VLAFlowMatching.forward only returns the loss, so this repeats its steps (same embeddings, masks and expert
# pass) and stops before the loss. Checked on CPU: (target - v_t)^2 equals SmolVLA's own loss exactly.
def flow_velocity(policy: SmolVLAPolicy, batch: dict, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Return the policy's predicted flow-matching velocity for the noised action chunk (same steps as
    SmolVLA's training forward pass, which itself only returns the loss)."""
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    t = time[:, None, None]
    x_t = t * noise + (1 - t) * actions
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"],
        state=state)
    suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att, suffix_att], dim=1)
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=torch.cumsum(pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    return model.action_out_proj(suffix_out[:, -policy.config.chunk_size:].to(torch.float32))


# Adapted from Shallow-pi: DualExpertPI0Pytorch.compute_nav_loss, i.e. lambda_task * MSE(u_t, v_student) +
# lambda_teach * MSE(v_teacher, v_student) with shared noise and time, both lambdas 1.0 as in pi0_config.py.
# Changes: a single modality, so no Kendall uncertainty weighting and no L2-SP backbone anchor; and the MSE is
# taken only over the 7 real action dimensions and in-episode steps, because SmolVLA pads actions to 32
# dimensions and pads chunks past the episode end (pi0's LIBERO pipeline had no padding to mask).
def distillation_loss(student: SmolVLAPolicy, teacher: SmolVLAPolicy, batch: dict,
                      lambda_task: float = 1.0, lambda_teach: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Shallow-pi loss: lambda_task * MSE(student, ground-truth target) + lambda_teach * MSE(student, teacher),
    with the teacher and student given the same noise and flow time."""
    actions = student.prepare_action(batch)
    noise = student.model.sample_noise(actions.shape, actions.device)
    time = student.model.sample_time(actions.shape[0], actions.device)
    v_student = flow_velocity(student, batch, noise, time)
    with torch.no_grad():
        v_teacher = flow_velocity(teacher, batch, noise, time)

    action_dim = student.config.action_feature.shape[0]  # ignore the zero-padded action dimensions
    valid = (~batch["action_is_pad"]).unsqueeze(-1).float()
    target = (noise - actions)[..., :action_dim]
    v_student, v_teacher = v_student[..., :action_dim], v_teacher[..., :action_dim]
    denom = valid.sum() * action_dim
    loss_task = ((v_student - target) ** 2 * valid).sum() / denom
    loss_teach = ((v_student - v_teacher) ** 2 * valid).sum() / denom
    total = lambda_task * loss_task + lambda_teach * loss_teach
    return total, {"loss_task": loss_task.item(), "loss_teach": loss_teach.item()}

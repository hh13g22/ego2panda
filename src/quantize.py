"""Quantise the weights of SmolVLA's vision encoder, to measure how much precision it needs.

Modes:
  bf16    unchanged (SmolVLA's default for the vision encoder)
  fp16    the vision encoder runs in half precision (real fp16 compute)
  int8    every Linear/Conv weight rounded to a symmetric 8-bit grid, one scale per output channel
  int4    the same with 4 bits
  binary  every weight replaced by sign(w) * mean|w| of its output channel (1 bit, XNOR-Net style)

int8, int4 and binary are simulated: the rounded weights are stored and run in float32, so the only error is the
quantisation itself. This measures accuracy, not speed (there are no integer kernels here); the memory a real
low-bit encoder would need is computed from the bit widths.
"""
import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

BITS = {"bf16": 16, "fp16": 16, "int8": 8, "int4": 4, "binary": 1}


class CastOutput(nn.Module):
    """Wraps the vision encoder so it can run in its own dtype: SmolVLA casts the image to `.dtype` on the way in,
    and this casts the features back to the connector's dtype on the way out."""

    def __init__(self, encoder: nn.Module, out_dtype: torch.dtype):
        super().__init__()
        self.encoder = encoder
        self.out_dtype = out_dtype

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def forward(self, **kwargs):
        out = self.encoder(**kwargs)
        out.last_hidden_state = out.last_hidden_state.to(self.out_dtype)
        return out


def fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Round a weight to a symmetric integer grid with one scale per output channel, and return it dequantised."""
    flat = weight.reshape(weight.shape[0], -1)
    qmax = 2 ** (bits - 1) - 1
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    return (torch.round(flat / scale).clamp(-qmax, qmax) * scale).reshape(weight.shape)


def binarize(weight: torch.Tensor) -> torch.Tensor:
    """Replace a weight by its sign times the mean absolute value of its output channel."""
    flat = weight.reshape(weight.shape[0], -1)
    return (torch.sign(flat) * flat.abs().mean(dim=1, keepdim=True)).reshape(weight.shape)


def quantize_vision_encoder(policy: SmolVLAPolicy, mode: str) -> dict:
    """Apply `mode` to the vision encoder's Linear and Conv2d weights in place; return what was changed."""
    if mode not in BITS:
        raise ValueError(f"Unknown mode {mode}; choose from {list(BITS)}")
    vlm = policy.model.vlm_with_expert.vlm.model
    encoder = vlm.vision_model
    layers = [m for m in encoder.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    n_weights = sum(m.weight.numel() for m in layers)
    original = [m.weight.detach().float().clone() for m in layers]
    if mode != "bf16":
        connector_dtype = next(vlm.connector.parameters()).dtype
        encoder.to(torch.float16 if mode == "fp16" else torch.float32)
        with torch.no_grad():
            for m in layers:
                if mode == "binary":
                    m.weight.copy_(binarize(m.weight))
                elif mode in ("int8", "int4"):
                    m.weight.copy_(fake_quantize(m.weight, BITS[mode]))
        vlm.vision_model = CastOutput(encoder, connector_dtype)
    scales = 0 if mode in ("bf16", "fp16") else sum(m.weight.shape[0] for m in layers)  # one fp16 scale per channel
    # Checks that the quantisation really happened: relative weight error, and the largest number of distinct
    # values in any of the first 32 output channels of each layer (int8 <= 255, int4 <= 15, binary <= 2).
    err = sum(((m.weight.detach().float() - w) ** 2).sum().item() for m, w in zip(layers, original))
    ref = sum((w ** 2).sum().item() for w in original)
    levels = max(torch.unique(row).numel() for m in layers for row in m.weight.detach().reshape(m.weight.shape[0], -1)[:32])
    return {
        "mode": mode,
        "bits_per_weight": BITS[mode],
        "quantized_layers": len(layers),
        "quantized_weights_million": round(n_weights / 1e6, 2),
        "vision_weight_mb": round((n_weights * BITS[mode] + scales * 16) / 8 / 2**20, 1),
        "weight_dtype": str(layers[0].weight.dtype).replace("torch.", ""),
        "relative_weight_error": round((err / ref) ** 0.5, 5),
        "max_distinct_values_per_channel": levels,
    }

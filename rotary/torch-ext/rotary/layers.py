import torch
import torch.nn as nn

from ._ops import ops


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conj: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    half_dim = x.shape[-1] // 2
    ops.apply_rotary(
        x[..., :half_dim], x[..., half_dim:], cos, sin, out[..., :half_dim], out[..., half_dim:], conj
    )
    return out


class _ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _rotate(x, cos, sin, conj=False)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        cos, sin = ctx.saved_tensors
        return _rotate(grad, cos, sin, conj=True), None, None


class apply_rotary_transformers(nn.Module):
    can_torch_compile = True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotary kernel implementation wrapper
        Adapts rotary kernel implementation to match transformers apply_rotary_pos_emb signature
        """
        half_dim = q.shape[-1] // 2
        cos = cos.unsqueeze(unsqueeze_dim)[..., :half_dim]
        sin = sin.unsqueeze(unsqueeze_dim)[..., :half_dim]
        return _ApplyRotary.apply(q, cos, sin), _ApplyRotary.apply(k, cos, sin)

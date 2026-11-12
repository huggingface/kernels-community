import torch
import torch.nn as nn

from ._ops import ops


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
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        q_rotated = q.clone()
        k_rotated = k.clone()

        # Support both full and partial RoPE:
        # If cos.shape[-1] < q.shape[-1], only rotate the first rotary_dim dimensions.
        rotary_dim = min(cos.shape[-1], q.shape[-1])
        half_rotary_dim = rotary_dim // 2

        q1 = q_rotated[..., :half_rotary_dim]
        q2 = q_rotated[..., half_rotary_dim:rotary_dim]
        k1 = k_rotated[..., :half_rotary_dim]
        k2 = k_rotated[..., half_rotary_dim:rotary_dim]

        cos_rot = cos[..., :half_rotary_dim]
        sin_rot = sin[..., :half_rotary_dim]

        ops.apply_rotary(q1, q2, cos_rot, sin_rot, q1, q2, False)
        ops.apply_rotary(k1, k2, cos_rot, sin_rot, k1, k2, False)
        return q_rotated, k_rotated


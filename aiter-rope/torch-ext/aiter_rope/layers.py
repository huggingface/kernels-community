import torch
import torch.nn as nn

from .rope import RotateStyle, rope_cached_fwd, rope_cached_positions_fwd


def _to_sbhd(t: torch.Tensor) -> torch.Tensor:
    # transformers passes attention tensors as (batch, heads, seq, head_dim);
    # AITER's kernels operate in (seq, batch, heads, head_dim).
    return t.permute(2, 0, 1, 3).contiguous()


def _to_bhsd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 2, 0, 3)


def _apply_rotary(q, k, cos, sin):
    q_sbhd = _to_sbhd(q)
    k_sbhd = _to_sbhd(k)

    common = dict(
        rotate_style=int(RotateStyle.NEOX),
        reuse_freqs_front_part=True,
        nope_first=False,
    )
    batch, seq_len, dim = cos.shape
    if batch == 1:
        # All batch rows share the same positions: index cos / sin by sequence position.
        cos_freqs = cos[0].unsqueeze(1).unsqueeze(1).contiguous()
        sin_freqs = sin[0].unsqueeze(1).unsqueeze(1).contiguous()
        q_out_sbhd = rope_cached_fwd(q_sbhd, cos_freqs, sin_freqs, **common)
        k_out_sbhd = rope_cached_fwd(k_sbhd, cos_freqs, sin_freqs, **common)
    else:
        # Rows can have different positions (e.g. left padding). The kernel ignores the batch
        # stride of cos / sin, so flatten them into a (batch * seq, dim) table and look up row
        # `b * seq_len + s` for token `s` of batch row `b`.
        cos_freqs = cos.reshape(batch * seq_len, 1, 1, dim).contiguous()
        sin_freqs = sin.reshape(batch * seq_len, 1, 1, dim).contiguous()
        positions = (
            torch.arange(batch * seq_len, device=cos.device).view(batch, seq_len).t()
        )
        q_out_sbhd = rope_cached_positions_fwd(
            q_sbhd, cos_freqs, sin_freqs, positions, **common
        )
        k_out_sbhd = rope_cached_positions_fwd(
            k_sbhd, cos_freqs, sin_freqs, positions, **common
        )

    return _to_bhsd(q_out_sbhd), _to_bhsd(k_out_sbhd)


class apply_rotary_transformers(nn.Module):
    can_torch_compile: bool = True

    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        return _apply_rotary(q, k, cos, sin)


__all__ = ["apply_rotary_transformers"]

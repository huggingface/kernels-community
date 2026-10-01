import torch
import torch.nn as nn

from .rope import RotateStyle, rope_cached_fwd


def _to_sbhd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(2, 0, 1, 3).contiguous()


def _to_bhsd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 2, 0, 3)


class apply_rotary_transformers(nn.Module):
    can_torch_compile: bool = True

    def forward(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_sbhd = _to_sbhd(q)
        k_sbhd = _to_sbhd(k)

        cos_freqs = cos[0].unsqueeze(1).unsqueeze(1).contiguous()
        sin_freqs = sin[0].unsqueeze(1).unsqueeze(1).contiguous()

        common = dict(
            rotate_style=int(RotateStyle.NEOX),
            reuse_freqs_front_part=True,
            nope_first=False,
        )
        q_out_sbhd = rope_cached_fwd(q_sbhd, cos_freqs, sin_freqs, **common)
        k_out_sbhd = rope_cached_fwd(k_sbhd, cos_freqs, sin_freqs, **common)

        return _to_bhsd(q_out_sbhd), _to_bhsd(k_out_sbhd)


__all__ = ["apply_rotary_transformers"]

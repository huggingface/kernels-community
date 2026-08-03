"""Hub kernel layer for ESMFold2's triangle multiplicative update.

This is a drop-in replacement for the pure-PyTorch ``forward`` of
``transformers.models.esmfold2.modeling_esmfold2.EsmFold2TriangleMultiplicativeUpdate``.
The `kernels` library swaps the in-tree module's ``forward`` for this one when the
model is kernelized (``model.set_use_kernels(True)``, CUDA + inference); ``self`` is
the original module instance, so this reads its parameters directly. The forward
signature and the attribute names (``norm_start``/``norm_mix``/``proj_bundle``/
``proj_emit``/``proj_gate``, plus ``dim``/``flow``) are the contract and must stay in
sync with the in-tree module.

Returns the residual-free delta (``TriMul(pair_grid)``); the trunk's residual add
stays in ``EsmFold2PairUpdateBlock`` in transformers.

Precision: this layer runs everything in bf16, including the norm weights, which the
in-tree module deliberately keeps in fp32 (``_keep_in_fp32_modules_strict``). The
in-tree contraction is additionally promoted to fp32 by the fp32 ``visibility`` mask.
Outputs therefore differ from the pure-PyTorch path by ~6e-3 relative -- bf16
rounding, flat in sequence length rather than accumulating. On real weights that sits
inside the model's own non-determinism (ubiquitin/GB1 ΔpLDDT ≤ 0.002).
"""

import torch
import torch.nn as nn

from .trimul_with_residual import triangle_multiplicative_update_with_residual

_EPS = 1e-5


def _bf16(t):
    return None if t is None else t.to(torch.bfloat16)


class ESMFold2TriangleMultiplication(nn.Module):
    def forward(self, pair_grid: torch.Tensor, visibility: torch.Tensor | None = None) -> torch.Tensor:
        lat = self.dim
        pb = self.proj_bundle.weight
        pair_bf = pair_grid.to(torch.bfloat16)
        # NOTE: delta-only boundary -> residual=zeros. A residual-optional kernel
        # entry would avoid this [B,N,N,C] alloc+read; see README "Follow-ups".
        out = triangle_multiplicative_update_with_residual(
            pair_bf,
            self.flow,
            residual=torch.zeros_like(pair_bf),
            drop_mask=None,
            norm_in_weight=_bf16(self.norm_start.weight),
            norm_in_bias=_bf16(self.norm_start.bias),
            p_in_weight=_bf16(pb[: 2 * lat, :]),
            g_in_weight=_bf16(pb[2 * lat :, :]),
            norm_out_weight=_bf16(self.norm_mix.weight),
            norm_out_bias=_bf16(self.norm_mix.bias),
            p_out_weight=_bf16(self.proj_emit.weight),
            g_out_weight=_bf16(self.proj_gate.weight),
            mask=visibility,
            eps=_EPS,
        )
        return out.to(pair_grid.dtype)


__all__ = ["ESMFold2TriangleMultiplication"]

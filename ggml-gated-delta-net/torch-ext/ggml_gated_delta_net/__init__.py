"""Fused sequence-mixing kernels from ggml, as torch ops.

The quantization ops live in `gguf-kernels`; this package is the other half — the layers a decode step
spends its time in, where eager torch has to spell out what ggml does in one kernel.
"""

import os
from pathlib import Path

import torch


# A local (non-nix) build leaves the metallib on disk beside this module instead of embedding it, and
# `ggml_dispatch.mm` looks it up through this variable when nothing is embedded. Set before the extension
# is imported, and never over an existing value, so an explicit choice still wins.
_METALLIB = Path(__file__).parent / "ggml-metal.metallib"
if _METALLIB.is_file():
    os.environ.setdefault("GGML_ATTN_METALLIB", str(_METALLIB))

from ._ops import add_op_namespace_prefix, ops  # noqa: E402
from .layers import Qwen3_5GatedDeltaNet  # noqa: E402,F401


# The gate layouts upstream's kernel supports. `1` is a scalar gate per head (Qwen3-Next, Qwen3.5);
# a gate of `head_dim` values per head is the KDA variant, which needs a different specialisation.
GATE_WIDTHS = frozenset({1})


def gated_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One kernel for a run of gated-delta-rule steps.

    Args:
        q, k, v: `(n_seqs, n_tokens, n_heads, head_dim)`. `q` and `k` must already carry one head per
            value head — expand them yourself, so your own head order is the one that applies.
        g: `(n_seqs, n_tokens, n_heads)`, the log-domain gate; the kernel exponentiates it.
        beta: `(n_seqs, n_tokens, n_heads)`.
        state: `(n_seqs, n_heads, head_dim, head_dim)`, the initial recurrent state, indexed
            `[value_index][key_index]` -- transposed relative to the `k` outer `v` product that
            builds it, because upstream stores it that way to keep a thread's row contiguous.
            `final_state` comes back in the same layout, so a caller that stores what this
            returns never has to transpose: a fresh state is zeros, which is symmetric.

    Returns:
        `(out, final_state)`. `out` is `(n_seqs, n_tokens, n_heads, head_dim)` and carries the
        kernel's own `1/sqrt(head_dim)` scaling. Both are views into one allocation.
    """
    n_seqs, n_tokens, n_heads, head_dim = v.shape
    # One flat allocation comes back: the outputs, then the final state. The op cannot return the two
    # views itself -- two returns of a custom op may not alias each other -- so the split is here.
    dst = ops.gated_delta_net(q, k, v, g, beta, state)
    n_out = n_seqs * n_tokens * n_heads * head_dim
    out = dst[:n_out].view(n_seqs, n_tokens, n_heads, head_dim)
    final_state = dst[n_out:].view(n_seqs, n_heads, head_dim, head_dim)
    return out, final_state
def l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x / max(|x|, eps)` along the last dimension, in one dispatch.

    Five torch ops become one, and one of those five was `aten::sum`, which on MPS *blocks the host*
    (its wall time equals its CPU time). Note the epsilon placement is ggml's, which is also
    `F.normalize`'s: a caller writing `x * rsqrt(sum + eps)` differs by ~3e-8, including for all-zero
    rows.
    """
    return ops.l2_norm(x, eps)


def supports_gated_delta_net(head_dim: int) -> bool:
    """Whether this build has a gated-delta-rule kernel for `head_dim`.

    Ask instead of assuming: upstream covers a state row with 32 threads times `head_dim/32` values,
    so the head dim has to be a multiple of 32, and only a few widths are instantiated.
    """
    return ops.supports_gated_delta_net(head_dim)


@torch.library.register_fake(add_op_namespace_prefix("l2_norm"))
def _(x, eps):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_net"))
def _(q, k, v, g, beta, state):
    n_seqs, n_tokens, n_heads, head_dim = v.shape
    n_out = n_seqs * n_tokens * n_heads * head_dim
    n_state = n_seqs * n_heads * head_dim * head_dim
    return v.new_empty((n_out + n_state,))


__all__ = [
    "GATE_WIDTHS",
    "gated_delta_net",
    "l2_norm",
    "supports_gated_delta_net",
]

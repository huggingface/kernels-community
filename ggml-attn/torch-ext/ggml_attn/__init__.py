"""ggml's flash attention as a torch op, and as a `transformers` attention implementation.

This wraps `ggml_attn_ext` out of llama.cpp's Metal backend -- the same kernel llama.cpp runs. ggml
picks between two implementations of that op by query count (`ne01 < 20` selects the *vector* path, wider
`q` the *tiled* one); both are ported, so decode and prefill both run on ggml's kernels. Head-dim pairs
upstream has no template for are refused, not silently served by torch's attention -- ask
`supports_flash_attn` before selecting this implementation.
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


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Flash attention through ggml's vector kernel — the path upstream picks for decode.

    Args:
        q: `(n_seqs, n_heads, n_q, head_dim)`.
        k, v: `(n_seqs, n_heads_kv, n_kv, head_dim)`. Grouped-query attention is native, so do **not**
            expand them to `n_heads` first — that copy is exactly what this avoids.
        mask: `(n_seqs, 1, n_q, n_kv)` additive mask, or None. Cast to f16 internally, as the kernel
            requires. **A None mask means attend to everything**: ggml has no `is_causal` argument, so
            causality has to arrive as a mask. `flash_attn_forward` builds one when it must.
        scale: softmax scale; defaults to `head_dim ** -0.5`.

    Returns:
        `(n_seqs, n_q, n_heads, head_dim)` — tokens before heads, which is what SDPA gives after its
        own `.transpose(1, 2)`, so a caller usually wants precisely this and no further permute.

    Ask `supports_flash_attn` first: it answers for whichever of upstream's two paths the shape selects.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ops.flash_attn(q, k, v, mask, scale)


def supports_flash_attn(n_q: int, head_dim_k: int, head_dim_v: int) -> bool:
    """Whether this build has a flash-attention kernel for these shapes.

    Upstream picks its vector kernel for `n_q < 20` and its tiled one above that; both are ported, so
    what is declined is head-dim pairs neither has a template for, and `head_dim_k < head_dim_v`.
    """
    return ops.supports_flash_attn(n_q, head_dim_k, head_dim_v)


@torch.library.register_fake(add_op_namespace_prefix("flash_attn"))
def _(q, k, v, mask, scale):
    n_seqs, n_heads, n_q, _ = q.shape
    return q.new_empty((n_seqs, n_q, n_heads, v.shape[-1]))


__all__ = ["flash_attn", "supports_flash_attn", "flash_attn_forward"]


_CAUSAL_CACHE: dict = {}


def _causal_mask(query: torch.Tensor, n_kv: int) -> torch.Tensor:
    """The additive causal mask ggml requires when the caller supplied none.

    `ggml_attn_ext` takes no `is_causal` argument: in ggml the mask *is* the causality, and a null mask
    means attend to everything. `transformers` is entitled to hand us None -- `sdpa` carries causality in its
    own `is_causal` flag instead, so the mask is dropped as an optimisation -- and that flag does not reach an
    attention function at all. Rebuilding the mask here restores what was dropped; without it, any `n_q > 1`
    call attends to future positions, which is silent: the logits move but greedy text often does not.

    Lower-right aligned, matching `sdpa`'s convention when a cache is in play: the `n_q` queries are the last
    `n_q` of the `n_kv` positions, so query `i` may see key `j <= i + (n_kv - n_q)`.
    """
    n_q = query.shape[2]
    key = (n_q, n_kv, str(query.device))
    cached = _CAUSAL_CACHE.get(key)
    if cached is None:
        # f16, not the query dtype: the kernel casts an f32 mask to f16 anyway, and at prefill widths
        # that cast is a second n_q*n_kv allocation per layer per forward -- enough to cost more than
        # the kernel gains. One entry, because a generation reuses one prefill shape and then decodes.
        future = torch.ones(n_q, n_kv, dtype=torch.bool, device=query.device).triu(1 + n_kv - n_q)
        cached = torch.zeros(n_q, n_kv, dtype=torch.float16, device=query.device).masked_fill_(
            future, float("-inf")
        )[None, None]
        _CAUSAL_CACHE.clear()
        _CAUSAL_CACHE[key] = cached
    return cached


def flash_attn_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention entry point shaped for `transformers`' attention interface.

    Selected with `attn_implementation="<repo>:flash_attn_forward"`, which is how a kernel provides an
    attention implementation without the caller reaching into `ALL_ATTENTION_FUNCTIONS`.

    Both of upstream's paths are ported -- the vector kernel for `n_q < 20`, the tiled one above it -- so
    decode and prefill both run on ggml's kernels. A head-dim pair neither has a template for raises, the
    way `flash_attention_2` and the block-sparse kernels do, rather than quietly running torch's attention
    under a name that promises ggml's: what upstream instantiates does not depend on the input, so a model
    whose head dims are unsupported is unsupported for its whole run, and saying so at the first forward
    beats reporting a speedup that never happened. Grouped-query attention is native to the kernel, so
    `key`/`value` keep their own head count.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    n_q, head_dim_k, head_dim_v = query.shape[2], query.shape[-1], value.shape[-1]
    if not supports_flash_attn(n_q, head_dim_k, head_dim_v):
        raise ValueError(
            f"ggml-attn has no kernel for head_dim {head_dim_k}/{head_dim_v} at {n_q} queries. ggml "
            f"instantiates its attention kernels per head-dim pair, and this build carries the pairs "
            f"upstream provides templates for; `supports_flash_attn(n_q, head_dim_k, head_dim_v)` answers "
            f"for any shape. Use `attn_implementation='sdpa'` for this model."
        )

    mask = attention_mask
    if mask is not None and mask.dtype == torch.bool:
        # The kernel wants an additive mask; a boolean one says which positions are allowed.
        mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill_(~mask, float("-inf"))
    elif mask is None and n_q > 1:
        mask = _causal_mask(query, key.shape[2])
    return flash_attn(query, key, value, mask, scaling), None

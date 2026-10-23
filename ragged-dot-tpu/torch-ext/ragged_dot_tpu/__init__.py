"""
MoE experts on torch_tpu with `tokamax.ragged_dot` (Pallas Mosaic TPU grouped
matmul, Megablocks-style).

`experts_forward` is a transformers experts implementation, a drop-in for
`grouped_mm_experts_forward`:

    ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
    model = AutoModelForCausalLM.from_pretrained(
        ..., experts_implementation="tokamax_ragged_dot")

The whole experts forward (sort by expert, both grouped matmuls, gate
activation, routing weights, un-sort and sum) is one JAX function, so
torch_tpu runs a single compiled op per layer instead of dispatching ~20
small ones.
"""

import functools

import jax
import jax.numpy as jnp
import tokamax
import torch
from torch_tpu._internal.pallas import jax_op

from ._ops import add_op_namespace_prefix

# tokamax's v2 Pallas TPU grouped matmul, with its VMEM-aware tiling
# heuristic. v1 ("mosaic") only has tuned tiles for TPU7x.
IMPLEMENTATION = "mosaic_tpu_v2"

# HF experts store weights as [num_experts, out_features, in_features]. These
# dimension numbers contract lhs dim 1 with rhs dim 2, so the weights are used
# as stored, without a transposed copy.
_TRANS_RHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


def _ragged_dot_jax(
    lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
) -> jax.Array:
    return tokamax.ragged_dot_general(
        lhs, rhs, group_sizes, _TRANS_RHS_DIM_NUMS, implementation=IMPLEMENTATION
    )


def _experts_jax(
    hidden_states: jax.Array,
    top_k_index: jax.Array,
    top_k_weights: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    activation,
) -> jax.Array:
    top_k = top_k_index.shape[-1]
    num_experts = gate_up_proj.shape[0]

    # Sort (token, expert) pairs by expert. Under expert parallelism, pairs
    # routed to another rank carry the sentinel id `num_experts`: they sort
    # last and fall outside every group, so their rows are never computed.
    expert_ids = top_k_index.reshape(-1)
    perm = jnp.argsort(expert_ids)
    expert_ids = expert_ids[perm]
    group_sizes = jnp.sum(
        expert_ids[:, None] == jnp.arange(num_experts), axis=0, dtype=jnp.int32
    )
    token_ids = perm // top_k

    gate_up = _ragged_dot_jax(hidden_states[token_ids], gate_up_proj, group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    # Activation in fp32 and rounded once, like torch on bf16 inputs.
    act = activation(gate.astype(jnp.float32)).astype(gate.dtype) * up
    out = _ragged_dot_jax(act, down_proj, group_sizes)

    # Sentinel rows are uninitialized: zero them before weighting.
    is_local = (expert_ids < num_experts)[:, None]
    out = jnp.where(is_local, out, 0) * top_k_weights.reshape(-1)[perm][:, None]
    # Un-sort and sum each token's top-k expert outputs, accumulating in fp32.
    # A gather by the inverse permutation rather than a scatter-add: on TPU,
    # XLA's scatter-add returns wrong rows for some shapes (e.g. 512 x 2816
    # fp32, Gemma 4's hidden size).
    out = out[jnp.argsort(perm)].astype(jnp.float32)
    final = out.reshape(hidden_states.shape[0], top_k, -1).sum(axis=1)
    return final.astype(hidden_states.dtype)


# Grouped matmul: lhs[M, K] @ rhs[G, N, K].mT -> [M, N]. Rows of `lhs` are
# sorted by group: the first group_sizes[0] rows use rhs[0], and so on.
# `group_sizes` is int32. Rows past group_sizes.sum() are left uninitialized.
ragged_dot = jax_op(add_op_namespace_prefix("ragged_dot"), _ragged_dot_jax)

# Gate activations, keyed by the class of the experts module's `act_fn`
# (transformers' ACT2FN): SiLU for Qwen3-MoE, GELU with the tanh approximation
# (`gelu_pytorch_tanh`) for Gemma 4. One fused op per activation.
_ACTIVATIONS = {"SiLUActivation": "silu", "SiLU": "silu", "GELUTanh": "gelu_tanh"}
_ACTIVATION_FNS = {
    "silu": jax.nn.silu,
    "gelu_tanh": functools.partial(jax.nn.gelu, approximate=True),
}


def _make_experts_op(activation):
    def _experts_op(
        hidden_states: jax.Array,
        top_k_index: jax.Array,
        top_k_weights: jax.Array,
        gate_up_proj: jax.Array,
        down_proj: jax.Array,
    ) -> jax.Array:
        return _experts_jax(
            hidden_states,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            activation=activation,
        )

    return _experts_op


_experts = {
    name: jax_op(add_op_namespace_prefix(f"experts_{name}"), _make_experts_op(fn))
    for name, fn in _ACTIVATION_FNS.items()
}


def experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """transformers experts forward: gated experts without bias (SiLU or
    GELU-tanh gate), weights stored [E, N, K] with gate and up concatenated."""
    activation = _ACTIVATIONS.get(type(self.act_fn).__name__)
    if (
        self.has_bias
        or self.is_transposed
        or not (self.has_gate and self.is_concatenated)
        or activation is None
    ):
        raise NotImplementedError(
            "ragged-dot-tpu supports gated experts without bias, stored [E, N, K]"
            " with gate and up concatenated, with a SiLU or GELU-tanh gate; got"
            f" act_fn {type(self.act_fn).__name__}."
        )
    # jax_op traces with 32-bit ints; the router's topk indices are int64.
    return _experts[activation](
        hidden_states,
        top_k_index.int(),
        top_k_weights,
        self.gate_up_proj,
        self.down_proj,
    )


__all__ = ["experts_forward", "ragged_dot"]

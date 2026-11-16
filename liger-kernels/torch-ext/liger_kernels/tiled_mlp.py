import math
from contextlib import contextmanager, nullcontext

from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.distributed as dist

from torch.utils.checkpoint import checkpoint


if dist.is_available():
    from torch.distributed._functional_collectives import AsyncCollectiveTensor
    from torch.distributed.fsdp import FSDPModule


def _prepare_fsdp_checkpoint(mlp_module, x):
    """Prepare FSDP parameters and checkpoint recomputation contexts."""
    if not (dist.is_available() and dist.is_initialized()):
        return x, {}

    fsdp_modules = [module for module in mlp_module.modules() if isinstance(module, FSDPModule)]
    if not fsdp_modules:
        return x, {}

    for module in fsdp_modules:
        module.unshard()

    # An outer checkpoint can re-enter after FSDP skips its input-casting
    # hooks. Keep the same computation dtype at our checkpoint boundary.
    parameter = next(mlp_module.parameters(), None)
    if parameter is not None and parameter.dtype in (torch.float16, torch.bfloat16):
        x = x.to(parameter.dtype)

    @contextmanager
    def recompute_context():
        # A projection can have been resharded by an earlier tile's backward.
        # FSDP2 skips its pre-forward unshard during checkpoint recomputation,
        # so explicitly materialize those parameters for each new tile.
        for module in fsdp_modules:
            module.unshard()
        yield

    # checkpoint's context_fn returns (forward context, recomputation context).
    # Only recomputation needs the extra FSDP unsharding.
    return x, {"context_fn": lambda: (nullcontext(), recompute_context())}


class LigerTiledMLPFunction:
    """Compatibility entry point for checkpointed tiled MLP computation.

    Each shard is checkpointed and recomputed during backward to reduce activation
    memory, matching the original custom autograd implementation.
    Non-reentrant checkpointing records the parameter dependencies even when
    the input does not require gradients, and supports ``torch.autograd.grad``.
    """

    @staticmethod
    def apply(
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        # Keep compute_params in the public API; autograd discovers the weights
        # used by fn without registering them through a separate parameter list.
        x = x.contiguous()
        x, checkpoint_kwargs = _prepare_fsdp_checkpoint(mlp_module, x)
        collective_tensor_types = (AsyncCollectiveTensor,) if dist.is_available() and dist.is_initialized() else ()

        output_shards = []
        for x_shard in torch.tensor_split(x, shards, dim=-2):
            output = checkpoint(fn, mlp_module, x_shard, use_reentrant=False, **checkpoint_kwargs)
            # An outer checkpoint can stop before cat consumes the TP outputs.
            # Resolve each tile now so its collective is never abandoned.
            if isinstance(output, collective_tensor_types):
                # Keep the wrapper: wait() returns its detached local tensor.
                output.wait()
            output_shards.append(output)
        return torch.cat(output_shards, dim=-2)


def apply_tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    Apply tiled MLP computation for memory efficiency.

    Args:
        fn: the function to call on sharded inputs (e.g., lambda module, x: module(x))
        mlp_module: the MLP nn.Module object
        x: the input tensor with shape [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        num_shards: number of shards to use. If None, automatically calculated as ceil(seqlen / hidden_size)
        compute_params: optional parameter list retained for API compatibility

    Returns:
        output tensor with the same shape as input
    """
    if num_shards is None:
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)

    # Ensure num_shards is at least 1
    num_shards = max(1, num_shards)

    # Sharded parameters can be gathered during each tile's recomputation. Keep
    # the number of tiles identical across ranks, including unequal CP inputs.
    # torch.chunk may return fewer than the requested number of chunks, so the
    # checkpoint loop uses tensor_split instead.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        limits = torch.tensor([x.shape[-2], -num_shards], device=x.device)
        dist.all_reduce(limits, op=dist.ReduceOp.MIN)
        min_length, negative_max_shards = limits.tolist()
        num_shards = min(-negative_max_shards, max(1, min_length))

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )

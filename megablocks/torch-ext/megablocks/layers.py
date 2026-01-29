import torch
import torch.distributed as dist

from typing import Optional, Any, TYPE_CHECKING

from . import _layers
from . import ops

# Conditional import for meta kernel registration
if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn

else:
    try:
        from torch.library import register_fake
    except ImportError:
        try:
            from torch.library import impl_abstract as register_fake
        except ImportError:
            # Fallback for older PyTorch versions
            def register_fake(op_name):
                def decorator(fn):
                    return fn

                return decorator


# Meta kernel implementations for torch.compile compatibility
def _install_meta_kernels():
    """Install meta kernels for existing MegaBlocks operations"""

    # Create wrapper functions that check for compilation and return meta tensors

    # Patch ops.sort
    if hasattr(ops, "sort"):
        original_sort = ops.sort

        def sort_with_meta(x, end_bit=None):
            if torch.compiler.is_compiling():
                print("Using meta kernel for sort")
                # Meta implementation - return tensors with correct shape/dtype/device
                return torch.empty_like(x), torch.empty_like(x)
            # print("Using original sort kernel")
            return original_sort(x, end_bit)

        ops.sort = sort_with_meta

    # Patch ops.histogram
    if hasattr(ops, "histogram"):
        original_histogram = ops.histogram

        def histogram_with_meta(x, max_val):
            if torch.compiler.is_compiling():
                # Meta implementation
                return torch.empty((max_val,), dtype=torch.int32, device=x.device)
            return original_histogram(x, max_val)

        ops.histogram = histogram_with_meta

    # Patch ops.inclusive_cumsum
    if hasattr(ops, "inclusive_cumsum"):
        original_inclusive_cumsum = ops.inclusive_cumsum

        def inclusive_cumsum_with_meta(x, dim):
            if torch.compiler.is_compiling():
                # Meta implementation
                return torch.empty_like(x)
            return original_inclusive_cumsum(x, dim)

        ops.inclusive_cumsum = inclusive_cumsum_with_meta

    # Patch ops.binned_gather
    if hasattr(ops, "binned_gather"):
        original_binned_gather = ops.binned_gather

        def binned_gather_with_meta(x, indices, bins, bin_size, top_k):
            if torch.compiler.is_compiling():
                # Meta implementation - output shape based on bin_size
                if x.dim() >= 2:
                    hidden_size = x.size(-1)
                    return torch.empty(
                        (bin_size, x.size(1), hidden_size),
                        dtype=x.dtype,
                        device=x.device,
                    )
                else:
                    return torch.empty((bin_size,), dtype=x.dtype, device=x.device)
            return original_binned_gather(x, indices, bins, bin_size, top_k)

        ops.binned_gather = binned_gather_with_meta

    # Patch ops.binned_scatter
    if hasattr(ops, "binned_scatter"):
        original_binned_scatter = ops.binned_scatter

        def binned_scatter_with_meta(x, indices, weights, bins, top_k):
            if torch.compiler.is_compiling():
                # Meta implementation - typically reduces to 2D
                if x.dim() >= 3:
                    return torch.empty(
                        (x.size(1), x.size(2)), dtype=x.dtype, device=x.device
                    )
                else:
                    return torch.empty_like(x)
            return original_binned_scatter(x, indices, weights, bins, top_k)

        ops.binned_scatter = binned_scatter_with_meta

    # Patch ops.gather
    if hasattr(ops, "gather"):
        original_gather = ops.gather

        def gather_with_meta(x, indices, bin_ids, bins, top_k):
            if torch.compiler.is_compiling():
                # Meta implementation
                if x.dim() >= 2:
                    hidden_size = x.size(-1)
                    return torch.empty(
                        (indices.numel(), hidden_size), dtype=x.dtype, device=x.device
                    )
                else:
                    return torch.empty(indices.shape, dtype=x.dtype, device=x.device)
            return original_gather(x, indices, bin_ids, bins, top_k)

        ops.gather = gather_with_meta

    # Patch ops.scatter
    if hasattr(ops, "scatter"):
        original_scatter = ops.scatter

        def scatter_with_meta(x, indices, bin_ids, weights, bins, top_k):
            if torch.compiler.is_compiling():
                # Meta implementation - restore sequence shape
                seq_len = (
                    indices.size(0) // top_k
                    if indices.numel() > 0 and top_k > 0
                    else x.size(0)
                )
                if x.dim() >= 2:
                    return torch.empty(
                        (seq_len, x.size(-1)), dtype=x.dtype, device=x.device
                    )
                else:
                    return torch.empty((seq_len,), dtype=x.dtype, device=x.device)
            return original_scatter(x, indices, bin_ids, weights, bins, top_k)

        ops.scatter = scatter_with_meta

    # Patch ops.replicate
    if hasattr(ops, "replicate"):
        original_replicate = ops.replicate

        def replicate_with_meta(x, bins, num_outputs):
            if torch.compiler.is_compiling():
                # Meta implementation
                return torch.empty(
                    (x.shape[0], num_outputs), dtype=x.dtype, device=x.device
                )
            return original_replicate(x, bins, num_outputs)

        ops.replicate = replicate_with_meta

    # Patch ops.repeat (if it's a regular function)
    if hasattr(ops, "repeat"):
        original_repeat = ops.repeat

        def repeat_with_meta(x, repeats):
            if torch.compiler.is_compiling():
                # Meta implementation
                if isinstance(repeats, (tuple, list)):
                    new_shape = list(x.shape)
                    for i, rep in enumerate(repeats):
                        if i < len(new_shape):
                            new_shape[i] *= rep
                    return torch.empty(new_shape, dtype=x.dtype, device=x.device)
                else:
                    new_shape = [x.size(0) * repeats] + list(x.shape[1:])
                    return torch.empty(new_shape, dtype=x.dtype, device=x.device)
            return original_repeat(x, repeats)

        ops.repeat = repeat_with_meta


# Install meta kernels on import
try:
    _install_meta_kernels()
except Exception as e:
    # If meta kernel installation fails, continue without them
    # torch.compile may not work but the library will still function
    import warnings

    warnings.warn(
        f"Failed to install meta kernels for torch.compile support: {e}", UserWarning
    )


# Set the expert model parallel attributes on a tensor
def set_expert_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
):
    assert not hasattr(tensor, "expert_model_parallel")
    setattr(tensor, "expert_model_parallel", is_parallel)


# Get the expert model parallel attributes from a tensor
def expert_sharding_degree(
    world_size: int,
    moe_num_experts: int,
) -> int:
    esd = min(world_size, moe_num_experts)
    if (moe_num_experts % esd) != 0:
        raise ValueError(f"Cannot shard {moe_num_experts} experts {esd} ways.")
    return esd


# Calculate the hidden sharding degree based on world size and expert sharding degree
def hidden_sharding_degree(
    world_size: int,
    moe_num_experts: int,
    ffn_hidden_size: int,
) -> int:
    esd = expert_sharding_degree(world_size, moe_num_experts)
    hsd = world_size // esd
    if (ffn_hidden_size % hsd) != 0:
        raise ValueError(f"Cannot shard {ffn_hidden_size} features {hsd} ways.")
    if (esd * hsd) != world_size:
        raise ValueError(
            f"Invalid sharding. expert_sharding_degree ({esd}) * hidden_sharding_degree ({hsd}) != world_size ({world_size})."
        )
    return hsd


# Calculate the number of experts per rank based on world size and expert sharding degree
def experts_per_rank(
    moe_num_experts: int,
    world_size: int,
) -> int:
    return moe_num_experts // expert_sharding_degree(world_size, moe_num_experts)


# Calculate the number of features per rank based on ffn hidden size and hidden sharding degree
def features_per_rank(
    ffn_hidden_size: int, world_size: int, moe_num_experts: int
) -> int:
    return ffn_hidden_size // hidden_sharding_degree(
        world_size, moe_num_experts, ffn_hidden_size
    )


# Apply jitter to the input tensor
def apply_jitter(x: torch.Tensor, moe_jitter_eps: float) -> torch.Tensor:
    low = 1.0 - moe_jitter_eps
    high = 1.0 + moe_jitter_eps
    noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
    return x * (low + noise * (high - low))


# Compute the top-k scores from the logits
def compute_top_k(scores: torch.Tensor, moe_top_k: int):
    if moe_top_k == 1:
        return scores.max(dim=-1, keepdim=True)
    return torch.topk(scores, moe_top_k, dim=-1)


# Route tokens to experts and compute expert weights and indices
def route_tokens(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    moe_top_k: int,
    moe_num_experts: int,
    moe_jitter_eps: float = None,
    moe_normalize_expert_weights: int = None,
    uniform_expert_assignment: bool = False,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if training and moe_jitter_eps is not None:
        x = apply_jitter(x, moe_jitter_eps)

    x_flat = x.view(-1, x.shape[-1])
    logits = torch.nn.functional.linear(x_flat, router_weight, router_bias)
    expert_weights, expert_indices = compute_top_k(logits, moe_top_k)
    expert_weights = expert_weights.softmax(dim=-1)
    if moe_normalize_expert_weights is not None:
        expert_weights = expert_weights / torch.norm(
            expert_weights,
            p=moe_normalize_expert_weights,
            dim=-1,
            keepdim=True,
        )
    if uniform_expert_assignment:
        expert_indices = _layers.router._uniform_expert_assignment(
            expert_indices,
            moe_num_experts,
        )

    return logits, expert_weights, expert_indices


# Scale the gradient of the weights
def scale_grad(
    w: torch.Tensor,
    gradient_scale: Optional[float] = None,
) -> torch.Tensor:
    if gradient_scale is None:
        return w
    return _layers.mlp.scale_gradient(w, gradient_scale)


# Forward pass for the MLP layer
def mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    gradient_scale: Optional[float] = None,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    # Scale weights
    w1 = scale_grad(w1, gradient_scale)
    w2 = scale_grad(w2, gradient_scale)
    w1_bias = scale_grad(w1_bias, gradient_scale)
    w2_bias = scale_grad(w2_bias, gradient_scale)

    # Resolve dtensors
    w1 = _layers.mlp.resolve_dtensor(w1)
    w2 = _layers.mlp.resolve_dtensor(w2)
    w1_bias = _layers.mlp.resolve_dtensor(w1_bias)
    w2_bias = _layers.mlp.resolve_dtensor(w2_bias)

    # Forward pass
    gate_up = torch.bmm(x, w1) + w1_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    next_states = torch.bmm(((up + 1) * glu), w2)
    next_states += w2_bias[..., None, :]
    return next_states

# Shared expert MLP forward pass
def shared_mlp_forward(
    x: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    activation_fn: Optional[Any] = None,
    gradient_scale: Optional[float] = None,
) -> torch.Tensor:
    # Default activation function
    if activation_fn is None:
        activation_fn = torch.nn.functional.gelu

    # Scale weights
    up_proj_weight = scale_grad(up_proj_weight, gradient_scale)
    down_proj_weight = scale_grad(down_proj_weight, gradient_scale)
    if up_proj_bias is not None:
        up_proj_bias = scale_grad(up_proj_bias, gradient_scale)
    if down_proj_bias is not None:
        down_proj_bias = scale_grad(down_proj_bias, gradient_scale)

    # Resolve dtensors
    up_proj_weight = _layers.mlp.resolve_dtensor(up_proj_weight)
    down_proj_weight = _layers.mlp.resolve_dtensor(down_proj_weight)
    if up_proj_bias is not None:
        up_proj_bias = _layers.mlp.resolve_dtensor(up_proj_bias)
    if down_proj_bias is not None:
        down_proj_bias = _layers.mlp.resolve_dtensor(down_proj_bias)

    # Up projection
    x = torch.nn.functional.linear(x, up_proj_weight, up_proj_bias)

    # Activation
    x = activation_fn(x)

    # Down projection
    x = torch.nn.functional.linear(x, down_proj_weight, down_proj_bias)

    return x


# Combine outputs from shared expert and regular experts
def combine_expert_shared_outputs(
    shared_expert_out: torch.Tensor,
    expert_out: torch.Tensor,
    shared_expert_weighted_sum: bool = False,
    moe_top_k: int = 1,
) -> torch.Tensor:
    if shared_expert_weighted_sum:
        # Weighted sum based on number of experts used
        total_experts = moe_top_k + 1
        shared_weight = 1.0 / total_experts
        expert_weight = moe_top_k / total_experts
        return shared_expert_out * shared_weight + expert_out * expert_weight
    else:
        # Simple addition
        return shared_expert_out + expert_out


# Global variable to store load balancing loss
_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def batched_load_balancing_loss(args):
    if args.moe_loss_weight == 0:
        return 0.0

    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}",
        )
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}",
        )

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all(
        (x.ndim == 1 and x.numel() == args.moe_num_experts for x in tokens_per_expert)
    )

    tokens = expert_scores[0].shape[0]
    assert all(
        (
            (
                x.ndim == 2
                and x.shape[1] == args.moe_num_experts
                and x.shape[0] == tokens
            )
            for x in expert_scores
        )
    )

    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    expert_scores = torch.cat(expert_scores, dim=1)
    if args.moe_lbl_in_fp32:
        expert_scores = expert_scores.float()
    if tokens != 0:
        expert_scores = expert_scores.mean(dim=0)
    else:
        expert_scores = expert_scores.sum(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = args.moe_num_experts * args.moe_loss_weight
    scale_denominator = args.num_layers * tokens * args.moe_top_k
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


# Calculate the expert capacity based on tokens, top_k, number of experts,
# expert parallel group, capacity factor, and whether expert model parallelism is used.
def expert_capacity(
    tokens: int,
    top_k: int,
    num_experts: int,
    expert_parallel_group: int,
    moe_capacity_factor: float,
    moe_expert_model_parallelism: bool,
) -> int:
    world_size = (
        dist.get_world_size(expert_parallel_group)
        if moe_expert_model_parallelism
        else 1
    )

    tokens_per_expert = top_k * tokens * world_size / num_experts
    return int(moe_capacity_factor * tokens_per_expert)


def load_balancing_loss(
    tokens_per_expert: torch.Tensor,
    expert_scores: torch.Tensor,
    top_k: int,
    num_experts: int,
):
    assert len(expert_scores.size()) == 2
    tokens, num_experts = expert_scores.size()
    assert num_experts == num_experts
    assert len(tokens_per_expert.size()) == 1
    (num_experts,) = tokens_per_expert.size()
    assert num_experts == num_experts
    scale = num_experts / (tokens * top_k)
    return scale * torch.dot(
        tokens_per_expert.to(expert_scores.dtype),
        expert_scores.mean(dim=0),
    )


def indices_and_bins(
    top_expert: torch.Tensor,
    sort_end_bit: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top_expert = top_expert.int()

    # Ensure contiguous memory layout
    top_expert = top_expert.contiguous()

    # Ensure CUB knows which device to use
    with torch.cuda.device(top_expert.device):
        output = ops.sort(top_expert, sort_end_bit)
        bin_ids, indices = output
        tokens_per_expert = ops.histogram(top_expert, num_experts)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

    bins = bins.view(1) if not len(bins.size()) else bins
    return indices, bin_ids, bins, tokens_per_expert


def expert_capacity_fn(
    tokens: int,
    top_k: int,
    num_experts: int,
    expert_parallel_group: torch.distributed.ProcessGroup,
    moe_capacity_factor: float = 1.0,
    moe_expert_model_parallelism: bool = False,
) -> int:
    world_size = (
        dist.get_world_size(expert_parallel_group)
        if moe_expert_model_parallelism
        else 1
    )
    tokens_per_expert = top_k * tokens * world_size / num_experts
    return int(moe_capacity_factor * tokens_per_expert)


def permute_and_compute(
    x,
    tokens_per_expert,
    indices,
    bin_ids,
    expert_weights,
    bins,
    expert_capacity,
    top_k,
    w1,
    w2,
    w1_bias,
    w2_bias,
    gradient_scale,
    alpha,
):
    # Route tokens to experts
    x = x.view(-1, x.shape[-1])

    # Ensure CUB knows which device to use
    with torch.cuda.device(x.device):
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

    # Expert computation
    x = mlp_forward(x, w1, w2, w1_bias, w2_bias, gradient_scale, alpha)

    # Ensure CUB knows which device to use
    with torch.cuda.device(x.device):
        # Route tokens back
        out = ops.binned_scatter(x, indices, expert_weights, bins, top_k)
    return out


def forward_once(
    x: torch.Tensor,
    expert_weights: torch.Tensor,
    top_experts: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    gradient_scale: Optional[float] = None,
    alpha: float = 1.702,
    sort_end_bit: int = 0,
    top_k: int = 4,
    num_experts: int = 128,
    expert_parallel_group: int = None,
    moe_capacity_factor: float = 1.0,
    moe_expert_model_parallelism: bool = False,
    mlp_impl: Optional[str] = None,
):
    # x: [sl, bs, hs]
    # expert_weights: [sl * bs, top-k]
    # top_experts: [sl * bs, top-k]
    expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()

    with torch.no_grad():
        indices, bin_ids, bins, tokens_per_expert = indices_and_bins(
            top_experts, sort_end_bit, num_experts
        )

        # Calculate expert capacity
        sl, bs, _ = x.size()

        expert_capacity = expert_capacity_fn(
            sl * bs,
            top_k,
            num_experts,
            expert_parallel_group,
            moe_capacity_factor,
            moe_expert_model_parallelism,
        )

        if expert_capacity == 0:
            expert_capacity = torch.max(tokens_per_expert).item()

    x = permute_and_compute(
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capacity,
        top_k,
        w1,
        w2,
        w1_bias,
        w2_bias,
        gradient_scale,
        alpha,
    )
    return x, tokens_per_expert


def parallel_forward_once(
    x: torch.Tensor,
    expert_weights: torch.Tensor,
    top_experts: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    gradient_scale: Optional[float] = None,
    alpha: float = 1.702,
    sort_end_bit: int = 0,
    top_k: int = 4,
    num_experts: int = 128,
    expert_parallel_group: torch.distributed.ProcessGroup = None,
    moe_capacity_factor: float = 1.0,
    moe_expert_model_parallelism: bool = True,
    hidden_size: int = 1152,
    mlp_impl: Optional[str] = "grouped",
):
    # Flatten inputs
    expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()

    # TODO: remove debugging var
    # my_rank = dist.get_rank(expert_parallel_group) if expert_parallel_group else 0

    with torch.no_grad():
        # Step 1: Local permutation setup
        indices, bin_ids, bins, tokens_per_expert = indices_and_bins(
            top_experts, sort_end_bit, num_experts
        )

        # Calculate sharding parameters
        world_size = dist.get_world_size(expert_parallel_group)
        hidden_sharding_deg = hidden_sharding_degree(
            world_size, num_experts, hidden_size
        )
        experts_per_rank_val = experts_per_rank(num_experts, world_size)

        # Replicate token counts for hidden sharding
        repeated_tokens_per_expert = ops.repeat(
            tokens_per_expert, (hidden_sharding_deg,)
        )

        # Exchange token counts across devices
        parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)

        # Ensure CUB knows which device to use
        tpe_handle = dist.all_to_all_single(
            parallel_tokens_per_expert,
            repeated_tokens_per_expert,
            group=expert_parallel_group,
            async_op=True,
        )

    # Step 2: Local permutation - group tokens by target device
    x = x.view(-1, x.shape[-1])  # [sl * bs, hs]
    x = ops.gather(x, indices, bin_ids, bins, top_k)

    # Step 3: Compute communication counts and exchange tokens
    with torch.no_grad():
        tpe_handle.wait()

        # Reshape for per-device calculations
        repeated_tokens_per_expert = repeated_tokens_per_expert.view(
            world_size, experts_per_rank_val
        )
        parallel_tokens_per_expert = parallel_tokens_per_expert.view(
            world_size, experts_per_rank_val
        )

        # Calculate send/recv counts
        send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1).tolist()
        # recv_counts = parallel_tokens_per_expert.cpu().sum(dim=-1).tolist()
        parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
        recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1).tolist()
        tokens_received = sum(recv_counts)

    # Replicate for hidden sharding
    x = ops.repeat(x, (hidden_sharding_deg, 1))

    # Cross-device token exchange
    parallel_x, parallel_x_handle = _layers.all_to_all.all_to_all(
        x, recv_counts, send_counts, expert_parallel_group, async_op=True
    )

    with torch.no_grad():
        # Step 4: Setup for local expert computation
        replicate_bins = ops.inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)
        replicate_bins = (
            replicate_bins.view(1) if not len(replicate_bins.size()) else replicate_bins
        )

        # Create expert indices for received tokens
        parallel_top_expert = torch.remainder(
            torch.arange(
                num_experts * hidden_sharding_deg,
                dtype=torch.int32,
                device=indices.device,
            ),
            experts_per_rank_val,
        )
        parallel_top_expert = ops.replicate(
            parallel_top_expert.unsqueeze(dim=0),
            replicate_bins,
            tokens_received,
        ).flatten()

        # Sort tokens by expert assignment
        parallel_bin_ids, parallel_indices = ops.sort(
            parallel_top_expert,
            sort_end_bit,
        )

        # Calculate bins for local experts
        parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
            dim=0, dtype=torch.int
        )
        parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
        parallel_bins = (
            parallel_bins.view(1) if not len(parallel_bins.size()) else parallel_bins
        )

        # Calculate expert capacity
        expert_capacity = expert_capacity_fn(
            tokens_received,
            top_k,
            experts_per_rank_val,
            expert_parallel_group,
            moe_capacity_factor,
            moe_expert_model_parallelism,
        )
        if expert_capacity == 0:
            expert_capacity = torch.max(parallel_tokens_per_expert).item()

    # Locally permute the tokens and perform the expert computation.
    # Block to make sure that the cross-device permutation is complete.
    if mlp_impl == "grouped":
        # GroupedMLP requires counts on CPU. We can use the tensor already
        # moved to CPU for the prior all_to_all, which avoids an extra
        # device synchronization.
        parallel_tokens_per_expert = parallel_tokens_per_expert_cpu.sum(
            dim=0,
            dtype=torch.int,
        )

    # Step 5: Expert computation
    parallel_x_handle.wait()

    parallel_x = permute_and_compute(
        parallel_x,
        parallel_tokens_per_expert,
        parallel_indices,
        parallel_bin_ids,
        None,  # expert_weights
        parallel_bins,
        expert_capacity,
        top_k=1,
        w1=w1,
        w2=w2,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gradient_scale=gradient_scale,
        alpha=alpha,
    )

    # Step 6: Reverse communication - send results back
    x, _ = _layers.all_to_all.all_to_all(
        parallel_x, send_counts, recv_counts, expert_parallel_group
    )

    # Step 7: Reduce across hidden sharding dimension
    shape = (hidden_sharding_deg, -1, hidden_size)
    x = x.view(shape).sum(dim=0)

    # Step 8: Final local unpermutation
    x = ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)

    return x, tokens_per_expert.flatten()


def moe_forward(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    moe_top_k: int,
    moe_num_experts: int,
    moe_jitter_eps: float = None,
    moe_normalize_expert_weights: int = None,
    uniform_expert_assignment: bool = False,
    training: bool = False,
    w1: torch.Tensor = None,
    w2: torch.Tensor = None,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    gradient_scale: Optional[float] = None,
    alpha: float = 1.702,
    sort_end_bit: int = 0,
    expert_parallel_group: torch.distributed.ProcessGroup = None,
    moe_capacity_factor: float = 1.0,
    moe_expert_model_parallelism: bool = False,
    forward_fn: Any = None,
    hidden_size: int = None,
    mlp_impl: str = "grouped",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Route tokens to experts
    logits, expert_weights, expert_indices = route_tokens(
        x,
        router_weight,
        router_bias,
        moe_top_k,
        moe_num_experts,
        moe_jitter_eps,
        moe_normalize_expert_weights,
        uniform_expert_assignment,
        training,
    )

    # Create router scores for output
    router_scores = (
        torch.zeros_like(logits)
        .scatter_(1, expert_indices, expert_weights)
        .transpose(0, 1)
    )

    in_shape = x.size()

    # Prepare forward function arguments
    forward_args = {
        "x": x,
        "expert_weights": expert_weights,
        "top_experts": expert_indices,
        "w1": w1,
        "w2": w2,
        "w1_bias": w1_bias,
        "w2_bias": w2_bias,
        "gradient_scale": gradient_scale,
        "alpha": alpha,
        "sort_end_bit": sort_end_bit,
        "top_k": moe_top_k,
        "num_experts": moe_num_experts,
        "expert_parallel_group": expert_parallel_group,
        "moe_capacity_factor": moe_capacity_factor,
        "moe_expert_model_parallelism": moe_expert_model_parallelism,
        "mlp_impl": mlp_impl,
    }

    # Add hidden_size for parallel forward
    if moe_expert_model_parallelism and hidden_size is not None:
        forward_args["hidden_size"] = hidden_size
    elif moe_expert_model_parallelism and hidden_size is None:
        # Infer hidden_size from input shape
        forward_args["hidden_size"] = x.shape[-1]

    # Compute expert outputs
    x, tokens_per_expert = forward_fn(**forward_args)

    # Save load balancing loss if needed
    moe_loss_weight = 0.0  # Can be made configurable
    if training and moe_loss_weight > 0:
        save_load_balancing_loss((tokens_per_expert, logits))

    # Restore original shape
    x = x.view(in_shape)

    return x, expert_weights, router_scores


def moe_forward_with_shared_expert(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    moe_top_k: int,
    moe_num_experts: int,
    moe_jitter_eps: float = None,
    moe_normalize_expert_weights: int = None,
    uniform_expert_assignment: bool = False,
    training: bool = False,
    w1: torch.Tensor = None,
    w2: torch.Tensor = None,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    gradient_scale: Optional[float] = None,
    alpha: float = 1.702,
    sort_end_bit: int = 0,
    expert_parallel_group: torch.distributed.ProcessGroup = None,
    moe_capacity_factor: float = 1.0,
    moe_expert_model_parallelism: bool = False,
    forward_fn: Any = None,
    hidden_size: int = None,
    mlp_impl: str = "grouped",
    # Shared expert parameters
    shared_up_proj_weight: Optional[torch.Tensor] = None,
    shared_down_proj_weight: Optional[torch.Tensor] = None,
    shared_up_proj_bias: Optional[torch.Tensor] = None,
    shared_down_proj_bias: Optional[torch.Tensor] = None,
    shared_expert_weighted_sum: bool = False,
    shared_activation_fn: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # First, compute regular MoE forward pass
    expert_out, expert_weights, router_scores = moe_forward(
        x=x,
        router_weight=router_weight,
        router_bias=router_bias,
        moe_top_k=moe_top_k,
        moe_num_experts=moe_num_experts,
        moe_jitter_eps=moe_jitter_eps,
        moe_normalize_expert_weights=moe_normalize_expert_weights,
        uniform_expert_assignment=uniform_expert_assignment,
        training=training,
        w1=w1,
        w2=w2,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        gradient_scale=gradient_scale,
        alpha=alpha,
        sort_end_bit=sort_end_bit,
        expert_parallel_group=expert_parallel_group,
        moe_capacity_factor=moe_capacity_factor,
        moe_expert_model_parallelism=moe_expert_model_parallelism,
        forward_fn=forward_fn,
        hidden_size=hidden_size,
        mlp_impl=mlp_impl,
    )

    # If shared expert weights provided, compute shared expert output
    if shared_up_proj_weight is not None and shared_down_proj_weight is not None:
        shared_expert_out = shared_mlp_forward(
            x=x,
            up_proj_weight=shared_up_proj_weight,
            down_proj_weight=shared_down_proj_weight,
            up_proj_bias=shared_up_proj_bias,
            down_proj_bias=shared_down_proj_bias,
            activation_fn=shared_activation_fn,
            gradient_scale=gradient_scale,
        )

        # Combine expert outputs
        combined_out = combine_expert_shared_outputs(
            shared_expert_out=shared_expert_out,
            expert_out=expert_out,
            shared_expert_weighted_sum=shared_expert_weighted_sum,
            moe_top_k=moe_top_k,
        )

        return combined_out, expert_weights, router_scores

    # Return regular MoE output if no shared expert
    return expert_out, expert_weights, router_scores


def create_shared_expert_weights(
    hidden_size: int,
    shared_expert_hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    init_method: Any,
    output_layer_init_method: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

    if output_layer_init_method is None:
        output_layer_init_method = init_method

    # Create weight tensors
    up_proj_weight = torch.empty(
        shared_expert_hidden_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    down_proj_weight = torch.empty(
        hidden_size,
        shared_expert_hidden_size,
        device=device,
        dtype=dtype,
    )

    # Initialize weights
    init_method(up_proj_weight)
    output_layer_init_method(down_proj_weight)

    # No bias by default
    return up_proj_weight, down_proj_weight, None, None


# HACK: Extract device_mesh from pre-hook closure - required for transformers integration
# This exists because device_mesh is trapped in hook closures with no model attribute
# Fragile - breaks if hook structure changes or Python internals change
# TODO: Replace with a more robust solution when available
def get_device_mesh(model):
    # Extract device_mesh from child's unused pre_hook closure
    try:
        # Find the pre-hook that contains 'device_mesh' in its closure
        hook = next(
            h
            for h in model.experts._forward_pre_hooks.values()
            if "device_mesh" in h.__code__.co_freevars
        )
        # Extract the device_mesh from the closure
        return hook.__closure__[
            hook.__code__.co_freevars.index("device_mesh")
        ].cell_contents
    except Exception:
        return None


class MegaBlocksMoeMLP(torch.nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moe_top_k = getattr(self.router, "top_k", 4)
        moe_num_experts = getattr(self.experts, "num_experts", 128)
        gradient_scale = getattr(self.experts, "gradient_scale", None)
        alpha = getattr(self.experts, "alpha", 1.0)
        moe_capacity_factor = getattr(self.experts, "capacity_factor", 1.0)
        moe_jitter_eps = getattr(self.experts, "jitter_eps", None)
        moe_normalize_expert_weights = getattr(
            self.experts, "normalize_expert_weights", None
        )
        uniform_expert_assignment = getattr(self, "uniform_expert_assignment", False)

        expert_parallel_group = getattr(self, "expert_parallel_group", None)
        if expert_parallel_group is None:
            device_mesh = get_device_mesh(self)
            expert_parallel_group = device_mesh.get_group() if device_mesh else None

        has_parallel = (
            expert_parallel_group is not None
            and dist.is_initialized()
            and dist.get_world_size(expert_parallel_group) > 1
        )
        forward_fn = parallel_forward_once if has_parallel else forward_once

        sort_end_bit = max(
            int(torch.ceil(torch.log2(torch.tensor(moe_num_experts)))), 1
        )
        mlp_impl = getattr(self, "mlp_impl", "grouped")
        output, expert_weights_out, *_ = moe_forward(
            x=x,
            router_weight=self.router.weight,
            router_bias=self.router.bias,
            moe_top_k=moe_top_k,
            moe_num_experts=moe_num_experts,
            moe_jitter_eps=moe_jitter_eps,
            moe_normalize_expert_weights=moe_normalize_expert_weights,
            uniform_expert_assignment=uniform_expert_assignment,
            training=self.training,
            w1=self.experts.gate_up_proj,
            w2=self.experts.down_proj,
            w1_bias=self.experts.gate_up_proj_bias,
            w2_bias=self.experts.down_proj_bias,
            gradient_scale=gradient_scale,
            alpha=alpha,
            sort_end_bit=sort_end_bit,
            expert_parallel_group=expert_parallel_group,
            moe_capacity_factor=moe_capacity_factor,
            moe_expert_model_parallelism=has_parallel,
            forward_fn=forward_fn,
            hidden_size=self.experts.hidden_size,
            mlp_impl=mlp_impl,
        )
        return output, expert_weights_out


# Export main classes
__all__ = ["MegaBlocksMoeMLP", "MegaBlocksMoeMLPWithSharedExpert"]


class MegaBlocksMoeMLPWithSharedExpert(MegaBlocksMoeMLP):

    def __init__(self):
        super().__init__()
        # Shared expert weights will be set by the user
        self.shared_up_proj_weight = None
        self.shared_down_proj_weight = None
        self.shared_up_proj_bias = None
        self.shared_down_proj_bias = None
        self.shared_expert_weighted_sum = False
        self.shared_activation_fn = None

    def set_shared_expert_weights(
        self,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        up_proj_bias: Optional[torch.Tensor] = None,
        down_proj_bias: Optional[torch.Tensor] = None,
        weighted_sum: bool = False,
        activation_fn: Optional[Any] = None,
    ):
        self.shared_up_proj_weight = up_proj_weight
        self.shared_down_proj_weight = down_proj_weight
        self.shared_up_proj_bias = up_proj_bias
        self.shared_down_proj_bias = down_proj_bias
        self.shared_expert_weighted_sum = weighted_sum
        self.shared_activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moe_top_k = getattr(self.router, "top_k", 4)
        moe_num_experts = getattr(self.experts, "num_experts", 128)
        gradient_scale = getattr(self.experts, "gradient_scale", None)
        alpha = getattr(self.experts, "alpha", 1.0)
        moe_capacity_factor = getattr(self.experts, "capacity_factor", 1.0)
        moe_jitter_eps = getattr(self.experts, "jitter_eps", None)
        moe_normalize_expert_weights = getattr(
            self.experts, "normalize_expert_weights", None
        )
        uniform_expert_assignment = getattr(self, "uniform_expert_assignment", False)

        expert_parallel_group = getattr(self, "expert_parallel_group", None)
        if expert_parallel_group is None:
            device_mesh = get_device_mesh(self)
            expert_parallel_group = device_mesh.get_group() if device_mesh else None

        has_parallel = (
            expert_parallel_group is not None
            and dist.is_initialized()
            and dist.get_world_size(expert_parallel_group) > 1
        )
        forward_fn = parallel_forward_once if has_parallel else forward_once

        sort_end_bit = max(
            int(torch.ceil(torch.log2(torch.tensor(moe_num_experts)))), 1
        )
        mlp_impl = getattr(self, "mlp_impl", "grouped")

        output, expert_weights_out, *_ = moe_forward_with_shared_expert(
            x=x,
            router_weight=self.router.weight,
            router_bias=self.router.bias,
            moe_top_k=moe_top_k,
            moe_num_experts=moe_num_experts,
            moe_jitter_eps=moe_jitter_eps,
            moe_normalize_expert_weights=moe_normalize_expert_weights,
            uniform_expert_assignment=uniform_expert_assignment,
            training=self.training,
            w1=self.experts.gate_up_proj,
            w2=self.experts.down_proj,
            w1_bias=self.experts.gate_up_proj_bias,
            w2_bias=self.experts.down_proj_bias,
            gradient_scale=gradient_scale,
            alpha=alpha,
            sort_end_bit=sort_end_bit,
            expert_parallel_group=expert_parallel_group,
            moe_capacity_factor=moe_capacity_factor,
            moe_expert_model_parallelism=has_parallel,
            forward_fn=forward_fn,
            hidden_size=self.experts.hidden_size,
            mlp_impl=mlp_impl,
            # Shared expert parameters
            shared_up_proj_weight=self.shared_up_proj_weight,
            shared_down_proj_weight=self.shared_down_proj_weight,
            shared_up_proj_bias=self.shared_up_proj_bias,
            shared_down_proj_bias=self.shared_down_proj_bias,
            shared_expert_weighted_sum=self.shared_expert_weighted_sum,
            shared_activation_fn=self.shared_activation_fn,
        )
        return output, expert_weights_out


# Patch for XPU or CPU support
if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .xpu_fused_moe import MegaBlocksMoeMLP
elif not torch.cuda.is_available():
    from .cpu_moe_cpp import MegaBlocksMoeMLP

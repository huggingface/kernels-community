import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cross_entropy import LigerCrossEntropyFunction
from .dyt import LigerDyTFunction
from .fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from .geglu import LigerGELUMulFunction
from .group_norm import LigerGroupNormFunction
from .jsd import LigerJSDFunction
from .kl_div import LigerKLDivLossFunction
from .layer_norm import LigerLayerNormFunction
from .qwen2vl_mrope import LigerQwen2VLMRopeFunction
from .rms_norm import LigerRMSNormFunction
from .rope import LigerRopeFunction
from .swiglu import LigerSiLUMulFunction
from .tiled_mlp import apply_tiled_mlp
from .tvd import LigerTVDLossFunction


class LigerRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 0.0,
        casting_mode: str = "llama",
        init_fn: str = "ones",
        in_place: bool = True,
        row_mode: Optional[bool] = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        assert init_fn in ("ones", "zeros"), f"init_fn must be 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.offset = offset
        self.casting_mode = casting_mode
        self.in_place = in_place
        self.row_mode = row_mode
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            init = torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
            self.weight = nn.Parameter(init)
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, offset={self.offset}, "
            f"in_place={self.in_place}, row_mode={self.row_mode}"
        )


class LigerLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        bias: bool = False,
        init_fn: str = "ones",
    ):
        super().__init__()
        assert init_fn in ("ones", "zeros"), f"init_fn must be 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size) if bias else torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return LigerLayerNormFunction.apply(hidden_states, self.weight, self.bias, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}"


class LigerGroupNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-6,
        bias: bool = False,
        init_fn: str = "ones",
    ):
        super().__init__()
        assert init_fn in ("ones", "zeros"), f"init_fn must be 'ones' or 'zeros', got {init_fn}"
        assert num_channels % num_groups == 0, (
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(num_channels) if init_fn == "ones" else torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.randn(num_channels) if bias else torch.zeros(num_channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() >= 3, f"Input must have at least 3 dimensions, got {hidden_states.dim()}"
        assert hidden_states.size(1) == self.num_channels, (
            f"Input must have {self.num_channels} channels, got {hidden_states.size(1)}"
        )
        return LigerGroupNormFunction.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.num_channels,
            self.num_groups,
            self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, num_groups={self.num_groups}, eps={self.variance_epsilon}"


class LigerDyT(nn.Module):
    def __init__(self, hidden_size: int, beta: bool = True, init_alpha: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, init_alpha={self.init_alpha}, beta={self.beta is not None}"


class LigerCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
    ):
        super().__init__()
        assert 0.0 <= label_smoothing <= 1.0, f"label_smoothing must be in [0, 1], got {label_smoothing}"
        assert reduction in ("mean", "sum", "none"), f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        assert softcap is None or softcap > 0, f"softcap must be > 0 or None, got {softcap}"
        self.weight = weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _, _, _ = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            False,
            False,
            False,
        )
        return loss


class LigerFusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
    ):
        super().__init__()
        assert 0.0 <= label_smoothing <= 1.0, f"label_smoothing must be in [0, 1], got {label_smoothing}"
        assert reduction in ("mean", "sum", "none"), f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        assert softcap is None or softcap > 0, f"softcap must be > 0 or None, got {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling

    def forward(
        self,
        lin_weight: torch.Tensor,
        _input: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss, _, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            False,
            self.accum_dtype,
            self.use_token_scaling,
            False,
            False,
        )
        return loss


class LigerJSD(nn.Module):
    def __init__(self, beta: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(
        self,
        log_q: torch.Tensor,
        log_p: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return LigerJSDFunction.apply(log_q, log_p, shift_labels, self.beta, self.ignore_index)


class LigerKLDIVLoss(nn.KLDivLoss):
    def __init__(self, eps: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return LigerKLDivLossFunction.apply(y_pred, y_true, self.reduction, self.log_target, self.eps)


class LigerTVDLoss(nn.Module):
    def __init__(self, reduction: str = "batchmean", ignore_index: int = -100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return LigerTVDLossFunction.apply(p, q, shift_labels, self.reduction, self.ignore_index)


class LigerSwiGLUMLP(nn.Module):
    """SwiGLU MLP block. ``config`` must expose ``hidden_size``, ``intermediate_size``,
    and ``hidden_act`` (must be ``silu`` or ``swish``)."""

    def __init__(self, config):
        super().__init__()
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerGEGLUMLP(nn.Module):
    """GEGLU MLP block. ``config`` must expose ``hidden_size`` and ``intermediate_size``.
    Uses the tanh approximation of GELU (matches Gemma 1/1.1/2)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerTiledGEGLUMLP(nn.Module):
    """Memory-efficient GEGLU MLP using tiled computation.

    Combines GEGLU activation with tiled processing to handle very long
    sequences efficiently. The forward pass is recomputed during backward to
    save memory. ``config`` must expose ``hidden_size`` and ``intermediate_size``
    (and optionally ``hidden_act``, which must be a GELU variant).

    Args:
        config: Model configuration with ``hidden_size`` and ``intermediate_size``.
        num_shards: Number of shards to split the sequence into. If ``None``,
            automatically calculated as ``ceil(seqlen / hidden_size)``.
    """

    def __init__(self, config, num_shards: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Validate activation function
        if hasattr(config, "hidden_act") and config.hidden_act not in [
            "gelu",
            "gelu_new",
            "gelu_pytorch_tanh",
        ]:
            raise ValueError(f"LigerTiledGEGLUMLP requires GELU activation, got {config.hidden_act}")

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(LigerGELUMulFunction.apply(gate, up))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_params = [p for p in self.parameters() if p.requires_grad]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


class LigerTiledSwiGLUMLP(nn.Module):
    """Memory-efficient SwiGLU MLP using tiled computation.

    Combines SwiGLU activation with tiled processing to handle very long
    sequences efficiently. The forward pass is recomputed during backward to
    save memory. ``config`` must expose ``hidden_size`` and ``intermediate_size``
    (and optionally ``hidden_act``, which must be ``silu`` or ``swish``).

    Args:
        config: Model configuration with ``hidden_size`` and ``intermediate_size``.
        num_shards: Number of shards to split the sequence into. If ``None``,
            automatically calculated as ``ceil(seqlen / hidden_size)``.
    """

    def __init__(self, config, num_shards: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Validate activation function
        if hasattr(config, "hidden_act") and config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"LigerTiledSwiGLUMLP requires SiLU/Swish activation, got {config.hidden_act}")

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(LigerSiLUMulFunction.apply(gate, up))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_params = [p for p in self.parameters() if p.requires_grad]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


@dataclass
class CrossEntropyOutput:
    loss: torch.Tensor
    z_loss: Optional[torch.Tensor] = None
    token_accuracy: Optional[torch.Tensor] = None
    predicted_tokens: Optional[torch.Tensor] = None


def liger_fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    loss, z_loss, token_accuracy, predicted_tokens = LigerFusedLinearCrossEntropyFunction.apply(
        input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    if not return_z_loss and not return_token_accuracy and not return_predicted_tokens:
        return loss
    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def LigerForCausalLMLoss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    final_logit_softcapping: Optional[float] = None,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs,
):
    """Drop-in replacement for ``transformers.loss.ForCausalLMLoss`` that fuses the
    final ``lm_head`` projection with the cross-entropy loss. Returns a scalar
    ``loss`` by default; returns a :class:`CrossEntropyOutput` when
    ``return_token_accuracy`` or ``return_predicted_tokens`` is set."""
    applicable_params = inspect.signature(liger_fused_linear_cross_entropy).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    hidden_states = hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1).to(hidden_states.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    result = liger_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        shift_labels,
        reduction=reduction,
        ignore_index=ignore_index,
        softcap=final_logit_softcapping,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        **kwargs,
    )

    if isinstance(result, CrossEntropyOutput):
        loss = result.loss
        token_accuracy = result.token_accuracy
        predicted_tokens = result.predicted_tokens
    else:
        loss = result
        token_accuracy = None
        predicted_tokens = None

    if reduction == "sum":
        loss = loss / num_items_in_batch

    if return_token_accuracy or return_predicted_tokens:
        return CrossEntropyOutput(
            loss=loss,
            token_accuracy=token_accuracy,
            predicted_tokens=predicted_tokens,
        )
    return loss


def liger_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply standard rotary positional embedding to ``q`` and ``k``."""
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def liger_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen2-VL multimodal rotary positional embedding (M-RoPE) to ``q`` and ``k``."""
    return LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section, unsqueeze_dim)


__all__ = [
    "LigerRMSNorm",
    "LigerLayerNorm",
    "LigerGroupNorm",
    "LigerDyT",
    "LigerCrossEntropyLoss",
    "LigerFusedLinearCrossEntropyLoss",
    "LigerJSD",
    "LigerKLDIVLoss",
    "LigerTVDLoss",
    "LigerSwiGLUMLP",
    "LigerGEGLUMLP",
    "LigerTiledGEGLUMLP",
    "LigerTiledSwiGLUMLP",
    "CrossEntropyOutput",
    "liger_fused_linear_cross_entropy",
    "LigerForCausalLMLoss",
    "liger_rotary_pos_emb",
    "liger_multimodal_rotary_pos_emb",
]

import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from .rms_norm import LigerRMSNormFunction
from .swiglu import LigerSiLUMulFunction


# NOTE: Not compile-friendly --> large deviations to the original implementation under compile
class LigerRMSNorm(nn.Module):
    weight: nn.Parameter
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            0,
            "llama",
            True,
            None,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LigerSwiGLUMLP(nn.Module):
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    can_torch_compile = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


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


# NOTE: We have this intentional graph break as we encounter issues such as IMAs and Cublas errors.
#       We know that this is an optimized kernel already so there is less ways to
#       fuse it either way; we rely on torch compile to go through the base model to optimize.
@torch.compiler.disable
def LigerForCausalLMLoss(
    logits: None,  # to match transformers signature
    labels: torch.Tensor,
    vocab_size: int,  # to match transformers signature
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    hidden_states: torch.Tensor | None = None,
    lm_head_weight: torch.Tensor | None = None,
    **kwargs,
):
    # To match signature we hide these behind the kwargs but we expect a few kwargs to exist
    hidden_size = kwargs.pop("hidden_size", None)
    final_logit_softcapping = kwargs.pop("final_logit_softcapping", None)

    if hidden_size is None or hidden_states is None or lm_head_weight is None:
        raise ValueError(
            f"`LigerForCausalLMLoss` requires the LLM's weight (found `{lm_head_weight is not None}`),"
            f"the last hidden state (found `{hidden_states is not None}`), and the `hidden_size`"
            f"(found `{hidden_size is not None}`). Please make sure to pass the necessary kwargs."
        )

    applicable_params = inspect.signature(liger_fused_linear_cross_entropy).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    hidden_states = hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1).to(hidden_states.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = liger_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        shift_labels,
        reduction=reduction,
        ignore_index=ignore_index,
        softcap=final_logit_softcapping,
        return_token_accuracy=False,
        return_predicted_tokens=False,
        **kwargs,
    )

    if reduction == "sum":
        loss = loss / num_items_in_batch

    return loss


# Add torch compile support for functions - in this case it's to allow this to be used
# but the function itself will not be compiled (see the note at the function)
LigerForCausalLMLoss.can_torch_compile = True


__all__ = [
    "LigerRMSNorm",
    "LigerSwiGLUMLP",
    "LigerForCausalLMLoss",
]

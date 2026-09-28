"""TDT (Token-and-Duration Transducer) loss CUDA kernel."""

from typing import Sequence, Union

import torch

from ._ops import ops

__all__ = ["tdt_loss"]

_REDUCTIONS = ("mean_volume", "mean_batch", "mean", "sum", "none")


def _last_dim_contiguous(x: torch.Tensor) -> torch.Tensor:
    # The kernels accept arbitrary strides for the (batch, T, U) dims, so slices of a
    # joint `(..., vocab_size + num_durations)` output can be passed without a copy.
    return x if x.stride(-1) == 1 else x.contiguous()


class TDTLoss(torch.autograd.Function):
    """Per-sample TDT loss (negative log-likelihood) with a CUDA forward and backward."""

    @staticmethod
    def forward(
        ctx,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        durations: torch.Tensor,
        blank_id: int,
        sigma: float,
    ) -> torch.Tensor:
        token_logits = _last_dim_contiguous(token_logits)
        duration_logits = _last_dim_contiguous(duration_logits)

        B, max_T, max_U, _ = token_logits.shape
        D = duration_logits.shape[-1]
        f32 = dict(device=token_logits.device, dtype=torch.float32)

        blank_lp = torch.empty(B, max_T, max_U, **f32)
        label_lp = torch.empty(B, max_T, max_U, **f32)
        dur_lp = torch.empty(B, max_T, max_U, D, **f32)
        token_lse = torch.empty(B, max_T, max_U, **f32)
        ops.tdt_logprobs_fwd(
            token_logits,
            duration_logits,
            targets,
            source_lengths,
            target_lengths,
            blank_id,
            sigma,
            blank_lp,
            label_lp,
            dur_lp,
            token_lse,
        )

        alphas = torch.empty(B, max_T, max_U, **f32)
        log_ll = torch.empty(B, **f32)
        ops.tdt_loss_fwd(blank_lp, label_lp, dur_lp, source_lengths, target_lengths, durations, alphas, log_ll)

        ctx.save_for_backward(
            token_logits,
            targets,
            source_lengths,
            target_lengths,
            durations,
            blank_lp,
            label_lp,
            dur_lp,
            token_lse,
            alphas,
            log_ll,
        )
        ctx.blank_id = blank_id
        ctx.duration_shape = duration_logits.shape
        ctx.duration_dtype = duration_logits.dtype
        return -log_ll

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (
            token_logits,
            targets,
            source_lengths,
            target_lengths,
            durations,
            blank_lp,
            label_lp,
            dur_lp,
            token_lse,
            alphas,
            log_ll,
        ) = ctx.saved_tensors
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return (None,) * 8

        betas = torch.empty_like(alphas)
        log_ll_bwd = torch.empty_like(log_ll)
        ops.tdt_loss_bwd(blank_lp, label_lp, dur_lp, source_lengths, target_lengths, durations, betas, log_ll_bwd)

        grad_token_logits = torch.empty(token_logits.shape, device=token_logits.device, dtype=token_logits.dtype)
        grad_duration_logits = torch.empty(ctx.duration_shape, device=token_logits.device, dtype=ctx.duration_dtype)
        ops.tdt_logits_grad(
            token_logits,
            targets,
            source_lengths,
            target_lengths,
            durations,
            blank_lp,
            label_lp,
            dur_lp,
            token_lse,
            alphas,
            betas,
            log_ll,
            grad_loss.float().contiguous(),
            ctx.blank_id,
            grad_token_logits,
            grad_duration_logits,
        )
        return grad_token_logits, grad_duration_logits, None, None, None, None, None, None


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    durations: Union[Sequence[int], torch.Tensor],
    blank_id: int,
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the TDT (Token-and-Duration Transducer) loss (https://arxiv.org/abs/2304.06795).

    Args:
        token_logits: Token logits of shape `(batch, T, U+1, vocab_size+1)`, in float32, float16 or bfloat16.
            Only the last dimension needs to be contiguous.
        duration_logits: Duration logits of shape `(batch, T, U+1, num_durations)`, same dtype as `token_logits`.
        targets: Target labels of shape `(batch, U)`.
        source_lengths: Encoder output lengths of shape `(batch,)`.
        target_lengths: Target lengths of shape `(batch,)`.
        durations: Duration values, e.g. `[0, 1, 2, 3, 4]`.
        blank_id: Blank token id.
        sigma: Logit undernormalization constant (see TDT paper). Defaults to `0.0`.
        reduction: One of `"mean_volume"`, `"mean_batch"`, `"mean"`, `"sum"` or `"none"`, mirroring NeMo's
            `RNNTLoss`. `"mean"` divides each loss by its target length before averaging over the batch.

    Returns:
        Scalar loss tensor (or per-example losses of shape `(batch,)` if `reduction="none"`). Samples without a
        valid alignment get an infinite loss and a zero gradient.
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f'Invalid reduction mode "{reduction}". Expected one of {", ".join(repr(r) for r in _REDUCTIONS)}.'
        )

    device = token_logits.device
    if isinstance(durations, torch.Tensor):
        durations = durations.to(device=device, dtype=torch.int32).contiguous()
    else:
        durations = torch.tensor(list(durations), device=device, dtype=torch.int32)
    targets = targets.to(device=device, dtype=torch.int32).contiguous()
    source_lengths = source_lengths.to(device=device, dtype=torch.int32).contiguous()
    target_lengths = target_lengths.to(device=device, dtype=torch.int32).contiguous()

    losses = TDTLoss.apply(
        token_logits,
        duration_logits,
        targets,
        source_lengths,
        target_lengths,
        durations,
        int(blank_id),
        float(sigma),
    )

    if reduction == "mean_volume":
        return losses.sum() / target_lengths.sum().float()
    if reduction == "mean_batch":
        return losses.mean()
    if reduction == "mean":
        return (losses / target_lengths.float()).mean()
    if reduction == "sum":
        return losses.sum()
    return losses

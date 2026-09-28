"""TDT (Token-and-Duration Transducer) loss: autograd function and functional API."""

from typing import Dict, Sequence, Tuple, Union

import torch

from ._ops import ops

_REDUCTIONS = ("mean_volume", "mean_batch", "mean", "sum", "none")


_DURATIONS_CACHE: Dict[Tuple[Tuple[int, ...], torch.device], torch.Tensor] = {}


def _durations_tensor(durations: Union[Sequence[int], torch.Tensor], device: torch.device) -> torch.Tensor:
    """Validated int32 tensor of durations on `device`, cached since the durations are fixed for a model."""
    if isinstance(durations, torch.Tensor):
        durations = durations.tolist()
    key = (tuple(int(d) for d in durations), device)
    if key not in _DURATIONS_CACHE:
        if any(d < 0 for d in key[0]):
            raise ValueError(f"Durations must be non-negative, got {list(key[0])}.")
        _DURATIONS_CACHE[key] = torch.tensor(key[0], device=device, dtype=torch.int32)
    return _DURATIONS_CACHE[key]


def _last_dim_contiguous(x: torch.Tensor) -> torch.Tensor:
    # The kernels accept arbitrary strides for the (batch, T, U) dims, so slices of a
    # joint `(..., vocab_size + num_durations)` output can be passed without a copy.
    return x if x.stride(-1) == 1 else x.contiguous()


class _TDTLossFunction(torch.autograd.Function):
    """Per-sample TDT loss (negative log-likelihood) with a CUDA forward and backward."""

    @staticmethod
    def forward(
        ctx,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
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
            logit_lengths,
            target_lengths,
            blank_id,
            sigma,
            blank_lp,
            label_lp,
            dur_lp,
            token_lse,
        )

        # The lattice recursions run in float64, see lattice.cu.
        f64 = dict(device=token_logits.device, dtype=torch.float64)
        alphas = torch.empty(B, max_T, max_U, **f64)
        log_ll = torch.empty(B, **f64)
        ops.tdt_loss_fwd(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations, alphas, log_ll)

        ctx.save_for_backward(
            token_logits,
            targets,
            logit_lengths,
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
        return -log_ll.float()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (
            token_logits,
            targets,
            logit_lengths,
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
        ops.tdt_loss_bwd(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations, betas)

        grad_token_logits = torch.empty(token_logits.shape, device=token_logits.device, dtype=token_logits.dtype)
        grad_duration_logits = torch.empty(ctx.duration_shape, device=token_logits.device, dtype=ctx.duration_dtype)
        ops.tdt_logits_grad(
            token_logits,
            targets,
            logit_lengths,
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
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    durations: Union[Sequence[int], torch.Tensor],
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the TDT (Token-and-Duration Transducer) loss (https://arxiv.org/abs/2304.06795).

    Args:
        token_logits: Token logits of shape `(batch, T, U+1, vocab_size+1)`, in float32, float16 or bfloat16.
            Only the last dimension needs to be contiguous.
        duration_logits: Duration logits of shape `(batch, T, U+1, num_durations)`, same dtype as `token_logits`.
        targets: Target labels of shape `(batch, U)`.
        logit_lengths: Encoder output lengths of shape `(batch,)`.
        target_lengths: Target lengths of shape `(batch,)`.
        blank_token_id: Blank token id.
        durations: Non-negative duration values, e.g. `[0, 1, 2, 3, 4]`.
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
    durations = _durations_tensor(durations, device)
    targets = targets.to(device=device, dtype=torch.int32).contiguous()
    logit_lengths = logit_lengths.to(device=device, dtype=torch.int32).contiguous()
    target_lengths = target_lengths.to(device=device, dtype=torch.int32).contiguous()

    losses = _TDTLossFunction.apply(
        token_logits,
        duration_logits,
        targets,
        logit_lengths,
        target_lengths,
        durations,
        int(blank_token_id),
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

"""TDT (Token-and-Duration Transducer) loss CUDA kernel."""

from typing import List, Union

import torch

from ._ops import ops


class TDTLoss(torch.autograd.Function):
    """Custom autograd function for TDT loss."""

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
        sigma: float = 0.0,
    ) -> torch.Tensor:
        B, max_T, max_U, V = token_logits.shape
        D = duration_logits.shape[3]
        device = token_logits.device

        token_logits = token_logits.contiguous().float()
        duration_logits = duration_logits.contiguous().float()
        targets = targets.contiguous().int()
        source_lengths = source_lengths.contiguous().int()
        target_lengths = target_lengths.contiguous().int()
        durations = durations.contiguous().int()

        blank_lp = torch.empty(B, max_T, max_U, device=device, dtype=torch.float32)
        label_lp = torch.empty(B, max_T, max_U, device=device, dtype=torch.float32)
        dur_lp = torch.empty(B, max_T, max_U, D, device=device, dtype=torch.float32)
        alphas = torch.full((B, max_T, max_U), -1e30, device=device, dtype=torch.float32)
        log_ll = torch.empty(B, device=device, dtype=torch.float32)

        ops.tdt_logprobs_fwd(
            token_logits, duration_logits, targets,
            source_lengths, target_lengths, blank_id, sigma,
            blank_lp, label_lp, dur_lp,
        )
        ops.tdt_loss_fwd(
            blank_lp, label_lp, dur_lp,
            source_lengths, target_lengths, durations,
            alphas, log_ll,
        )

        ctx.save_for_backward(
            token_logits, duration_logits, targets,
            source_lengths, target_lengths, durations,
            blank_lp, label_lp, dur_lp, alphas, log_ll,
        )
        ctx.blank_id = blank_id
        return -log_ll

    @staticmethod
    def backward(ctx, grad_output):
        (token_logits, duration_logits, targets,
         source_lengths, target_lengths, durations,
         blank_lp, label_lp, dur_lp, alphas, log_ll) = ctx.saved_tensors
        blank_id = ctx.blank_id

        B, max_T, max_U, V = token_logits.shape
        D = duration_logits.shape[3]
        device = token_logits.device

        betas = torch.full((B, max_T, max_U), -1e30, device=device, dtype=torch.float32)
        ll_bwd = torch.empty(B, device=device, dtype=torch.float32)
        ops.tdt_loss_bwd(
            blank_lp, label_lp, dur_lp,
            source_lengths, target_lengths, durations,
            betas, ll_bwd,
        )

        grad_blank = torch.zeros(B, max_T, max_U, device=device, dtype=torch.float32)
        grad_label = torch.zeros(B, max_T, max_U, device=device, dtype=torch.float32)
        grad_dur = torch.zeros(B, max_T, max_U, D, device=device, dtype=torch.float32)
        ops.tdt_loss_grad(
            alphas, betas, blank_lp, label_lp, dur_lp, log_ll,
            source_lengths, target_lengths, durations,
            grad_blank, grad_label, grad_dur,
        )

        grad_blank = grad_blank * grad_output.unsqueeze(-1).unsqueeze(-1)
        grad_label = grad_label * grad_output.unsqueeze(-1).unsqueeze(-1)
        grad_dur = grad_dur * grad_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        grad_token_logits = torch.zeros_like(token_logits)
        grad_duration_logits = torch.zeros_like(duration_logits)
        ops.tdt_logprobs_bwd(
            token_logits, duration_logits, targets,
            source_lengths, target_lengths, blank_id,
            grad_blank, grad_label, grad_dur,
            grad_token_logits, grad_duration_logits,
        )

        return grad_token_logits, grad_duration_logits, None, None, None, None, None, None


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    durations: Union[List[int], torch.Tensor],
    blank_id: int,
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute TDT (Token-and-Duration Transducer) loss using CUDA kernels.

    Args:
        token_logits: Token logits of shape (batch, T, U+1, vocab_size+1).
        duration_logits: Duration logits of shape (batch, T, U+1, num_durations).
        targets: Target labels of shape (batch, U).
        source_lengths: Encoder output lengths of shape (batch,).
        target_lengths: Target lengths of shape (batch,).
        durations: List or 1-D tensor of duration values (e.g. [0, 1, 2, 3, 4]).
        blank_id: Blank token id.
        sigma: Logit undernormalization constant (see TDT paper). Defaults to 0.0.
        reduction: Loss reduction method: "mean", "sum", or "none".

    Returns:
        Scalar loss tensor (or per-example losses if reduction="none").
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f'Invalid reduction mode "{reduction}". Expected one of "mean", "sum", or "none".')

    if isinstance(durations, (list, tuple)):
        durations = torch.tensor(durations, dtype=torch.int32, device=token_logits.device)
    else:
        durations = durations.to(device=token_logits.device, dtype=torch.int32).contiguous()

    per_sample_loss = TDTLoss.apply(
        token_logits, duration_logits, targets,
        source_lengths, target_lengths, durations,
        blank_id, sigma,
    )
    if reduction == "mean":
        return (per_sample_loss / target_lengths.float()).mean()
    elif reduction == "sum":
        return per_sample_loss.sum()
    return per_sample_loss

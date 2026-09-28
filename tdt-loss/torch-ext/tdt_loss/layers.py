from typing import Sequence, Union

import torch
import torch.nn as nn

from .loss import tdt_loss


class TDTLoss(nn.Module):
    """
    Stateless layer computing the TDT loss with the CUDA kernels. Its forward has the same signature as
    `transformers.loss.loss_tdt.tdt_loss`, so it can replace it through `kernels.kernelize`.
    """

    has_backward = True
    can_torch_compile = False

    def forward(
        self,
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
        return tdt_loss(
            token_logits,
            duration_logits,
            targets,
            logit_lengths,
            target_lengths,
            blank_token_id,
            durations,
            sigma=sigma,
            reduction=reduction,
        )

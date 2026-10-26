"""Pure PyTorch reference of the TDT loss, shared by the tests and the benchmark."""

import torch


def tdt_loss_reference(
    token_logits,
    duration_logits,
    targets,
    logit_lengths,
    target_lengths,
    blank_id,
    durations,
    sigma=0.0,
    dtype=torch.float32,
):
    """Per-sample TDT loss, as implemented in `transformers.loss.loss_tdt.tdt_loss` (computed in `dtype`)."""
    device = token_logits.device
    batch_size, max_t, max_u, _ = token_logits.shape

    token_log_probs = torch.log_softmax(token_logits.to(dtype), dim=-1) - sigma
    duration_log_probs = torch.log_softmax(duration_logits.to(dtype), dim=-1)

    log_alpha = torch.full((batch_size, max_t, max_u), float("-inf"), device=device, dtype=dtype)
    log_alpha[:, 0, 0] = 0.0
    blank_log_probs = token_log_probs[:, :, :, blank_id]
    if max_u > 1:
        targets_expanded = targets.long().unsqueeze(1).expand(-1, max_t, -1)
        label_log_probs = torch.gather(
            token_log_probs[:, :, : max_u - 1, :], dim=3, index=targets_expanded.unsqueeze(-1)
        ).squeeze(-1)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    for n in range(1, max_t + max_u - 1):
        u_indices = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=device)
        t_indices = n - u_indices
        candidates = []
        for i, dur in enumerate(durations):
            t_prev = t_indices - dur
            valid_t = t_prev >= 0
            if not valid_t.any():
                continue
            t_src = t_prev.clamp(min=0)
            if dur > 0:
                contrib = (
                    log_alpha[:, t_src, u_indices]
                    + blank_log_probs[:, t_src, u_indices]
                    + duration_log_probs[:, t_src, u_indices, i]
                )
                candidates.append(torch.where(valid_t.unsqueeze(0), contrib, neg_inf))
            valid_both = valid_t & (u_indices > 0)
            if valid_both.any():
                u_src = (u_indices - 1).clamp(min=0)
                u_src_label = u_src.clamp(max=max_u - 2) if max_u > 1 else u_src
                contrib = (
                    log_alpha[:, t_src, u_src]
                    + label_log_probs[:, t_src, u_src_label]
                    + duration_log_probs[:, t_src, u_src, i]
                )
                candidates.append(torch.where(valid_both.unsqueeze(0), contrib, neg_inf))
        if candidates:
            log_alpha[:, t_indices, u_indices] = torch.logsumexp(torch.stack(candidates), dim=0)

    batch_idx = torch.arange(batch_size, device=device)
    target_lengths = target_lengths.long()
    log_probs = torch.full((batch_size,), float("-inf"), device=device, dtype=dtype)
    for i, dur in enumerate(durations):
        if dur == 0:
            continue
        t_final = logit_lengths.long() - dur
        valid = t_final >= 0
        if not valid.any():
            continue
        t_clamped = t_final.clamp(min=0)
        terminal = (
            log_alpha[batch_idx, t_clamped, target_lengths]
            + token_log_probs[batch_idx, t_clamped, target_lengths, blank_id]
            + duration_log_probs[batch_idx, t_clamped, target_lengths, i]
        )
        log_probs = torch.where(valid, torch.logsumexp(torch.stack([log_probs, terminal]), dim=0), log_probs)
    return -log_probs

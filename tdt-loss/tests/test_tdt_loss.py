"""Tests for the TDT loss kernel.

The kernel is checked against two pure PyTorch references:
- `tdt_loss_reference`: the vectorized implementation used in transformers, whose autograd
  gradients are used to validate the kernel backward;
- `tdt_loss_naive`: a loop-based implementation that is easy to verify by hand.
"""

from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    import tdt_loss as tdt_loss_kernel
except ImportError:
    from kernels import get_local_kernel

    tdt_loss_kernel = get_local_kernel(repo_path=Path(__file__).parent.parent, package_name="tdt_loss")

DEVICE = torch.device("cuda")
DURATIONS = [0, 1, 2, 3, 4]
REDUCTIONS = ["mean_volume", "mean_batch", "mean", "sum", "none"]
# float32 lattice sums over ~1000 labels lose a few digits.
GRAD_ATOL_LONG = 1e-2


def reduce(losses, target_lengths, reduction):
    target_lengths = target_lengths.float()
    if reduction == "mean_volume":
        return losses.sum() / target_lengths.sum()
    if reduction == "mean_batch":
        return losses.mean()
    if reduction == "mean":
        return (losses / target_lengths).mean()
    if reduction == "sum":
        return losses.sum()
    return losses


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


def tdt_loss_naive(token_logits, duration_logits, targets, logit_lengths, target_lengths, blank_id, durations, sigma=0.0):
    """Per-sample TDT loss with explicit loops over the lattice, in float64 on the CPU."""
    token_lp = (torch.log_softmax(token_logits.double().cpu(), dim=-1) - sigma).tolist()
    dur_lp = torch.log_softmax(duration_logits.double().cpu(), dim=-1).tolist()
    targets = targets.tolist()
    neg_inf = float("-inf")

    def log_add(a, b):
        return torch.logaddexp(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)).item()

    losses = []
    for b in range(len(targets)):
        T, U = int(logit_lengths[b]), int(target_lengths[b])
        alpha = [[neg_inf] * (U + 1) for _ in range(T)]
        alpha[0][0] = 0.0
        for t in range(T):
            for u in range(U + 1):
                if t == 0 and u == 0:
                    continue
                for i, d in enumerate(durations):
                    t_src = t - d
                    if t_src < 0:
                        continue
                    if d > 0:
                        arc = alpha[t_src][u] + token_lp[b][t_src][u][blank_id] + dur_lp[b][t_src][u][i]
                        alpha[t][u] = log_add(alpha[t][u], arc)
                    if u > 0:
                        label = targets[b][u - 1]
                        arc = alpha[t_src][u - 1] + token_lp[b][t_src][u - 1][label] + dur_lp[b][t_src][u - 1][i]
                        alpha[t][u] = log_add(alpha[t][u], arc)
        log_ll = neg_inf
        for i, d in enumerate(durations):
            if d > 0 and T - d >= 0:
                t_src = T - d
                log_ll = log_add(log_ll, alpha[t_src][U] + token_lp[b][t_src][U][blank_id] + dur_lp[b][t_src][U][i])
        losses.append(-log_ll)
    return torch.tensor(losses, dtype=torch.float64)


def assert_grad_close(actual, expected, **kwargs):
    # The reference gets NaN gradients on lattice nodes that cannot be reached (e.g. u > t when there is no
    # zero duration), from the backward of a logsumexp over only -inf. The true gradient there is zero.
    torch.testing.assert_close(actual.to(expected.dtype), expected.nan_to_num(nan=0.0), **kwargs)


def make_inputs(B, T, U, V, durations, dtype=torch.float32, seed=0, joint=False):
    """Random inputs with variable lengths. `joint=True` returns slices of a single joint tensor,
    like `ParakeetForTDT` does."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    D = len(durations)
    logits = torch.randn(B, T, U + 1, V + D, generator=generator).to(DEVICE, dtype)
    if joint:
        token_logits, duration_logits = logits[..., :V], logits[..., V:]
    else:
        token_logits, duration_logits = logits[..., :V].contiguous(), logits[..., V:].contiguous()
    targets = torch.randint(1, V, (B, U), generator=generator).to(DEVICE, torch.int32)
    logit_lengths = torch.tensor([T - (b % 3) for b in range(B)], device=DEVICE, dtype=torch.int32)
    target_lengths = torch.tensor([U - (b % 2) for b in range(B)], device=DEVICE, dtype=torch.int32)
    return token_logits, duration_logits, targets, logit_lengths, target_lengths


def run_kernel(token_logits, duration_logits, targets, logit_lengths, target_lengths, durations, sigma, reduction):
    return tdt_loss_kernel.tdt_loss(
        token_logits,
        duration_logits,
        targets,
        logit_lengths,
        target_lengths,
        durations,
        0,
        sigma=sigma,
        reduction=reduction,
    )


@pytest.mark.parametrize("sigma", [0.0, 0.05])
def test_matches_naive(sigma):
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(3, 12, 5, 16, DURATIONS)
    expected = tdt_loss_naive(
        token_logits, duration_logits, targets, logit_lengths, target_lengths, 0, DURATIONS, sigma=sigma
    )
    losses = run_kernel(
        token_logits, duration_logits, targets, logit_lengths, target_lengths, DURATIONS, sigma, "none"
    )
    torch.testing.assert_close(losses.double().cpu(), expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("sigma", [0.0, 0.05])
@pytest.mark.parametrize("durations", [DURATIONS, [1, 2], [0, 1, 3, 5]])
def test_forward_backward_match_reference(durations, sigma, reduction):
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(4, 30, 8, 64, durations)

    ref_tok = token_logits.clone().requires_grad_(True)
    ref_dur = duration_logits.clone().requires_grad_(True)
    ref_losses = tdt_loss_reference(
        ref_tok, ref_dur, targets, logit_lengths, target_lengths, 0, durations, sigma=sigma
    )
    expected = reduce(ref_losses, target_lengths, reduction)
    expected.sum().backward()

    tok = token_logits.clone().requires_grad_(True)
    dur = duration_logits.clone().requires_grad_(True)
    loss = run_kernel(tok, dur, targets, logit_lengths, target_lengths, durations, sigma, reduction)
    loss.sum().backward()

    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)
    assert_grad_close(tok.grad, ref_tok.grad, atol=1e-5, rtol=1e-4)
    assert_grad_close(dur.grad, ref_dur.grad, atol=1e-5, rtol=1e-4)


def test_joint_logits_slices():
    """Non-contiguous slices of a joint output (as in ParakeetForTDT) are handled without a copy."""
    V = 128
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(
        3, 25, 6, V, DURATIONS, joint=True
    )
    assert not token_logits.is_contiguous() and not duration_logits.is_contiguous()

    joint = token_logits._base.clone().requires_grad_(True)
    loss = run_kernel(
        joint[..., :V], joint[..., V:], targets, logit_lengths, target_lengths, DURATIONS, 0.0, "mean"
    )
    loss.backward()

    ref_joint = token_logits._base.clone().requires_grad_(True)
    expected = reduce(
        tdt_loss_reference(
            ref_joint[..., :V], ref_joint[..., V:], targets, logit_lengths, target_lengths, 0, DURATIONS
        ),
        target_lengths,
        "mean",
    )
    expected.backward()

    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)
    assert_grad_close(joint.grad, ref_joint.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision(dtype):
    """Half-precision logits are read directly and gradients come back in the input dtype."""
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bfloat16 requires compute capability >= 8.0")
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(
        4, 30, 8, 256, DURATIONS, dtype=dtype
    )

    tok = token_logits.clone().requires_grad_(True)
    dur = duration_logits.clone().requires_grad_(True)
    loss = run_kernel(tok, dur, targets, logit_lengths, target_lengths, DURATIONS, 0.0, "mean")
    loss.backward()
    assert tok.grad.dtype == dtype and dur.grad.dtype == dtype

    # The reference upcasts the same half-precision values to float32.
    ref_tok = token_logits.float().requires_grad_(True)
    ref_dur = duration_logits.float().requires_grad_(True)
    expected = reduce(
        tdt_loss_reference(ref_tok, ref_dur, targets, logit_lengths, target_lengths, 0, DURATIONS),
        target_lengths,
        "mean",
    )
    expected.backward()

    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)
    assert_grad_close(tok.grad, ref_tok.grad, atol=1e-3, rtol=1e-2)
    assert_grad_close(dur.grad, ref_dur.grad, atol=1e-3, rtol=1e-2)


def test_parakeet_like_shapes():
    """Vocabulary and label lengths of the size used by Parakeet TDT."""
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(
        2, 60, 20, 8193, DURATIONS, joint=True
    )
    tok = token_logits.clone().requires_grad_(True)
    dur = duration_logits.clone().requires_grad_(True)
    losses = run_kernel(tok, dur, targets, logit_lengths, target_lengths, DURATIONS, 0.0, "none")
    losses.sum().backward()

    ref_tok = token_logits.clone().requires_grad_(True)
    ref_dur = duration_logits.clone().requires_grad_(True)
    expected = tdt_loss_reference(ref_tok, ref_dur, targets, logit_lengths, target_lengths, 0, DURATIONS)
    expected.sum().backward()

    torch.testing.assert_close(losses, expected, atol=1e-3, rtol=1e-4)
    assert_grad_close(tok.grad, ref_tok.grad, atol=1e-5, rtol=1e-4)
    assert_grad_close(dur.grad, ref_dur.grad, atol=1e-5, rtol=1e-4)


def test_long_targets():
    """More labels than threads in a block (the lattice kernels stride over u)."""
    U = 1100
    token_logits, duration_logits, targets, logit_lengths, target_lengths = make_inputs(1, 6, U, 8, DURATIONS)
    tok = token_logits.clone().requires_grad_(True)
    loss = run_kernel(tok, duration_logits, targets, logit_lengths, target_lengths, DURATIONS, 0.0, "sum")
    loss.backward()

    # The log-likelihood is in the thousands, so compare against a float64 reference.
    ref_tok = token_logits.double().requires_grad_(True)
    expected = tdt_loss_reference(
        ref_tok, duration_logits, targets, logit_lengths, target_lengths, 0, DURATIONS, dtype=torch.float64
    ).sum()
    expected.backward()

    torch.testing.assert_close(loss.double(), expected, atol=1e-2, rtol=1e-5)
    assert_grad_close(tok.grad, ref_tok.grad, atol=GRAD_ATOL_LONG, rtol=0)


def test_empty_targets():
    """Samples without labels: the only path is a sequence of blanks."""
    token_logits, duration_logits, _, logit_lengths, _ = make_inputs(2, 10, 0, 16, DURATIONS)
    targets = torch.empty(2, 0, device=DEVICE, dtype=torch.int32)
    target_lengths = torch.zeros(2, device=DEVICE, dtype=torch.int32)

    losses = run_kernel(token_logits, duration_logits, targets, logit_lengths, target_lengths, DURATIONS, 0.0, "none")
    expected = tdt_loss_naive(token_logits, duration_logits, targets, logit_lengths, target_lengths, 0, DURATIONS)
    torch.testing.assert_close(losses.double().cpu(), expected, atol=1e-4, rtol=1e-5)


def test_infeasible_alignment():
    """No valid path: the loss is infinite and the gradient is zero rather than NaN."""
    durations = [0, 4]
    token_logits, duration_logits, targets, _, target_lengths = make_inputs(2, 6, 3, 16, durations)
    # Sample 0 emits all labels at t=0 and exits with a blank of duration 4; sample 1 has T=3 < 4.
    logit_lengths = torch.tensor([4, 3], device=DEVICE, dtype=torch.int32)

    tok = token_logits.clone().requires_grad_(True)
    losses = run_kernel(tok, duration_logits, targets, logit_lengths, target_lengths, durations, 0.0, "none")
    assert torch.isfinite(losses[0]) and torch.isinf(losses[1])
    losses.backward(torch.ones_like(losses))
    assert torch.isfinite(tok.grad).all()
    assert tok.grad[0].abs().sum() > 0
    assert (tok.grad[1] == 0).all()


def test_deterministic():
    inputs = make_inputs(4, 30, 8, 64, DURATIONS)
    grads = []
    for _ in range(2):
        tok = inputs[0].clone().requires_grad_(True)
        run_kernel(tok, *inputs[1:], DURATIONS, 0.0, "mean").backward()
        grads.append(tok.grad)
    assert torch.equal(grads[0], grads[1])


def test_invalid_reduction():
    inputs = make_inputs(1, 5, 2, 4, [0, 1])
    with pytest.raises(ValueError, match="Invalid reduction"):
        run_kernel(*inputs, [0, 1], 0.0, "invalid")

"""Tests for TDT loss kernel, validated against a naive loop-based reference."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    from tdt_loss import tdt_loss
except ImportError as e:
    pytest.skip(f"tdt_loss not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Naive loop-based reference (no vectorisation, easy to verify by hand)
# ---------------------------------------------------------------------------

def _tdt_loss_naive(
    token_logits,       # (B, T, U, V)
    duration_logits,    # (B, T, U, D)
    targets,            # (B, U-1)
    source_lengths,     # (B,)
    target_lengths,     # (B,)
    durations,          # list[int]
    blank_id,
    sigma=0.0,
):
    """Single-sample, triple-nested-loop reference. Returns per-sample losses."""
    B, max_T, max_U, V = token_logits.shape
    D = len(durations)
    device = token_logits.device

    token_logits = token_logits.float()
    duration_logits = duration_logits.float()

    tok_lp = torch.log_softmax(token_logits, dim=-1) - sigma
    dur_lp = torch.log_softmax(duration_logits, dim=-1)

    losses = []
    for b in range(B):
        T_b = int(source_lengths[b].item())
        U_b = int(target_lengths[b].item()) + 1

        alpha = torch.full((T_b, U_b), float("-inf"), device=device, dtype=torch.float64)
        alpha[0, 0] = 0.0

        for t in range(T_b):
            for u in range(U_b):
                if t == 0 and u == 0:
                    continue
                val = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
                for i, dur in enumerate(durations):
                    t_src = t - dur
                    if t_src < 0:
                        continue
                    # Blank arc (dur > 0): (t_src, u) -> (t, u)
                    if dur > 0:
                        arc = alpha[t_src, u] + tok_lp[b, t_src, u, blank_id] + dur_lp[b, t_src, u, i]
                        val = torch.logaddexp(val, arc.to(torch.float64))
                    # Label arc (any dur): (t_src, u-1) -> (t, u)
                    if u > 0:
                        label = int(targets[b, u - 1].item())
                        arc = alpha[t_src, u - 1] + tok_lp[b, t_src, u - 1, label] + dur_lp[b, t_src, u - 1, i]
                        val = torch.logaddexp(val, arc.to(torch.float64))
                alpha[t, u] = val

        # Terminal: blank arcs with dur > 0 that exit the lattice
        ll = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
        for i, dur in enumerate(durations):
            if dur == 0:
                continue
            t_src = T_b - dur
            if t_src < 0:
                continue
            arc = alpha[t_src, U_b - 1] + tok_lp[b, t_src, U_b - 1, blank_id] + dur_lp[b, t_src, U_b - 1, i]
            ll = torch.logaddexp(ll, arc.to(torch.float64))

        losses.append(-ll)

    return torch.stack(losses)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.0, 0.05])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_tdt_loss_forward(sigma, reduction):
    """Forward loss matches the naive loop-based reference."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, U_labels, V = 4, 30, 8, 32
    durations = [0, 1, 2, 3, 4]
    D = len(durations)
    max_U = U_labels + 1
    blank_id = 0

    token_logits = torch.randn(B, T, max_U, V, device=device)
    duration_logits = torch.randn(B, T, max_U, D, device=device)
    targets = torch.randint(1, V, (B, U_labels), device=device)
    source_lengths = torch.tensor([T, T - 2, T - 5, T], device=device, dtype=torch.int32)
    target_lengths = torch.tensor(
        [U_labels, U_labels - 1, U_labels - 2, U_labels], device=device, dtype=torch.int32
    )

    # Naive reference (float64 for accuracy)
    ref_per_sample = _tdt_loss_naive(
        token_logits, duration_logits, targets,
        source_lengths, target_lengths, durations, blank_id,
        sigma=sigma,
    ).float()

    # Kernel under test
    kernel_out = tdt_loss(
        token_logits, duration_logits, targets,
        source_lengths, target_lengths, durations,
        blank_id, sigma=sigma, reduction=reduction,
    )

    if reduction == "none":
        ref = ref_per_sample
    elif reduction == "mean":
        ref = (ref_per_sample / target_lengths.float()).mean()
    else:
        ref = ref_per_sample.sum()

    torch.testing.assert_close(kernel_out.cpu(), ref.cpu(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sigma", [0.0, 0.05])
def test_tdt_loss_backward(sigma):
    """Backward gradients are finite and consistent between two identical calls."""
    torch.manual_seed(123)
    device = torch.device("cuda")

    B, T, U_labels, V = 2, 20, 5, 16
    durations = [0, 1, 2, 3]
    D = len(durations)
    max_U = U_labels + 1
    blank_id = 0

    tok_data = torch.randn(B, T, max_U, V, device=device, dtype=torch.float32)
    dur_data = torch.randn(B, T, max_U, D, device=device, dtype=torch.float32)
    targets = torch.randint(1, V, (B, U_labels), device=device)
    source_lengths = torch.tensor([T, T - 3], device=device, dtype=torch.int32)
    target_lengths = torch.tensor([U_labels, U_labels - 1], device=device, dtype=torch.int32)

    # Run 1
    tok1 = tok_data.clone().requires_grad_(True)
    dur1 = dur_data.clone().requires_grad_(True)
    loss1 = tdt_loss(
        tok1, dur1, targets, source_lengths, target_lengths,
        durations, blank_id, sigma=sigma, reduction="mean",
    )
    loss1.backward()

    # Run 2 (should produce identical gradients)
    tok2 = tok_data.clone().requires_grad_(True)
    dur2 = dur_data.clone().requires_grad_(True)
    loss2 = tdt_loss(
        tok2, dur2, targets, source_lengths, target_lengths,
        durations, blank_id, sigma=sigma, reduction="mean",
    )
    loss2.backward()

    # Losses should match
    torch.testing.assert_close(loss1, loss2)

    # Gradients should be identical and finite
    assert torch.isfinite(tok1.grad).all(), "token_logits gradients contain non-finite values"
    assert torch.isfinite(dur1.grad).all(), "duration_logits gradients contain non-finite values"
    torch.testing.assert_close(tok1.grad, tok2.grad)
    torch.testing.assert_close(dur1.grad, dur2.grad)


def test_tdt_loss_batch_size_one():
    """Sanity check with batch size 1."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    B, T, U_labels, V = 1, 15, 4, 10
    durations = [0, 1, 2]
    D = len(durations)
    max_U = U_labels + 1
    blank_id = 0

    token_logits = torch.randn(B, T, max_U, V, device=device)
    duration_logits = torch.randn(B, T, max_U, D, device=device)
    targets = torch.randint(1, V, (B, U_labels), device=device)
    source_lengths = torch.tensor([T], device=device, dtype=torch.int32)
    target_lengths = torch.tensor([U_labels], device=device, dtype=torch.int32)

    ref = _tdt_loss_naive(
        token_logits, duration_logits, targets,
        source_lengths, target_lengths, durations, blank_id,
    ).float()

    out = tdt_loss(
        token_logits, duration_logits, targets,
        source_lengths, target_lengths, durations,
        blank_id, reduction="none",
    )
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-4, rtol=1e-4)


def test_tdt_loss_invalid_reduction():
    """Should raise on invalid reduction."""
    device = torch.device("cuda")
    B, T, U, V, D = 1, 5, 3, 4, 2
    with pytest.raises(ValueError, match="Invalid reduction"):
        tdt_loss(
            torch.randn(B, T, U, V, device=device),
            torch.randn(B, T, U, D, device=device),
            torch.randint(1, V, (B, U - 1), device=device),
            torch.tensor([T], device=device, dtype=torch.int32),
            torch.tensor([U - 1], device=device, dtype=torch.int32),
            [0, 1], 0, reduction="invalid",
        )

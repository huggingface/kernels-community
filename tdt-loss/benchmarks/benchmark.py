import torch

from kernels.benchmark import Benchmark

DURATIONS = [0, 1, 2, 3, 4]


def tdt_loss_reference(
    token_logits, duration_logits, targets, logit_lengths, target_lengths, blank_id, durations, sigma=0.0
):
    """Per-sample TDT loss, as implemented in `transformers.loss.loss_tdt.tdt_loss`."""
    device = token_logits.device
    batch_size, max_t, max_u, _ = token_logits.shape

    token_log_probs = torch.log_softmax(token_logits.float(), dim=-1) - sigma
    duration_log_probs = torch.log_softmax(duration_logits.float(), dim=-1)

    log_alpha = torch.full((batch_size, max_t, max_u), float("-inf"), device=device)
    log_alpha[:, 0, 0] = 0.0
    blank_log_probs = token_log_probs[:, :, :, blank_id]
    if max_u > 1:
        targets_expanded = targets.long().unsqueeze(1).expand(-1, max_t, -1)
        label_log_probs = torch.gather(
            token_log_probs[:, :, : max_u - 1, :], dim=3, index=targets_expanded.unsqueeze(-1)
        ).squeeze(-1)

    neg_inf = torch.tensor(float("-inf"), device=device)
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
    log_probs = torch.full((batch_size,), float("-inf"), device=device)
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


class TDTLossBenchmark(Benchmark):
    seed: int = 42

    def _make_inputs(self, B, T, U, V, dtype):
        D = len(DURATIONS)
        # Slices of a joint output, as produced by ParakeetForTDT.
        self.logits = torch.randn(B, T, U + 1, V + D, device=self.device, dtype=dtype, requires_grad=True)
        self.targets = torch.randint(1, V, (B, U), device=self.device, dtype=torch.int32)
        self.logit_lengths = torch.full((B,), T, device=self.device, dtype=torch.int32)
        self.target_lengths = torch.full((B,), U, device=self.device, dtype=torch.int32)
        self.V = V

    def _run(self):
        self.logits.grad = None
        losses = self.kernel.tdt_loss(
            self.logits[..., : self.V],
            self.logits[..., self.V :],
            self.targets,
            self.logit_lengths,
            self.target_lengths,
            DURATIONS,
            0,
            reduction="none",
        )
        losses.sum().backward()
        self.out = losses.detach()

    def _reference(self):
        with torch.no_grad():
            return tdt_loss_reference(
                self.logits[..., : self.V],
                self.logits[..., self.V :],
                self.targets,
                self.logit_lengths,
                self.target_lengths,
                0,
                DURATIONS,
            )

    # Small vocabulary, short utterances.
    def setup(self):
        self._make_inputs(B=8, T=100, U=20, V=1025, dtype=torch.float32)

    def benchmark_base(self):
        self._run()

    def verify_base(self) -> torch.Tensor:
        return self._reference()

    # Parakeet TDT v3 sizes (8192 tokens + blank), float32.
    def setup_large(self):
        self._make_inputs(B=4, T=200, U=40, V=8193, dtype=torch.float32)

    def benchmark_large(self):
        self._run()

    def verify_large(self) -> torch.Tensor:
        return self._reference()

    # Same sizes with bfloat16 logits.
    def setup_large_bf16(self):
        self._make_inputs(B=4, T=200, U=40, V=8193, dtype=torch.bfloat16)

    def benchmark_large_bf16(self):
        self._run()

    def verify_large_bf16(self) -> torch.Tensor:
        return self._reference()

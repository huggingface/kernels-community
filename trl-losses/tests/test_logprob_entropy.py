import pytest
import torch

import kernels


trl_losses = kernels.get_kernel("kernels-community/trl-losses", version=1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = get_device()
requires_accelerator = pytest.mark.skipif(
    DEVICE.type == "cpu", reason="the kernel needs an accelerator"
)


def reference(logits, index):
    logits = logits.float()
    logprobs = logits.log_softmax(-1)
    selected_logprobs = logprobs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1)
    return selected_logprobs, entropy


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("logprob_weight", "entropy_weight"), [(2, None), (None, 0.5), (2, 0.5)]
)
def test_forward_and_backward(dtype, logprob_weight, entropy_weight):
    # Cross two full Triton blocks and leave a partial final block.
    vocab_size = 2053
    base_logits = torch.randn(2, 5, vocab_size, device=DEVICE, dtype=dtype)
    base_logits[..., :1024].add_(20)
    # Exercise online normalization across blocks with sharply different maxima.
    base_logits[..., 1024:2048].sub_(20)
    base_logits.requires_grad_()
    # TRL slices sequence logits and token IDs without making them contiguous.
    logits = base_logits[:, 1:4]
    index = torch.randint(vocab_size, (2, 5), device=DEVICE)[:, 1:4]
    index[0, 0] = vocab_size - 1  # select a target from the partial block

    logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(logits, index)
    completion_mask = torch.tensor(
        [[1, 0, 1], [0, 1, 1]], device=DEVICE, dtype=torch.bool
    )
    logprobs[~completion_mask] = 0.0  # TRL masks prompt and padding tokens in place
    loss = 0.0
    if logprob_weight is not None:
        loss = loss + logprob_weight * logprobs.sum()
    if entropy_weight is not None:
        loss = loss + entropy_weight * entropy.sum()
    loss.backward()
    actual_grad = base_logits.grad.clone()

    reference_logits = base_logits.detach().clone().requires_grad_()
    reference_logprobs, reference_entropy = reference(reference_logits[:, 1:4], index)
    reference_logprobs[~completion_mask] = 0.0
    reference_loss = 0.0
    if logprob_weight is not None:
        reference_loss = reference_loss + logprob_weight * reference_logprobs.sum()
    if entropy_weight is not None:
        reference_loss = reference_loss + entropy_weight * reference_entropy.sum()
    reference_loss.backward()

    torch.testing.assert_close(logprobs, reference_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, reference_entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_grad, reference_logits.grad, rtol=1e-3, atol=1e-3)


@requires_accelerator
@pytest.mark.kernels_ci
def test_torch_compile_fullgraph():
    logits = torch.randn(
        2, 3, 257, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    index = torch.randint(257, (2, 3), device=DEVICE)

    def loss(logits, index):
        logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(logits, index)
        return (logprobs + 0.1 * entropy).sum()

    torch.compile(loss, fullgraph=True)(logits, index).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

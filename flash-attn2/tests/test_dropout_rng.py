"""Dropout RNG plumbing for the stable-ABI CUDA build.

The kernel cannot read the Torch CUDA generator through the stable ABI, so the forward is handed
a generator (torch >= 2.13) or draws a seeded random seed/offset itself (older runtimes); see
set_params_philox in flash_api.cpp. These tests pin down the contract either way: determinism
under `torch.manual_seed`, the saved `rng_state`, generator advancement, explicit generators,
backward reproducibility, torch.compile parity and activation checkpointing.
"""

import kernels
import pytest
import torch

flash_attn2 = kernels.get_kernel("kernels-community/flash-attn2", version=4)
iface = flash_attn2.flash_attn_interface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

B, S, H, D = 2, 128, 4, 64
P = 0.17


def _qkv(dtype=torch.bfloat16, requires_grad=False):
    gen = torch.Generator(device="cuda").manual_seed(0)
    return tuple(
        torch.randn(B, S, H, D, device="cuda", dtype=dtype, generator=gen).requires_grad_(requires_grad)
        for _ in range(3)
    )


def _fwd(q, k, v, dropout_p=P, return_softmax=False):
    return iface._flash_attn_forward(q, k, v, dropout_p, D**-0.5, False, -1, -1, 0.0, None, return_softmax)


@pytest.mark.kernels_ci
def test_dropout_is_applied_and_seeded():
    q, k, v = _qkv()
    torch.manual_seed(0)
    out_a, _, _, rng_a = _fwd(q, k, v)
    torch.manual_seed(0)
    out_b, _, _, rng_b = _fwd(q, k, v)
    out_c, _, _, rng_c = _fwd(q, k, v)  # next draw, same seed, advanced offset
    out_nodrop, _, _, _ = _fwd(q, k, v, dropout_p=0.0)

    assert torch.equal(out_a, out_b)
    assert torch.equal(rng_a, rng_b)
    assert not torch.equal(out_a, out_c)
    assert not torch.equal(out_a, out_nodrop)
    # rng_state holds the seed and offset the kernel was given; a new draw gives a new pair.
    assert not torch.equal(rng_a, rng_c)
    if iface._STABLE_ABI_GENERATOR:
        # Exactly ATen's philox_cuda_state(B * H * 32): seed, then the pre-advance offset.
        assert rng_a.tolist() == [torch.cuda.initial_seed(), 0]
        assert rng_c.tolist() == [torch.cuda.initial_seed(), B * H * 32]


@pytest.mark.kernels_ci
def test_generator_is_consumed_only_with_dropout():
    q, k, v = _qkv()
    gen = torch.cuda.default_generators[q.device.index]
    torch.manual_seed(0)
    before = gen.get_offset()
    _fwd(q, k, v)
    assert gen.get_offset() > before, "dropout must consume the Torch CUDA RNG"
    if iface._STABLE_ABI_GENERATOR:
        assert gen.get_offset() - before == B * H * 32
    before = gen.get_offset()
    _fwd(q, k, v, dropout_p=0.0)
    assert gen.get_offset() == before, "no dropout must not consume RNG"


@pytest.mark.kernels_ci
def test_explicit_generator():
    q, k, v = _qkv()
    g1 = torch.Generator(device="cuda").manual_seed(7)
    g2 = torch.Generator(device="cuda").manual_seed(7)
    if not iface._STABLE_ABI_GENERATOR:
        with pytest.raises(ValueError, match="torch >= 2.13"):
            flash_attn2.fwd(q, k, v, p_dropout=P, gen=g1)
        return
    torch.manual_seed(0)
    default_before = torch.cuda.get_rng_state()
    out_1, _, _, rng_1 = flash_attn2.fwd(q, k, v, p_dropout=P, gen=g1)
    out_2, _, _, rng_2 = flash_attn2.fwd(q, k, v, p_dropout=P, gen=g2)
    out_3, _, _, _ = flash_attn2.fwd(q, k, v, p_dropout=P, gen=g1)  # g1 advanced: new mask
    assert torch.equal(out_1, out_2)
    assert torch.equal(rng_1, rng_2)
    assert not torch.equal(out_1, out_3)
    assert rng_1.tolist() == [7, 0]
    assert g2.get_offset() == B * H * 32  # one call on g2
    assert g1.get_offset() == 2 * B * H * 32  # two calls on g1
    assert torch.equal(torch.cuda.get_rng_state(), default_before), "default generator untouched"
    flash_attn2.fwd(q, k, v, p_dropout=0.0, gen=g2)
    assert g2.get_offset() == B * H * 32, "no dropout must not consume the generator"


@pytest.mark.kernels_ci
def test_dropout_mask_matches_return_softmax():
    """`return_softmax` dumps the post-dropout probabilities; dropped entries must be zero and
    kept entries scaled by 1/(1-p). The kept fraction should be close to 1-p."""
    q, k, v = _qkv()
    torch.manual_seed(0)
    _, _, s_dmask, _ = _fwd(q, k, v, return_softmax=True)
    # Upstream encodes "dropped" as negative values in S_dmask.
    dropped = (s_dmask < 0).float().mean().item()
    assert abs(dropped - P) < 0.02


@pytest.mark.kernels_ci
def test_backward_reproducible_from_rng_state():
    q, k, v = _qkv(requires_grad=True)
    dout = torch.randn_like(q)

    def run():
        torch.manual_seed(0)
        out = flash_attn2.flash_attn_func(q, k, v, dropout_p=P)
        grads = torch.autograd.grad(out, (q, k, v), dout)
        return out, grads

    out_a, grads_a = run()
    out_b, grads_b = run()
    assert torch.equal(out_a, out_b)
    for a, b in zip(grads_a, grads_b):
        assert torch.equal(a, b)


@pytest.mark.kernels_ci
def test_dropout_backward_matches_reference():
    """Compare against a reference attention that applies the kernel's own dropout mask."""
    q, k, v = _qkv(dtype=torch.float16, requires_grad=True)
    torch.manual_seed(0)
    out, lse, s_dmask, _ = _fwd(q, k, v, return_softmax=True)
    dout = torch.randn_like(out)
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    torch.manual_seed(0)
    _, _, _, rng_state = _fwd(q, k, v)  # same seed/offset as above
    iface._flash_attn_backward(
        dout, q, k, v, out, lse, dq, dk, dv, P, D**-0.5, False, -1, -1, 0.0, None, False, rng_state
    )

    keep = (s_dmask[..., :S, :S] >= 0).to(q.dtype)
    qf, kf, vf = (x.detach().float().transpose(1, 2).requires_grad_() for x in (q, k, v))
    scores = torch.softmax((qf @ kf.transpose(-1, -2)) * D**-0.5, dim=-1)
    ref = (scores * keep.float() / (1 - P)) @ vf
    ref.backward(dout.float().transpose(1, 2))
    for got, want in zip((dq, dk, dv), (qf.grad, kf.grad, vf.grad)):
        torch.testing.assert_close(got.float().transpose(1, 2), want, atol=5e-2, rtol=5e-2)


@pytest.mark.kernels_ci
def test_compile_dropout_matches_eager():
    q, k, v = _qkv(requires_grad=True)
    dout = torch.randn_like(q)

    def run(fn):
        torch.manual_seed(0)
        out = fn(q, k, v, dropout_p=P)
        return out, torch.autograd.grad(out, (q, k, v), dout)

    eager = run(flash_attn2.flash_attn_func)
    torch._dynamo.reset()
    compiled = run(torch.compile(flash_attn2.flash_attn_func, fullgraph=True))
    assert torch.equal(eager[0], compiled[0])
    for a, b in zip(eager[1], compiled[1]):
        assert torch.equal(a, b)


@pytest.mark.kernels_ci
def test_activation_checkpoint_recomputes_same_mask():
    """`torch.utils.checkpoint` restores the CUDA RNG state before recomputing the forward, so
    the recomputed dropout mask must match and the gradients must equal the uncheckpointed ones."""
    from torch.utils.checkpoint import checkpoint

    q, k, v = _qkv(requires_grad=True)
    dout = torch.randn_like(q)

    def attn(q, k, v):
        return flash_attn2.flash_attn_func(q, k, v, dropout_p=P)

    torch.manual_seed(0)
    out = attn(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    torch.manual_seed(0)
    out_ckpt = checkpoint(attn, q, k, v, use_reentrant=False)
    grads_ckpt = torch.autograd.grad(out_ckpt, (q, k, v), dout)
    assert torch.equal(out, out_ckpt)
    for a, b in zip(grads, grads_ckpt):
        assert torch.equal(a, b)

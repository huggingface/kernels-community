import kernels
import pytest
import torch

from .test_flash_attn import generate_qkv

flash_attn2 = kernels.get_kernel("kernels-community/flash-attn2", version=4)
iface = flash_attn2.flash_attn_interface

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# These ops also return scratch tensors that are partly uninitialized (`rng_state` without dropout,
# the padded tail of `softmax_d`), which makes opcheck's eager vs. AOT output comparison flaky, so
# that check is covered by the compile tests below instead.
OPCHECK_TEST_UTILS = ("test_schema", "test_autograd_registration", "test_faketensor")

# Every test runs once against the dense ops and once against the varlen ops.
VARLEN = pytest.mark.parametrize("varlen", [False, True], ids=["dense", "varlen"])


def _ops(varlen):
    if varlen:
        return iface._flash_attn_varlen_forward, iface._flash_attn_varlen_backward
    return iface._flash_attn_forward, iface._flash_attn_backward


def _inputs(varlen):
    """q, k, v, dout and, for varlen, (cu_seqlens, max_seqlen) of an unpadded batch."""
    torch.manual_seed(0)
    q, k, v, dout = (torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    if not varlen:
        return q, k, v, dout, None
    q, k, v, cu_seqlens, _, max_seqlen, *_ = generate_qkv(q, k, v)
    # generate_qkv sets requires_grad; these tests call the raw ops without autograd.
    return q.detach(), k.detach(), v.detach(), dout.flatten(0, 1), (cu_seqlens, max_seqlen)


def _fwd_args(q, k, v, seqlens):
    scale = q.shape[-1] ** -0.5
    if seqlens is None:
        return (q, k, v, 0.0, scale, False, -1, -1, 0.0, None, False)
    cu_seqlens, max_seqlen = seqlens
    return (q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, scale, False)


def _bwd_args(dout, q, k, v, out, lse, grads, rng_state, seqlens):
    scale = q.shape[-1] ** -0.5
    tail = (0.0, scale, False, -1, -1, 0.0, None, False, rng_state)
    if seqlens is None:
        return (dout, q, k, v, out, lse, *grads, *tail)
    cu_seqlens, max_seqlen = seqlens
    return (dout, q, k, v, out, lse, *grads, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, *tail)


def _fwd_bwd(q, k, v, dout, seqlens):
    fwd, bwd = _ops(seqlens is not None)
    out, lse, _, rng_state = fwd(*_fwd_args(q, k, v, seqlens))
    grads = tuple(torch.empty_like(x) for x in (q, k, v))
    bwd(*_bwd_args(dout, q, k, v, out, lse, grads, rng_state, seqlens))
    return out, lse, *grads


@VARLEN
def test_low_level_ops_are_custom_ops(varlen):
    for op in _ops(varlen):
        assert isinstance(op, torch._library.custom_ops.CustomOpDef)


def test_upstream_aliases():
    for name in ("flash_attn_forward", "flash_attn_backward", "flash_attn_varlen_forward", "flash_attn_varlen_backward"):
        alias = getattr(iface, f"_wrapped_{name}")
        op = getattr(iface, f"_{name}")
        assert alias is getattr(flash_attn2.flash_attn_ops, f"_{name}")
        assert alias._qualified_op_name == op._qualname


@VARLEN
def test_opcheck(varlen):
    q, k, v, dout, seqlens = _inputs(varlen)
    fwd, bwd = _ops(varlen)
    fwd_args = _fwd_args(q, k, v, seqlens)
    torch.library.opcheck(fwd, fwd_args, test_utils=OPCHECK_TEST_UTILS)

    out, lse, _, rng_state = fwd(*fwd_args)
    grads = tuple(torch.empty_like(x) for x in (q, k, v))
    bwd_args = _bwd_args(dout, q, k, v, out, lse, grads, rng_state, seqlens)
    torch.library.opcheck(bwd, bwd_args, test_utils=OPCHECK_TEST_UTILS)


@VARLEN
def test_compile_fullgraph_matches_eager(varlen):
    args = _inputs(varlen)
    expected = _fwd_bwd(*args)
    actual = torch.compile(_fwd_bwd, fullgraph=True)(*args)
    for e, a in zip(expected, actual):
        torch.testing.assert_close(a, e)

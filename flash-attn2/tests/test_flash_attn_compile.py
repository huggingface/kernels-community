import kernels
import pytest
import torch

flash_attn2 = kernels.get_kernel("kernels-community/flash-attn2", version=4)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# These ops also return scratch tensors that are partly uninitialized (`rng_state` without dropout,
# the padded tail of `softmax_d`), which makes opcheck's eager vs. AOT output comparison flaky, so
# that check is covered by the compile tests below instead.
OPCHECK_TEST_UTILS = ("test_schema", "test_autograd_registration", "test_faketensor")


def _qkv(shape, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    return tuple(torch.randn(shape, device=device, dtype=dtype) for _ in range(4))


def _fwd_bwd(q, k, v, dout):
    out, lse, _, rng_state = flash_attn2.flash_attn_interface._flash_attn_forward(
        q, k, v, 0.0, q.shape[-1] ** -0.5, False, -1, -1, 0.0, None, False
    )
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    flash_attn2.flash_attn_interface._flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        0.0,
        q.shape[-1] ** -0.5,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state,
    )
    return out, lse, dq, dk, dv


def _varlen_fwd_bwd(q, k, v, dout, cu_seqlens, max_seqlen):
    scale = q.shape[-1] ** -0.5
    out, lse, _, rng_state = flash_attn2.flash_attn_interface._flash_attn_varlen_forward(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, scale, False
    )
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    flash_attn2.flash_attn_interface._flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        0.0,
        scale,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state,
    )
    return out, lse, dq, dk, dv


def test_low_level_ops_are_custom_ops():
    for op in (
        flash_attn2.flash_attn_interface._flash_attn_forward,
        flash_attn2.flash_attn_interface._flash_attn_backward,
        flash_attn2.flash_attn_interface._flash_attn_varlen_forward,
        flash_attn2.flash_attn_interface._flash_attn_varlen_backward,
    ):
        assert isinstance(op, torch._library.custom_ops.CustomOpDef)


def test_upstream_aliases():
    iface = flash_attn2.flash_attn_interface
    for name in ("flash_attn_forward", "flash_attn_backward", "flash_attn_varlen_forward", "flash_attn_varlen_backward"):
        alias = getattr(iface, f"_wrapped_{name}")
        op = getattr(iface, f"_{name}")
        assert alias is getattr(flash_attn2.flash_attn_ops, f"_{name}")
        assert alias._qualified_op_name == op._qualname


def test_opcheck():
    q, k, v, dout = _qkv((2, 128, 4, 64))
    scale = q.shape[-1] ** -0.5
    fwd_args = (q, k, v, 0.0, scale, False, -1, -1, 0.0, None, False)
    torch.library.opcheck(
        flash_attn2.flash_attn_interface._flash_attn_forward, fwd_args, test_utils=OPCHECK_TEST_UTILS
    )

    out, lse, _, rng_state = flash_attn2.flash_attn_interface._flash_attn_forward(*fwd_args)
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    bwd_args = (
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        0.0,
        scale,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state,
    )
    torch.library.opcheck(
        flash_attn2.flash_attn_interface._flash_attn_backward, bwd_args, test_utils=OPCHECK_TEST_UTILS
    )


def test_varlen_opcheck():
    q, k, v, dout = (x.flatten(0, 1) for x in _qkv((2, 128, 4, 64)))
    cu_seqlens = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    fwd_args = (q, k, v, cu_seqlens, cu_seqlens, 128, 128, 0.0, scale, False)
    torch.library.opcheck(
        flash_attn2.flash_attn_interface._flash_attn_varlen_forward, fwd_args, test_utils=OPCHECK_TEST_UTILS
    )

    out, lse, _, rng_state = flash_attn2.flash_attn_interface._flash_attn_varlen_forward(*fwd_args)
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    bwd_args = (
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens,
        cu_seqlens,
        128,
        128,
        0.0,
        scale,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state,
    )
    torch.library.opcheck(
        flash_attn2.flash_attn_interface._flash_attn_varlen_backward, bwd_args, test_utils=OPCHECK_TEST_UTILS
    )


def test_compile_fullgraph_matches_eager():
    q, k, v, dout = _qkv((2, 128, 4, 64))
    expected = _fwd_bwd(q, k, v, dout)
    actual = torch.compile(_fwd_bwd, fullgraph=True)(q, k, v, dout)
    for e, a in zip(expected, actual):
        torch.testing.assert_close(a, e)


def test_varlen_compile_fullgraph_matches_eager():
    q, k, v, dout = (x.flatten(0, 1) for x in _qkv((2, 128, 4, 64)))
    cu_seqlens = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    expected = _varlen_fwd_bwd(q, k, v, dout, cu_seqlens, 128)
    actual = torch.compile(_varlen_fwd_bwd, fullgraph=True)(
        q, k, v, dout, cu_seqlens, 128
    )
    for e, a in zip(expected, actual):
        torch.testing.assert_close(a, e)

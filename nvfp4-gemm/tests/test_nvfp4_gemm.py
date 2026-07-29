import inspect

import pytest
import torch

import nvfp4_gemm

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
requires_blackwell = pytest.mark.skipif(not BLACKWELL, reason="requires sm100+")


def test_swizzled_sf_shape_math():
    # 128-row tiles; 4 fp8 scales packed per int32.
    assert nvfp4_gemm.swizzled_sf_shape(1, 64) == (128, 1)
    assert nvfp4_gemm.swizzled_sf_shape(6656, 19968) == (6656, 312)
    assert nvfp4_gemm.swizzled_sf_shape(130, 64) == (256, 1)


def test_quantize_reference_emulation():
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.bfloat16) * 0.02
    wq = nvfp4_gemm.quantize_reference(w)
    assert wq.shape == w.shape and wq.dtype == w.dtype
    err = (wq.float() - w.float()).abs().mean()
    assert err < 0.25 * w.float().std()  # ~4-bit fidelity
    assert err > 0  # actually quantized


@requires_blackwell
@pytest.mark.parametrize("m", [1, 4, 64, 2048])
def test_pack_and_gemm_matches_emulation(m):
    """Compare against a W4A4 emulation, not a W4A16 one.

    Quantizing the activations costs ~9.5% relative error on its own, so a
    reference built from exact bf16 activations sits right at the old rel<0.1
    threshold -- it measured the emulation gap, not the kernel, and a real
    scaling or swizzle bug could not have tripped it. Against a reference that
    quantizes activations the same way the kernel does, the kernel tracks to
    ~1.5%, which leaves room to actually catch a regression.

    Parametrized over m because the swizzled scale layout pads the m dimension
    (128-row tiles) and the CUTLASS config is dispatched on m.
    """
    torch.manual_seed(0)
    n, k = 768, 1024
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.5
    pw = nvfp4_gemm.pack(w)
    y = nvfp4_gemm.gemm(pw, x).float()

    wq = nvfp4_gemm.quantize_reference(w).float().cuda()
    xq = nvfp4_gemm.quantize_reference(x.cpu()).float().cuda()
    ref_w4a4 = xq @ wq.T
    ref_w4a16 = x.float() @ wq.T

    rel_w4a4 = ((y - ref_w4a4).norm() / ref_w4a4.norm().clamp(min=1e-6)).item()
    rel_w4a16 = ((y - ref_w4a16).norm() / ref_w4a16.norm().clamp(min=1e-6)).item()

    if m <= 2 and pw.sf_rowmajor is not None:
        # gemm() dispatches tiny m to the W4A16 GEMV: exact activations, so
        # the W4A16 emulation is the right reference.
        assert rel_w4a16 < 0.03, f"GEMV deviates from W4A16 emulation: {rel_w4a16:.4f}"
        return

    assert rel_w4a4 < 0.03, f"kernel deviates from W4A4 emulation: {rel_w4a4:.4f}"
    # Sanity: the W4A4 reference must be the closer of the two, otherwise the
    # activation path is not doing what the emulation thinks it is.
    assert rel_w4a4 < 0.5 * rel_w4a16


def _dequant_reference(
    qweight: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    """Dequantize row-major NVFP4 in pure torch: byte j holds element 2j in
    the low nibble; e2m1 magnitudes {0,.5,1,1.5,2,3,4,6} with sign in bit 3;
    times the FP8-E4M3 per-16 block scale, divided by the global scale."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=qweight.device)
    lo = qweight & 0xF
    hi = qweight >> 4
    nib = torch.stack([lo, hi], dim=-1).reshape(n, k).long()
    sign = torch.where((nib & 0x8) != 0, -1.0, 1.0)
    vals = grid[nib & 0x7] * sign
    scale = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return vals * scale / gs.float()


@requires_blackwell
@pytest.mark.parametrize("n,k", [(4096, 6656), (19968, 6656), (6656, 19968)])
@pytest.mark.parametrize("m", [1, 2])
def test_gemv_matches_dequant(n, k, m):
    from nvfp4_gemm import global_scale_for
    from nvfp4_gemm._ops import ops

    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    gs = global_scale_for(w)
    qweight, sf = ops.scaled_fp4_quant(w, gs, False)
    dequant = _dequant_reference(qweight, sf, gs, n, k)

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    alpha = (1.0 / gs).to(torch.float32).reshape(1)
    y = ops.nvfp4_gemv(x, qweight, sf, alpha).float()
    assert y.shape == (m, n)

    ref = x.float() @ dequant.T
    rel = ((y - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel < 1e-2, f"GEMV deviates from dequant reference: {rel:.4e}"


@requires_blackwell
@pytest.mark.parametrize("m", [1, 2])
def test_gemv_swiglu_matches_dequant(m):
    """Cover the row-concatenated SwiGLU launch geometry used by Onyx."""
    from nvfp4_gemm import global_scale_for
    from nvfp4_gemm._ops import ops

    torch.manual_seed(0)
    n, k = 19968, 6656
    gate = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    up = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    weight = torch.cat([gate, up], dim=0)
    gs = global_scale_for(weight)
    qweight, sf = ops.scaled_fp4_quant(weight, gs, False)
    dequant = _dequant_reference(qweight, sf, gs, 2 * n, k)

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    alpha = (1.0 / gs).to(torch.float32).reshape(1)
    y = ops.nvfp4_gemv_swiglu(x, qweight, sf, alpha).float()
    gate_ref, up_ref = (x.float() @ dequant.T).chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate_ref) * up_ref

    assert y.shape == (m, n)
    rel = ((y - ref).norm() / ref.norm().clamp(min=1e-6)).item()
    assert rel < 1e-2, f"SwiGLU GEMV deviates from dequant reference: {rel:.4e}"


def test_hub_layer_is_pure():
    # Contract: no constructor, no extra methods on the hub layer.
    cls = inspect.getsource(nvfp4_gemm.layers.NVFP4Linear)
    assert "__init__" not in cls
    assert cls.count("def ") == 1  # forward only

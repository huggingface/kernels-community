"""Loader and gfx1201 numerical tests; gates match upstream FlyDSL tests."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import kernels

REPO = "kernels-community/aiter-flash-attn-flydsl"


@pytest.fixture(scope="module")
def kernel():
    return kernels.get_kernel(REPO, version=1)


def gfx1201():
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"
    )


gpu = pytest.mark.skipif(not gfx1201(), reason="requires gfx1201")


@pytest.mark.kernels_ci
def test_load_without_aiter(kernel):
    assert set(kernel.__all__) == {"attention", "flash_attn_func", "fp8_quant"}
    assert importlib.util.find_spec("aiter") is None


@pytest.mark.kernels_ci
@pytest.mark.parametrize("explicit_toolkit", [False, True])
def test_runtime_setup_toolkit_resolution(
    kernel, monkeypatch, tmp_path, explicit_toolkit
):
    check = kernel.flash_attn_func.__globals__["check_runtime_setup"]
    resolve = check.__wrapped__.__globals__["_toolkit_root"]
    RocmBackend = check.__wrapped__.__globals__["RocmBackend"]
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "environment-sdk"))
    toolkit = tmp_path / "nix-sdk"
    option = f"toolkit={toolkit} " if explicit_toolkit else ""
    monkeypatch.setattr(
        RocmBackend,
        "pipeline_fragments",
        lambda self, **kwargs: [f"gpu-module-to-binary{{{option}format=fatbin}}"],
    )
    assert resolve() == (toolkit if explicit_toolkit else tmp_path / "environment-sdk")


@pytest.mark.kernels_ci
def test_runtime_setup_diagnostics(kernel, monkeypatch, tmp_path):
    check = kernel.flash_attn_func.__globals__["check_runtime_setup"]
    monkeypatch.setitem(
        check.__wrapped__.__globals__, "_toolkit_root", lambda: tmp_path
    )
    monkeypatch.setitem(
        check.__wrapped__.__globals__, "_detected_arch", lambda: "gfx1201"
    )
    check.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=r"rocm\[devel\].*ROCM_PATH"):
            check()
        linker = tmp_path / "llvm/bin/ld.lld"
        linker.parent.mkdir(parents=True)
        linker.write_text("#!/bin/sh\nexit 0\n")
        with pytest.raises(RuntimeError, match="not executable"):
            check()
        linker.chmod(0o755)
        monkeypatch.setitem(
            check.__wrapped__.__globals__, "_detected_arch", lambda: "gfx942"
        )
        with pytest.raises(RuntimeError, match="FLYDSL_GPU_ARCH=gfx1201"):
            check()
        monkeypatch.setitem(
            check.__wrapped__.__globals__, "_detected_arch", lambda: "gfx1201"
        )
        check()
        # A successful setup is checked once, rather than on every launch.
        linker.unlink()
        check()
    finally:
        check.cache_clear()


def qkv(dtype, dim=128, sq=257, sk=None):
    torch.manual_seed(42)
    q = torch.randn(1, sq, 4, dim, device="cuda", dtype=dtype)
    k = torch.randn(1, sk or sq, 4, dim, device="cuda", dtype=dtype)
    return q, k, torch.randn_like(k)


def reference(q, k, v, causal=False, scale=None):
    return F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2)


def accuracy(out, ref, *, fp8=False):
    assert torch.isfinite(out).all()
    cos = F.cosine_similarity(
        out.float().flatten(0, 2), ref.float().flatten(0, 2), dim=-1
    )
    if fp8:
        assert ((out.float() - ref).norm() / ref.norm()).item() < 0.08
        assert cos.min().item() > 0.98
    else:
        assert ((out.float() - ref).norm() / ref.norm()).item() < 0.01
        assert cos.min().item() > 0.99
        assert cos.mean().item() > 0.999


@pytest.mark.kernels_ci
def test_accuracy_rejects_wrong_magnitude():
    ref = torch.ones(1, 2, 1, 64)
    with pytest.raises(AssertionError):
        accuracy(2 * ref, ref)


@gpu
@pytest.mark.kernels_ci
def test_ci_attention_paths(kernel):
    q, k, v = qkv(torch.bfloat16, dim=64, sq=65)
    ref = reference(q, k, v)
    accuracy(kernel.flash_attn_func(q, k, v), ref)
    q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v)
    out = kernel.flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    accuracy(out, ref, fp8=True)


@gpu
@pytest.mark.parametrize("fp8", [False, True, "auto"])
def test_compile_fullgraph(kernel, fp8):
    q, k, v = qkv(torch.bfloat16, dim=64, sq=65)
    compiled = torch.compile(
        lambda q, k, v: kernel.attention(q, k, v, fp8=fp8), fullgraph=True
    )
    out = compiled(q, k, v)
    accuracy(out, reference(q, k, v), fp8=fp8 is True)


@gpu
def test_compile_prequantized(kernel):
    q, k, v = qkv(torch.bfloat16, dim=64, sq=65)
    q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v)
    compiled = torch.compile(kernel.attention, fullgraph=True)
    out = compiled(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    accuracy(out, reference(q, k, v), fp8=True)


def test_fp8_selection(kernel):
    selection = kernel.attention.__globals__["fp8_eligible"]
    assert not selection(torch.bfloat16, 2687, 2687, 24, 128)
    assert selection(torch.bfloat16, 2688, 2688, 24, 128)
    assert selection(torch.bfloat16, 1920, 4096, 24, 128)
    assert not selection(torch.bfloat16, 4096, 512, 24, 128)
    assert not selection(torch.bfloat16, 16384, 16384, 24, 256)
    assert not selection(torch.float16, 16384, 16384, 24, 128)


def test_two_builds_can_coexist(kernel, tmp_path):
    builder = shutil.which("kernel-builder")
    if builder is None:
        pytest.skip("requires kernel-builder to generate a second namespace")
    source = Path(__file__).resolve().parents[1]
    second = tmp_path / "second"
    shutil.copytree(
        source, second, ignore=shutil.ignore_patterns("build", "__pycache__", ".git")
    )
    subprocess.run(
        [
            builder,
            "create-pyproject",
            "--force",
            "--unique-id",
            "coexistence_test",
            "--kernel-sha",
            "84075df8ace3d446e2b54a72a478255a55c9811b",
            "--kernel-dirty",
            str(second),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "setup.py", "build_kernel"],
        cwd=second,
        check=True,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    first_path = Path(kernel.__file__).parent
    env["LOCAL_KERNELS"] = (
        f"{REPO}={first_path}:{REPO}-second={second / 'build' / 'torch-rocm'}"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import kernels; a=kernels.get_kernel({REPO!r},version=1); "
                f"b=kernels.get_kernel({(REPO + '-second')!r},version=1); assert a.attention is not b.attention"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("causal,sk", [(False, None), (True, None), (False, 129)])
def test_attention(kernel, dtype, dim, causal, sk):
    q, k, v = qkv(dtype, dim, sk=sk)
    out = kernel.flash_attn_func(q, k, v, causal=causal)
    assert out.shape == q.shape and out.dtype == dtype
    accuracy(out, reference(q, k, v, causal))


@gpu
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_fp8(kernel, rotation, dim):
    q, k, v = qkv(torch.bfloat16, dim)
    q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v, rotation=rotation)
    assert all(t.dtype == torch.float8_e4m3fn for t in (q8, k8, v8))
    assert all(
        s.dtype == torch.float32 and s.numel() == 1 and s.item() > 0
        for s in (sq, sk, sv)
    )
    out = kernel.flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    assert out.dtype == torch.bfloat16
    accuracy(out, reference(q, k, v), fp8=True)


@gpu
@pytest.mark.parametrize("causal,sk", [(True, None), (False, 193)])
def test_fp8_masking(kernel, causal, sk):
    q, k, v = qkv(torch.bfloat16, sk=sk)
    q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v)
    out = kernel.flash_attn_func(
        q8, k8, v8, causal=causal, q_descale=sq, k_descale=sk, v_descale=sv
    )
    accuracy(out, reference(q, k, v, causal), fp8=True)


@gpu
def test_noncontiguous_and_out(kernel):
    q, k, v = (
        x.transpose(1, 2).contiguous().transpose(1, 2) for x in qkv(torch.bfloat16)
    )
    out = torch.empty_like(q)
    with pytest.warns(UserWarning, match="materializes"):
        actual = kernel.flash_attn_func(q, k, v, out=out)
    assert actual is out
    accuracy(out, reference(q, k, v))


@gpu
@pytest.mark.parametrize("fp8", [False, True])
def test_nondefault_stream(kernel, fp8):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q, k, v = qkv(torch.bfloat16)
        if fp8:
            q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v)
            out = kernel.flash_attn_func(
                q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv, stream=stream
            )
        else:
            out = kernel.flash_attn_func(q, k, v, stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    accuracy(out, reference(q, k, v), fp8=fp8)


@gpu
@pytest.mark.parametrize("dim", [64, 128])
def test_quantization_preserves_cross_stream_input_lifetime(kernel, dim):
    kernel.fp8_quant(*qkv(torch.bfloat16, dim=dim, sq=1025))
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        q, k, v = qkv(torch.bfloat16, dim=dim, sq=1025)
        expected = tuple(t.cpu() for t in kernel.fp8_quant(q, k, v))
    producer.synchronize()
    shape = q.shape
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(200_000_000)
        actual = kernel.fp8_quant(q, k, v)
    del q, k, v
    with torch.cuda.stream(producer):
        replacements = [
            torch.full(shape, 10.0, device="cuda", dtype=torch.bfloat16)
            for _ in range(16)
        ]
    torch.cuda.synchronize()
    for index, (output, ref) in enumerate(zip(actual, expected)):
        if index < 3:
            assert torch.equal(output.cpu().view(torch.uint8), ref.view(torch.uint8))
        else:
            torch.testing.assert_close(output.cpu(), ref, rtol=0, atol=0)
    del replacements


@gpu
def test_quantization_reference(kernel):
    q, k, v = qkv(torch.bfloat16)
    actual = kernel.fp8_quant(q, k, v, rotation=False)
    for original, quantized, scale in zip((q, k, v), actual[:3], actual[3:]):
        expected_scale = (original.float().abs().max() / 448.0).clamp_min(1e-12)
        torch.testing.assert_close(scale.reshape(()), expected_scale)
        expected = (original.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(quantized.float(), expected.float(), rtol=0, atol=0)


@gpu
def test_invalid_inputs(kernel):
    q, k, v = qkv(torch.bfloat16)
    with pytest.raises(ValueError, match="causal cross-attention"):
        kernel.flash_attn_func(q, k[:, :128], v[:, :128], causal=True)
    with pytest.raises(ValueError, match="head_dim"):
        kernel.flash_attn_func(*qkv(torch.bfloat16, dim=48))
    with pytest.raises(ValueError, match="dtype"):
        kernel.fp8_quant(q.half(), k.half(), v.half())
    with pytest.raises(ValueError, match="descale"):
        kernel.flash_attn_func(*(x.to(torch.float8_e4m3fn) for x in (q, k, v)))
    with pytest.raises(ValueError, match="overlap|alias"):
        kernel.flash_attn_func(q, k, v, out=q)


@gpu
def test_non_power_of_two_quantization(kernel):
    q, k, v = qkv(torch.bfloat16, dim=96)
    with pytest.raises(ValueError, match="power-of-two|rotation"):
        kernel.fp8_quant(q, k, v, rotation=True)
    with pytest.warns(UserWarning, match="pads"):
        q8, k8, v8, sq, sk, sv = kernel.fp8_quant(q, k, v, rotation=False)
    assert q8.shape == q.shape
    out = kernel.flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    accuracy(out, reference(q, k, v), fp8=True)


@gpu
@pytest.mark.parametrize("fp8", [False, True, "auto"])
def test_adapter_custom_scale(kernel, fp8):
    q, k, v = qkv(torch.bfloat16, dim=64, sq=257)
    out = kernel.attention(q, k, v, fp8=fp8, softmax_scale=0.2)
    accuracy(out, reference(q, k, v, scale=0.2), fp8=fp8 is True)


@gpu
def test_auto_quantization_and_compile(kernel):
    torch.manual_seed(42)
    q = torch.randn(1, 2688, 24, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    def attention(q, k, v):
        return kernel.attention(q, k, v, fp8="auto", softmax_scale=0.1)

    eager = attention(q, k, v)
    forced = kernel.attention(q, k, v, fp8=True, softmax_scale=0.1)
    compiled = torch.compile(attention, fullgraph=True)(q, k, v)
    torch.testing.assert_close(eager, forced, rtol=0, atol=0)
    torch.testing.assert_close(compiled, eager)
    accuracy(eager, reference(q, k, v, scale=0.1), fp8=True)


@gpu
@pytest.mark.parametrize("mode", ["native", "fp8", "prequantized"])
def test_compile_noncontiguous_inputs(kernel, mode):
    q, k, v = qkv(torch.bfloat16, dim=64, sq=65)
    kwargs = {}
    if mode == "prequantized":
        q, k, v, sq, sk, sv = kernel.fp8_quant(q, k, v)
        kwargs = {"q_descale": sq, "k_descale": sk, "v_descale": sv}
    elif mode == "fp8":
        kwargs = {"fp8": True}
    q, k, v = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v))
    assert not q.is_contiguous()

    def attention(q, k, v):
        return kernel.attention(q, k, v, **kwargs)

    eager = attention(q, k, v)
    compiled = torch.compile(attention, fullgraph=True)(q, k, v)
    assert eager.is_contiguous() and compiled.is_contiguous()
    torch.testing.assert_close(compiled, eager)

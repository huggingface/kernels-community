"""
Test script for quantization-mlx Metal kernels.

Usage:
    # From a nix shell with the right torch version:
    nix develop '.#torch210' -c python test_quantization.py

    # Or if you have a matching torch in your conda env:
    conda activate kernel
    python test_quantization.py
"""

import sys
import os
import torch

# ---------------------------------------------------------------------------
# Auto-detect the right build directory based on installed torch version
# ---------------------------------------------------------------------------

def find_build_dir():
    major, minor = torch.__version__.split(".")[:2]
    torch_tag = f"torch{major}{minor}"
    build_root = os.path.join(os.path.dirname(__file__), "build")

    # Try exact match first, then fall back to any available build
    candidates = []
    if os.path.isdir(build_root):
        for name in sorted(os.listdir(build_root)):
            full = os.path.join(build_root, name)
            if os.path.isdir(full) and "metal" in name:
                candidates.append((name, full))

    for name, full in candidates:
        if name.startswith(torch_tag):
            return full

    # Fall back to latest available
    if candidates:
        print(f"[WARN] No exact build for torch {torch.__version__}, using {candidates[-1][0]}")
        return candidates[-1][1]

    raise RuntimeError(f"No build directory found in {build_root}")


build_dir = find_build_dir()
print(f"Using build: {os.path.basename(build_dir)}")
sys.path.insert(0, build_dir)

import quantization_mlx  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: simulate affine quantization (pack weights + compute scales/biases)
# ---------------------------------------------------------------------------

def affine_quantize(w_float, group_size=128, bits=4):
    """
    Quantize a float weight matrix to uint32-packed affine format.

    w_float: [N, K] (transposed layout) or [K, N] (non-transposed)
    Returns: w_packed (uint32), scales, biases
    """
    assert bits in (2, 4, 8), f"Unsupported bits: {bits}"
    N, K = w_float.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    n_groups = K // group_size
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1

    # Reshape into groups: [N, n_groups, group_size]
    w_grouped = w_float.reshape(N, n_groups, group_size)

    # Per-group min/max → scales and biases
    w_min = w_grouped.min(dim=-1).values  # [N, n_groups]
    w_max = w_grouped.max(dim=-1).values  # [N, n_groups]

    scales = (w_max - w_min) / max_val  # [N, n_groups]
    scales = scales.clamp(min=1e-8)
    biases = w_min  # [N, n_groups]

    # Quantize to integers
    w_int = ((w_grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    w_int = w_int.round().clamp(0, max_val).to(torch.int32)

    # Pack into uint32
    w_flat = w_int.reshape(N, -1)  # [N, K]
    K_packed = K // elems_per_int
    w_packed = torch.zeros(N, K_packed, dtype=torch.int32, device=w_float.device)
    for i in range(elems_per_int):
        w_packed |= (w_flat[:, i::elems_per_int] << (bits * i))

    return w_packed.to(torch.uint32), scales, biases


def affine_dequantize(w_packed, scales, biases, group_size=128, bits=4, K=None):
    """Dequantize for reference comparison."""
    N = w_packed.shape[0]
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1

    if K is None:
        K = w_packed.shape[1] * elems_per_int

    w_packed_i = w_packed.to(torch.int32)
    w_flat = torch.zeros(N, K, dtype=torch.float32, device=w_packed.device)
    for i in range(elems_per_int):
        w_flat[:, i::elems_per_int] = ((w_packed_i >> (bits * i)) & max_val).float()

    # [N, n_groups, group_size]
    w_grouped = w_flat.reshape(N, -1, group_size)
    w_deq = w_grouped * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    return w_deq.reshape(N, K)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_affine_qmm_t(M=32, K=256, N=128, group_size=128, bits=4, dtype=torch.float16):
    """Test: y = x @ dequant(w).T  where w is [N, K_packed]"""
    print(f"\n--- affine_qmm_t (M={M}, K={K}, N={N}, gs={group_size}, bits={bits}, {dtype}) ---")

    device = "mps"
    x = torch.randn(M, K, dtype=dtype, device=device)

    # Create and quantize a [N, K] weight on CPU, then move
    w_float = torch.randn(N, K, dtype=torch.float32)
    w_packed_cpu, scales_cpu, biases_cpu = affine_quantize(w_float, group_size, bits)

    w_packed = w_packed_cpu.to(device)
    scales = scales_cpu.to(dtype).to(device)
    biases = biases_cpu.to(dtype).to(device)

    # Kernel output
    y = quantization_mlx.affine_qmm_t(x, w_packed, scales, biases, group_size, bits)

    # Reference: dequantize on CPU, compute matmul
    w_deq = affine_dequantize(w_packed_cpu, scales_cpu.float(), biases_cpu.float(), group_size, bits)
    w_deq = w_deq.to(dtype).to(device)
    y_ref = x @ w_deq.T

    # Compare
    diff = (y.float() - y_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (y_ref.float().abs() + 1e-6)).mean().item()

    print(f"  max_abs_err:  {max_err:.6f}")
    print(f"  mean_abs_err: {mean_err:.6f}")
    print(f"  mean_rel_err: {rel_err:.6f}")
    print(f"  output shape: {y.shape}")

    # For low-bit quantization, tolerance is relatively loose
    ok = rel_err < 0.1
    print(f"  PASS" if ok else f"  FAIL")
    return ok


def test_affine_qmm_n(M=32, K=256, N=128, group_size=128, bits=4, dtype=torch.float16):
    """Test: y = x @ dequant(w)  where w is [K, N_packed] (N is the packed dim)"""
    print(f"\n--- affine_qmm_n (M={M}, K={K}, N={N}, gs={group_size}, bits={bits}, {dtype}) ---")

    device = "mps"
    x = torch.randn(M, K, dtype=dtype, device=device)

    # For qmm_n, the logical weight is [K, N].
    # The kernel packs along the N (second) dimension, with groups along N.
    # w_packed: [K, N/(32/bits)], scales/biases: [K, N/group_size]
    # affine_quantize packs along the second dimension, so we can reuse it.
    w_float = torch.randn(K, N, dtype=torch.float32)
    w_packed_cpu, scales_cpu, biases_cpu = affine_quantize(w_float, group_size, bits)

    w_packed = w_packed_cpu.to(device)
    scales = scales_cpu.to(dtype).to(device)
    biases = biases_cpu.to(dtype).to(device)

    y = quantization_mlx.affine_qmm_n(x, w_packed, scales, biases, N, group_size, bits)

    # Reference: dequantize on CPU, compute matmul (no transpose for qmm_n)
    w_deq = affine_dequantize(w_packed_cpu, scales_cpu.float(), biases_cpu.float(), group_size, bits)
    w_deq = w_deq.to(dtype).to(device)
    y_ref = x @ w_deq  # [M, K] @ [K, N] = [M, N]

    diff = (y.float() - y_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (y_ref.float().abs() + 1e-6)).mean().item()

    print(f"  max_abs_err:  {max_err:.6f}")
    print(f"  mean_abs_err: {mean_err:.6f}")
    print(f"  mean_rel_err: {rel_err:.6f}")
    print(f"  output shape: {y.shape}")

    ok = rel_err < 0.1
    print(f"  PASS" if ok else f"  FAIL")
    return ok


def test_affine_qmv(K=256, N=128, group_size=128, bits=4, dtype=torch.float16):
    """Test: y = dequant(w) @ x  (matrix-vector)"""
    print(f"\n--- affine_qmv (K={K}, N={N}, gs={group_size}, bits={bits}, {dtype}) ---")

    device = "mps"
    x = torch.randn(K, dtype=dtype, device=device)

    w_float = torch.randn(N, K, dtype=torch.float32)
    w_packed_cpu, scales_cpu, biases_cpu = affine_quantize(w_float, group_size, bits)

    w_packed = w_packed_cpu.to(device)
    scales = scales_cpu.to(dtype).to(device)
    biases = biases_cpu.to(dtype).to(device)

    y = quantization_mlx.affine_qmv(x, w_packed, scales, biases, N, group_size, bits)

    # Reference
    w_deq = affine_dequantize(w_packed_cpu, scales_cpu.float(), biases_cpu.float(), group_size, bits)
    w_deq = w_deq.to(dtype).to(device)
    y_ref = w_deq @ x

    diff = (y.float() - y_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (y_ref.float().abs() + 1e-6)).mean().item()

    print(f"  max_abs_err:  {max_err:.6f}")
    print(f"  mean_abs_err: {mean_err:.6f}")
    print(f"  mean_rel_err: {rel_err:.6f}")
    print(f"  output shape: {y.shape}")

    ok = rel_err < 0.1
    print(f"  PASS" if ok else f"  FAIL")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        print("MPS not available, skipping tests")
        sys.exit(1)

    results = []

    # affine_qmm_t tests
    results.append(test_affine_qmm_t(M=32, K=256, N=128, bits=4))
    results.append(test_affine_qmm_t(M=1, K=512, N=256, bits=4))
    results.append(test_affine_qmm_t(M=64, K=512, N=512, bits=2))

    # affine_qmm_n tests
    results.append(test_affine_qmm_n(M=32, K=256, N=128, bits=4))

    # affine_qmv tests
    results.append(test_affine_qmv(K=256, N=128, bits=4))
    results.append(test_affine_qmv(K=512, N=256, bits=4))

    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)

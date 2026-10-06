"""The host dispatch against the upstream code it was transcribed from.

`mlx_metal/mlx_dispatch.mm` re-implements the part of MLX's `quantized.cpp` a linear layer reaches:
which kernel runs for a shape, on which grid, with which buffers bound where. `vendor.py --rev`
replaces upstream's side of that, and a pin bump can change any of it. A renamed kernel fails at
runtime with a clear message, but a changed threshold, grid or buffer index still runs and silently
returns wrong numbers or slower kernels. This parses upstream and fails loudly instead.

No device and no built extension needed, so it runs anywhere.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "vendor/mlx/backend/metal/kernels"
QUANTIZED_CPP = ROOT / "vendor/mlx/backend/metal/quantized.cpp"
QUANTIZED_H = KERNELS / "quantized.h"
QUANTIZED_METAL = KERNELS / "quantized.metal.h"
DISPATCH_MM = ROOT / "mlx_metal/mlx_dispatch.mm"

pytestmark = pytest.mark.skipif(not QUANTIZED_CPP.exists(), reason="vendor/ is not checked out")


def squash(text):
    return " ".join(text.split())


def function(source, signature):
    """The text of the function starting at `signature`, through its matching closing brace."""
    start = source.index(signature)
    depth, i = 0, source.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        i += 1
        if depth == 0:
            return source[start:i]


@pytest.mark.parametrize(
    "signature",
    [
        "inline int get_qmv_batch_limit(",
        "inline int qmv_fast_k_alignment(",
        "inline bool use_qmv_wide(",
    ],
)
def test_helpers_are_verbatim(signature):
    upstream = function(QUANTIZED_CPP.read_text(), signature)
    ours = function(DISPATCH_MM.read_text(), signature)
    assert squash(ours) == squash(upstream), f"{signature} drifted from upstream; re-copy it"


# Upstream lines the plan() transcription relies on. Each must still be in quantized.cpp as written;
# when one is not, re-read that function upstream and update plan() to match.
TRANSCRIBED = [
    # QuantizedMatmul::eval_gpu
    "bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;",
    "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
    "if (M >= vector_limit) {",
    "if (transpose_ && B == 1) { qmm_splitk(",
    # qmm_splitk
    "int bm = 32, bn = 32;",
    "int split_k = std::max(1, 512 / current_tgs);",
    "int k_align = group_size > 32 ? group_size : 32;",
    "split_k = std::min(split_k, K / k_align);",
    "while (split_k > 1 && (K % (split_k * k_align) != 0)) { split_k--; }",
    "if (split_k <= 1) { return qmm(",
    "MTL::Size group_dims(32, 2, 2); MTL::Size grid_dims(n_tiles, m_tiles, split_k);",
    'mode + "_qmm_t_splitk_",',
    # qmm
    "MTL::Size group_dims(32, wn, wm); MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
    "bool aligned = N % 32 == 0;",
    'transpose ? (aligned ? "_alN_true" : "_alN_false") : "", batched ? "_batch_1" : "_batch_0",',
    # dispatch_qmv
    "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) { qmv_quad(",
    "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) { qmv_wide(",
    # qmv_quad
    "constexpr int quads_per_simd = 8; constexpr int results_per_quadgroup = 8;",
    "MTL::Size group_dims(simdgroup_size, 1, 1); MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
    '"_d_", K, B > 1 ? "_batch_1" : "_batch_0");',
    # qmv_wide
    "int n_tiles = (M + 4) / 5; // ceil(M / 5); tile size caps at 5",
    "int vecs_per_tg = (M + n_tiles - 1) / n_tiles;",
    'int k_lanes = mode == "affine" ? 8 : 16;',
    "int rows_per_tg = (32 / k_lanes) * num_simdgroups;",
    '"_nv_", vecs_per_tg, "_kl_", k_lanes, batched ? "_batch_1" : "_batch_0");',
    # qmv
    "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
    'mode == "nvfp4";',  # the narrow tile is nvfp4-only, so affine keeps 4 results per simdgroup
    "MTL::Size group_dims(bk, 2, 1); MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
    'use_narrow_qmv ? "_r_2" : "", B > 1 ? "_batch_1" : "_batch_0", global_scale ? "_hgs" : "");',
]


@pytest.mark.parametrize("line", TRANSCRIBED)
def test_transcribed_lines_still_upstream(line):
    assert squash(line) in squash(QUANTIZED_CPP.read_text()), f"upstream changed: {line}"


def buffers(kernel):
    """Parameter names of `[[kernel]] void <kernel>(...)` in order, with their [[buffer(i)]] if any."""
    src = QUANTIZED_H.read_text()
    m = re.search(r"\[\[kernel\]\]\s*void\s+" + kernel + r"\s*\((.*?)\)\s*\{", src, re.S)
    assert m, f"{kernel} not found in quantized.h"
    params = []
    for p in m.group(1).split(","):
        if "[[threadgroup" in p or "[[simdgroup" in p or "[[thread_" in p or "[[quadgroup" in p:
            continue
        name = re.search(r"[&*\s](\w+)\s*(\[\[buffer\((\d+)\)\]\])?\s*$", p.strip())
        params.append((name.group(1), int(name.group(3)) if name.group(3) else len(params)))
    return params


COMMON = [("w", 0), ("scales", 1), ("biases", 2), ("x", 3), ("y", 4)]


@pytest.mark.parametrize(
    "kernel,bound",
    [
        # what mlx_dispatch.mm binds, per kernel family, in order
        ("affine_qmv_quad", COMMON + [("in_vec_size", 5), ("out_vec_size", 6)]),
        ("affine_qmv_fast", COMMON + [("in_vec_size", 5), ("out_vec_size", 6)]),
        ("affine_qmv", COMMON + [("in_vec_size", 5), ("out_vec_size", 6)]),
        ("affine_qmv_wide", COMMON + [("in_vec_size", 5), ("out_vec_size", 6), ("M", 7)]),
        ("affine_qmm_t", COMMON + [("K", 5), ("N", 6), ("M", 7)]),
        (
            "affine_qmm_t_splitk",
            COMMON + [("K", 5), ("N", 6), ("M", 7), ("k_partition_size", 8), ("split_k_partition_stride", 9)],
        ),
    ],
)
def test_buffer_layout(kernel, bound):
    assert buffers(kernel)[: len(bound)] == bound


@pytest.mark.parametrize(
    "fragment",
    [
        # the kernel names plan() builds, as quantized.metal spells them
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched',
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched',
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched',
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched',
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,',
        "instantiate_quantized_batched_wrap(affine_qmv_fast, type, group_size, bits)",
        "instantiate_quantized_batched_wrap(affine_qmv, type, group_size, bits)",
        "instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)",
        "instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)",
        *[f"instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, {v}, 8)" for v in (2, 3, 4, 5)],
        *[f"instantiate_quantized_types({g}, bits)" for g in (128, 64, 32)],
        *[f"instantiate_quantized_groups({b})" for b in (2, 3, 4, 5, 6, 8)],
    ],
)
def test_instantiations(fragment):
    assert squash(fragment) in squash(QUANTIZED_METAL.read_text())


def test_math_mode_matches_upstream_build():
    """The wrapper's pragma stands in for the flag MLX compiles its kernels with."""
    cmake = ROOT / "vendor/mlx/backend/metal/kernels/CMakeLists.txt"
    if cmake.exists():
        assert "-fno-fast-math" in cmake.read_text()
    assert "#pragma METAL fp math_mode(safe)" in (ROOT / "mlx_metal/mlx_quantized.metal").read_text()

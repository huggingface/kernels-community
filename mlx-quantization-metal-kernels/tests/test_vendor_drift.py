"""The host dispatch against the upstream code it was transcribed from.

`mlx_metal/mlx_dispatch.mm` and `mlx_metal/mlx_quantization.cpp` re-implement MLX's launchers and op
layer: which kernel runs for a shape, on which grid, with which buffers bound where, and the
defaults and checks in front of that. `vendor.py --rev` replaces upstream's side, and a pin bump can
change any of it. A renamed kernel fails at runtime with a clear message, but a changed threshold,
grid or buffer index still runs and silently returns wrong numbers or slower kernels. This parses
upstream and fails loudly instead.

No device and no built extension needed, so it runs anywhere.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor/mlx"
KERNELS = VENDOR / "backend/metal/kernels"
QUANTIZED_CPP = VENDOR / "backend/metal/quantized.cpp"
REDUCE_CPP = VENDOR / "backend/metal/reduce.cpp"
DEVICE_CPP = VENDOR / "backend/metal/device.cpp"
OPS_CPP = VENDOR / "ops.cpp"
DISPATCH_MM = ROOT / "mlx_metal/mlx_dispatch.mm"
TORCH_SIDE = ROOT / "mlx_metal/mlx_quantization.cpp"

# Not in the kernels_ci subset: CI builds from the files build.toml lists, which leaves out vendor/'s
# host sources, so these run locally (after `vendor.py`) instead.
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
    ["inline int get_qmv_batch_limit(", "inline int qmv_fast_k_alignment(", "inline bool use_qmv_wide("],
)
def test_helpers_are_verbatim(signature):
    upstream = function(QUANTIZED_CPP.read_text(), signature)
    ours = function(DISPATCH_MM.read_text(), signature)
    assert squash(ours) == squash(upstream), f"{signature} drifted from upstream; re-copy it"


# Upstream lines the transcription relies on. Each must still be in the vendored file as written;
# when one is not, re-read that function upstream and update ours to match.
TRANSCRIBED = {
    QUANTIZED_CPP: [
        # QuantizedMatmul::eval_gpu
        "bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;",
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (M >= vector_limit) {",
        "if (transpose_ && B == 1) { qmm_splitk(",
        "// Run of the mill qvm if (K < 1024) { qvm(",
        # GatherQMM::eval_gpu
        "if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) { gather_qmm_rhs(",
        "x.size() / K,",
        # qmm_splitk
        "int split_k = std::max(1, 512 / current_tgs);",
        "int k_align = group_size > 32 ? group_size : 32;",
        "split_k = std::min(split_k, K / k_align);",
        "while (split_k > 1 && (K % (split_k * k_align) != 0)) { split_k--; }",
        "MTL::Size group_dims(32, 2, 2); MTL::Size grid_dims(n_tiles, m_tiles, split_k);",
        # qmm and qmm_nax
        "bool has_nax_kernel = metal::is_nax_available() && (transpose || mode == \"affine\");",
        "bool nax_aligned = (K % 64 == 0) && (transpose || N % 64 == 0);",
        "if (has_nax_kernel && nax_aligned && (env::enable_tf32() || x.dtype() != float32)) {",
        "int bm = (transpose && M <= 32) ? 32 : 64;",
        "MTL::Size group_dims(32, wn, wm); MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        "} else if (transpose) { if (global_scale) { compute_encoder.set_input_array(*global_scale, c); } c++; }",
        # dispatch_qmv, qmv_quad, qmv_wide, qmv
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) { qmv_quad(",
        "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) { qmv_wide(",
        "constexpr int quads_per_simd = 8; constexpr int results_per_quadgroup = 8;",
        "int n_tiles = (M + 4) / 5; // ceil(M / 5); tile size caps at 5",
        'int k_lanes = mode == "affine" ? 8 : 16;',
        "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
        'use_narrow_qmv ? "_r_2" : "", B > 1 ? "_batch_1" : "_batch_0", global_scale ? "_hgs" : "");',
        # qvm and qvm_split_k
        "int bn = std::min(group_size, 32) * num_simdgroups;",
        "int split_k = K > 8192 ? 32 : 8;",
        "x_strides[x_ndim - 1] = split_D;",
        "int final_block_size = K - (split_k - 1) * split_D;",
        "int axis = intermediate.ndim() - 3;",
        # gather_qmm and gather_qmm_rhs
        "if (metal::is_nax_available() && transpose && (K % 64 == 0) && (env::enable_tf32() || x.dtype() != float32)) {",
        "int bm = 16, bn = 32, bk = 32; int wm = 1, wn = 2;",
        "int bm = (M / E < 64) ? 32 : 64;",
        "{&align_N, MTL::DataType::DataTypeBool, 201}, {&align_K, MTL::DataType::DataTypeBool, 202},",
        "MTL::Size grid_dims( (N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1);",
        # quantize_impl / get_quantize_kernel_dims
        "int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;",
        "dequantize ? packs_per_int : std::max(group_size / simd_size, 1);",
        'concatenate(kname, "_hgs_", has_global_scale ? "true" : "false");',
    ],
    REDUCE_CPP: [
        "if (args.reduction_size * args.non_col_reductions < 32) { return strided_reduce_small(",
        "if (args.reduction_stride < 32 && args.reduction_size * args.non_col_reductions >= 1024) {",
        "if (args.reduction_size * args.non_col_reductions > 256 && out.size() / 32 < 1024) {",
        "size_t threadgroup_y = std::min( 8ul, std::min(kernel->maxTotalThreadsPerThreadgroup() / threadgroup_x, total));",
        "int BN = 32; int BM = 1024 / BN; int threadgroup_size = 8 * 32;",
        "compute_encoder.set_bytes(out_size, 11);",
    ],
    DEVICE_CPP: [
        "if (__builtin_available( macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {",
        "can_use_nax &= gen >= (arch == 'p' ? 18 : 17);",
    ],
    OPS_CPP: [
        # quantization_params_from_mode
        "case QuantizationMode::Affine: default_group_size = 64; default_bits = 4;",
        "case QuantizationMode::Nvfp4: default_group_size = 16; default_bits = 4;",
        "case QuantizationMode::Mxfp4: default_group_size = 32; default_bits = 4;",
        "case QuantizationMode::Mxfp8: default_group_size = 32; default_bits = 8;",
        # quantized_matmul / gather_qmm dtype and batching
        "dtype = promote_types(x.dtype(), dtype);",
        "if (x.ndim() > 2 && w.ndim() > 2) { inputs = broadcast_arrays(inputs, {-2, -1}, s); }",
        "sorted_indices && !lhs_indices_),",
        # validate_mode_with_type: fp modes dequantize to bfloat16 by default
        "return {bfloat16, qmode};",
    ],
}


@pytest.mark.parametrize(
    "path,line", [(p, line) for p, lines in TRANSCRIBED.items() for line in lines], ids=lambda v: str(v)[:60]
)
def test_transcribed_lines_still_upstream(path, line):
    assert squash(line) in squash(path.read_text()), f"{path.name} changed: {line}"


def kernel_params(header, kernel):
    """Parameter names of `[[kernel]] void <kernel>(...)`, in order, without thread-position ones."""
    m = re.search(r"\[\[kernel\]\]\s*void\s+" + kernel + r"\s*\((.*?)\)\s*\{", (KERNELS / header).read_text(), re.S)
    assert m, f"{kernel} not found in {header}"
    names = []
    for p in m.group(1).split(","):
        if re.search(r"\[\[(threadgroup|simdgroup|thread|quadgroup|threads)_", p):
            continue
        names.append(re.search(r"(\w+)\s*(\[\[buffer\(\d+\)\]\])?\s*$", p.strip()).group(1))
    return names


def explicit_indices(header, kernel):
    """{name: index} for the parameters that pin one with [[buffer(i)]]."""
    m = re.search(r"\[\[kernel\]\]\s*void\s+" + kernel + r"\s*\((.*?)\)\s*\{", (KERNELS / header).read_text(), re.S)
    return {n: int(i) for n, i in re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", m.group(1))}


# What the launchers bind, in index order, for each kernel family: affine binds biases at 2; the fp
# modes bind an optional global scale there (or skip it), so their indices are positional.
BOUND = {
    ("quantized.h", "affine_qmv_quad"): "w scales biases x y in_vec_size out_vec_size",
    ("quantized.h", "affine_qmv_fast"): "w scales biases x y in_vec_size out_vec_size",
    ("quantized.h", "affine_qmv"): "w scales biases x y in_vec_size out_vec_size",
    ("quantized.h", "affine_qmv_wide"): "w scales biases x y in_vec_size out_vec_size M",
    ("quantized.h", "affine_qvm"): "w scales biases x y in_vec_size out_vec_size",
    ("quantized.h", "affine_qvm_split_k"): "w scales biases x y in_vec_size out_vec_size x_batch_ndims x_shape x_strides w_batch_ndims w_shape w_strides s_strides b_strides final_block_size",
    ("quantized.h", "affine_qmm_t"): "w scales biases x y K N M",
    ("quantized.h", "affine_qmm_n"): "w scales biases x y K N M",
    ("quantized.h", "affine_qmm_t_splitk"): "w scales biases x y K N M k_partition_size split_k_partition_stride",
    ("quantized.h", "affine_gather_qmv"): "w scales biases x lhs_indices rhs_indices y in_vec_size out_vec_size",
    ("quantized.h", "affine_gather_qmm_t"): "w scales biases x lhs_indices rhs_indices y K N M",
    ("quantized.h", "affine_gather_qmm_rhs"): "x w scales biases offsets y M N K num_groups",
    ("quantized.h", "affine_quantize"): "w out scales biases",
    ("quantized.h", "affine_dequantize"): "w scales biases out",
    ("fp_quantized.h", "fp_qmv_quad"): "w scales x y in_vec_size out_vec_size",
    ("fp_quantized.h", "fp_qmv_fast"): "w scales global_scale x y in_vec_size out_vec_size",
    ("fp_quantized.h", "fp_qmv_wide"): "w scales x y in_vec_size out_vec_size M",
    ("fp_quantized.h", "fp_qvm"): "w scales global_scale x y in_vec_size out_vec_size",
    ("fp_quantized.h", "fp_qvm_split_k"): "w scales x y in_vec_size out_vec_size x_batch_ndims x_shape x_strides w_batch_ndims w_shape w_strides s_strides final_block_size",
    ("fp_quantized.h", "fp_qmm_t"): "w scales global_scale x y K N M",
    ("fp_quantized.h", "fp_qmm_n"): "w scales x y K N M",
    ("fp_quantized.h", "fp_qmm_t_splitk"): "w scales x y K N M k_partition_size split_k_partition_stride",
    ("fp_quantized.h", "fp_gather_qmv"): "w scales global_scale x lhs_indices rhs_indices y in_vec_size out_vec_size",
    ("fp_quantized.h", "fp_gather_qmm_t"): "w scales global_scale x lhs_indices rhs_indices y K N M",
    ("fp_quantized.h", "fp_gather_qmm_rhs"): "x w scales global_scale offsets y M N K num_groups",
    ("fp_quantized.h", "fp_quantize"): "w out scales global_scale",
    ("fp_quantized.h", "fp_dequantize"): "w scales global_scale out",
}


@pytest.mark.parametrize("header,kernel", BOUND, ids=lambda v: v)
def test_buffer_layout(header, kernel):
    expected = BOUND[(header, kernel)].split()
    params = kernel_params(header, kernel)
    assert params[: len(expected)] == expected, params
    # where upstream pins indices, they must be the positions the launchers bind at
    for name, index in explicit_indices(header, kernel).items():
        assert params.index(name) == index, (name, index)


@pytest.mark.parametrize(
    "path,fragment",
    [
        # the kernel names the launchers build, as the .metal files spell them
        ("quantized.metal.h", '#name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched'),
        ("quantized.metal.h", '#name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched'),
        ("quantized.metal.h", '#name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched'),
        ("quantized.metal.h", '"_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched'),
        ("quantized.metal.h", '"_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn'),
        ("fp_quantized.metal.h", '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched'),
        ("fp_quantized.metal.h", '#mode "_quantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #has_global_scale'),
        ("quantized_nax.metal.h", '"_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched'),
        ("reduce.metal.h", '"col_reduce_small_" #dim "_reduce_" #name'),
        ("reduce.metal.h", '"col_reduce_looped_" #dim "_" #bm "_" #bn "_reduce_" #name'),
    ],
)
def test_instantiations(path, fragment):
    assert squash(fragment) in squash((KERNELS / path).read_text())


def test_math_mode_matches_upstream_build():
    """The wrappers' pragma stands in for the flag MLX compiles its kernels with."""
    assert "-fno-fast-math" in (KERNELS / "CMakeLists.txt").read_text()
    wrappers = sorted((ROOT / "mlx_metal").glob("*.metal"))
    assert len(wrappers) == len(list(KERNELS.glob("*.metal.h")))
    for w in wrappers:
        assert "#pragma METAL fp math_mode(safe)" in w.read_text(), w.name

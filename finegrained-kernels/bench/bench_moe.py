"""finegrained-kernels bench — local build vs upstream finegrained-fp8 (rev v4) + reference impls.

The **finegrained-kernels** arm is the local kernel; **finegrained-fp8** is the upstream hub
build (``kernels-community/finegrained-fp8`` @ ``v4``, which has the fused MoE + MX paths).
By default the finegrained-kernels arm feeds PRE-SWIZZLED (SWIZZLE_32_4_4) MX weight scales, so
its numbers reflect the tcgen05 fast path (set ``PRESWIZZLE=0`` for the affine path). Writes
``bench/bench_moe.csv`` (all 3 modes) + ``bench/bench_moe.png`` beside this file.

Rows (each row = decode | prefill subplot pair in the figure):
  quantized           moe_fused_*    finegrained-kernels vs finegrained-fp8 vs transformers@main vs DeepGEMM
                                     — every impl at its best: fused where it has one, and
                                     transformers@main contributes its two-GEMM experts dispatch
  unquantized (BF16)  finegrained-kernels fused vs transformers grouped_mm/batched_mm vs SonicMoE
                      vs DeepGEMM grouped BF16
  attn quantized      matmul_2d, one qkv-proj-shaped linear (N=3H, K=H) per model in
                      its deployment format — FP8 128x128 (finegrained-kernels/finegrained-fp8/DeepGEMM), MXFP4
                      W4A4 (finegrained-kernels W4A4, finegrained-fp8 W4A8, DeepGEMM FP4), NVFP4 (finegrained-kernels only),
                      MXFP8 (finegrained-kernels/finegrained-fp8)

MoE problems (real model shapes; one base model per format, same roster BF16'd for
the unquantized row; baselines per problem):
  deepseek-ai/DeepSeek-V4  MXFP4 W4A8         finegrained-fp8, DeepGEMM FP4, TRT-LLM (MxFP4xMxFP8)
  openai/GPT-OSS-120B      full MXFP4 (W4A4)  OpenAI triton_kernels, TRT-LLM (MxFP4xBf16) —
                                              finegrained-fp8 lacks W4A4 AND its kernels
                                              can't run K=2880 (no BK-divides-K guard)
  nvidia/GLM-5.2-NVFP4     NVFP4 (W4A4)       TRT-LLM (NvFP4xNvFP4), torch.scaled_grouped_mm
  deepseek-ai/DeepSeek-V3  FP8 W8A8 (128x128) finegrained-fp8, DeepGEMM FP8, vLLM, TRT-LLM — UE8M0
                                              block scales (the B200 deployment format;
                                              DeepGEMM SM100 rejects fp32 scales by design)
  MiniMaxAI/MiniMax-M3     MXFP8 W8A8         finegrained-fp8 (no TRT-LLM: clamped SwiGLU)

The TRT-LLM arms are FlashInfer's ``trtllm_*_routed_moe`` kernels — the backends vLLM's
serving path dispatches to on Blackwell for these quant modes (its own triton fused-MoE
covers only bf16 and block-FP8). Weight prep and activation quant come from THEIR stack,
so each bar is that integration end-to-end.

Every (row, problem, regime, impl) cell runs in THREE modes:
  eager      do_bench on the plain call
  cudagraph  do_bench_cudagraph (decode's deployment mode)
  compile    torch.compile(mode="max-autotune", fullgraph=True), warmed, then do_bench

Regimes: decode T=1 and prefill T=8192 (routed through top_k experts; attn row: M=T).

SMOKE=1 env: fast everything-compiles pass — 3-trial tunes (via
FINEGRAINED_AUTOTUNE_TRIALS, which must be set before the package import) and a
256-token prefill.

Baselines ("all kinds"): finegrained-fp8 (upstream), DeepGEMM (fp8/fp4/bf16), transformers
grouped_mm/batched_mm (torch._grouped_mm / torch.bmm, the BF16 torch path), SonicMoE, OpenAI
triton_kernels (MXFP4), and torch.scaled_grouped_mm (the cuBLAS quantized-prefill path). Each
is import-guarded; a missing dependency skips that baseline.

Run: python bench/bench_moe.py             (all rows, single GPU)
     GPUS=8 python bench/bench_moe.py      (shard problems across 8 GPUs, one process per GPU)
     SMOKE=1 python bench/bench_moe.py     (fast compile check)
     PRESWIZZLE=0 python bench/bench_moe.py (affine MX scales instead of the fast path)
     python bench/bench_moe.py gpt-oss     (substring filter on row/problem names)
     REPLOT=1 python bench/bench_moe.py    (rebuild the figure from bench_moe.csv)
"""

import os
import sys
from types import SimpleNamespace

# GPUS>1 shards the per-problem tasks across that many GPUs (one process per GPU, coordinator
# merges + plots). BENCH_SHARD="g/n" marks a worker subprocess (owns tasks where i % n == g).
GPUS = int(os.environ.get("GPUS", "1"))
_SHARD = os.environ.get("BENCH_SHARD")
SMOKE = os.environ.get("SMOKE") == "1"
# MOCK=1: no GPU, no kernels — every cell gets a random-but-plausible latency and
# parity so the FIGURE (layout, crash markers, parity hatching) can be validated in
# seconds. Writes bench_moe_mock.png.
MOCK = os.environ.get("MOCK") == "1"
# REPLOT=1: skip all benching, rebuild the figure from an existing bench_moe.csv.
# Lets the layout/config (model order, which baselines are shown) be re-rendered in
# seconds without re-running the multi-hour sweep.
REPLOT = os.environ.get("REPLOT") == "1"
if SMOKE:
    os.environ.setdefault("FINEGRAINED_AUTOTUNE_TRIALS", "3")

import torch  # noqa: E402
from triton.testing import do_bench, do_bench_cudagraph  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "torch-ext"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
import finegrained_kernels as fgm  # noqa: E402  local branch
from kernels import get_kernel  # noqa: E402

# All baselines here are kernels-community repos we already trust; the publisher-trust check
# hits a rate-limited org-overview API (429 under the 8-way shard fan-out) and can't be reached
# for sonic-moe (loaded via transformers' lazy_load_kernel, no trust_remote_code hook). Neutralize
# the check process-wide so every get_kernel — ours, sonic, deepgemm, gpt-oss — loads from cache.
import kernels.utils as _kernels_utils  # noqa: E402

_kernels_utils._check_trust_remote_code = lambda *a, **k: None

# PRESWIZZLE=0 to bench the affine (row-major) MX scale path instead of the pre-swizzled
# SWIZZLE_32_4_4 tcgen05 fast path. Default on: the finegrained-kernels arm feeds pre-swizzled
# weight scales so the numbers reflect the max-perf path (the guard rejects non-128 gate/N,
# so we only swizzle MX weights on 128-aligned dims; everything else stays affine).
PRESWIZZLE = os.environ.get("PRESWIZZLE", "1") == "1"
_MX_WEIGHTS = {"mxfp8", "mxfp8_u8", "mxfp4", "nvfp4"}


def _can_preswizzle(cfg):
    # Deployment feeds ONE pre-swizzled checkpoint to both prefill (grouped) and decode (batched):
    # the interleaved gate|up + non-gate swizzle round-trips bit-exact on every fused op. Only MX
    # weights on 128-aligned dims swizzle (the descriptor reads whole 128-row blocks).
    return (PRESWIZZLE and cfg["weights"] in _MX_WEIGHTS
            and cfg["H"] % 128 == 0 and cfg["I"] % 128 == 0)


# the activation recipe fed to the MoE forwards. Explicit cfg["recipe"] wins (e.g. DeepSeek-V4
# W4A8 pins "mxfp8"); recipe="weights" follows the weight family, and recipe=None is weight-only
# (raw bf16 activations). Weight families with no entry below resolve to None.
def _recipe(cfg):
    """Pass the cfg recipe straight through: ``"weights"`` (the op resolves it off the weight
    dtypes), ``None`` (raw bf16 activations), or an explicit format (e.g. dsv4 pins "mxfp8" for
    W4A8). No weight-family table here — ``weight_recipe`` in the op is the single resolver."""
    return cfg["recipe"]


_NVFP4_GLOBALS = {}


def _nvfp4_input_globals(cfg, hidden, gu, gus, gu_g):
    """Calibrated NVFP4 activation globals ``(act, intermediate)`` — the offline PTQ step every
    NVFP4 deployment ships. Without the intermediate one the fused requant clips: the SwiGLU
    output's block amax divided by 6 overruns e4m3's 448 ceiling, the intermediate collapses,
    and the row measures a saturated forward (measured: rel 1.0 / cos 0.64 vs the bf16 master,
    0.16 / 0.99 once calibrated). Both stacks read the same pair, so parity stays meaningful.
    The activation amax is exact; the intermediate one comes from a 64-token / 4-expert
    sample, cached per problem."""
    key = (cfg["E"], cfg["H"], cfg["I"], hidden.shape[0])
    if key not in _NVFP4_GLOBALS:
        sample = hidden[:64].float()
        weights = dq_nvfp4_two_level(gu[:4], gus[:4], gu_g[:4]).float()
        gate_up = torch.einsum("th,eih->eti", sample, weights)
        inter = _glu(gate_up[..., : cfg["I"]], gate_up[..., cfg["I"]:], cfg)
        amax = lambda t: (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)
        _NVFP4_GLOBALS[key] = (amax(hidden), amax(inter))
    return _NVFP4_GLOBALS[key]


def _nvfp4_kwargs(cfg, hidden, gu, gus, gu_g):
    """The calibrated globals as forward kwargs (empty for non-NVFP4 rows)."""
    if cfg["weights"] != "nvfp4":
        return {}
    act_g, inter_g = _nvfp4_input_globals(cfg, hidden, gu, gus, gu_g)
    return dict(gate_up_input_global_scale=act_g, down_input_global_scale=inter_g)


def _preswizzle_moe_scale(scale):
    """Per-expert SWIZZLE_32_4_4 of a grouped MX weight scale ``(E, rows, K//G)`` -> the swizzled
    artifact the tcgen05 fast path reads (the library helper owns the expert-stack and byte-view
    contracts). Done once at arm setup (not timed).

    No ``gate`` argument: the interleaved gate|up layout made a gated scale slab just the ungated
    one at doubled extent, so the 6-D gate-interleaved artifact is gone. The parameter outlived it
    as a REQUIRED arg that two of four call sites did not pass, which took the finegrained-kernels
    arm out of every preswizzle row with a setup TypeError."""
    return fgm.swizzle_mx_scales(scale)


from utils import WEIGHTS, dq_nvfp4_two_level, make_weights  # noqa: E402  tests/utils.py registry
from transformers.integrations.deepgemm import (  # noqa: E402
    deepgemm_bf16_experts_forward,
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
)
from transformers.integrations.moe import (  # noqa: E402
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
)
from transformers.integrations.finegrained_fp8 import (  # noqa: E402
    fp8_batched_mm_experts_forward,
    fp8_grouped_mm_experts_forward,
)
from transformers.integrations.sonicmoe import sonicmoe_experts_forward  # noqa: E402

# vLLM's triton fused-MoE kernel — its bf16 serving path (DeepGEMM/CUTLASS/FlashInfer are its
# quantized backends). Soft import off the local checkout; the arm is enlisted only when it
# resolves. Tuned tile configs for the roster's shapes ship in bench/vllm_configs — vLLM has
# no B200 bf16 JSONs for them and would silently fall back to heuristic defaults (measured
# 12-23% slow at prefill).
os.environ.setdefault("VLLM_TUNED_CONFIG_FOLDER", os.path.join(_HERE, "vllm_configs"))
sys.path.insert(0, os.path.expanduser("~/vllm"))
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts  # noqa: E402
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation as _VllmAct  # noqa: E402
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config as _vllm_fp8_qc  # noqa: E402
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8 as _vllm_group_quant  # noqa: E402
except ImportError:
    vllm_fused_experts = None
# kernels-community/nvfp4-gemm — the dense NVFP4 GEMM behind transformers' `NVFP4Linear`
# (upstream #47883). Linear-only: it never sees an MoE, because experts are fused 3-D
# parameters rather than nn.Linear. Benched here two ways — as a 2D GEMM against our
# matmul_2d (apples to apples), and looped per expert over the routed rows, which is what
# an MoE would cost if it were served by that integration.
try:
    from transformers.integrations.hub_kernels import lazy_load_kernel as _hub_kernel  # noqa: E402
    _nvfp4_gemm = _hub_kernel("nvfp4")
except Exception:
    _nvfp4_gemm = None
# megablocks dropless-MoE (dMoE) — the hub build, NOT a local pip install: megablocks pins an
# older torch, so installing it from source downgrades the environment out from under every other
# arm. Loaded like the rest of the roster.
try:
    _megablocks = get_kernel("kernels-community/megablocks", version=1)
except Exception:
    _megablocks = None

# FlashInfer TRT-LLM fused MoE (NVIDIA's Blackwell serving kernels) — installed --no-deps
# (its PyPI pin would downgrade torch; the JIT has no torch ABI coupling). JIT needs
# CUDA_HOME >= 12.8 at first compile; the built module is cached under ~/.cache/flashinfer.
try:
    # the JIT'd module dlopens libcudart.so.12 by name — preload it into global scope
    # (torch keeps its copy private)
    import ctypes, glob  # noqa: E402
    _cudart = (glob.glob(os.path.join(os.environ.get("CUDA_HOME", ""), "lib64/libcudart.so.12*"))
               or glob.glob(os.path.join(os.path.dirname(torch.__file__),
                            "../nvidia/cuda_runtime/lib/libcudart.so.12*")))
    ctypes.CDLL(_cudart[0], mode=ctypes.RTLD_GLOBAL)
    from flashinfer.fused_moe import Fp8QuantizationType as _FiQuantType  # noqa: E402
    from flashinfer.fused_moe import WeightLayout as _FiWeightLayout  # noqa: E402
    from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe  # noqa: E402
    from flashinfer import mxfp8_quantize as _fi_mxfp8_quantize  # noqa: E402
    from flashinfer import reorder_rows_for_gated_act_gemm as _fi_reorder  # noqa: E402
    from flashinfer import shuffle_matrix_a as _fi_shuffle_a  # noqa: E402
    from flashinfer import shuffle_matrix_sf_a as _fi_shuffle_sf_a  # noqa: E402
    from flashinfer import fp4_quantize as _fi_fp4_quantize  # noqa: E402
    from flashinfer.fp4_quantization import block_scale_interleave as _fi_sf_interleave  # noqa: E402
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe  # noqa: E402
    from flashinfer.fused_moe.core import (  # noqa: E402
        _maybe_get_cached_w3_w1_permute_indices as _fi_w13_permute,
        get_w2_permute_indices_with_cache as _fi_w2_permute,
    )
except Exception:
    trtllm_fp8_block_scale_routed_moe = None


UPSTREAM_FP8_REV = "v4"  # the pinned hub revision; also the legend suffix
upstream_fp8 = (None if (MOCK or REPLOT)
          else get_kernel("kernels-community/finegrained-fp8", revision=UPSTREAM_FP8_REV,
                          trust_remote_code=True))

# OpenAI triton_kernels (matmul_ogs) — the MXFP4 experts path transformers uses for
# GPT-OSS. Loaded like finegrained-fp8; its module-level handle drives the mxfp4 swizzle helpers.
if not (MOCK or REPLOT):
    import transformers.integrations.mxfp4 as _tfmx
    triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1, trust_remote_code=True)
    _tfmx.triton_kernels_hub = triton_kernels_hub

DEV = "cuda" if torch.cuda.is_available() else "xpu"
ACCEL = getattr(torch, DEV)  # torch.cuda / torch.xpu: synchronize(), get_device_name()
# Level-Zero (Intel XPU) masks devices with ZE_AFFINITY_MASK, CUDA/ROCm with CUDA_VISIBLE_DEVICES.
DEV_MASK_ENV = "CUDA_VISIBLE_DEVICES" if DEV == "cuda" else "ZE_AFFINITY_MASK"
DECODE_TOKENS = 1
PREFILL_TOKENS = 256 if SMOKE else 8192

# fixed left-to-right model order for every figure row (matched by base-model prefix,
# so GLM-5.2-NVFP4 and GLM-5.2 both land in the GLM-5.2 slot). Roughly most-baseline-
# support first, finegrained-kernels-only (GPT-OSS, GLM-NVFP4) last.
CANONICAL_MODEL_ORDER = ["DeepSeek-V4", "DeepSeek-V3", "MiniMax-M3", "GPT-OSS-120B", "GLM-5.2"]

MOE_PROBLEMS = {
    # Scaled-down stand-in for the DeepSeek-V4 geometry (same recipe, /8 experts, /2 dims) so the
    # MoE rows fit on a 32GB part: build() materializes the pre-quant weights in fp32, which needs
    # ~16GB for a single E256 H4096 gate_up grid alone.
    "small/DeepSeek-V4-shaped FP8 block-dyn W8A8 ue8m0 (E32 H2048 I1024 top6)": dict(
        E=32, H=2048, I=1024, top_k=6, weights="fp8_128x128_ue8m0", recipe="weights",
        baselines=("finegrained-fp8", "deepgemm"), fp8_block=[128, 128], block_size=(128, 128),
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "small/MiniMax-M3-shaped MXFP8 (E32 H2048 I1024 top4)": dict(
        E=32, H=2048, I=1024, top_k=4, weights="mxfp8", recipe="weights",
        baselines=("finegrained-fp8",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
    "deepseek-ai/DeepSeek-V4-Base FP8 block-dyn W8A8 ue8m0 (E256 H4096 I2048 top6)": dict(
        # config.json: fp8 e4m3, scale_fmt ue8m0, weight_block_size [128,128], dynamic acts.
        # Same expert geometry as the MXFP4 V4 row below — the difference is the deployed
        # recipe, so the two rows isolate W4A8 vs W8A8 on identical shapes. UE8M0 scales are
        # what DeepGEMM's SM100 experts kernel wants, so unlike the V3 fp32 row it gets a
        # deepgemm baseline. (Router is sqrtsoftplus upstream; the bench's shared softmax
        # top-k feeds every arm the same weights, so it does not affect the comparison.)
        E=256, H=4096, I=2048, top_k=6, weights="fp8_128x128_ue8m0", recipe="weights",
        baselines=("finegrained-fp8", "deepgemm", "vllm", "trtllm"), fp8_block=[128, 128], block_size=(128, 128),
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V4 MXFP4 W4A8 (E256 H4096 I2048 top6)": dict(
        E=256, H=4096, I=2048, top_k=6, weights="mxfp4", recipe="mxfp8",
        baselines=("finegrained-fp8", "deepgemm", "trtllm"), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "openai/GPT-OSS-120B MXFP4 W4A16 (E128 H2880 I2880 top4)": dict(
        # GPT-OSS deploys mxfp4 WEIGHTS with BF16 activations (W4A16) — transformers' matmul_ogs does
        # a bf16-act x mxfp4-weight matmul (fp4 weight upcast in-MMA). fgm runs the SAME recipe via the
        # dedicated weight-only kernels (recipe=None: raw bf16 acts, mxfp4 weight upcast to bf16
        # per-group in-loop, plain bf16 dot). The triton_kernels baseline is the reference W4A16.
        # (finegrained-fp8 lacks W4A16 AND its MX kernels have no BK-divides-K guard — BK {128,256}
        # doesn't divide 2880 -> NaN.)
        E=128, H=2880, I=2880, top_k=4, weights="mxfp4", recipe=None,
        baselines=("trtllm",), fused_extra=("triton_kernels",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
    "nvidia/GLM-5.2-NVFP4 W4A4 (E256 H6144 I2048 top8)": dict(
        E=256, H=6144, I=2048, top_k=8, weights="nvfp4", recipe="weights",
        baselines=("trtllm",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V3 FP8 block-dyn W8A8 fp32 (E256 H7168 I2048 top8)": dict(
        # DeepSeek-V3 experts deploy fp32 block scales (software rescale). DeepGEMM's SM100
        # experts kernel requires UE8M0 and fails loud on fp32, so no deepgemm baseline here.
        E=256, H=7168, I=2048, top_k=8, weights="fp8_128x128", recipe="weights",
        baselines=("finegrained-fp8", "vllm", "trtllm"), fp8_block=[128, 128], block_size=(128, 128),
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "MiniMaxAI/MiniMax-M3 MXFP8 (E128 H6144 I3072 top4)": dict(
        # No TRT-LLM arm: this row's activation is the clamped/scaled SwiGLU (alpha 1.702,
        # limit 7.0) and their fp8-block kernel answers a different function with those scalars
        # (see the gate in _run_task) — their fp4 kernel implements them, so GPT-OSS keeps its arm.
        E=128, H=6144, I=3072, top_k=4, weights="mxfp8", recipe="weights",
        baselines=("finegrained-fp8",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
}
# the same base-model roster, run as if dequantized to BF16 (one shape per model)
BF16_PROBLEMS = {
    "small/DeepSeek-V4-shaped BF16 (E32 H2048 I1024 top6)": dict(
        E=32, H=2048, I=1024, top_k=6, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V4 BF16 (E256 H4096 I2048 top6)": dict(
        E=256, H=4096, I=2048, top_k=6, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "openai/GPT-OSS-120B BF16 (E128 H2880 I2880 top4)": dict(
        E=128, H=2880, I=2880, top_k=4, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
    "zai-org/GLM-5.2 BF16 (E256 H6144 I2048 top8)": dict(
        E=256, H=6144, I=2048, top_k=8, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V3 BF16 (E256 H7168 I2048 top8)": dict(
        E=256, H=7168, I=2048, top_k=8, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "MiniMaxAI/MiniMax-M3 BF16 (E128 H6144 I3072 top4)": dict(
        E=128, H=6144, I=3072, top_k=4, weights="bf16", recipe="weights",
        baselines=("transformers", "sonicmoe", "vllm", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
}
ATTN_PROBLEMS = {
    "deepseek-ai/DeepSeek-V4 attn FP8 W8A8 ue8m0 qkv-shaped (N=12288 K=4096)": dict(
        # DeepSeek-V4's attention deploys block-FP8 W8A8 with UE8M0 (power-of-two) scales
        # (only the EXPERTS are mixed W4A8) — routes through the tcgen05 dot_scaled arm.
        # K=4096 is on the 128 grid, so finegrained-fp8 (block-dyn) and DeepGEMM FP8 both run it.
        N=12288, K=4096, weights="fp8_128x128_ue8m0", recipe="weights",
        block=[128, 128], baselines=("finegrained-fp8", "deepgemm"),
    ),
    # (No standalone dense 2D W4A16 row: no prominent model ships dense mxfp4 — gpt-oss's only mxfp4
    # is its experts (benchmarked as the MoE W4A16 row), and dense 4-bit-weight LLMs deploy GPTQ/AWQ
    # int4, not mxfp4. The 2D weight-only kernel stays covered by test_ops + the unfused MoE path.)
    "nvidia/GLM-5.2-NVFP4 attn W4A4 qkv-shaped (N=18432 K=6144)": dict(
        N=18432, K=6144, weights="nvfp4", recipe="weights",
        block=None, baselines=(),  # no baseline supports NVFP4
    ),
    "deepseek-ai/DeepSeek-V3 attn FP8 W8A8 fp32 128x128 qkv-shaped (N=21504 K=7168)": dict(
        # DeepSeek-V3 deploys fp32 block scales everywhere (software rescale, not dot_scaled).
        # No deepgemm baseline, for the same reason the V3 MoE row has none: its SM100 path can
        # only consume UE8M0, so the linear rounds each fp32 block scale to a power of two — up to
        # 2x per block, measured 0.53 relative vs every other arm (0.001 on the UE8M0 row below).
        # That is a different computation, so timing it here would compare unlike work; the honest
        # UE8M0 deepgemm comparison is the DeepSeek-V4 attn row.
        N=21504, K=7168, weights="fp8_128x128", recipe="weights",
        block=[128, 128], baselines=("finegrained-fp8",),
    ),
    "MiniMaxAI/MiniMax-M3 attn MXFP8 W8A8 qkv-shaped (N=18432 K=6144)": dict(
        N=18432, K=6144, weights="mxfp8", recipe="weights",
        block=None, baselines=("finegrained-fp8",),  # DeepGEMM FP8 is 128-block, not group-32 MX
    ),
}
MODES = ["eager", "cudagraph", "compile"]

IMPL_COLORS = {
    "finegrained-kernels": "#1f77b4",
    "finegrained-fp8": "#bbbbbb",
    "deepgemm": "#2ca02c",
    "deepgemm_bf16": "#2ca02c",
    "transformers": "#9467bd",
    "transformers@main": "#7f7f7f",  # transformers main on a quantized checkpoint (status quo)
    "sonicmoe": "#ff7f0e",
    "vllm": "#17becf",  # vLLM triton fused-MoE (its bf16 serving kernel)
    "trtllm": "#76b900",  # FlashInfer TRT-LLM fused MoE (NVIDIA Blackwell serving kernels)
    "triton_kernels": "#8c564b",  # OpenAI mxfp4 (GPT-OSS) reference
    "nvfp4_gemm": "#d62728",  # transformers NVFP4Linear kernel (dense 2D row only)
    "megablocks": "#7f7f7f",  # databricks dropless-MoE (bf16 reference)
    "torch": "#e377c2",  # torch/cuBLAS F.scaled_grouped_mm (quantized prefill reference)
    "torch_mm": "#e377c2",  # torch/cuBLAS F.scaled_mm (the attn-row sibling)
}


def _impl_label(impl, regime):
    """Legend name. The reference forwards ARE plain torch ops — name them by the op:
    transformers batched_mm -> torch.bmm (decode) / grouped_mm -> torch._grouped_mm (bf16);
    ``torch`` -> torch.scaled_grouped_mm (the quantized cuBLAS path). Every other impl
    (finegrained-kernels, finegrained-fp8, deepgemm, ...) is its own label."""
    if impl == "transformers":
        return "torch.bmm" if regime == "decode" else "torch._grouped_mm"
    if impl == "torch":
        return "torch.scaled_grouped_mm"
    if impl == "torch_mm":
        return "torch.scaled_mm"
    # transformers' NVFP4Linear kernel (kernels-community/nvfp4-gemm). One name in both rows:
    # this label function only sees the regime, not the row, and "(per-expert)" — true of the MoE
    # rows, where it is looped because that integration has no MoE kernel — is wrong on the dense
    # attn row. The per-expert framing lives in the row docs instead of a misapplied suffix.
    if impl == "nvfp4_gemm":
        return "nvfp4-gemm"
    if impl == "vllm":
        return "vLLM fused_moe"
    if impl == "trtllm":
        return "TRT-LLM (FlashInfer)"
    if impl == "finegrained-fp8":
        return f"finegrained-fp8@{UPSTREAM_FP8_REV}"
    return impl


def _mark_static(*tensors):
    for t in tensors:
        if t is not None:
            # a real model holds weights as Parameters (cudagraph-static); closure
            # tensors would be re-copied into the cudagraph buffers EVERY compiled
            # call (~150us/GB DtoD, measured 95% of compiled decode) — mark static.
            torch._dynamo.mark_static_address(t)


def build(cfg):
    """Quantized weights for one MoE problem, shared by every impl and row. The GLM-FP8
    problem uses UE8M0 block scales (finegrained-fp8/finegrained-kernels decode them natively; DeepGEMM SM100
    requires them)."""
    E, H, inter = cfg["E"], cfg["H"], cfg["I"]
    if cfg["weights"] == "fp8_128x128_ue8m0":
        def make(n, k, e):
            return (*make_weights(n, k, DEV, [128, 128],
                                  scale_dtype=torch.float8_e8m0fnu, num_experts=e), None)
    else:
        make = WEIGHTS[cfg["weights"]]["make"]
    # NVFP4 is two-level: make() returns the per-expert fp32 global as its 3rd value (None for the
    # single-level FP8/MX recipes). Thread it through — dropping it silently ran nvfp4 at global=1.
    gu, gus, gu_g = make(2 * inter, H, E)
    dn, dns, dn_g = make(H, inter, E)
    _mark_static(gu, gus, dn, dns)
    return gu, gus, dn, dns, gu_g, dn_g


def routing(cfg, tokens):
    torch.manual_seed(0)
    hidden = torch.randn(tokens, cfg["H"], device=DEV, dtype=torch.bfloat16)
    logits = torch.randn(tokens, cfg["E"], device=DEV)
    w, idx = torch.topk(torch.softmax(logits, -1), cfg["top_k"], dim=-1)
    return hidden, idx.to(torch.int32), w, logits


def _glu(gate, up, cfg):
    a, lim = cfg["swiglu_alpha"], cfg["swiglu_limit"]
    if lim is not None:
        gate, up = gate.clamp(max=lim), up.clamp(-lim, lim)
    if a is not None:
        return (up + 1.0) * (gate * torch.sigmoid(a * gate))
    return torch.nn.functional.silu(gate) * up


class _Experts:
    """Duck-typed experts module for the transformers-integration forwards
    (grouped_mm/batched_mm, SonicMoE, DeepGEMM): our (E, out, in) layout is their
    ``is_transposed=False``; gate|up rows are INTERLEAVED (gate on even rows, up on odd),
    the same artifact the fused kernels read.

    ``is_concatenated`` is inert here -- the integration forwards defer the split to
    ``_apply_gate``, which this class overrides -- so the layout is expressed there."""

    def __init__(self, cfg, gu, dn, gus=None, dns=None):
        self.num_experts = cfg["E"]
        self.has_gate, self.has_bias = True, False
        self.is_transposed, self.is_concatenated = False, True
        self.gate_up_proj, self.down_proj = gu, dn
        self.gate_up_proj_scale_inv, self.down_proj_scale_inv = gus, dns
        self.block_size = cfg["block_size"]
        self.activation_scheme = "dynamic"
        self._deepgemm_disabled = False
        self.act_fn = torch.nn.functional.silu
        self.config = SimpleNamespace(hidden_act="silu")
        self.swiglu_alpha = cfg["swiglu_alpha"]
        self.swiglu_limit = cfg["swiglu_limit"]
        self._cfg = cfg

    def _apply_gate(self, gate_up_out):
        # The gate|up weight is interleaved, so the GEMM's output columns are too: a
        # chunk(2) split pairs the wrong halves and silently scrambles the result
        # (parity 1.2e+00, cosine ~0 -- not a crash). Split on the stride instead.
        # De-interleaving the weight itself is not an option: a 128-row block scale
        # spans 64 gate and 64 up rows, so the halves cannot be separated.
        gate, up = gate_up_out[..., 0::2], gate_up_out[..., 1::2]
        return _glu(gate, up, self._cfg).to(gate_up_out.dtype)


# ── MoE impl arms: each returns a no-arg closure computing the full forward ──


def moe_fused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g, dn_g, *_):
    """``recipe`` sets the activation precision; None follows the weight recipe
    (mxfp4/nvfp4 -> the all-fp4 W4A4 chain, bf16 -> unquantized). dsv4 deploys
    W4A8, so it pins recipe="mxfp8". Under ``PRESWIZZLE`` the MX weight scales are
    pre-swizzled into SWIZZLE_32_4_4 so the forward takes the tcgen05 fast path."""
    fn = fgm.moe_fused_grouped if grouped else fgm.moe_fused_batched
    nvfp4_kw = _nvfp4_kwargs(cfg, hidden, gu, gus, gu_g)  # raw scales: before any preswizzle
    if _can_preswizzle(cfg):
        gus = _preswizzle_moe_scale(gus)   # fused gate GEMM reads the interleaved layout
        dns = _preswizzle_moe_scale(dns)
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg),
              gate_up_proj_global_scale=gu_g, down_proj_global_scale=dn_g,
              **nvfp4_kw)
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, **kw)


def moe_unfused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g, dn_g, *_):
    fn = fgm.moe_unfused_grouped if grouped else fgm.moe_unfused_batched
    nvfp4_kw = _nvfp4_kwargs(cfg, hidden, gu, gus, gu_g)  # raw scales: before any preswizzle
    if _can_preswizzle(cfg):
        # ONE checkpoint layout: gate_up scales are always the gate-interleaved artifact; the
        # unfused plain 2N GEMM reads it via the in-kernel INTERLEAVED_SCALES block remap.
        gus = _preswizzle_moe_scale(gus)
        dns = _preswizzle_moe_scale(dns)
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg),
              gate_up_proj_global_scale=gu_g, down_proj_global_scale=dn_g,
              **nvfp4_kw)
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, **kw)


def _torch_preblock_weight_scale(ws):
    """Block-rearrange a WEIGHT scale ONCE (torchao's SWIZZLE_32_4_4 builder — the exact op+layout
    ``moe_torch_grouped`` feeds ``scaled_grouped_mm``), so the timed forward uses it as-is instead of
    re-running the rearrange every call. A real deployment blocks weight scales offline; leaving it in
    the loop added a per-call kernel to the torch baseline and inflated its latency. Bit-identical
    (same deterministic transform); the activation scale stays per-forward inside ``moe_torch_grouped``
    (it changes each call)."""
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_per_group_3d,
    )
    return triton_mx_block_rearrange_per_group_3d(ws.view(torch.uint8)).view(ws.dtype)  # keep dtype (E4M3=NVFP4)


def torch_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g, dn_g, *_):
    """torch / cuBLAS reference: ``moe_torch_grouped`` over the PUBLIC ``F.scaled_grouped_mm``
    (two scaled grouped GEMMs + host GLU + weighted reduce) — the quantized-path torch baseline
    (the ``transformers`` grouped_mm arm is the BF16 one). Runs BOTH regimes: it groups routed
    tokens by expert regardless of token count, so decode works too. Weight scales are pre-blocked
    ONCE (``_torch_preblock_weight_scale``) so the timed forward doesn't re-swizzle them each call;
    the per-expert NVFP4 global rides alongside."""
    nvfp4_kw = _nvfp4_kwargs(cfg, hidden, gu, gus, gu_g)
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg),
              gate_up_proj_global_scale=gu_g, down_proj_global_scale=dn_g,
              **nvfp4_kw)
    gus_b, dns_b = _torch_preblock_weight_scale(gus), _torch_preblock_weight_scale(dns)
    return lambda: fgm.moe_torch_grouped(hidden, idx, w, gu, dn, gus_b, dns_b, **kw)


def _fp8_scales(t, block):
    """finegrained-fp8's BLOCK-FP8 path predates UE8M0 scales (KeyError on e8m0) — hand it the
    exact fp32 values instead (UE8M0 holds pure exponents: lossless, same math).
    Its MX path reads UE8M0 natively, so MX scales (block None) pass through."""
    if block is not None and t is not None and t.dtype == torch.float8_e8m0fnu:
        t = t.float()
        _mark_static(t)
    return t


def fp8_fused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """Hub finegrained-fp8 fused forward (block_size positional; None = MX, [128,128] = block FP8)."""
    fn = upstream_fp8.moe_fused_grouped if grouped else upstream_fp8.moe_fused_batched
    gus = _fp8_scales(gus, cfg["fp8_block"])
    dns = _fp8_scales(dns, cfg["fp8_block"])
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"])
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, cfg["fp8_block"], **kw)


def fp8_unfused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """finegrained-fp8 unfused: two plain finegrained-fp8 GEMMs + host GLU + weighted reduce. finegrained-fp8's grouped op
    wants A expert-sorted with per-expert cumulative row-ends (``offsets``) and counts;
    its batched op wants pre-expanded A + flat expert ids."""
    E, top_k, T = cfg["E"], cfg["top_k"], hidden.shape[0]
    gus = _fp8_scales(gus, cfg["fp8_block"])
    dns = _fp8_scales(dns, cfg["fp8_block"])
    flat = idx.reshape(-1).long()
    if grouped:
        order = torch.argsort(flat, stable=True)
        src = order // top_k
        counts = torch.histc(flat.float(), bins=E, min=0, max=E - 1).to(torch.int32)
        offsets = torch.cumsum(counts, 0).to(torch.int32)

        def run():
            a_sorted = hidden[src]
            gu_out = upstream_fp8.matmul_grouped(a_sorted, gu, gus, offsets, counts,
                                           cfg["fp8_block"], torch.bfloat16)
            gate, up = gu_out.chunk(2, dim=-1)
            inter = _glu(gate, up, cfg).to(torch.bfloat16)
            down = upstream_fp8.matmul_grouped(inter, dn, dns, offsets, counts,
                                         cfg["fp8_block"], torch.bfloat16)
            routed = torch.empty_like(down)
            routed[order] = down
            return (routed.view(T, top_k, -1) * w[..., None]).sum(1)
    else:
        src = torch.arange(T, device=hidden.device).repeat_interleave(top_k)
        flat_i32 = flat.to(torch.int32)

        def run():
            a = hidden[src]
            gu_out = upstream_fp8.matmul_batched(a, gu, gus, flat_i32, cfg["fp8_block"],
                                           torch.bfloat16)
            gate, up = gu_out.chunk(2, dim=-1)
            inter = _glu(gate, up, cfg).to(torch.bfloat16)
            down = upstream_fp8.matmul_batched(inter, dn, dns, flat_i32, cfg["fp8_block"],
                                         torch.bfloat16)
            return (down.view(T, top_k, -1) * w[..., None]).sum(1)

    return run


def transformers_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """transformers.integrations.moe reference: grouped_mm (prefill) / batched_mm (decode)."""
    mod = _Experts(cfg, gu, dn)
    fwd = grouped_mm_experts_forward if grouped else batched_mm_experts_forward
    return lambda: fwd(mod, hidden, idx, w)


def transformers_main_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """``transformers`` main on a quantized checkpoint — its ``finegrained_fp8`` integration
    (fp8_grouped_mm / fp8_batched_mm experts dispatch), which loads the finegrained-fp8 hub kernel
    behind a Python wrapper. This is the status quo an fgm integration would replace, so it is the
    reference for what integrating fgm buys end-to-end; the bare ``finegrained-fp8`` arm is the
    same kernel WITHOUT the wrapper, which separates integration overhead from kernel speed."""
    mod = _Experts(cfg, gu, dn, gus, dns)
    fwd = fp8_grouped_mm_experts_forward if grouped else fp8_batched_mm_experts_forward
    return lambda: fwd(mod, hidden, idx, w)


def sonicmoe_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    mod = _Experts(cfg, gu, dn)
    return lambda: sonicmoe_experts_forward(mod, hidden, idx, w)


def _trtllm_mxfp8_prep(gu, gus, dn, dns):
    """TRT-LLM MxFp8 weight prep, verbatim from flashinfer's passing reference tests
    (v0.6.15.post1): halves swapped to their [up; gate] convention, gate|up rows
    reordered for the fused gated GEMM (weights AND scales), then the epilogue-tile-128
    shuffles — ``shuffle_matrix_a`` on values, ``shuffle_matrix_sf_a`` on scale bytes."""
    E, I2, _ = gu.shape
    I = I2 // 2
    gu_u8, gus_u8 = gu.view(torch.uint8), gus.view(torch.uint8)
    gu_ug = torch.cat([gu_u8[:, I:], gu_u8[:, :I]], dim=1).contiguous()
    gus_ug = torch.cat([gus_u8[:, I:], gus_u8[:, :I]], dim=1).contiguous()
    g1 = torch.stack([_fi_shuffle_a(_fi_reorder(gu_ug[i].reshape(I2, -1)), 128)
                      for i in range(E)]).view(torch.float8_e4m3fn)
    g1s = torch.stack([_fi_shuffle_sf_a(_fi_reorder(gus_ug[i].reshape(I2, -1)), 128)
                       for i in range(E)]).reshape(gus.shape[0], I2, -1)
    d1 = torch.stack([_fi_shuffle_a(dn.view(torch.uint8)[i], 128)
                      for i in range(E)]).view(torch.float8_e4m3fn)
    d1s = torch.stack([_fi_shuffle_sf_a(dns.view(torch.uint8)[i].reshape(dn.shape[1], -1), 128)
                       for i in range(E)]).reshape(dns.shape[0], dn.shape[1], -1)
    return g1, g1s, d1, d1s


def _trtllm_gated_scalars(cfg, num_experts, device):
    """Per-expert ``(alpha, beta, clamp_limit)`` for TRT-LLM's gated activation. Their form is
    ``x2*sigmoid(alpha*x2)*(x1+beta)`` with x1 clamped to [-limit, limit] and x2 to limit — the
    GPT-OSS/MiniMax SwiGLU our ``_glu`` computes, so a row with a clamp limit MUST pass these
    (dropping them silently computes plain SwiGLU and the arm answers a different function)."""
    if cfg["swiglu_alpha"] is None and cfg["swiglu_limit"] is None:
        return None, None, None
    full = lambda v: torch.full((num_experts,), v, device=device, dtype=torch.float32)
    return (full(cfg["swiglu_alpha"]) if cfg["swiglu_alpha"] is not None else None,
            full(1.0) if cfg["swiglu_alpha"] is not None else None,
            full(cfg["swiglu_limit"]) if cfg["swiglu_limit"] is not None else None)


def _trtllm_fp4_pad(t, rows, cols, gated):
    """Zero-pad an fp4 byte/scale tensor to the padded expert geometry. Gated tensors pad
    each half separately (the [up; gate] halves stay contiguous); zero e2m1 bytes and zero
    scale bytes both decode to 0, so the padded lanes contribute nothing."""
    if gated:
        half = t.shape[1] // 2
        up, gate = t[:, :half], t[:, half:]
        zeros = torch.zeros(t.shape[0], rows // 2 - half, t.shape[2],
                            dtype=t.dtype, device=t.device)
        t = torch.cat([up, zeros, gate, zeros], dim=1)
    return torch.nn.functional.pad(t, (0, cols - t.shape[2], 0, rows - t.shape[1])).contiguous()


def _trtllm_fp4_prep(w13_bytes, w13_scale_bytes, w2_bytes, w2_scale_bytes):
    """TRT-LLM fp4 weight prep, verbatim from FP4Moe.prepare_static_weights_for_kernel
    (flashinfer v0.6.15.post1): per-expert gated-activation row reorder fused into the
    epilogue-tile-128 permutation (weights AND scales), then block_scale_interleave on
    the scale bytes. Inputs are already-quantized bytes — the SAME checkpoint our arm
    reads, so the two stacks run identical weights."""
    cache = {}
    dev = w13_bytes.device
    w13, w13s, w2, w2s = [], [], [], []
    for i in range(w13_bytes.shape[0]):
        p = _fi_w13_permute(cache, w13_bytes[i], 128, is_gated_act_gemm=True)
        w13.append(w13_bytes[i][p.to(dev)].contiguous())
        p = _fi_w13_permute(cache, w13_scale_bytes[i], 128, num_elts_per_sf=16,
                            is_gated_act_gemm=True)
        w13s.append(_fi_sf_interleave(w13_scale_bytes[i][p.to(dev)].contiguous()))
        p = _fi_w2_permute(cache, w2_bytes[i], 128)
        w2.append(w2_bytes[i][p.to(dev)].contiguous())
        p = _fi_w2_permute(cache, w2_scale_bytes[i], 128, num_elts_per_sf=16)
        w2s.append(_fi_sf_interleave(w2_scale_bytes[i][p.to(dev)].contiguous()))
    return (torch.stack(w13), torch.stack(w13s).view(torch.float8_e4m3fn),
            torch.stack(w2), torch.stack(w2s).view(torch.float8_e4m3fn))


def _trtllm_fp4_arm(cfg, hidden, idx, packed, gu, gus, dn, dns, gu_g, dn_g):
    """FlashInfer TRT-LLM fp4 routed MoE: NVFP4xNVFP4 (W4A4), MXFP4xMXFP8 (W4A8) and
    MXFP4xBF16 (W4A16) all run this kernel, differing only in the activation quant inside
    the timed call and in the scalars. Their gate|up order is [up; gate] (halves swapped);
    hidden/intermediate are rounded up to 256 the way vLLM's TRT-LLM MoE path pads GPT-OSS,
    since the kernel has no config below that granularity."""
    E, H, I = cfg["E"], cfg["H"], cfg["I"]  # gu/dn store PACKED fp4: last dim is half the logical K
    I2 = 2 * I
    nvfp4 = cfg["weights"] == "nvfp4"
    group = 16 if nvfp4 else 32
    Hp, Ip = -(-H // 256) * 256, -(-I // 256) * 256
    swap = lambda t: torch.cat([t[:, I:], t[:, :I]], dim=1).contiguous()
    w13 = _trtllm_fp4_pad(swap(gu.view(torch.uint8).reshape(E, I2, H // 2)),
                          2 * Ip, Hp // 2, True)
    w13s = _trtllm_fp4_pad(swap(gus.view(torch.uint8).reshape(E, I2, H // group)),
                           2 * Ip, Hp // group, True)
    w2 = _trtllm_fp4_pad(dn.view(torch.uint8).reshape(E, H, I // 2), Hp, Ip // 2, False)
    w2s = _trtllm_fp4_pad(dns.view(torch.uint8).reshape(E, H, I // group), Hp, Ip // group, False)
    g1, g1s, d1, d1s = _trtllm_fp4_prep(w13, w13s, w2, w2s)

    ones = torch.ones(E, device=hidden.device, dtype=torch.float32)
    if nvfp4:
        # their globals are the reciprocal of ours (they scale INTO fp4, we scale out), so their
        # dequant scalars — c_global/(w_global*a_global) and its FC2 mirror — reduce to products
        act_g, inter_g = _nvfp4_input_globals(cfg, hidden, gu, gus, gu_g)
        hidden_g = (1.0 / act_g).float().reshape(())
        scale_gate1 = gu_g.float() * act_g
        scale_c1 = scale_gate1 / inter_g
        scale_c2 = dn_g.float() * inter_g
    else:
        hidden_g = None
        scale_c1 = scale_gate1 = scale_c2 = ones
    alpha, beta, limit = _trtllm_gated_scalars(cfg, E, hidden.device)
    act_recipe = _recipe(cfg)

    def run():
        x = torch.nn.functional.pad(hidden, (0, Hp - H)) if Hp != H else hidden
        # the activation quant is THEIR kernel, inside the timed call
        if nvfp4:
            hq, hsf = _fi_fp4_quantize(x, hidden_g, 16, False, False)
        elif act_recipe == "mxfp8":
            hq, hsf = _fi_mxfp8_quantize(x, False)
        else:
            hq, hsf = x, None
        if hsf is not None:
            hsf = hsf.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
        out = trtllm_fp4_block_scale_routed_moe(
            packed, None, hq, hsf, g1, g1s, None, alpha, beta, limit, d1, d1s, None,
            scale_c1, scale_gate1, scale_c2,
            num_experts=E, top_k=idx.shape[1], n_group=None, topk_group=None,
            intermediate_size=Ip, local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
        )
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out[:, :H]

    return run


def trtllm_moe_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g=None, dn_g=None, *_):
    """FlashInfer TRT-LLM fp8-block routed MoE (the DeepSeek recipe on SM100). Weight prep
    (offline): TRT-LLM's gate|up order is [up; gate] — the halves swap; UE8M0 scales ride
    as fp32 (same values). Routing rides packed ``(expert_id << 16) | bf16-weight`` ids;
    the activation quant (1x128, TRANSPOSED [K/128, T] scale layout) is inside the timed
    call and is the STACK'S OWN kernel (vLLM's per_token_group_quant_fp8 — what their
    TRT-LLM integration runs), so the bar is the integration end-to-end. Numerics caveat (oracle-checked 2026-08-06):
    ~4.8e-2 from the exact block recipe vs our 2.8e-3 — looser than vLLM-triton's 1.7e-2."""
    E, I2, H = gu.shape
    I = I2 // 2
    packed = (idx.to(torch.int32) << 16) | (
        w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF)
    if cfg["weights"] in ("mxfp4", "nvfp4"):
        return _trtllm_fp4_arm(cfg, hidden, idx, packed, gu, gus, dn, dns, gu_g, dn_g)
    if cfg["weights"] == "mxfp8":
        g1, g1s, dn_s, d1s = _trtllm_mxfp8_prep(gu, gus, dn, dns)
        alpha, beta, limit = _trtllm_gated_scalars(cfg, E, hidden.device)

        def run():
            hq, hsr = _fi_mxfp8_quantize(hidden, False)
            hs = hsr.view(torch.uint8).reshape(hidden.shape[0], -1)
            out = trtllm_fp8_block_scale_routed_moe(
                packed, None, hq, hs, g1, g1s.view(torch.uint8), dn_s, d1s.view(torch.uint8),
                num_experts=E, top_k=idx.shape[1], n_group=None, topk_group=None,
                intermediate_size=I, local_expert_offset=0, local_num_experts=E,
                routed_scaling_factor=None, routing_method_type=1,
                use_shuffled_weight=True, weight_layout=_FiWeightLayout.MajorK,
                fp8_quantization_type=_FiQuantType.MxFp8,
                gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=limit,
            )
            return out[0] if isinstance(out, (list, tuple)) else out

        return run
    g1 = torch.cat([gu[:, I:], gu[:, :I]], dim=1).contiguous()
    gus_f = gus.float()  # UE8M0 first: cat has no e8m0 kernel
    g1s = torch.cat([gus_f[:, I // 128:], gus_f[:, : I // 128]], dim=1).contiguous()
    dns_f = dns.float().contiguous()

    def run():
        # the STACK's own act quant (vLLM's TRT-LLM integration runs this exact kernel) —
        # the arm is the integration end-to-end, never our kernels feeding theirs
        hq, hs = _vllm_group_quant(hidden, 128, column_major_scales=False)
        out = trtllm_fp8_block_scale_routed_moe(
            packed, None, hq, hs.t().contiguous(), g1, g1s, dn, dns_f,
            num_experts=E, top_k=idx.shape[1], n_group=None, topk_group=None,
            intermediate_size=I, local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None, routing_method_type=1,
        )
        return out[0] if isinstance(out, (list, tuple)) else out

    return run


def vllm_moe_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """vLLM triton fused-MoE — the kernel its serving path runs for bf16 and for
    block-fp8 W8A8 (the DeepSeek scheme; dynamic per-token-group act quant inside),
    tuned tiles from bench/vllm_configs. UE8M0 weight scales ride as fp32 (same values —
    vLLM stores block scales fp32). Plain SwiGLU only: the functional API doesn't thread
    clamp_limit/alpha (vLLM's class-based serving path takes them from the quant config).
    MX/NVFP4 rows are out of reach here: the functional API refuses ocp_mx_scheme
    (emulation-only); vLLM's native paths are its class-based FlashInfer/CUTLASS/Marlin
    backends, each with its own weight interleaving."""
    tki = idx.to(torch.int32)
    tkw = w.to(torch.float32)
    qc = None
    if cfg["fp8_block"] is not None:
        qc = _vllm_fp8_qc(w1_scale=gus.float(), w2_scale=dns.float(),
                          block_shape=list(cfg["fp8_block"]))
    return lambda: vllm_fused_experts(hidden, gu, dn, tkw, tki,
                                      activation=_VllmAct.SILU, quant_config=qc)


def deepgemm_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """DeepGEMM M-grouped experts (transformers integration): FP8 128-block (UE8M0
    scales on SM100) or FP4 (int8-packed weights, group-32 UE8M0) with FP8 acts."""
    mod = _Experts(cfg, gu, dn, gus, dns)
    return lambda: deepgemm_fp8_fp4_experts_forward(mod, hidden, idx, w)


def deepgemm_bf16_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    mod = _Experts(cfg, gu, dn)
    return lambda: deepgemm_bf16_experts_forward(mod, hidden, idx, w)


def triton_kernels_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g, dn_g, logits):
    """OpenAI triton_kernels MXFP4 experts (transformers' GPT-OSS path): fused
    ``matmul_ogs`` gate_up+SwiGLU → down over a swizzled MXFP4 weight layout. The
    quantize+swizzle is load-time weight prep (done here, before the timed closure —
    same as our offline packing); the timed call is routing + the two matmuls. One
    fused forward handles any T, so it serves both decode and prefill."""
    import transformers.integrations.mxfp4 as tfmx

    E, H, inter = cfg["E"], cfg["H"], cfg["I"]
    pc = triton_kernels_hub.matmul_ogs.PrecisionConfig
    flex = triton_kernels_hub.matmul_ogs.FlexCtx
    inflex = triton_kernels_hub.matmul_ogs.InFlexData

    def prep(bf16_w):  # (E, out, in) bf16 -> swizzled mxfp4 weight + precision config
        tw, ws = tfmx.quantize_to_mxfp4(
            bf16_w.transpose(-1, -2).contiguous(), triton_kernels_hub)
        tw, ws = tfmx.swizzle_mxfp4(tw, ws, triton_kernels_hub)
        return tw, pc(weight_scale=ws, flex_ctx=flex(rhs_data=inflex()))

    experts = tfmx.Mxfp4GptOssExperts(
        SimpleNamespace(num_local_experts=E, intermediate_size=inter, hidden_size=H,
                        swiglu_limit=cfg["swiglu_limit"] or 7.0)).to(DEV)
    # dequantize the SHARED mxfp4 weights to bf16 and re-quantize through their prep: the
    # values are already on the E2M1 grid, so the round-trip is exact and both impls run
    # bit-identical weights (drawing fresh randn here made parity meaningless).
    dq = WEIGHTS[cfg["weights"]]["dequant"]
    gu_stacked = dq(gu, gus).to(torch.bfloat16)  # (E, 2I, H) = [all gate rows; all up rows]
    # GPT-OSS INTERLEAVES gate/up (modeling_gpt_oss: gate_up[..., ::2] / [..., 1::2]) while our
    # layout stacks them — feeding stacked rows pairs the wrong halves in their SwiGLU.
    gu_bf16 = torch.empty_like(gu_stacked)
    gu_bf16[:, 0::2] = gu_stacked[:, :inter]
    gu_bf16[:, 1::2] = gu_stacked[:, inter:]
    dn_bf16 = dq(dn, dns).to(torch.bfloat16)
    for p in ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"):
        experts._parameters.pop(p, None)
    experts.gate_up_proj, experts.gate_up_proj_precision_config = prep(gu_bf16)
    experts.down_proj, experts.down_proj_precision_config = prep(dn_bf16)
    experts.gate_up_proj_bias = torch.zeros(E, 2 * inter, device=DEV)
    experts.down_proj_bias = torch.zeros(E, H, device=DEV)
    _mark_static(experts.gate_up_proj_bias, experts.down_proj_bias)
    # sm_first=True = softmax over ALL experts then top-k, matching the bench's routing().
    # Their default top-ks first then softmaxes over the k (weights sum to 1, the GPT-OSS
    # convention) — same experts either way (softmax is monotonic), but a per-token rescale
    # of the combine weights that left parity at ~1.1.
    rd, gi, si = triton_kernels_hub.routing.routing(logits, cfg["top_k"], sm_first=True)
    return lambda: experts(hidden, rd, gi, si)


def megablocks_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns, *_):
    """megablocks dropless-MoE: block-sparse grouped GEMM over the routed rows, its own
    permute/histogram/gather ops inside the timed call. Driven through
    ``ParallelDroplessMLP.forward(x, scores, expert_weights, top_experts)`` so the bench's OWN
    routing is used, matching every other arm — ``dMoE`` would run its internal router instead
    and measure a different problem.

    bf16 only: megablocks has no quantized path, so this is the unquantized MoE reference —
    compare it against ``deepgemm_bf16`` and the bf16 row, not against the quantized arms."""
    E, I2, H = gu.shape
    I = I2 // 2
    args = _megablocks.Arguments(
        hidden_size=H, ffn_hidden_size=I, moe_num_experts=E, moe_top_k=cfg["top_k"],
        bf16=True, fp16=False, device=hidden.device, moe_capacity_factor=0,
        # sparse (stk block-sparse) is refused on triton >= 3.2; "grouped" is the grouped-GEMM
        # backend, which is the path megablocks actually recommends on current stacks anyway
        # sparse (stk block-sparse) is refused on triton >= 3.2; "grouped" is the grouped-GEMM
        # backend megablocks recommends on current stacks. mlp_type="glu" is required for SwiGLU
        # models — the default MLP is UNGATED and sizes w1 as [E*I, H], so a gate|up stack cannot
        # load into it.
        mlp_impl="grouped", mlp_type="glu",
        # forward returns (out, bias) unless this is off, and a tuple breaks the parity check
        return_bias=False,
        # megablocks defaults to tanh-approx GELU; the roster's models are SiLU, and the mismatch
        # reads as a plausible-looking 3.7e-2 parity rather than as an error
        activation_fn=torch.nn.functional.silu,
    )
    mlp = _megablocks.ParallelDroplessMLP(args).to(hidden.device, torch.bfloat16)
    # the bench's gate|up is INTERLEAVED (gate at even rows, up at odd); megablocks wants the
    # [gate; up] halves, so de-interleave once, offline
    with torch.no_grad():
        # the bench's gate|up is INTERLEAVED (gate even rows, up odd); the GLU MLP holds them as
        # separate w1 (gate) and v1 (up) tensors, so de-interleave rather than stack
        gate, up = gu[..., 0::2, :], gu[..., 1::2, :]
        # every megablocks GLU buffer is [E*I, H]. gate/up already are (E, I, H), but the bench's
        # down is (E, H, I) — it must be TRANSPOSED, not reshaped: the element count matches either
        # way, so a bare reshape silently scrambles it (parity 1.4e+00, not a crash).
        for name, src in (("w1", gate), ("v1", up), ("w2", dn.transpose(1, 2))):
            buf = getattr(mlp.mlp, name, None)
            if buf is None:
                raise RuntimeError(f"megablocks GLU MLP has no {name}; API changed")
            if buf.numel() != src.numel():
                raise RuntimeError(
                    f"megablocks {name} wants {tuple(buf.shape)} ({buf.numel()} elems), "
                    f"source has {tuple(src.shape)} ({src.numel()})")
            buf.copy_(src.contiguous().reshape(buf.shape).to(torch.bfloat16))
    scores = torch.zeros(hidden.shape[0], E, device=hidden.device, dtype=torch.bfloat16)
    scores.scatter_(1, idx.long(), w.to(torch.bfloat16))
    # belt and braces: some builds still hand back (out, bias)
    def run():
        out = mlp(hidden, scores, w.to(torch.bfloat16), idx.long())
        return out[0] if isinstance(out, tuple) else out

    return run


ARMS = {
    "finegrained-kernels": moe_fused_arm,
    "finegrained-kernels_unfused": moe_unfused_arm,
    "finegrained-fp8": fp8_fused_arm,
    "finegrained-fp8_unfused": fp8_unfused_arm,
    "transformers": transformers_arm,
    "transformers@main": transformers_main_arm,
    "sonicmoe": sonicmoe_arm,
    "vllm": vllm_moe_arm,
    "trtllm": trtllm_moe_arm,
    "deepgemm": deepgemm_arm,
    "deepgemm_bf16": deepgemm_bf16_arm,
    "triton_kernels": triton_kernels_arm,
    "torch": torch_arm,
    "megablocks": megablocks_arm,
}


# ── timing: one warmed process, each mode measured on the same closure ──


def bench_modes(run, tag):
    """{mode: latency_us | None}, plus the eager output for parity. A mode that raises
    is a red CRASH cell — the other modes still run (fresh error printed inline)."""
    res, out = {}, None
    try:
        out = run()
        ACCEL.synchronize()  # warm + tune before ANY timing/capture
        res["eager"] = do_bench(run, return_mode="min") * 1e3
        print(f"      {tag:14s} eager      {res['eager']:9.1f}us", flush=True)
    except Exception as e:
        print(f"      [{tag} eager crashed: {type(e).__name__}: {str(e)[:90]}]", flush=True)
        res["eager"] = None
    try:
        res["cudagraph"] = do_bench_cudagraph(run, return_mode="min") * 1e3
        print(f"      {tag:14s} cudagraph  {res['cudagraph']:9.1f}us", flush=True)
    except Exception as e:
        print(f"      [{tag} cudagraph crashed: {type(e).__name__}: {str(e)[:90]}]", flush=True)
        res["cudagraph"] = None
    try:
        crun = torch.compile(run, mode="max-autotune", fullgraph=True)
        cout = crun()
        ACCEL.synchronize()
        # Self-check the compiled graph against THIS arm's own eager output before timing it.
        # The cross-impl parity below is computed from eager only, so without this a compiled
        # graph that drops work (e.g. an out-param matmul DCE'd because its mutation isn't
        # declared) posts a fast time and a clean parity. The bound catches SKIPPED WORK (zeros
        # or garbage, ~1.0 relative) and nothing tighter: recompiling legitimately re-rounds a
        # quantized path, and W4A8 spreads further than the cross-impl parity we already accept
        # on it (deepgemm-vs-ours 2.3e-2, its own compile-vs-eager 2.1e-2 — a 1e-2 bound flagged
        # the latter as a crash and put a false red X on the figure).
        if isinstance(cout, torch.Tensor) and isinstance(out, torch.Tensor):
            drift = rel_diff(cout, out)
            if drift > 0.25:
                raise RuntimeError(f"compiled output diverges from eager (rel {drift:.2e})")
        res["compile"] = do_bench(crun, return_mode="min") * 1e3
        print(f"      {tag:14s} compile    {res['compile']:9.1f}us", flush=True)
    except Exception as e:
        print(f"      [{tag} compile crashed: {type(e).__name__}: {str(e)[:90]}]", flush=True)
        res["compile"] = None
    return res, out


def rel_diff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-6)).item()


def _mock_rows(row, pname, arms, rows_out):
    """Figure-validation stand-in: plausible random latencies/parities through the
    exact plotting path — decode vs prefill scales, a crashed finegrained-fp8 prefill-compile
    cell (red X), and one wild parity (hatched bar) per row."""
    import random

    rng = random.Random(hash((row, pname)) & 0xFFFF)
    for regime, scale in (("decode", 100.0), ("prefill", 2000.0)):
        for i, name in enumerate(arms):
            res = {m: scale * rng.uniform(0.5, 3.0) for m in MODES}
            if "finegrained-fp8" in name and regime == "prefill":
                res["compile"] = None  # finegrained-fp8's sm_count fullgraph skip
            parity = None if i == 0 else rng.choice([1e-3, 3e-2, 0.12, float("nan")])
            rows_out.append((row, pname, regime, _impl(name), res, parity))


def bench_problem_row(row, pname, cfg, arms, weights, rows_out):
    """One (row, problem): both regimes, finegrained-kernels-first (parity anchor), streaming prints."""
    print(f"== [{row}] {pname}")
    if MOCK:
        _mock_rows(row, pname, arms, rows_out)
        return
    # every problem reuses the same closure code objects — isolate their dynamo
    # state so shapes don't go automatic-dynamic across problems and the shared
    # frames can't hit the fullgraph recompile limit
    torch._dynamo.reset()
    gu, gus, dn, dns, gu_g, dn_g = weights
    for regime, tokens, grouped in (("decode", DECODE_TOKENS, False),
                                    ("prefill", PREFILL_TOKENS, True)):
        print(f"   -- {regime}")
        hidden, idx, w, logits = routing(cfg, tokens)
        args = (cfg, grouped, hidden, idx, w, gu, gus, dn, dns, gu_g, dn_g, logits)
        anchor_res, anchor_out = None, None
        for name in arms:
            try:
                run = ARMS[name](*args)
            except Exception as e:
                print(f"      [{name} setup failed: {type(e).__name__}: {str(e)[:90]}]",
                      flush=True)
                rows_out.append((row, pname, regime, _impl(name),
                                 {m: None for m in MODES}, None))
                continue
            res, out = bench_modes(run, name)
            parity = None
            if anchor_res is None:
                anchor_res, anchor_out = res, out
            elif anchor_out is not None and out is not None:
                parity = rel_diff(anchor_out, out)
                sp = {m: f"{res[m] / anchor_res[m]:.2f}x" for m in MODES
                      if res.get(m) and anchor_res.get(m)}
                print(f"      {name:14s} parity-vs-finegrained-kernels {parity:.1e}"
                      f"  finegrained-kernels-speedup {sp}", flush=True)
            rows_out.append((row, pname, regime, _impl(name), res, parity))
    print()


def _impl(arm_name):
    """Arm -> legend name (fused/unfused variants share the impl color/label)."""
    return arm_name.replace("_unfused", "").replace("_bf16", "")


def bench_attn_row(row, pname, cfg, rows_out):
    """One attn linear per model, in its deployment format (same weights across
    impls; the finegrained-kernels arm's ``input_recipe`` follows the model — GPT-OSS runs W4A4)."""
    print(f"== [{row}] {pname}")
    if MOCK:
        _mock_rows(row, pname, ("finegrained-kernels",) + cfg["baselines"], rows_out)
        return
    torch._dynamo.reset()
    N, K, block = cfg["N"], cfg["K"], cfg["block"]
    W_g = None
    if cfg["weights"] == "fp8_128x128_ue8m0":
        W, Ws = make_weights(N, K, DEV, [128, 128],
                             scale_dtype=torch.float8_e8m0fnu)
    else:
        # registry makers are expert-batched; build E=1 and index the slab off. NVFP4 returns a
        # per-expert fp32 global as its 3rd value (None for single-level FP8/MX) — keep it so the
        # two-level global multiply isn't dropped (it was, silently running NVFP4 at global=1).
        W, Ws, W_g = WEIGHTS[cfg["weights"]]["make"](N, K, 1)
        W, Ws = W[0], Ws[0]
        W_g = W_g if W_g is None else W_g[0]
    _mark_static(W, Ws)
    Ws_fp8 = _fp8_scales(Ws, block)
    dg_block = tuple(block) if block else None
    # deployment layout for the LOCAL arm only (every baseline reads the raw row-major scale):
    # dense attn weights ship pre-swizzled like the MoE arms, so the 2D op benches the tcgen05
    # fast path (weight-only recipes stay affine — no swizzled read)
    Ws_fgm = Ws
    if PRESWIZZLE and cfg["weights"] in _MX_WEIGHTS and _recipe(cfg) and N % 128 == 0:
        Ws_fgm = fgm.swizzle_mx_scales(Ws)
    # OpenAI triton_kernels dense mxfp4 matmul (matmul_ogs, no routing): the qkv linear
    # in the GPT-OSS MXFP4 format. Weight is a single (1, K, N) expert, swizzled once
    # at load (same as the fused arm); latency-only (its own weights).
    if "triton_kernels" in cfg["baselines"]:
        tw_bf = torch.randn(1, K, N, device=DEV, dtype=torch.bfloat16) * 0.05
        tw, tws = _tfmx.quantize_to_mxfp4(tw_bf, triton_kernels_hub)
        tw, tws = _tfmx.swizzle_mxfp4(tw, tws, triton_kernels_hub)
        tk_pc = triton_kernels_hub.matmul_ogs.PrecisionConfig(
            weight_scale=tws,
            flex_ctx=triton_kernels_hub.matmul_ogs.FlexCtx(
                rhs_data=triton_kernels_hub.matmul_ogs.InFlexData()))
        tk_ogs = triton_kernels_hub.matmul_ogs.matmul_ogs
    # torch.nn.functional.scaled_mm (cuBLAS) reference on the MX-family attn rows (mxfp8 /
    # nvfp4) — the same layouts scaled_grouped_mm consumes: torchao-blocked SWIZZLE_32_4_4
    # scales, weight scale blocked once offline, act quant + its blocking inside the timed call
    # (they change per call, the local arm's inline-quant rule). NVFP4 is two-level (block e4m3
    # + TensorWise fp32 globals; dynamic acts ride identity). No torch bar on the BLOCK-FP8 attn
    # rows: torch HAS the DeepSeek scheme (BlockWise1x128 + 128x128) but its CUDA impl is
    # Hopper-only, and RowWise is a different quantization granularity (measured ~20 relative) —
    # timing unlike work on a shared axis is worse than an absent bar.
    torch_mm = None
    if cfg["weights"] in _MX_WEIGHTS and _recipe(cfg) and not (MOCK or REPLOT):
        from torch.nn.functional import ScalingType, SwizzleType
        from torchao.prototype.mx_formats.utils import to_blocked

        nvfp4_row = cfg["weights"] == "nvfp4"
        FP4 = getattr(torch, "float4_e2m1fn_x2", None)
        one = torch.ones(1, device=DEV, dtype=torch.float32)
        if nvfp4_row:
            sb = [to_blocked(Ws.view(torch.uint8)).view(torch.float8_e4m3fn),
                  (W_g if W_g is not None else one).reshape(1)]
            rb = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
            swz = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]
            mat_b = W.view(FP4).t()

            def torch_mm(a):
                aq, a_s = fgm.nvfp4_act_quant(a)
                sa = [to_blocked(a_s.view(torch.uint8)).view(torch.float8_e4m3fn), one]
                return torch.nn.functional.scaled_mm(
                    aq.view(FP4), mat_b, sa, rb, sb, rb,
                    swizzle_a=swz, swizzle_b=swz, output_dtype=torch.bfloat16)
        else:
            sb = to_blocked(Ws.view(torch.uint8)).view(torch.float8_e8m0fnu)
            rb = ScalingType.BlockWise1x32
            swz = SwizzleType.SWIZZLE_32_4_4
            mat_b = W.t()

            def torch_mm(a):
                aq, a_s = fgm.mxfp8_act_quant(a)
                sa = to_blocked(a_s).view(torch.float8_e8m0fnu)
                return torch.nn.functional.scaled_mm(
                    aq, mat_b, sa, rb, sb, rb,
                    swizzle_a=swz, swizzle_b=swz, output_dtype=torch.bfloat16)
    for regime, tokens in (("decode", DECODE_TOKENS), ("prefill", PREFILL_TOKENS)):
        print(f"   -- {regime}")
        torch.manual_seed(0)
        x = torch.randn(tokens, K, device=DEV, dtype=torch.bfloat16)
        # act is inline-quantized (As=None); Ws is the weight scale (Bs). The recipe rides a
        # Quantization (input_recipe = the activation precision); None follows the weight recipe.
        _q = fgm.Quantization(input_recipe=_recipe(cfg)) if _recipe(cfg) else None
        attn_arms = {
            "finegrained-kernels": lambda: fgm.matmul_2d(
                x, W, None, Ws_fgm, quantization=_q, output_dtype=torch.bfloat16, b_global_scale=W_g),
        }
        if torch_mm is not None:
            attn_arms["torch_mm"] = lambda: torch_mm(x)
        if "finegrained-fp8" in cfg["baselines"]:
            attn_arms["finegrained-fp8"] = lambda: upstream_fp8.matmul_2d(x, W, Ws_fp8, block,
                                                       torch.bfloat16)
        if "deepgemm" in cfg["baselines"]:
            attn_arms["deepgemm"] = lambda: deepgemm_fp8_fp4_linear(
                x, W, Ws, block_size=dg_block, output_dtype=torch.bfloat16)
        if "triton_kernels" in cfg["baselines"]:
            attn_arms["triton_kernels"] = lambda: tk_ogs(
                x, tw, None, None, precision_config=tk_pc)
        # The dense GEMM is where transformers' NVFP4 path actually lives, so this is the
        # apples-to-apples cell: same weight, its packer, its kernel. Packing is offline (the
        # checkpoint conversion does it), so only `gemm` is timed.
        if _nvfp4_gemm is not None and cfg["weights"] == "nvfp4":
            # their packer takes a bf16 weight and produces its OWN layout, so feed it the
            # dequantized tensor — `W` is already packed E2M1 (uint8, two values per byte), and
            # casting those bytes to bf16 hands it a K/2-wide matrix ("expected 3072 input
            # features, got 6144"). Same underlying weight, each stack quantizing it its own way,
            # which is how every other baseline here is treated.
            _w_bf16 = WEIGHTS["nvfp4"]["dequant"](
                W[None], Ws[None], (W_g if W_g is not None else torch.ones(1, device=W.device))
            )[0].to(torch.bfloat16)
            _pw = _nvfp4_gemm.pack(_w_bf16, device=x.device)
            attn_arms["nvfp4_gemm"] = lambda: _nvfp4_gemm.gemm(_pw, x)
        anchor_res, anchor_out = None, None
        for name, run in attn_arms.items():
            res, out = bench_modes(run, name)
            parity = None
            if anchor_res is None:
                anchor_res, anchor_out = res, out
            elif anchor_out is not None and out is not None:
                parity = rel_diff(anchor_out, out)
                sp = {m: f"{res[m] / anchor_res[m]:.2f}x" for m in MODES
                      if res.get(m) and anchor_res.get(m)}
                print(f"      {name:14s} parity-vs-finegrained-kernels {parity:.1e}"
                      f"  finegrained-kernels-speedup {sp}", flush=True)
            rows_out.append((row, pname, regime, name, res, parity))
    print()


device_name = "MOCK (random values)" if MOCK else ACCEL.get_device_name(0)
print(f"device: {device_name}  torch {torch.__version__}"
      f"{'  [SMOKE]' if SMOKE else ''}")
print("finegrained-kernels = local build; baselines: finegrained-fp8 (upstream), DeepGEMM, "
      "transformers grouped_mm/batched_mm, SonicMoE, torch.scaled_grouped_mm, "
      "nvfp4-gemm (transformers' NVFP4Linear kernel), megablocks (bf16 dMoE)"
      f"{f'  |  {GPUS} GPUs' if GPUS > 1 else ''}\n")

FILTERS = sys.argv[1:]


def wanted(*names):
    return not FILTERS or any(f in n for f in FILTERS for n in names)


# ── multi-GPU: GPUS>1 shards the per-problem tasks across GPUs, one process per GPU (each
# owning one device via CUDA_VISIBLE_DEVICES). The coordinator spawns GPUS workers, each
# writes a shard CSV, then the coordinator merges + plots. A single GPU (GPUS=1) runs inline. ──
_CSV = os.path.join(_HERE, "bench_moe.csv")
_CSV_HEADER = "category,problem,regime,impl,mode,latency_us,parity_vs_finegrained_kernels\n"


def _write_rows_csv(path, rows_out):
    with open(path, "w") as f:
        f.write(_CSV_HEADER)
        for rr, p, reg, i, res, par in rows_out:
            for mode in MODES:
                v = res.get(mode)
                f.write(f'"{rr}","{p}",{reg},{i},{mode},'
                        f'{"" if v is None else f"{v:.2f}"},'
                        f'{"" if par is None else par}\n')


def _load_rows_csv(path):
    """Rebuild `rows` from a CSV, honoring the CURRENT config's baseline sets so config edits
    (e.g. dropping a baseline) take effect on re-render/merge. impl names in the CSV are already
    the legend names (fused/unfused/bf16 collapsed)."""
    import csv

    def _allowed(cfg):
        return {"finegrained-kernels"} | {_impl(b) for b in cfg["baselines"]}

    # Arms enlisted DYNAMICALLY in _run_task (not in a problem's `baselines`) must be listed
    # here too, or they bench, land in the CSV, and are silently dropped at plot/merge time —
    # a bar that runs and never renders. `torch`/`torch_mm`/`transformers@main` were already
    # special-cased for this; nvfp4_gemm and megablocks are the same shape of arm.
    allowed = {}
    for pn, c in MOE_PROBLEMS.items():
        allowed["quantized", pn] = (
            _allowed(c) | {_impl(b) for b in c.get("fused_extra", ())}
            | {"torch", "transformers@main"})
    for pn, c in BF16_PROBLEMS.items():
        allowed["unquantized", pn] = _allowed(c) | {"megablocks"}
    for pn, c in ATTN_PROBLEMS.items():
        allowed["attn quantized", pn] = _allowed(c) | {"torch_mm", "nvfp4_gemm"}
    acc = {}  # (cat, problem, regime, impl) -> (res dict, parity)
    for r in csv.DictReader(open(path)):
        if r["impl"] not in allowed.get((r["category"], r["problem"]), {"finegrained-kernels"}):
            continue
        key = (r["category"], r["problem"], r["regime"], r["impl"])
        res, par = acc.setdefault(key, ({}, None))
        res[r["mode"]] = float(r["latency_us"]) if r["latency_us"] else None
        if r["parity_vs_finegrained_kernels"]:
            acc[key] = (res, float(r["parity_vs_finegrained_kernels"]))
    return [(cat, p, reg, impl, res, par)
            for (cat, p, reg, impl), (res, par) in acc.items()]


def _run_task(kind, pname, cfg, rows_out):
    """Run one problem's row(s) into rows_out (build its weights once, bench its arms)."""
    if kind in ("moe", "bf16"):
        try:
            weights = None if MOCK else build(cfg)
        except Exception as e:
            print(f"== {pname}\n      [build failed: {type(e).__name__}: {str(e)[:90]}]\n")
            return
    if kind == "moe":
        # torch = the cuBLAS scaled_grouped_mm quantized reference (moe_torch_grouped, its own
        # sort/GLU/reduce — there's no fused-vs-unfused distinction on the torch side), so it rides
        # BOTH rows as the same reference. MX weight families only: F.scaled_grouped_mm has no
        # bf16-act × mxfp4-weight form (W4A16's reference is the triton_kernels matmul_ogs
        # baseline) and no 128×128 block-FP8 form — enlisting those rows just paints a CRASH
        # marker that reads as a baseline failure. fused_extra = other fused-only baselines
        # (single-forward impls with no unfused form).
        # ... and same-family only: scaled_grouped_mm has no mixed-family form (the W4A8
        # mxfp4-weight x mxfp8-act row raises a contraction-dim mismatch on the packed rhs).
        torch_arm_t = ("torch",) if (
            _recipe(cfg) is not None
            and cfg["weights"] in _MX_WEIGHTS
            and cfg["recipe"] == "weights"
        ) else ()
        # the status quo an fgm integration replaces: transformers@main on this checkpoint, via its
        # finegrained_fp8 experts dispatch. It belongs on the UNFUSED row — it applies the GLU in
        # Python between two GEMMs (`self._apply_gate`), so it has no fused epilogue; comparing it
        # against our fused path would be apples-to-oranges. BLOCK-SCALE FP8 only: that dispatch is
        # built around block_size and is transformers' only quantized MoE experts path (GPT-OSS
        # MXFP4 goes through mxfp4.py/triton_kernels instead; MXFP8/NVFP4 MoE has none). Feeding it
        # an MX checkpoint faults the CUDA context (illegal access), killing the whole shard.
        # It runs wherever the kernel it loads is already a baseline AND the activations are
        # quantized: its forward hands `block_size` straight to grouped_matmul (None = MX), so
        # MXFP4/MXFP8 checkpoints work, not just block-scale FP8. It cannot express bf16
        # activations (W4A16) — it always quantizes — and feeding it those faults the CUDA context.
        tfm_arm_t = (("transformers@main",)
                     if "finegrained-fp8" in cfg["baselines"] and _recipe(cfg) is not None else ())
        # nvfp4-gemm is NOT enlisted on the MoE rows. That integration has no MoE kernel, so the
        # only way to run it here is a dense GEMM per expert — 8.8ms against our 47us, which is a
        # statement about the missing kernel rather than a comparison of kernels, and one bar that
        # tall flattens every other bar in the panel. It stays on the attn row, where the two are
        # doing the same job.
        quant_arms = (("finegrained-kernels",) + cfg["baselines"] + cfg.get("fused_extra", ())
                      + tfm_arm_t + torch_arm_t)
        # soft-imported arms drop out where their package is absent; the TRT-LLM fp8-block
        # kernel additionally drops on clamped/scaled SwiGLU rows — passing its gemm1_alpha/
        # beta/clamp_limit there still answers a different function (measured rel 4.4e-1 vs
        # 2e-2 unclamped, and the same scalars land at 2.6e-3 on its fp4 kernel), so a bar
        # would be timing the wrong math
        trtllm_gated_gap = (cfg["weights"] not in ("mxfp4", "nvfp4")
                            and not (cfg["swiglu_alpha"] is None and cfg["swiglu_limit"] is None))
        unavailable = (("vllm",) if vllm_fused_experts is None else ()) + (
            ("trtllm",) if trtllm_fp8_block_scale_routed_moe is None or trtllm_gated_gap else ())
        quant_arms = tuple(a for a in quant_arms if a not in unavailable)
        if wanted("quantized", pname):
            bench_problem_row("quantized", pname, cfg, quant_arms, weights, rows_out)
    elif kind == "bf16":
        # sonic's kernel computes plain SwiGLU only — its integration raises on the clamped/scaled
        # variants (alpha/limit), so enlisting those rows would paint a CRASH that reads as a
        # sonic bug rather than an unsupported activation
        plain_glu = cfg["swiglu_alpha"] is None and cfg["swiglu_limit"] is None
        arms = tuple(a for a in cfg["baselines"]
                     if (a not in ("sonicmoe", "vllm") or plain_glu)
                     and (a != "vllm" or vllm_fused_experts is not None))
        # megablocks is bf16-only (no quantized path), so it belongs on this row and nowhere else
        if _megablocks is not None and plain_glu:
            arms = arms + ("megablocks",)
        bench_problem_row("unquantized", pname, cfg, ("finegrained-kernels",) + arms,
                          weights, rows_out)
    else:  # attn
        bench_attn_row("attn quantized", pname, cfg, rows_out)


# flat, deterministic task list (one entry per problem × row-group), filtered by the CLI substrings
TASKS = ([("moe", p, c) for p, c in MOE_PROBLEMS.items()
          if wanted("quantized", p)]
         + [("bf16", p, c) for p, c in BF16_PROBLEMS.items() if wanted("unquantized", p)]
         + [("attn", p, c) for p, c in ATTN_PROBLEMS.items() if wanted("attn quantized", p)])

rows = []
if REPLOT:
    rows = _load_rows_csv(_CSV)
elif GPUS > 1 and _SHARD is None and not MOCK:
    # COORDINATOR: fan the tasks across GPUS subprocesses (one device each), merge, then plot.
    import subprocess

    shard_paths = [os.path.join(_HERE, f"bench_moe.shard{g}.csv") for g in range(GPUS)]
    for sp in shard_paths:  # a leftover shard from a prior run must never merge as fresh data
        if os.path.exists(sp):
            os.unlink(sp)
    procs = [subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)] + FILTERS,
        env={**os.environ, DEV_MASK_ENV: str(g), "BENCH_SHARD": f"{g}/{GPUS}",
             "GPUS": "1"}) for g in range(GPUS)]
    nfail = sum(p.wait() != 0 for p in procs)
    missing = [g for g, sp in enumerate(shard_paths) if not os.path.exists(sp)]
    if nfail or missing:
        raise SystemExit(
            f"{nfail}/{GPUS} shard worker(s) exited non-zero; shards missing: {missing} — "
            "results are incomplete, not merging (rerun; a partial figure misreads as a full one)"
        )
    with open(_CSV, "w") as out:  # merge shard CSVs (skip repeated headers)
        out.write(_CSV_HEADER)
        for sp in shard_paths:
            with open(sp) as f:
                next(f, None)
                out.writelines(f)
    rows = _load_rows_csv(_CSV)
else:
    # single process (GPUS=1) OR one shard worker (BENCH_SHARD="g/n")
    shard, nshards = (int(x) for x in _SHARD.split("/")) if _SHARD else (0, 1)
    for i, (kind, pname, cfg) in enumerate(TASKS):
        if i % nshards == shard:
            _run_task(kind, pname, cfg, rows)
    if _SHARD is not None:  # WORKER: write only this shard, then exit (coordinator merges+plots)
        _write_rows_csv(os.path.join(_HERE, f"bench_moe.shard{shard}.csv"), rows)
        sys.exit(0)


# ── figure: ONE png — 4 rows x (decode | prefill), linear axes, red CRASH markers ──

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ONE quantized row. Impls with a fused path are benched fused (their best); transformers@main
# has only the two-GEMM shape, so that is what it contributes. No fused/unfused split in the
# figure — it is an internal distinction of ours that most baselines do not have.
ROW_ORDER = ["quantized", "attn quantized", "unquantized"]
present_rows = [r for r in ROW_ORDER if any(rr == r for rr, *_ in rows)]

# bars must be PHYSICALLY identical across every subplot of every figure: one slot
# width (sized by the fullest impl set) and one x data-range on every axis (sparse
# rows centered) — a data-unit width on a stretched sparse axis renders fat bars
GLOBAL_WIDTH = 0.9 / max(
    (len({i for rr, _, reg, i, *_ in rows if rr == r and reg == reg2})
     for r in present_rows for reg2 in ("decode", "prefill")),
    default=1,
)
GLOBAL_SPAN = max(
    (len({p for rr, p, reg, *_ in rows if rr == r and reg == reg2})
     for r in present_rows for reg2 in ("decode", "prefill")),
    default=1,
) - 1

# every (row, problem, regime, impl) x mode -> CSV, so the full 3-mode numbers survive even
# though the figure shows one deployment mode per regime. REPLOT reads this CSV as its source
# (no write), and the multi-GPU coordinator already merged the shard CSVs into it — so only the
# single-process run writes here.
# BENCH_SUFFIX lets filtered rows run concurrently on separate GPUs without racing for
# the same partial CSV/PNG (splice the pieces into bench_moe.csv afterwards)
suffix = os.environ.get("BENCH_SUFFIX") or (
    "_mock" if MOCK else ("_partial" if FILTERS else "")
)
_via_coordinator = GPUS > 1 and _SHARD is None and not MOCK  # merged the CSV already
if not REPLOT and _SHARD is None and not _via_coordinator:
    _write_rows_csv(os.path.join(_HERE, f"bench_moe{suffix}.csv"), rows)

# ONE figure, 8 panels: rows = the 4 categories, cols = (decode | prefill). Each
# bar is colored by impl (finegrained-kernels leftmost, fixed slots). DECODE superposes its two
# graph-captured modes — cudagraph as the solid fill, compile as a black hatched
# outline over the same slot — so the gap between them is visible per impl. PREFILL
# is eager only (single solid) — prefill is not compiled in deployment, so a compile
# bar there would show a mode nobody ships. Red X = crashed (no latency). This chart is latency
# only; parity vs finegrained-kernels lives in the bench log + the CSV beside this png.
import matplotlib.patches as mpatches  # noqa: E402

# (solid_mode, overlay_mode) per regime
REGIME_MODES = {"decode": ("cudagraph", "compile"), "prefill": ("eager", None)}
fig, axes = plt.subplots(max(len(present_rows), 1), 2,
                         figsize=(18, 4.6 * max(len(present_rows), 1)),
                         squeeze=False)
for ri, row in enumerate(present_rows):
    for ci, regime in enumerate(("decode", "prefill")):
        ax = axes[ri][ci]
        solid_mode, overlay_mode = REGIME_MODES[regime]
        cells = [(p, i, r, par) for (rr, p, reg, i, r, par) in rows
                 if rr == row and reg == regime]
        problems = list(dict.fromkeys(p for p, *_ in cells))
        # ONE fixed model order across every row (was per-panel support-sort, which
        # reordered models between rows and read as confusing). Roughly most-supported
        # first, finegrained-kernels-only (GPT-OSS, GLM-NVFP4) last; keyed on the base model so the
        # quantized/attn (GLM-5.2-NVFP4) and unquantized (GLM-5.2) rows line up.
        def _model_rank(p):
            name = p.split(" ")[0].split("/")[-1]
            for i, m in enumerate(CANONICAL_MODEL_ORDER):
                if name.startswith(m):
                    return i
            return len(CANONICAL_MODEL_ORDER)
        problems.sort(key=_model_rank)
        # legend order stays canonical; the BARS are speed-ranked per group below
        row_impls = list(dict.fromkeys(i for _, i, *_ in cells))
        labeled = set()
        overlay_drawn = False  # any compile-beats-cudagraph overlay in this panel?
        eager_only_drawn = False  # any bar shown as eager because cudagraph is unsupported?
        ticks, ticklabels = [], []
        for gi, pname in enumerate(problems):
            ticks.append(gi)
            # compact 2-line tick: model short-name + format (full ids in the log)
            short = pname.split(" (")[0]
            model, _, rest = short.partition(" ")
            model = model.split("/")[-1]
            rest = rest.removeprefix("attn ").removesuffix(" qkv-shaped")
            ticklabels.append(f"{model}\n{rest}" if rest else model)
            # bars speed-ranked within the group (fastest leftmost, crashed last),
            # centered on the group's own arm count — colors identify the impls
            group = [(i, r) for p, i, r, _par in cells if p == pname]
            def _drawn(res):
                """The value this impl will actually show: its solid mode, else — on decode —
                its eager number. An impl that cannot be graph-captured still HAS a latency,
                and a red X hides it behind something that reads as total failure."""
                v = res.get(solid_mode)
                return v if v is not None else (res.get("eager") if regime == "decode" else None)

            group.sort(key=lambda t: (_drawn(t[1]) is None, _drawn(t[1]) or 0.0))
            for slot, (impl, res) in enumerate(group):
                off = (slot - (len(group) - 1) / 2) * GLOBAL_WIDTH
                sval = res.get(solid_mode)
                eager_fb = (res.get("eager") if sval is None and regime == "decode" else None)
                if sval is not None:
                    ax.bar(gi + off, sval, GLOBAL_WIDTH, color=IMPL_COLORS[impl],
                           label=impl if impl not in labeled else None, zorder=2)
                    labeled.add(impl)
                elif eager_fb is not None:
                    # Cannot be graph-captured (megablocks' routing does a device->host copy;
                    # nvfp4_gemm's per-expert loop has data-dependent shapes) but DOES run. Show
                    # the eager cost, dot-hatched so it is never read as a cudagraph number —
                    # a red X here would say "broken" about something merely 5x slower.
                    ax.bar(gi + off, eager_fb, GLOBAL_WIDTH, color=IMPL_COLORS[impl],
                           alpha=0.55, edgecolor="black", hatch="..", linewidth=0.7, zorder=2,
                           label=impl if impl not in labeled else None)
                    labeled.add(impl)
                    eager_only_drawn = True
                else:
                    # nothing ran in any mode -> red X at the baseline
                    ax.plot(gi + off, 0, "x", color="red", markersize=9,
                            clip_on=False, zorder=4)
                # compile overlay (decode): hatched outline — shown ONLY when compile
                # beats cudagraph (a real win); dropped otherwise (compile's usual
                # decode regression is the fixed per-call wrapper overhead, which
                # amortizes at model scale — see notes, not worth cluttering the bar)
                oval = res.get(overlay_mode) if overlay_mode else None
                if oval is not None and sval is not None and oval < sval:
                    ax.bar(gi + off, oval, GLOBAL_WIDTH, facecolor="none",
                           edgecolor="black", hatch="////", linewidth=0.8, zorder=3)
                    overlay_drawn = True
        # dotted separators between model groups (so adjacent models' edge bars
        # don't read as one cluster)
        for sep in range(len(problems) - 1):
            ax.axvline(sep + 0.5, color="0.7", linestyle=":", linewidth=0.8, zorder=0)
        # same x data-range on every axis; sparse rows sit centered
        extent = max(ticks) if ticks else 0
        margin = (GLOBAL_SPAN - extent) / 2
        ax.set_xlim(-0.6 - margin, extent + 0.6 + margin)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7)
        ax.set_ylabel("latency (us)")
        tok = DECODE_TOKENS if regime == "decode" else PREFILL_TOKENS
        mode_note = ("solid=cudagraph, hatch=compile-if-faster" if overlay_drawn
                     else (solid_mode if not overlay_mode else "cudagraph"))
        ax.set_title(f"{row} — {regime} (T={tok}; {mode_note})")
        # legend: impl colors + the mode/marker key (compile entry only if any
        # panel bar actually had a compile win to show)
        handles = [mpatches.Patch(color=IMPL_COLORS[i], label=_impl_label(i, regime))
                   for i in row_impls]
        if overlay_drawn:
            handles.append(mpatches.Patch(facecolor="none", edgecolor="black",
                                          hatch="////", label="compile (faster)"))
        if eager_only_drawn:
            handles.append(mpatches.Patch(facecolor="none", edgecolor="black",
                                          hatch="..", label="eager (no cudagraph)"))
        ax.legend(handles=handles, loc="upper left", fontsize=8)
fig.suptitle(f"MoE bench — finegrained-kernels vs finegrained-fp8 + references  "
             f"({device_name}, real model shapes; decode=cudagraph+compile, "
             f"prefill=eager)", y=0.9995)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out_png = os.path.join(_HERE, f"bench_moe{suffix}.png")
fig.savefig(out_png, dpi=120)
print(f"wrote {out_png}" + ("" if REPLOT else f" + bench_moe{suffix}.csv"))

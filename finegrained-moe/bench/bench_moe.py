"""finegrained-moe bench — local build vs upstream finegrained-fp8 (rev v4) + reference impls.

The **finegrained-moe** arm is the local kernel; **finegrained-fp8** is the upstream hub
build (``kernels-community/finegrained-fp8`` @ ``v4``, which has the fused MoE + MX paths).
By default the finegrained-moe arm feeds PRE-SWIZZLED (SWIZZLE_32_4_4) MX weight scales, so
its numbers reflect the tcgen05 fast path (set ``PRESWIZZLE=0`` for the affine path). Writes
``bench/bench_moe.csv`` (all 3 modes) + ``bench/bench_moe.png`` beside this file.

Rows (each row = decode | prefill subplot pair in the figure):
  fused quantized     moe_fused_*    finegrained-moe vs finegrained-fp8 vs DeepGEMM (transformers integration)
  unfused quantized   moe_unfused_*  finegrained-moe vs finegrained-fp8 (two plain GEMMs + host GLU) vs DeepGEMM
  unquantized (BF16)  finegrained-moe fused vs transformers grouped_mm/batched_mm vs SonicMoE
                      vs DeepGEMM grouped BF16
  attn quantized      matmul_2d, one qkv-proj-shaped linear (N=3H, K=H) per model in
                      its deployment format — FP8 128x128 (finegrained-moe/finegrained-fp8/DeepGEMM), MXFP4
                      W4A4 (finegrained-moe W4A4, finegrained-fp8 W4A8, DeepGEMM FP4), NVFP4 (finegrained-moe only),
                      MXFP8 (finegrained-moe/finegrained-fp8)

MoE problems (real model shapes; one base model per format, same roster BF16'd for
the unquantized row; baselines per problem):
  deepseek-ai/DeepSeek-V4  MXFP4 W4A8         finegrained-fp8, DeepGEMM FP4
  openai/GPT-OSS-120B      full MXFP4 (W4A4)  none — finegrained-fp8 lacks W4A4 AND its kernels
                                              can't run K=2880 (no BK-divides-K guard)
  nvidia/GLM-5.2-NVFP4     NVFP4 (W4A4)       finegrained-moe only — no baseline supports it
  deepseek-ai/DeepSeek-V3  FP8 W8A8 (128x128) finegrained-fp8, DeepGEMM FP8 — UE8M0 block scales (the
                                              B200 deployment format; DeepGEMM SM100
                                              rejects fp32 scales by design)
  MiniMaxAI/MiniMax-M3     MXFP8 W8A8         finegrained-fp8

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
import triton  # noqa: E402
import finegrained_moe as fgm  # noqa: E402  local branch
from finegrained_moe.recipes import ue8m0_as_uint8  # noqa: E402  scale -> uint8 view for swizzle
from kernels import get_kernel  # noqa: E402

# PRESWIZZLE=0 to bench the affine (row-major) MX scale path instead of the pre-swizzled
# SWIZZLE_32_4_4 tcgen05 fast path. Default on: the finegrained-moe arm feeds pre-swizzled
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
# W4A8 pins "mxfp8"); else an MX/NVFP4 weight follows its own recipe (recipe=None defaults the
# activation to fp8, which the kernel rejects for MX/NVFP4 weights), and everything else stays None.
_WEIGHT_RECIPE = {"mxfp8": "mxfp8", "mxfp8_u8": "mxfp8", "mxfp4": "mxfp4", "nvfp4": "nvfp4"}


def _recipe(cfg):
    return cfg["recipe"] or _WEIGHT_RECIPE.get(cfg["weights"])


def _preswizzle_moe_scale(scale, gate):
    """Per-expert SWIZZLE_32_4_4 of a grouped MX weight scale ``(E, rows, K//G)`` -> the 5D
    ``(1, E*blocks, cols//4, 2, 256)`` layout the tcgen05 fast path reads (mirrors
    ``tests/_swizzle_bs``). ``gate``: rows = ``2*I`` (stacked gate|up), block-interleaved
    ``[g0,u0,g1,u1,...]`` so a tile reads its gate + up 128-blocks contiguously; else plain
    per-expert stack. Done once at arm setup (not timed)."""
    bs_u8 = ue8m0_as_uint8(scale)
    E, rows, kg = scale.shape
    cb = triton.cdiv(kg, 4)
    if gate:
        nrbN = triton.cdiv(rows // 2, 128)
        per = [
            fgm.swizzle_mx_scales(bs_u8[e])
            .reshape(2, nrbN, cb, 2, 256)
            .transpose(0, 1)
            .reshape(2 * nrbN, cb, 2, 256)
            for e in range(E)
        ]
        return torch.cat(per).reshape(1, E * 2 * nrbN, cb, 2, 256)
    nrb = triton.cdiv(rows, 128)
    per = [fgm.swizzle_mx_scales(bs_u8[e]) for e in range(E)]
    return torch.cat(per).reshape(1, E * nrb, cb, 2, 256)
from utils import WEIGHTS, make_weights  # noqa: E402  tests/utils.py registry
from transformers.integrations.deepgemm import (  # noqa: E402
    deepgemm_bf16_experts_forward,
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_linear,
)
from transformers.integrations.moe import (  # noqa: E402
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
)
from transformers.integrations.sonicmoe import sonicmoe_experts_forward  # noqa: E402

upstream_fp8 = (None if (MOCK or REPLOT)
          else get_kernel("kernels-community/finegrained-fp8", revision="v4"))

# OpenAI triton_kernels (matmul_ogs) — the MXFP4 experts path transformers uses for
# GPT-OSS. Loaded like finegrained-fp8; its module-level handle drives the mxfp4 swizzle helpers.
if not (MOCK or REPLOT):
    import transformers.integrations.mxfp4 as _tfmx
    triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
    _tfmx.triton_kernels_hub = triton_kernels_hub

DEV = "cuda"
DECODE_TOKENS = 1
PREFILL_TOKENS = 256 if SMOKE else 8192

# fixed left-to-right model order for every figure row (matched by base-model prefix,
# so GLM-5.2-NVFP4 and GLM-5.2 both land in the GLM-5.2 slot). Roughly most-baseline-
# support first, finegrained-moe-only (GPT-OSS, GLM-NVFP4) last.
CANONICAL_MODEL_ORDER = ["DeepSeek-V4", "DeepSeek-V3", "MiniMax-M3", "GPT-OSS-120B", "GLM-5.2"]

MOE_PROBLEMS = {
    "deepseek-ai/DeepSeek-V4 MXFP4 W4A8 (E256 H4096 I2048 top6)": dict(
        E=256, H=4096, I=2048, top_k=6, weights="mxfp4", recipe="mxfp8",
        baselines=("finegrained-fp8", "deepgemm"), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "openai/GPT-OSS-120B MXFP4 W4A4 (E128 H2880 I2880 top4)": dict(
        # no finegrained-fp8 baseline: beyond lacking W4A4, finegrained-fp8's MX kernels have no BK-divides-K
        # guard and their BK span {128,256} doesn't divide 2880 — NaN output.
        # fused_extra: the OpenAI triton_kernels mxfp4 path (transformers' GPT-OSS
        # experts) — a single fused forward, so it's a fused-row-only baseline.
        E=128, H=2880, I=2880, top_k=4, weights="mxfp4", recipe=None,
        baselines=(), fused_extra=("triton_kernels",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
    "nvidia/GLM-5.2-NVFP4 W4A4 (E256 H6144 I2048 top8)": dict(
        E=256, H=6144, I=2048, top_k=8, weights="nvfp4", recipe=None,
        baselines=(), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V3 FP8 block-dyn W8A8 fp32 (E256 H7168 I2048 top8)": dict(
        # DeepSeek-V3 experts deploy fp32 block scales (software rescale). DeepGEMM's SM100
        # experts kernel requires UE8M0 and fails loud on fp32, so no deepgemm baseline here.
        E=256, H=7168, I=2048, top_k=8, weights="fp8_128x128", recipe=None,
        baselines=("finegrained-fp8",), fp8_block=[128, 128], block_size=(128, 128),
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "MiniMaxAI/MiniMax-M3 MXFP8 (E128 H6144 I3072 top4)": dict(
        E=128, H=6144, I=3072, top_k=4, weights="mxfp8", recipe=None,
        baselines=("finegrained-fp8",), fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
}
# the same base-model roster, run as if dequantized to BF16 (one shape per model)
BF16_PROBLEMS = {
    "deepseek-ai/DeepSeek-V4 BF16 (E256 H4096 I2048 top6)": dict(
        E=256, H=4096, I=2048, top_k=6, weights="bf16", recipe=None,
        baselines=("transformers", "sonicmoe", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "openai/GPT-OSS-120B BF16 (E128 H2880 I2880 top4)": dict(
        E=128, H=2880, I=2880, top_k=4, weights="bf16", recipe=None,
        baselines=("transformers", "sonicmoe", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
    "zai-org/GLM-5.2 BF16 (E256 H6144 I2048 top8)": dict(
        E=256, H=6144, I=2048, top_k=8, weights="bf16", recipe=None,
        baselines=("transformers", "sonicmoe", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "deepseek-ai/DeepSeek-V3 BF16 (E256 H7168 I2048 top8)": dict(
        E=256, H=7168, I=2048, top_k=8, weights="bf16", recipe=None,
        baselines=("transformers", "sonicmoe", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=None, swiglu_limit=None,
    ),
    "MiniMaxAI/MiniMax-M3 BF16 (E128 H6144 I3072 top4)": dict(
        E=128, H=6144, I=3072, top_k=4, weights="bf16", recipe=None,
        baselines=("transformers", "sonicmoe", "deepgemm_bf16"),
        fp8_block=None, block_size=None,
        act="silu", swiglu_alpha=1.702, swiglu_limit=7.0,
    ),
}
ATTN_PROBLEMS = {
    "deepseek-ai/DeepSeek-V4 attn FP8 W8A8 ue8m0 qkv-shaped (N=12288 K=4096)": dict(
        # DeepSeek-V4's attention deploys block-FP8 W8A8 with UE8M0 (power-of-two) scales
        # (only the EXPERTS are mixed W4A8) — routes through the tcgen05 dot_scaled arm.
        # K=4096 is on the 128 grid, so finegrained-fp8 (block-dyn) and DeepGEMM FP8 both run it.
        N=12288, K=4096, weights="fp8_128x128_ue8m0", recipe=None,
        block=[128, 128], baselines=("finegrained-fp8", "deepgemm"),
    ),
    "openai/GPT-OSS-120B attn MXFP4 W4A4 qkv-shaped (N=8640 K=2880)": dict(
        # finegrained-moe (W4A4) vs the OpenAI triton_kernels dense mxfp4 matmul. No finegrained-fp8 (no W4A4)
        # or DeepGEMM (its UE8M0 packing asserts K%128==0; 2880 → 90 groups off grid).
        N=8640, K=2880, weights="mxfp4", recipe="mxfp4",
        block=None, baselines=("triton_kernels",),
    ),
    "nvidia/GLM-5.2-NVFP4 attn W4A4 qkv-shaped (N=18432 K=6144)": dict(
        N=18432, K=6144, weights="nvfp4", recipe=None,
        block=None, baselines=(),  # no baseline supports NVFP4
    ),
    "deepseek-ai/DeepSeek-V3 attn FP8 W8A8 fp32 128x128 qkv-shaped (N=21504 K=7168)": dict(
        # DeepSeek-V3 deploys fp32 block scales everywhere (software rescale, not dot_scaled);
        # DeepGEMM's SM100 linear path rounds the fp32 SF to UE8M0 for its own kernel.
        N=21504, K=7168, weights="fp8_128x128", recipe=None,
        block=[128, 128], baselines=("finegrained-fp8", "deepgemm"),
    ),
    "MiniMaxAI/MiniMax-M3 attn MXFP8 W8A8 qkv-shaped (N=18432 K=6144)": dict(
        N=18432, K=6144, weights="mxfp8", recipe=None,
        block=None, baselines=("finegrained-fp8",),  # DeepGEMM FP8 is 128-block, not group-32 MX
    ),
}
MODES = ["eager", "cudagraph", "compile"]

IMPL_COLORS = {
    "finegrained-moe": "#1f77b4",
    "finegrained-fp8": "#bbbbbb",
    "deepgemm": "#2ca02c",
    "deepgemm_bf16": "#2ca02c",
    "transformers": "#9467bd",
    "sonicmoe": "#ff7f0e",
    "triton_kernels": "#8c564b",  # OpenAI mxfp4 (GPT-OSS) reference
    "torch": "#e377c2",  # torch/cuBLAS F.scaled_grouped_mm (quantized prefill reference)
}


def _impl_label(impl, regime):
    """Legend name. The reference forwards ARE plain torch ops — name them by the op:
    transformers batched_mm -> torch.bmm (decode) / grouped_mm -> torch._grouped_mm (bf16);
    ``torch`` -> torch.scaled_grouped_mm (the quantized cuBLAS path). Every other impl
    (finegrained-moe, finegrained-fp8, deepgemm, ...) is its own label."""
    if impl == "transformers":
        return "torch.bmm" if regime == "decode" else "torch._grouped_mm"
    if impl == "torch":
        return "torch.scaled_grouped_mm"
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
    problem uses UE8M0 block scales (finegrained-fp8/finegrained-moe decode them natively; DeepGEMM SM100
    requires them)."""
    E, H, inter = cfg["E"], cfg["H"], cfg["I"]
    if cfg["weights"] == "fp8_128x128_ue8m0":
        def make(n, k, e):
            return (*make_weights(n, k, DEV, [128, 128],
                                  scale_dtype=torch.float8_e8m0fnu, num_experts=e), None)
    else:
        make = WEIGHTS[cfg["weights"]]["make"]
    # finegrained-fp8 recipes are single-level FP8/MX (block scales only); the per-tensor global is None.
    gu, gus, _ = make(2 * inter, H, E)
    dn, dns, _ = make(H, inter, E)
    _mark_static(gu, gus, dn, dns)
    return gu, gus, dn, dns


def routing(cfg, tokens):
    torch.manual_seed(0)
    hidden = torch.randn(tokens, cfg["H"], device=DEV, dtype=torch.bfloat16)
    logits = torch.randn(tokens, cfg["E"], device=DEV)
    w, idx = torch.topk(torch.softmax(logits, -1), cfg["top_k"], dim=-1)
    return hidden, idx.to(torch.int32), w


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
    ``is_transposed=False``; gate|up rows are stacked (concatenated), not interleaved."""

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
        gate, up = gate_up_out.chunk(2, dim=-1)
        return _glu(gate, up, self._cfg).to(gate_up_out.dtype)


# ── MoE impl arms: each returns a no-arg closure computing the full forward ──


def moe_fused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    """``recipe`` sets the activation precision; None follows the weight recipe
    (mxfp4/nvfp4 -> the all-fp4 W4A4 chain, bf16 -> unquantized). dsv4 deploys
    W4A8, so it pins recipe="mxfp8". Under ``PRESWIZZLE`` the MX weight scales are
    pre-swizzled into SWIZZLE_32_4_4 so the forward takes the tcgen05 fast path."""
    fn = fgm.moe_fused_grouped if grouped else fgm.moe_fused_batched
    if _can_preswizzle(cfg):
        gus = _preswizzle_moe_scale(gus, gate=True)
        dns = _preswizzle_moe_scale(dns, gate=False)
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg))
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, **kw)


def moe_unfused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    fn = fgm.moe_unfused_grouped if grouped else fgm.moe_unfused_batched
    if _can_preswizzle(cfg):
        gus = _preswizzle_moe_scale(gus, gate=True)
        dns = _preswizzle_moe_scale(dns, gate=False)
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg))
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, **kw)


def torch_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    """torch / cuBLAS reference: ``moe_torch_grouped`` over the PUBLIC ``F.scaled_grouped_mm``
    (two scaled grouped GEMMs + host GLU + weighted reduce). This is the quantized-path torch
    baseline the ``transformers`` grouped_mm arm is for BF16. GROUPED/prefill only
    (scaled_grouped_mm has no per-token decode form and cuBLAS has no quantized MoE decode) — so
    it fails-skips on decode. Fed AFFINE scales: ``moe_torch_grouped`` builds its own (torchao)
    swizzle for scaled_grouped_mm — the kernel's ``_preswizzle_moe_scale`` layout is a different
    5D and scaled_grouped_mm rejects it."""
    if not grouped:
        raise NotImplementedError("scaled_grouped_mm is prefill/grouped only")
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"], recipe=_recipe(cfg))
    return lambda: fgm.moe_torch_grouped(hidden, idx, w, gu, dn, gus, dns, **kw)


def _fp8_scales(t, block):
    """finegrained-fp8's BLOCK-FP8 path predates UE8M0 scales (KeyError on e8m0) — hand it the
    exact fp32 values instead (UE8M0 holds pure exponents: lossless, same math).
    Its MX path reads UE8M0 natively, so MX scales (block None) pass through."""
    if block is not None and t is not None and t.dtype == torch.float8_e8m0fnu:
        t = t.float()
        _mark_static(t)
    return t


def fp8_fused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    """Hub finegrained-fp8 fused forward (block_size positional; None = MX, [128,128] = block FP8)."""
    fn = upstream_fp8.moe_fused_grouped if grouped else upstream_fp8.moe_fused_batched
    gus = _fp8_scales(gus, cfg["fp8_block"])
    dns = _fp8_scales(dns, cfg["fp8_block"])
    kw = dict(act_fn=cfg["act"], swiglu_alpha=cfg["swiglu_alpha"],
              swiglu_limit=cfg["swiglu_limit"])
    return lambda: fn(hidden, idx, w, gu, dn, gus, dns, cfg["fp8_block"], **kw)


def fp8_unfused_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
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


def transformers_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    """transformers.integrations.moe reference: grouped_mm (prefill) / batched_mm (decode)."""
    mod = _Experts(cfg, gu, dn)
    fwd = grouped_mm_experts_forward if grouped else batched_mm_experts_forward
    return lambda: fwd(mod, hidden, idx, w)


def sonicmoe_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    mod = _Experts(cfg, gu, dn)
    return lambda: sonicmoe_experts_forward(mod, hidden, idx, w)


def deepgemm_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    """DeepGEMM M-grouped experts (transformers integration): FP8 128-block (UE8M0
    scales on SM100) or FP4 (int8-packed weights, group-32 UE8M0) with FP8 acts."""
    mod = _Experts(cfg, gu, dn, gus, dns)
    return lambda: deepgemm_fp8_fp4_experts_forward(mod, hidden, idx, w)


def deepgemm_bf16_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
    mod = _Experts(cfg, gu, dn)
    return lambda: deepgemm_bf16_experts_forward(mod, hidden, idx, w)


def triton_kernels_arm(cfg, grouped, hidden, idx, w, gu, gus, dn, dns):
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
    torch.manual_seed(0)
    gu_bf16 = torch.randn(E, 2 * inter, H, device=DEV, dtype=torch.bfloat16) * 0.05
    dn_bf16 = torch.randn(E, H, inter, device=DEV, dtype=torch.bfloat16) * 0.05
    for p in ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"):
        experts._parameters.pop(p, None)
    experts.gate_up_proj, experts.gate_up_proj_precision_config = prep(gu_bf16)
    experts.down_proj, experts.down_proj_precision_config = prep(dn_bf16)
    experts.gate_up_proj_bias = torch.zeros(E, 2 * inter, device=DEV)
    experts.down_proj_bias = torch.zeros(E, H, device=DEV)
    _mark_static(experts.gate_up_proj_bias, experts.down_proj_bias)
    logits = torch.randn(hidden.shape[0], E, device=DEV)
    rd, gi, si = triton_kernels_hub.routing.routing(logits, cfg["top_k"])
    return lambda: experts(hidden, rd, gi, si)


ARMS = {
    "finegrained-moe": moe_fused_arm,
    "finegrained-moe_unfused": moe_unfused_arm,
    "finegrained-fp8": fp8_fused_arm,
    "finegrained-fp8_unfused": fp8_unfused_arm,
    "transformers": transformers_arm,
    "sonicmoe": sonicmoe_arm,
    "deepgemm": deepgemm_arm,
    "deepgemm_bf16": deepgemm_bf16_arm,
    "triton_kernels": triton_kernels_arm,
    "torch": torch_arm,
}


# ── timing: one warmed process, each mode measured on the same closure ──


def bench_modes(run, tag):
    """{mode: latency_us | None}, plus the eager output for parity. A mode that raises
    is a red CRASH cell — the other modes still run (fresh error printed inline)."""
    res, out = {}, None
    try:
        out = run()
        torch.cuda.synchronize()  # warm + tune before ANY timing/capture
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
        crun()
        torch.cuda.synchronize()
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
    """One (row, problem): both regimes, finegrained-moe-first (parity anchor), streaming prints."""
    print(f"== [{row}] {pname}")
    if MOCK:
        _mock_rows(row, pname, arms, rows_out)
        return
    # every problem reuses the same closure code objects — isolate their dynamo
    # state so shapes don't go automatic-dynamic across problems and the shared
    # frames can't hit the fullgraph recompile limit
    torch._dynamo.reset()
    gu, gus, dn, dns = weights
    for regime, tokens, grouped in (("decode", DECODE_TOKENS, False),
                                    ("prefill", PREFILL_TOKENS, True)):
        print(f"   -- {regime}")
        hidden, idx, w = routing(cfg, tokens)
        args = (cfg, grouped, hidden, idx, w, gu, gus, dn, dns)
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
                print(f"      {name:14s} parity-vs-finegrained-moe {parity:.1e}"
                      f"  finegrained-moe-speedup {sp}", flush=True)
            rows_out.append((row, pname, regime, _impl(name), res, parity))
    print()


def _impl(arm_name):
    """Arm -> legend name (fused/unfused variants share the impl color/label)."""
    return arm_name.replace("_unfused", "").replace("_bf16", "")


def bench_attn_row(row, pname, cfg, rows_out):
    """One attn linear per model, in its deployment format (same weights across
    impls; the finegrained-moe arm's ``input_recipe`` follows the model — GPT-OSS runs W4A4)."""
    print(f"== [{row}] {pname}")
    if MOCK:
        _mock_rows(row, pname, ("finegrained-moe",) + cfg["baselines"], rows_out)
        return
    torch._dynamo.reset()
    N, K, block = cfg["N"], cfg["K"], cfg["block"]
    if cfg["weights"] == "fp8_128x128_ue8m0":
        W, Ws = make_weights(N, K, DEV, [128, 128],
                             scale_dtype=torch.float8_e8m0fnu)
    else:
        # registry makers are expert-batched; build E=1 and index the slab off (single-level
        # FP8/MX recipe, so the per-tensor global is None and dropped).
        W, Ws, _ = WEIGHTS[cfg["weights"]]["make"](N, K, 1)
        W, Ws = W[0], Ws[0]
    _mark_static(W, Ws)
    Ws_fp8 = _fp8_scales(Ws, block)
    dg_block = tuple(block) if block else None
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
    for regime, tokens in (("decode", DECODE_TOKENS), ("prefill", PREFILL_TOKENS)):
        print(f"   -- {regime}")
        torch.manual_seed(0)
        x = torch.randn(tokens, K, device=DEV, dtype=torch.bfloat16)
        # act is inline-quantized (As=None); Ws is the weight scale (Bs). The recipe rides a
        # Quantization (input_recipe = the activation precision); None follows the weight recipe.
        _q = fgm.Quantization(input_recipe=_recipe(cfg)) if _recipe(cfg) else None
        attn_arms = {
            "finegrained-moe": lambda: fgm.matmul_2d(
                x, W, None, Ws, quantization=_q, output_dtype=torch.bfloat16),
        }
        if "finegrained-fp8" in cfg["baselines"]:
            attn_arms["finegrained-fp8"] = lambda: upstream_fp8.matmul_2d(x, W, Ws_fp8, block,
                                                       torch.bfloat16)
        if "deepgemm" in cfg["baselines"]:
            attn_arms["deepgemm"] = lambda: deepgemm_fp8_fp4_linear(
                x, W, Ws, block_size=dg_block, output_dtype=torch.bfloat16)
        if "triton_kernels" in cfg["baselines"]:
            attn_arms["triton_kernels"] = lambda: tk_ogs(
                x, tw, None, None, precision_config=tk_pc)
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
                print(f"      {name:14s} parity-vs-finegrained-moe {parity:.1e}"
                      f"  finegrained-moe-speedup {sp}", flush=True)
            rows_out.append((row, pname, regime, name, res, parity))
    print()


device_name = "MOCK (random values)" if MOCK else torch.cuda.get_device_name(0)
print(f"device: {device_name}  torch {torch.__version__}"
      f"{'  [SMOKE]' if SMOKE else ''}")
print("finegrained-moe = local build; baselines: finegrained-fp8 (upstream), DeepGEMM, "
      "transformers grouped_mm/batched_mm, SonicMoE, torch.scaled_grouped_mm"
      f"{f'  |  {GPUS} GPUs' if GPUS > 1 else ''}\n")

FILTERS = sys.argv[1:]


def wanted(*names):
    return not FILTERS or any(f in n for f in FILTERS for n in names)


# ── multi-GPU: GPUS>1 shards the per-problem tasks across GPUs, one process per GPU (each
# owning one device via CUDA_VISIBLE_DEVICES). The coordinator spawns GPUS workers, each
# writes a shard CSV, then the coordinator merges + plots. A single GPU (GPUS=1) runs inline. ──
_CSV = os.path.join(_HERE, "bench_moe.csv")
_CSV_HEADER = "category,problem,regime,impl,mode,latency_us,parity_vs_finegrained_moe\n"


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
        return {"finegrained-moe"} | {_impl(b) for b in cfg["baselines"]}

    allowed = {}
    for pn, c in MOE_PROBLEMS.items():
        allowed["unfused quantized", pn] = _allowed(c)
        allowed["fused quantized", pn] = (
            _allowed(c) | {_impl(b) for b in c.get("fused_extra", ())} | {"torch"})
    for pn, c in BF16_PROBLEMS.items():
        allowed["unquantized", pn] = _allowed(c)
    for pn, c in ATTN_PROBLEMS.items():
        allowed["attn quantized", pn] = _allowed(c)
    acc = {}  # (cat, problem, regime, impl) -> (res dict, parity)
    for r in csv.DictReader(open(path)):
        if r["impl"] not in allowed.get((r["category"], r["problem"]), {"finegrained-moe"}):
            continue
        key = (r["category"], r["problem"], r["regime"], r["impl"])
        res, par = acc.setdefault(key, ({}, None))
        res[r["mode"]] = float(r["latency_us"]) if r["latency_us"] else None
        if r["parity_vs_finegrained_moe"]:
            acc[key] = (res, float(r["parity_vs_finegrained_moe"]))
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
        unfused_arms = tuple(a + "_unfused" if a in ("finegrained-moe", "finegrained-fp8") else a
                             for a in ("finegrained-moe",) + cfg["baselines"])
        # torch = the cuBLAS scaled_grouped_mm quantized reference (fused-only); fused_extra =
        # other fused-only baselines (single-forward impls with no unfused form)
        fused_arms = (("finegrained-moe",) + cfg["baselines"]
                      + cfg.get("fused_extra", ()) + ("torch",))
        if wanted("fused quantized", pname):
            bench_problem_row("fused quantized", pname, cfg, fused_arms, weights, rows_out)
        if wanted("unfused quantized", pname):
            bench_problem_row("unfused quantized", pname, cfg, unfused_arms, weights, rows_out)
    elif kind == "bf16":
        bench_problem_row("unquantized", pname, cfg, ("finegrained-moe",) + cfg["baselines"],
                          weights, rows_out)
    else:  # attn
        bench_attn_row("attn quantized", pname, cfg, rows_out)


# flat, deterministic task list (one entry per problem × row-group), filtered by the CLI substrings
TASKS = ([("moe", p, c) for p, c in MOE_PROBLEMS.items()
          if wanted("fused quantized", p) or wanted("unfused quantized", p)]
         + [("bf16", p, c) for p, c in BF16_PROBLEMS.items() if wanted("unquantized", p)]
         + [("attn", p, c) for p, c in ATTN_PROBLEMS.items() if wanted("attn quantized", p)])

rows = []
if REPLOT:
    rows = _load_rows_csv(_CSV)
elif GPUS > 1 and _SHARD is None and not MOCK:
    # COORDINATOR: fan the tasks across GPUS subprocesses (one device each), merge, then plot.
    import subprocess

    procs = [subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)] + FILTERS,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "BENCH_SHARD": f"{g}/{GPUS}",
             "GPUS": "1"}) for g in range(GPUS)]
    nfail = sum(p.wait() != 0 for p in procs)
    if nfail:
        print(f"[warning] {nfail}/{GPUS} shard worker(s) exited non-zero", flush=True)
    with open(_CSV, "w") as out:  # merge shard CSVs (skip repeated headers)
        out.write(_CSV_HEADER)
        for g in range(GPUS):
            sp = os.path.join(_HERE, f"bench_moe.shard{g}.csv")
            if os.path.exists(sp):
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

ROW_ORDER = ["fused quantized", "unfused quantized", "attn quantized", "unquantized"]
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
suffix = "_mock" if MOCK else ("_partial" if FILTERS else "")
_via_coordinator = GPUS > 1 and _SHARD is None and not MOCK  # merged the CSV already
if not REPLOT and _SHARD is None and not _via_coordinator:
    _write_rows_csv(os.path.join(_HERE, f"bench_moe{suffix}.csv"), rows)

# ONE figure, 8 panels: rows = the 4 categories, cols = (decode | prefill). Each
# bar is colored by impl (finegrained-moe leftmost, fixed slots). DECODE superposes its two
# graph-captured modes — cudagraph as the solid fill, compile as a black hatched
# outline over the same slot — so the gap between them is visible per impl. PREFILL
# is eager only (single solid). Red X = crashed (no latency). This chart is latency
# only; parity vs finegrained-moe lives in the bench log + the CSV beside this png.
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
        # first, finegrained-moe-only (GPT-OSS, GLM-NVFP4) last; keyed on the base model so the
        # quantized/attn (GLM-5.2-NVFP4) and unquantized (GLM-5.2) rows line up.
        def _model_rank(p):
            name = p.split(" ")[0].split("/")[-1]
            for i, m in enumerate(CANONICAL_MODEL_ORDER):
                if name.startswith(m):
                    return i
            return len(CANONICAL_MODEL_ORDER)
        problems.sort(key=_model_rank)
        # FIXED impl slots: every impl keeps the same offset under every tick
        # (finegrained-moe leftmost); unsupported impls leave their slot empty
        row_impls = list(dict.fromkeys(i for _, i, *_ in cells))
        labeled = set()
        overlay_drawn = False  # any compile-beats-cudagraph overlay in this panel?
        ticks, ticklabels = [], []
        for gi, pname in enumerate(problems):
            ticks.append(gi)
            # compact 2-line tick: model short-name + format (full ids in the log)
            short = pname.split(" (")[0]
            model, _, rest = short.partition(" ")
            model = model.split("/")[-1]
            rest = rest.removeprefix("attn ").removesuffix(" qkv-shaped")
            ticklabels.append(f"{model}\n{rest}" if rest else model)
            for impl in row_impls:
                off = (row_impls.index(impl)
                       - (len(row_impls) - 1) / 2) * GLOBAL_WIDTH
                cell = next(((r, par) for p, i, r, par in cells
                             if p == pname and i == impl), None)
                if cell is None:
                    continue  # impl doesn't support this problem — empty slot
                res = cell[0]
                sval = res.get(solid_mode)
                if sval is not None:
                    ax.bar(gi + off, sval, GLOBAL_WIDTH, color=IMPL_COLORS[impl],
                           label=impl if impl not in labeled else None, zorder=2)
                    labeled.add(impl)
                else:
                    # solid (deployment) mode crashed -> red X at the baseline
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
        ax.legend(handles=handles, loc="upper left", fontsize=8)
fig.suptitle(f"MoE bench — finegrained-moe vs finegrained-fp8 + references  "
             f"({device_name}, real model shapes; decode=cudagraph+compile, "
             f"prefill=eager)", y=0.9995)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out_png = os.path.join(_HERE, f"bench_moe{suffix}.png")
fig.savefig(out_png, dpi=120)
print(f"wrote {out_png}" + ("" if REPLOT else f" + bench_moe{suffix}.csv"))

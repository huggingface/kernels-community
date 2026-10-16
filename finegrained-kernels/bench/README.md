# finegrained-kernels bench

`bench_moe.py` benchmarks the local **finegrained-kernels** kernel against the upstream
**finegrained-fp8** kernel (`kernels-community/finegrained-fp8` @ `v4`) and external
reference implementations, on real model shapes.

## What it compares

Three figure rows, each a **decode | prefill** subplot pair:

- **quantized** — the MoE at every impl's best: `moe_fused_*` against finegrained-fp8, DeepGEMM,
  vLLM, TRT-LLM, `transformers@main` and `triton_kernels`. Each impl contributes the shape it has —
  fused where there is one, the two-GEMM shape for `transformers@main`, which has only that. There
  is no fused/unfused split in the figure: it is an internal distinction of ours that most of these
  baselines do not have. The unfused arms (`moe_unfused_*`) stay in `ARMS`, but no problem's
  baseline set enlists them, so they do not run by default.
- **linear quantized** — one qkv-shaped `matmul_2d` linear per model, in its deployment format
- **unquantized (BF16)** — fused vs transformers `grouped_mm`/`batched_mm`, SonicMoE, vLLM,
  DeepGEMM BF16 and megablocks

Baselines per problem ("all kinds"): upstream **finegrained-fp8** (`@ v4`), **DeepGEMM**
(fp8/fp4/bf16), **vLLM**'s fused MoE, **FlashInfer**'s TRT-LLM routed-MoE kernels,
**transformers** `grouped_mm`/`batched_mm` (= `torch._grouped_mm` / `torch.bmm`, the BF16
torch/cuBLAS path), **SonicMoE**, the OpenAI **triton_kernels** MXFP4 path (GPT-OSS),
**megablocks** (BF16 dMoE), **nvfp4-gemm** (the kernel behind transformers' `NVFP4Linear`) and
**`torch.scaled_grouped_mm`** / **`torch.scaled_mm`** (the cuBLAS references). Each is
import-guarded — a missing dependency skips that baseline instead of failing the run.

Every cell runs in three modes: `eager`, `cudagraph` (decode's deployment mode), and
`compile` (`torch.compile(max-autotune, fullgraph)`). A cell that raises is a red ✕ marker;
the others still run.

## The pre-swizzled fast path

By default the finegrained-kernels arm feeds **pre-swizzled** (`SWIZZLE_32_4_4`) MX weight scales,
so its numbers reflect the tcgen05 fast path. Only MX weights on 128-aligned dims are swizzled
(the routed guard rejects non-128 gate/N); block-fp8 and BF16 stay affine. Pass `--no-preswizzle`
to measure the affine path instead.

Correctness is cross-checked in-run: each baseline's output is compared to the finegrained-kernels
anchor (`parity-vs-finegrained-kernels`, relative L2 distance). What the number means, by regime:

- **fp8 ~3e-3 / MX ~1e-2 / fp4 ~5e-2** — arms that share our activation quant: only reduction order
  and epilogue rounding differ.
- **up to ~1e-1** — stacks with their own act-quant kernels (TRT-LLM, vLLM, finegrained-fp8,
  NVFP4Linear): the two arms quantize the activations independently (E4M3 has a 3-bit mantissa,
  E2M1 a 1-bit one, so two roundings of the same value differ by up to ~6% / ~25% per element) and
  the two quantized intermediates go through a second GEMM. These are measured values, not a
  bound, and this oracle cannot tell that regime apart from a bug of the same magnitude — the model
  smokes (generation) and the torch-reference tests carry that.
- **of order 1 (measured 1.3–1.7)** — the outputs are uncorrelated: the relative distance between two
  independent same-magnitude vectors is √2. A wrong gate|up row order (pairing the wrong halves in
  the GLU) or a routing error lands here, which is how the stacked-vs-interleaved feed was caught.

The bench's gate|up weights are STACKED (`[gate; up]`); the finegrained-kernels arms are fed the
row-interleaved view (`_interleave_gate_up`) because the kernels read gate|up interleaved.

## Running

```bash
python bench/bench_moe.py                    # full grid, single GPU -> bench_moe.csv + bench_moe.png
python bench/bench_moe.py --gpus 8           # shard problems across 8 GPUs (one process per GPU), then merge + plot
python bench/bench_moe.py --gpus 5 --devices 3,4,5,6,7   # pin the shards to GPUs 3-7
python bench/bench_moe.py --smoke            # fast everything-compiles pass (3-trial tunes, 256-tok prefill)
python bench/bench_moe.py --no-preswizzle    # affine MX scales instead of the fast path
python bench/bench_moe.py gpt-oss            # substring filter on row/problem names
python bench/bench_moe.py --replot           # rebuild the figure from an existing bench_moe.csv
python bench/bench_moe.py --mock             # no GPU: random latencies to validate the figure layout
```

Outputs land beside the script (`bench/bench_moe.csv`, `bench/bench_moe.png`). The figure is
committed (the top-level README embeds it), so refresh it in the same commit as a kernel change
that moves the numbers; the CSV is git-ignored (`bench/*.csv` in the repo's `.gitignore`) and regenerated per machine. Requires the bench env: `transformers`
(with the `integrations.{deepgemm,moe,sonicmoe,mxfp4}` helpers), `kernels`, DeepGEMM, and a
Blackwell (sm_100) GPU. **Don't run under concurrent GPU load** — the latencies won't be trustworthy.

## Rowwise FP8 and coarse-scale W8A16 linears

`bench_fp8_linear.py` measures dense linears through `kernels.get_kernel`. Select a
local build with `LOCAL_KERNELS` when comparing revisions. Performance runs require
at least 100 autotuning trials; the one-trial correctness policy is not used.

```bash
FINEGRAINED_AUTOTUNE_TRIALS=100 python bench/bench_fp8_linear.py \
  --family weight_only \
  --shapes '1,4096,4096;256,4096,4096;2048,4096,4096;1,16384,5120;256,16384,5120;2048,16384,5120' \
  --output weight_only.jsonl
FINEGRAINED_AUTOTUNE_TRIALS=100 python bench/bench_fp8_linear.py \
  --family rowwise \
  --shapes '1,16384,5120;16,16384,5120;256,16384,5120;4096,16384,5120;1,5120,8192;16,5120,8192;256,5120,8192;4096,5120,8192' \
  --output rowwise.jsonl
```

The JSONL records medians of five warm CUDA graph measurements and three cold
measurements. Cold measurements flush L2 between single-operation graph replays,
so Python and TMA descriptor construction are excluded from both timings. Warm
replays can keep these weights in B200 L2; neither measurement is whole-model
throughput. Rowwise results include activation quantization and compare against
cuBLAS rowwise FP8, including its quantization cost. Weight-only results compare
against both per-call dequantization plus cuBLAS and persistent BF16 weights.
Persistent BF16 weights use twice the weight storage; they are a speed reference,
not an equivalent memory configuration.

B200 W8A16 results (Torch 2.14.1+cu130, Triton 3.8.0; microseconds, BF16 activations,
128×128 weight scales):

| M | N | K | W8A16 warm | W8A16 cold | Persistent BF16 warm |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 4096 | 8.09 | 14.34 | 6.24 |
| 256 | 4096 | 4096 | 15.62 | 26.62 | 8.46 |
| 2048 | 4096 | 4096 | 59.56 | 61.63 | 50.43 |
| 1 | 16384 | 5120 | 26.87 | 36.86 | 26.70 |
| 256 | 16384 | 5120 | 92.56 | 92.10 | 42.73 |
| 2048 | 16384 | 5120 | 292.19 | 288.93 | 241.94 |

One-token W8A16 uses a scalar split-K reduction, including its finishing kernel
in tuning. Larger dense calls use a temporary BF16 weight workspace plus cuBLAS
when eligible; the table includes conversion and GEMM work. CUDA graph replay
reuses captured allocations. The workspace requires two bytes per weight for the
duration of the call.
Fused gate/bias epilogues retain the existing full-precision Triton path after
conversion. Persistent weights stay FP8. These coarse-scale W8A16 operations are
inference-only; their custom operators do not register an autograd formula.

Sparse routed decode reuses the split-K path for at most 16 routed rows and at
most two rows per expert on average. It reads only the selected expert weights
and fuses gather/scatter and bias/GLU, including invalid off-rank route masking.
This path does not create a BF16 expert-weight workspace. Reproduce routed
measurements with:

```bash
FINEGRAINED_AUTOTUNE_TRIALS=100 python bench/bench_fp8_routed.py \
  --family weight_only --kind grouped --rows 1,2 --output routed.jsonl
FINEGRAINED_AUTOTUNE_TRIALS=100 python bench/bench_fp8_routed.py \
  --family weight_only --kind batched --rows 1 --gate --gather --output gate.jsonl
```

With eight active experts, N1280/K5120 (52 MB of FP8 weights):

| Rows per expert | W8A16 warm / cold | Persistent BF16 bmm warm / cold | Per-call dequant + bmm warm / cold |
|---:|---:|---:|---:|
| 1 | 19.35 / 28.64 | 17.32 / 30.72 | 49.21 / 53.28 |
| 2 | 31.94 / 40.93 | 17.55 / 32.74 | 50.76 / 57.18 |

The BF16 reference retains twice the weight storage. The warm eight-expert case
fits in B200 L2; cold replay is also reported to avoid relying on that favorable
cache residency. Larger routed batches retain the original in-tile W8A16 path;
its throughput and large number of shared-memory rejections during tuning remain
known limitations. The dense and sparse decode improvements do not imply tuned
routed prefill performance.

The batched gate+gather case (eight experts, logical N1280/K5120; 105 MB FP8
weights) measured 36.64 µs warm / 45.06 cold, versus persistent BF16 bmm plus
GLU at 41.00 / 57.34 and per-call dequant+bmm+GLU at 102.37 / 108.54. The
committed routed harness reproduces this comparison, including the fused
finishing reduction in the FP8 path and each baseline's epilogue cost.

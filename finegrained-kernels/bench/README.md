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

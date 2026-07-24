# finegrained-moe bench

`bench_moe.py` benchmarks the local **finegrained-moe** kernel against the upstream
**finegrained-fp8** kernel (`kernels-community/finegrained-fp8` @ `v4`) and external
reference implementations, on real model shapes.

## What it compares

Four figure rows, each a **decode | prefill** subplot pair:

- **fused quantized** — `moe_fused_*` vs v4 vs DeepGEMM
- **unfused quantized** — `moe_unfused_*` (two GEMMs + host GLU) vs v4 vs DeepGEMM
- **unquantized (BF16)** — fused vs transformers `grouped_mm`/`batched_mm`, SonicMoE, DeepGEMM BF16
- **attn quantized** — one qkv-shaped `matmul_2d` linear per model, in its deployment format

Baselines per problem ("all kinds"): upstream **finegrained-fp8** (`@ v4`), **DeepGEMM**
(fp8/fp4/bf16), **transformers** `grouped_mm`/`batched_mm` (= `torch._grouped_mm` / `torch.bmm`,
the BF16 torch/cuBLAS path), **SonicMoE**, the OpenAI **triton_kernels** MXFP4 path (GPT-OSS),
and **`torch.scaled_grouped_mm`** (the quantized-prefill cuBLAS reference). Each is
import-guarded — a missing dependency skips that baseline instead of failing the run.

Every cell runs in three modes: `eager`, `cudagraph` (decode's deployment mode), and
`compile` (`torch.compile(max-autotune, fullgraph)`). A cell that raises is a red ✕ marker;
the others still run.

## The pre-swizzled fast path

By default the finegrained-moe arm feeds **pre-swizzled** (`SWIZZLE_32_4_4`) MX weight scales,
so its numbers reflect the tcgen05 fast path. Only MX weights on 128-aligned dims are swizzled
(the routed guard rejects non-128 gate/N); block-fp8 and BF16 stay affine. Set `PRESWIZZLE=0`
to measure the affine path instead.

Correctness is cross-checked in-run: each baseline's output is compared to the finegrained-moe
anchor (`parity-vs-v5` in the log), so a wrong scale layout shows up as a large parity diff.

## Running

```bash
python bench/bench_moe.py                 # full grid, single GPU -> bench_moe.csv + bench_moe.png
GPUS=8 python bench/bench_moe.py           # shard problems across 8 GPUs (one process per GPU), then merge + plot
SMOKE=1 python bench/bench_moe.py          # fast everything-compiles pass (3-trial tunes, 256-tok prefill)
PRESWIZZLE=0 python bench/bench_moe.py     # affine MX scales instead of the fast path
python bench/bench_moe.py gpt-oss          # substring filter on row/problem names
REPLOT=1 python bench/bench_moe.py         # rebuild the figure from an existing bench_moe.csv
MOCK=1 python bench/bench_moe.py           # no GPU: random latencies to validate the figure layout
```

Outputs land beside the script (`bench/bench_moe.csv`, `bench/bench_moe.png`); both are
`.gitignore`d by the repo (regenerate as needed). Requires the bench env: `transformers`
(with the `integrations.{deepgemm,moe,sonicmoe,mxfp4}` helpers), `kernels`, DeepGEMM, and a
Blackwell (sm_100) GPU. **Don't run under concurrent GPU load** — the latencies won't be trustworthy.

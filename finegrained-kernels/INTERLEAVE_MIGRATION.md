# gate|up row interleave — migration plan

Status: **kernel-side migration CODE-COMPLETE; correctness not yet confirmed on GPU.**

All of it parses and the package imports. What landed:

- `swizzle.py` — interleave/deinterleave helpers; `gate=` removed from all three functions; the
  6-D artifact is gone (a plain swizzle of the interleaved grid is correct)
- `tiles.py` — `weight_tile_ptrs` and `matmul_weight_ptrs` are one-line delegations to the plain
  path; `swizzle_offsets` gained `GATE` and emits the doubled weight-side span
- `epilogue.py` — `split_gate_up` is a trailing-axis `tl.split` (permute dropped)
- `scales.py` — both `SWIZZLED_SCALES and GATE` arms (incl. the two-block sub-128 gather) and the
  affine GATE arm collapsed to the plain doubled-row read; `gate_stacked_block_scale_ptrs` deleted
- `matmul.py` — all six kernels; `n_rows` added; affine scale extents doubled
- `batched.py` — five sites; `bs_off`'s gate/up `tl.where` select deleted; dead `n_blocks` removed
- `grouped.py` — six sites; scale block index doubled; the weight descriptor view moved from
  `(2E, N, K)` to `(E, 2N, K)` with a `[1, 2*BN, BK]` box; gate/up scale streams are now adjacent
- `scheduling.py` — both tile resolvers emit the doubled `offs_bn`/`n_off`, `row0` is the plain expert
- `INTERLEAVED_SCALES` removed everywhere (constexpr, tune keys, in-kernel remaps)
- callers dropped `gate=True`: `moe.py`, `tests/test_moe.py`, `bench/bench_moe.py`

**Not done:** pruners still registered (`gate_pointer_only`, `gate_stacked_tmem_trap`,
`gated_pointer_weight_warp_spec`, `allow_gate_subblock`) — deliberately left until correctness is
confirmed, since removing them changes the config space and forces another retune;
`flatten_weight_tile` (now a harmless no-op reshape on the already-2-D tile); the transformers side;
parity re-baselining.

**Next step is a GPU correctness run, not more editing.** Every gated tune key was invalidated by
the layout change, so the first gated call per shape pays a full autotune (minutes).

## Motivation

Simplification, not throughput. Today gate|up is a *logical* stack the kernels assemble from two
row blocks `N` apart, and that assumption is threaded through **33 `@jit` functions**, 4 pruners,
a `gate=` mode in the scale swizzler, and a 6-D scale artifact.

## The layout

Element (row) granularity, for weights **and** their scale grids:

    gate row j -> buffer row 2j      up row j -> buffer row 2j+1      i.e. [g0,u0,g1,u1,...]

This is already what GPT-OSS ships on disk, so the integration *deletes* `_deinterleave_gate_up_rows`
rather than gaining a converter.

## Why it collapses the special case

Tile `pid_n` needs gate rows `[pid_n*BN, (pid_n+1)*BN)` and the matching up rows. Interleaved those
are buffer rows `[2*pid_n*BN, 2*(pid_n+1)*BN)` — **contiguous**. And since `pid_n * 2BN == 2 * pid_n * BN`,
that range is exactly what a *plain* tile of `2*BN` rows at the **same** `pid_n` already reads.

> A gated load is the ungated load over a doubled extent.

So `GATE` reduces to two derived locals per kernel (the proven inline-constexpr pattern, cf.
`scales.py` `n_width`, `epilogue.py` `rows`) —

```python
N_W = 2 * N if GATE else N
BN_W: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
```

— and every loader below that becomes the plain path. `GATE` survives only in the epilogue, where
it genuinely means something.

**Do not** implement this as a helper returning a `tl.constexpr` tuple; constexpr propagation
through a `@jit` return boundary is unreliable on Triton 3.7.1. Inline the two locals.

## Verified facts (`scratchpad/check_interleave.py`, B200, 2026-08-15)

1. `interleave_gate_up_rows` / `deinterleave_gate_up_rows` round-trip exactly; gate `j` lands at
   row `2j`, up `j` at `2j+1`.
2. A **plain** `swizzle_mx_scales` of the interleaved grid round-trips bit-exactly and yields the
   ordinary 5-D artifact — no `gate=` mode, no 6-D shape.
3. **Decisive:** at `BN=128` a tile's interleaved rows occupy swizzle blocks `2*pid_n, 2*pid_n+1` —
   *the exact block indices the old block-interleaved artifact produced*. The descriptor read
   pattern is unchanged, so there is no perf risk on the main path; the `gate=` swizzle step was
   manufacturing by hand what row-interleaving gives for free.
4. Alignment floor drops `N % 128` -> `N % 64` (only the doubled extent must meet the 128-row
   block). `N=320` and GPT-OSS `N=2880` become legal.

## Sites

**Deletes outright**

- `tiles.py`: `weight_tile_ptrs` GATE arm (identical to `oriented_tile_ptrs` once gone),
  `flatten_weight_tile` (no 3-D tile to flatten), `matmul_weight_ptrs` (becomes `operand_tile_ptrs`)
- `scales.py`: both `SWIZZLED_SCALES and GATE` arms, including the two-block sub-128 gather — under
  interleave a `BN=32` tile is a *single-block* slice, so `reference_swizzled_gateup_subblock_decode`'s
  76->51us path becomes the default (re-measure to confirm it holds)
- `swizzle.py`: `gate=` from `_swizzle_to_blocks` / `swizzle_mx_scales` / `unswizzle_mx_scales`;
  the 6-D artifact and its `INTERLEAVED_SCALES` consumers in `batched.py`/`grouped.py`/`recipes.py:142`
- `pruners.py`: `gate_pointer_only_pruner`, `gate_stacked_tmem_trap_pruner`,
  `gated_pointer_weight_warp_spec_pruner`, `swizzled_scale_config_pruner(allow_gate_subblock)` —
  confirm each is genuinely dead rather than assuming

**Changes**

- kernels (`matmul.py` x6, `batched.py` x5, `grouped.py` x6): the two derived locals above
- `epilogue.py` `split_gate_up`: `reshape(rows, BN, 2)` + split on the last dim; today's
  `reshape(rows,2,BN)` + `permute` loses the permute
- `recipes.py` `expert_weight_shape`: unchanged in arithmetic, docstring only
- callers of `swizzle_mx_scales(..., gate=True)`: `moe.py:177,270`, `tests/test_moe.py:170`,
  `bench/bench_moe.py:160,474,489` — these just drop the kwarg, since interleaving now happens
  upstream at load
- transformers `integrations/finegrained.py`: delete `_deinterleave_gate_up_rows`; interleave the
  non-GPT-OSS families at load; reverse at save

## Risks

- **Parity gets a new variable.** Fused/unfused gate bit-exactness currently holds iff
  `BLOCK_SIZE_M` and `COMPUTE_MODE` match; layout joins that list. Re-baseline rather than inherit.
- **Tune caches invalidate** for every gated key.
- Checkpoint round-tripping touches four model families — save-side reverse must be tested, not assumed.

## Sequencing

Do this **before** writing any gate backward. Building the gate special-case into `autograd.py`
and then deleting it is the one ordering that wastes work. The backward payoff is the point:
interleaved, `B.t()` is a real view of the gate|up slab, so `dA = matmul_2d(dY_stacked, W.t(), ...)`
is one call with no GATE flag, `dW = dY.T @ A` lands directly in storage order, and
`quantize_both_orientations` works unchanged (no transposed-swizzle helper needed for MX gate dgrad).

Follow-up, separately: `SAVE_PREACT` constexpr + `SaveY` store so the fused GLU emits its
pre-activation for backward — measured ~15x cheaper than recompute at DSV3 prefill shape
(~10us of HBM vs ~145us of GEMM). Needs an explicit tune-key entry. Under interleave that store is
one contiguous `2*BN` write instead of two disjoint N-apart writes.

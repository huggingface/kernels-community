# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Op-level scenario tests: the (weight recipe x epilogue x input/output recipe)
support matrix of ``matmul_grouped`` / ``matmul_batched``, each cell checked against an
independent dequantize-and-matmul torch reference (``tests/utils.py``'s ``WEIGHTS``
registry — shared with ``test_moe``'s fused-vs-unfused problems).

The scenario list is GENERATED from the same support matrix the ``Quantization``
docstring documents — every weight recipe crosses its valid input and output recipes
once, activation-function variants ride one recipe (the GLU math is recipe-independent),
and routing variants (sentinel, noncontiguous ids, empty expert, decode and launch-scale
shapes, torch.compile) ride one recipe each. ``matmul_2d`` gets its own torch-reference
test here — nothing in this file uses a kernel under test as the oracle."""

from dataclasses import dataclass

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TO_TOL,
    REQUANT_FN,
    REQUANT_GROUP,
    TEST_DEVICE,
    WEIGHTS,
    dq_grouped,
    fp8_act_quant_block_dynamic,
    maybe_compile,
)

import finegrained_fp8  # type: ignore
from finegrained_fp8 import Epilogue, Quantization  # type: ignore
from finegrained_fp8.utils import apply_glu  # type: ignore


# ── the scenario spec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Problem:
    """One op-level scenario: a weight recipe (a ``WEIGHTS`` registry row) plus the
    transform and routing knobs the public ops expose. Validity of the recipe fields
    against the registry row is enforced at generation, not runtime."""

    weights: str
    S: int = 64
    E: int = 4
    N: int = 128
    K: int = 256
    gate: bool = False
    act_fn: str = "silu"
    swiglu_alpha: float | None = None
    swiglu_limit: float | None = None
    input_recipe: str | None = None
    output_recipe: str | None = None
    prequant: bool = False  # pass As explicitly (must be bit-identical to raw A)
    sentinel_fraction: float = 0.0
    noncontiguous: bool = False
    empty_expert: bool = False
    compile: bool = False
    dtype: torch.dtype = torch.bfloat16

    @property
    def id(self):
        tag = self.weights
        if self.gate:
            tag += f"_gate_{self.act_fn}"
            if self.swiglu_alpha is not None:
                tag += "_alpha"
            if self.swiglu_limit is not None:
                tag += "_limit"
        if self.input_recipe:
            tag += f"_in{self.input_recipe}"
        if self.output_recipe:
            tag += f"_out{self.output_recipe}"
        if self.prequant:
            tag += "_prequant"
        if self.sentinel_fraction:
            tag += "_sentinel"
        if self.noncontiguous:
            tag += "_noncontig"
        if self.empty_expert:
            tag += "_emptyexpert"
        if self.compile:
            tag += "_compile"
        return f"{tag}_S{self.S}_E{self.E}_N{self.N}_K{self.K}"


def scenarios() -> list[Problem]:
    """The curated matrix. Coverage rules:
    - every weight recipe: one plain GEMM cell;
    - every valid (weights, output_recipe) pair: one gate cell (requant lives under gate);
    - every non-default input recipe: one cell (the W4A4 chains);
    - one prequant mirror per quantized family (raw-vs-As bit-equality);
    - GLU variants (gelu, swiglu alpha+limit) on one recipe — the math is recipe-blind;
    - routing/launch variants (sentinel, noncontiguous, empty expert, decode shape,
      launch-scale shape, torch.compile) one recipe each."""
    out: list[Problem] = []
    for w, row in WEIGHTS.items():
        if w == "fp16":
            # emitted explicitly below with fp16 activations (matching dtypes required)
            continue
        out.append(Problem(weights=w))
        for orecipe in row["output_recipes"]:
            if orecipe is not None:
                out.append(Problem(weights=w, gate=True, output_recipe=orecipe))
        default_in = {"fp8_128x128": "fp8", "fp8_tensor": "fp8"}.get(w)
        for irecipe in row["input_recipes"]:
            if irecipe is not None and irecipe != default_in:
                out.append(Problem(weights=w, input_recipe=irecipe))
    out += [
        Problem(weights="fp16", dtype=torch.float16),
        # gate WITHOUT requant (raw GLU intermediate) — one quantized + one full recipe
        Problem(weights="mxfp8", gate=True),
        Problem(weights="bf16", gate=True),
        Problem(weights="mxfp8", gate=True, act_fn="gelu", output_recipe="mxfp8"),
        Problem(
            weights="mxfp8",
            gate=True,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            output_recipe="mxfp8",
        ),
        Problem(weights="fp8_128x128", prequant=True),
        Problem(weights="mxfp8", prequant=True),
        Problem(weights="nvfp4", prequant=True),
        Problem(weights="mxfp8", sentinel_fraction=0.25),
        Problem(weights="mxfp8", noncontiguous=True),
        Problem(weights="mxfp8", empty_expert=True),
        Problem(weights="mxfp8", S=8),  # decode shape
        Problem(weights="nvfp4", S=8),  # decode on the software/swap arms
        Problem(weights="fp8_128x128", S=8),
        # launch-scale smoke (the matrix rides small shapes; this catches scale-dependent
        # scheduling/tiling regressions)
        Problem(weights="mxfp8", S=2048, E=16, N=512, K=1024),
        Problem(weights="fp8_128x128", compile=True),
        Problem(weights="mxfp4", compile=True),
        Problem(weights="bf16", compile=True),  # the fp kernel's pre_hook under compile
    ]
    return out


PROBLEMS = scenarios()


# ── inputs, reference, and checks ────────────────────────────────────────────────


def _routed(problem: Problem):
    torch.manual_seed(0)
    A = torch.randn(problem.S, problem.K, device=TEST_DEVICE, dtype=problem.dtype)
    high = problem.E - 1 if problem.empty_expert else problem.E
    expert_ids = torch.randint(
        0, high, (problem.S,), device=TEST_DEVICE, dtype=torch.int32
    )
    if problem.sentinel_fraction:
        idx = torch.randperm(problem.S, device=TEST_DEVICE)[
            : int(problem.S * problem.sentinel_fraction)
        ]
        expert_ids[idx] = problem.E
    return A, expert_ids


def _make_noncontig(x):
    base = torch.empty((x.numel(), 2), dtype=x.dtype, device=x.device)
    base[:, 0] = x
    base[:, 1] = x
    return base[:, 0]


def _reference(problem: Problem, A, expert_ids, B, Bs):
    """fp32 oracle on the SAME quantized operands the op consumes: quantize A with the
    recipe's host fn (the exact code the op calls), dequantize both sides, matmul in
    fp32, then GLU in fp32 — the production epilogue applies it to the fp32 accumulator
    directly (``simulate_unfused`` is the rounding-matched variant, tested at moe level).
    Sentinel rows are zeroed (excluded from the checks). Returns
    ``(ref_out, (Aq, As) or None)``."""
    row = WEIGHTS[problem.weights]
    quant = row["act_quant"][problem.input_recipe]
    if quant is None:
        A_dq, prequant_args = A.float(), None
    else:
        Aq, As = quant(A)
        A_dq, prequant_args = row["dq_act"](Aq, As), (Aq, As)
    W = row["dequant"](B, Bs)  # (E, rows, K) fp32
    local = expert_ids.long().clamp(max=problem.E - 1)
    ref = torch.einsum("sk,snk->sn", A_dq, W[local])
    ref[expert_ids.long() >= problem.E] = 0
    if problem.gate:
        gate_v, up_v = ref.chunk(2, dim=-1)
        ref = apply_glu(
            gate_v, up_v, problem.act_fn, problem.swiglu_alpha, problem.swiglu_limit
        ).float()
    return ref, prequant_args


def _run_op(problem: Problem, layout, A, expert_ids, B, Bs, As=None):
    epilogue = (
        Epilogue(
            gate=True,
            act_fn=problem.act_fn,
            swiglu_alpha=problem.swiglu_alpha,
            swiglu_limit=problem.swiglu_limit,
        )
        if problem.gate
        else None
    )
    quantization = (
        Quantization(
            input_recipe=problem.input_recipe, output_recipe=problem.output_recipe
        )
        if (problem.input_recipe or problem.output_recipe)
        else None
    )
    kw = dict(epilogue=epilogue, quantization=quantization)
    if problem.output_recipe is None:
        kw["output_dtype"] = problem.dtype
    if layout == "batched":
        op = maybe_compile(finegrained_fp8.matmul_batched, problem.compile)
        return op(A, B, As, Bs, expert_ids=expert_ids, **kw)
    expert_start, gather_idx, scatter_idx = finegrained_fp8.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    op = maybe_compile(finegrained_fp8.matmul_grouped, problem.compile)
    return op(
        A,
        B,
        As,
        Bs,
        expert_start=expert_start,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        **kw,
    )


def _check(problem: Problem, out, ref, expert_ids):
    keep = expert_ids.long() < problem.E
    if problem.output_recipe is None:
        atol, rtol = DTYPE_TO_TOL[problem.dtype]
        torch.testing.assert_close(
            out[keep].float(), ref[keep].to(problem.dtype).float(), atol=atol, rtol=rtol
        )
        return
    # requant: compare DEQUANTIZED kernel output against the offline quant of the
    # reference intermediate — agreement is grid-boundary flips only (bit-equality
    # across independently tuned launches is not the contract; the fp32 accumulation
    # order differs per config).
    C, Cs = out
    if problem.output_recipe == "fp8":
        # per-(row, N-block) scales: dequant directly against the fp32 reference
        dq = C.float() * torch.repeat_interleave(
            Cs.float(), C.shape[1] // Cs.shape[1], dim=-1
        )
        ref_cmp = ref
    else:
        group = REQUANT_GROUP[problem.output_recipe]
        dq = dq_grouped(C, Cs, group)
        q_ref, s_ref = REQUANT_FN[problem.output_recipe](ref.to(problem.dtype))
        ref_cmp = dq_grouped(q_ref, s_ref, group)
    rel = (
        (dq[keep] - ref_cmp[keep]).abs().mean()
        / ref_cmp[keep].abs().mean().clamp(min=1e-6)
    ).item()
    assert rel < 0.06, f"requant dequant mean-rel {rel:.4f} vs offline reference"


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("layout", ["batched", "grouped"])
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_op_scenarios(problem: Problem, layout):
    A, expert_ids = _routed(problem)
    if problem.noncontiguous:
        expert_ids = _make_noncontig(expert_ids)
    row = WEIGHTS[problem.weights]
    B, Bs = row["make"](
        2 * problem.N if problem.gate else problem.N, problem.K, problem.E
    )
    ref, prequant_args = _reference(problem, A, expert_ids, B, Bs)
    out = _run_op(problem, layout, A, expert_ids, B, Bs)
    _check(problem, out, ref, expert_ids)
    if problem.prequant:
        # handing the op the SAME pre-quantized (Aq, As) must be bit-identical to the
        # raw-A run (the op quantizes raw A with the same host fn)
        Aq, As = prequant_args
        out2 = _run_op(problem, layout, Aq, expert_ids, B, Bs, As=As)
        assert torch.equal(out, out2), "pre-quantized As path diverged from raw A"


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("m_rows", [1, 64], ids=["M1_inline", "M64_offline"])
@pytest.mark.parametrize(
    "weights,input_recipe",
    [
        ("fp8_128x128", None),
        ("mxfp8", None),
        ("mxfp4", None),  # W4A8, the MX default
        ("mxfp4", "mxfp4"),  # W4A4 (packed acts)
        ("nvfp4", None),
    ],
    ids=lambda v: str(v),
)
def test_matmul_2d_vs_torch(weights, input_recipe, m_rows):
    """matmul_2d against the pure-torch dequant oracle — it used to be the reference for
    the routed tests, so it gets its own independent check (nothing kernel-side in the
    oracle). ``m_rows`` spans the ``maybe_act_quant`` gate: M=1 runs the in-kernel
    inline quant arm, M=64 the offline pre-quant pass (bit-exact pair by construction);
    ``input_recipe`` spans the activation grids per weight family."""
    torch.manual_seed(0)
    M, N, K = m_rows, 128, 256
    row = WEIGHTS[weights]
    B, Bs = row["make"](N, K, 1)
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    block_size = [128, 128] if weights == "fp8_128x128" else None
    out = finegrained_fp8.matmul_2d(A, B[0], Bs[0], block_size,
                                    input_recipe=input_recipe)
    quant = row["act_quant"][input_recipe]
    Aq, As = (
        quant(A) if weights != "fp8_128x128" else fp8_act_quant_block_dynamic(A, 128)
    )
    ref = row["dq_act"](Aq, As) @ row["dequant"](B, Bs)[0].T
    atol, rtol = DTYPE_TO_TOL[torch.bfloat16]
    torch.testing.assert_close(
        out.float(), ref.to(out.dtype).float(), atol=atol, rtol=rtol
    )

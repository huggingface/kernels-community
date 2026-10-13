import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn


LIN_ATTENTION = (
    "chunk_kimi_delta_attention",
    "recurrent_kimi_delta_attention",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
)
TRAINABLE_LIN_ATTENTION = ("chunk_kimi_delta_attention", "chunk_gated_delta_rule")
# dense: normal, packed: packed training, fused: fused paths (similar to mamba inner fn)
MODES = ("dense", "packed", "fused")

# (bsz, num_heads, head_dim)
DENSE_SHAPES = (
    (1, 1, 32),
    (1, 2, 64),
    (1, 4, 128),
    (2, 2, 32),
    (2, 4, 64),
    (2, 8, 128),
    (4, 1, 64),
    (4, 4, 32),
)
# (num_heads, head_dim) -> bsz will default to 1 (flattened inpit)
PACKED_SHAPES = ((1, 32), (2, 64), (4, 128), (8, 64))

# (seq_len,)
CHUNK_LENGTHS = (32, 64, 128, 256)
RECURRENT_LENGTHS = (1, 2, 4, 8)
PACKED_CHUNK_LENGTHS = ((32, 32), (64, 32), (128, 64), (192, 64))
PACKED_RECURRENT_LENGTHS = ((1, 1), (1, 2), (2, 3), (4, 8))

# subset for CI
CI_CASES = {
    "chunk": {
        ("dense", 1, 2, 64, (64,)),
        ("dense", 2, 4, 64, (128,)),
        ("packed", 1, 2, 64, (64, 32)),
        ("packed", 1, 4, 128, (128, 64)),
        ("fused", 1, 2, 64, (64,)),
        ("fused", 2, 8, 128, (128,)),
    },
    "recurrent": {
        ("dense", 1, 2, 64, (1,)),
        ("dense", 2, 4, 64, (4,)),
        ("packed", 1, 2, 64, (1, 2)),
        ("fused", 1, 2, 64, (2,)),
        ("fused", 2, 8, 128, (8,)),
    },
}


def _lin_attention_cases(names):
    cases = []
    for name in names:
        is_chunk = name.startswith("chunk_")
        kind = "chunk" if is_chunk else "recurrent"
        for mode in MODES:
            if mode == "packed":
                packed_lengths = (
                    PACKED_CHUNK_LENGTHS if is_chunk else PACKED_RECURRENT_LENGTHS
                )
                shapes = (
                    (1, h, d, lengths)
                    for h, d in PACKED_SHAPES
                    for lengths in packed_lengths
                )
            else:
                seq_lens = CHUNK_LENGTHS if is_chunk else RECURRENT_LENGTHS
                shapes = (
                    (bsz, h, d, (t,))
                    for bsz, h, d in DENSE_SHAPES
                    for t in seq_lens
                )

            for bsz, h, d, lengths in shapes:
                case = (mode, bsz, h, d, lengths)
                marks = pytest.mark.kernels_ci if case in CI_CASES[kind] else ()
                cases.append(
                    pytest.param(
                        name, mode, bsz, h, d, lengths,
                        marks=marks,
                        id=f"{mode}-{name}-b{bsz}-h{h}-d{d}-t{'x'.join(map(str, lengths))}",
                    )
                )
    return cases


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    pytest.skip("FLA requires CUDA or XPU")


@pytest.fixture(scope="module")
def fla(device):
    # TODO: Replace with get_kernel after local staging / building
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
    return importlib.import_module("fla")


@pytest.fixture(autouse=True)
def reset_compiler():
    # avoids recompiling as we pass different kwargs for each mode etc
    torch.compiler.reset()


def _leaf(x):
    return x.detach().clone().requires_grad_(True)


def _check(actual, expected, *, rtol=5e-3, atol=1e-3, max_rel_l2=5e-3):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    a, b = actual.detach().float(), expected.detach().float()
    rel_l2 = (a - b).norm() / b.norm().clamp_min(1e-8)
    assert rel_l2.item() <= max_rel_l2, f"relative L2 error: {rel_l2.item():.4g}"


def _lin_attention_case(name, device, mode, bsz, h, d, lengths):
    torch.manual_seed(42)
    is_kimi = "kimi" in name
    dtype = (
        torch.bfloat16
        if device.type == "xpu" or torch.cuda.is_bf16_supported()
        else torch.float16
    )
    t = sum(lengths)
    shape = (bsz, t, h, d)

    q, k, v = (
        torch.randn(shape, device=device, dtype=dtype) * 0.1 for _ in range(3)
    )
    g_shape = shape if is_kimi else (bsz, t, h)
    g = -torch.rand(g_shape, device=device, dtype=dtype) * 0.1
    beta = 0.25 + torch.rand((bsz, t, h), device=device, dtype=dtype) * 0.5
    nseq = len(lengths) if mode == "packed" else bsz
    state = torch.randn((nseq, h, d, d), device=device, dtype=torch.float32) * 0.01

    kwargs = {}
    if mode == "packed":
        offsets = (0, lengths[0], sum(lengths))
        kwargs["cu_seq_lens_q"] = torch.tensor(offsets, device=device, dtype=torch.int32)
    elif mode == "fused":
        kwargs.update(
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=torch.full((h,), -1.5, device=device),
            dt_bias=torch.zeros((h * d if is_kimi else h,), device=device),
        )

    return (q, k, v, g, beta), state, kwargs


@pytest.mark.parametrize(
    "bsz,seq_len,d",
    (
        pytest.param(1, 64, 128, marks=pytest.mark.kernels_ci),
        (2, 128, 256),
        (4, 64, 512),
        pytest.param(2, 256, 1024, marks=pytest.mark.kernels_ci),
    ),
)
def test_rmsnorm_gated_fullgraph_forward_backward(fla, device, bsz, seq_len, d):
    torch.manual_seed(0)
    module = fla.layers.FusedRMSNormGated().to(device)
    module.weight = nn.Parameter(torch.ones(d, device=device))
    module.activation = "silu"
    module.variance_epsilon = 1e-5

    x = torch.randn((bsz, seq_len, d), device=device)
    gate = torch.randn_like(x)
    ref_weight = _leaf(module.weight)

    def forward(x, gate):
        return x + module(x, gate)

    def reference(x, gate):
        return x + F.rms_norm(x, (d,), ref_weight, 1e-5) * F.silu(gate)

    eager_inputs = (_leaf(x), _leaf(gate))
    compiled_inputs = (_leaf(x), _leaf(gate))
    ref_inputs = (_leaf(x), _leaf(gate))
    compiled = torch.compile(forward, backend="inductor", fullgraph=True)

    eager_out = forward(*eager_inputs)
    compiled_out = compiled(*compiled_inputs)
    ref_out = reference(*ref_inputs)

    tol = dict(rtol=2e-3, atol=2e-3, max_rel_l2=2e-3)
    _check(compiled_out, eager_out, **tol)
    _check(eager_out, ref_out, **tol)
    _check(compiled_out, ref_out, **tol)

    eager_grads = torch.autograd.grad(eager_out.sum(), (*eager_inputs, module.weight))
    compiled_grads = torch.autograd.grad(compiled_out.sum(), (*compiled_inputs, module.weight))
    ref_grads = torch.autograd.grad(ref_out.sum(), (*ref_inputs, ref_weight))
    for actual, eager, ref in zip(compiled_grads, eager_grads, ref_grads, strict=True):
        _check(actual, eager, **tol)
        _check(eager, ref, **tol)
        _check(actual, ref, **tol)


@pytest.mark.parametrize(
    "name,mode,bsz,heads,head_dim,lengths",
    _lin_attention_cases(LIN_ATTENTION),
)
def test_lin_attention_fullgraph_inference(
    fla, device, name, mode, bsz, heads, head_dim, lengths
):
    module = getattr(fla.layers, name)().eval()
    tensors, state, kwargs = _lin_attention_case(
        name, device, mode, bsz, heads, head_dim, lengths
    )

    def forward(q, k, v, g, beta, initial_state):
        return module(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            **kwargs,
        )

    compiled = torch.compile(forward, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        eager = forward(*tensors, state)
        actual = compiled(*(x.clone() for x in tensors), state.clone())

    for a, b in zip(actual, eager, strict=True):
        _check(a, b)


@pytest.mark.parametrize(
    "name,mode,bsz,heads,head_dim,lengths",
    _lin_attention_cases(TRAINABLE_LIN_ATTENTION),
)
def test_lin_attention_fullgraph_forward_backward(
    fla, device, name, mode, bsz, heads, head_dim, lengths
):
    # XPU bug, also marked in transformers as incompatible
    if device.type == "xpu" and name == "chunk_kimi_delta_attention":
        pytest.skip("Chunk KDA backward is not supported on XPU")

    module = getattr(fla.layers, name)().train()
    tensors, state, kwargs = _lin_attention_case(
        name, device, mode, bsz, heads, head_dim, lengths
    )
    head = nn.Linear(heads * head_dim, 32, bias=False).to(device)
    targets = torch.randint(32, (bsz, tensors[0].shape[1]), device=device)
    state_target = torch.randn_like(state) * 0.01

    def forward(q, k, v, g, beta, initial_state):
        y, final_state = module(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            **kwargs,
        )
        residual = v + y
        logits = head(residual.float().flatten(2))
        return residual, logits, final_state

    def loss_fn(result):
        _, logits, final_state = result
        # Mean token loss keeps gradient scale stable across sequence lengths
        return (
            F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
            + 0.01 * (final_state.float() - state_target).square().sum()
        )

    eager_inputs = tuple(_leaf(x) for x in (*tensors, state))
    compiled_inputs = tuple(_leaf(x) for x in (*tensors, state))
    compiled = torch.compile(forward, backend="inductor", fullgraph=True)

    eager_out = forward(*eager_inputs)
    compiled_out = compiled(*compiled_inputs)
    for a, b in zip(compiled_out, eager_out, strict=True):
        _check(a, b)

    eager_grads = torch.autograd.grad(loss_fn(eager_out), (*eager_inputs, head.weight))
    compiled_grads = torch.autograd.grad(loss_fn(compiled_out), (*compiled_inputs, head.weight))
    for label, actual, expected in zip(
        ("dq", "dk", "dv", "dg", "dbeta", "dh0", "dweight"),
        compiled_grads, eager_grads, strict=True,
    ):
        assert expected.abs().max() > 0, f"No gradient signal for {label}"
        _check(actual, expected)

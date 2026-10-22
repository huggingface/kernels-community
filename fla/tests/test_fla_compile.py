import pytest
import torch
import torch.nn.functional as F
from kernels import get_kernel
from torch import nn


import importlib
import sys
from pathlib import Path


LIN_ATTENTION = (
    "chunk_kimi_delta_attention",
    "recurrent_kimi_delta_attention",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
)
TRAINABLE_LIN_ATTENTION = ("chunk_kimi_delta_attention", "chunk_gated_delta_rule")
# packed is the cuseqlens path as in FA (training usually or CB)
MODES = ("dense", "packed")

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
PACKED_SHAPES = ((1, 32), (2, 32), (2, 64), (4, 128), (8, 64))

# (seq_len,)
CHUNK_LENGTHS = (32, 64, 128, 256)
RECURRENT_LENGTHS = (1, 2, 4, 8)
PACKED_CHUNK_LENGTHS = ((32, 16), (32, 32), (64, 32), (128, 64), (192, 64))
PACKED_RECURRENT_LENGTHS = ((1, 1), (1, 2), (2, 3), (4, 8))

# CI subset
CI_INFERENCE = {
    ("recurrent_gated_delta_rule", "dense", 1, 2, 64, (4,)),
    ("recurrent_kimi_delta_attention", "dense", 1, 2, 64, (1,)),
}

def _lin_attention_cases(names, ci_cases):
    cases = []
    for name in names:
        chunk = name.startswith("chunk_")
        for mode in MODES:
            if mode == "packed":
                lengths_list = PACKED_CHUNK_LENGTHS if chunk else PACKED_RECURRENT_LENGTHS
                shapes = (
                    (1, h, d, lengths)
                    for h, d in PACKED_SHAPES
                    for lengths in lengths_list
                )
            else:
                lengths_list = CHUNK_LENGTHS if chunk else RECURRENT_LENGTHS
                shapes = (
                    (bsz, h, d, (t,))
                    for bsz, h, d in DENSE_SHAPES
                    for t in lengths_list
                )

            for bsz, h, d, lengths in shapes:
                case = (mode, bsz, h, d, lengths)
                cases.append(
                    pytest.param(
                        name, mode, bsz, h, d, lengths,
                        marks=pytest.mark.kernels_ci if (name, *case) in ci_cases else (),
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
    # return get_kernel("kernels-community/fla", version=2)
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
    """
    Returns the different linear attn cores like QKV and lin attn specific g/decay and beta

    Adiitionally, we return an initial state (important for certain gradient flows) and kwargs for e.g. packed (cuseqlens)
    """
    torch.manual_seed(42)

    dtype = (
        torch.bfloat16
        if device.type == "xpu" or torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # metadata
    t = sum(lengths)
    shape = (bsz, t, h, d)
    nseq = len(lengths) if mode == "packed" else bsz

    q, k, v = (torch.randn(shape, device=device, dtype=dtype) * 0.1 for _ in range(3))
    g_shape = shape if "kimi" in name else (bsz, t, h)  # KDA has learnable per head decay
    g = -torch.rand(g_shape, device=device, dtype=dtype) * 0.1
    beta = 0.25 + torch.rand((bsz, t, h), device=device, dtype=dtype) * 0.5
    initial_state = torch.randn((nseq, h, d, d), device=device) * 0.01

    kwargs = {"use_qk_l2norm_in_kernel": True}  # we do it as kwarg in transformers
    if mode == "packed":
        offsets = (0, lengths[0], sum(lengths))
        kwargs["cu_seq_lens_q"] = torch.tensor(offsets, device=device, dtype=torch.int32)

    return (q, k, v, g, beta), initial_state, kwargs


@pytest.mark.parametrize(
    "bsz,seq_len,d",
    (
        pytest.param(1, 64, 128, marks=pytest.mark.kernels_ci),
        (2, 128, 256),
        (4, 64, 512),
        (2, 256, 1024),
    ),
)
def test_rmsnorm_gated_compiled_outputs_and_gradients_match_reference(fla, device, bsz, seq_len, d):
    torch.manual_seed(0)

    # As we would find it under transformers
    module = fla.layers.FusedRMSNormGated().to(device)
    module.weight = nn.Parameter(torch.ones(d, device=device))
    module.activation = "silu"
    module.variance_epsilon = 1e-5

    # Dummy input
    x = torch.randn((bsz, seq_len, d), device=device)
    gate = torch.randn_like(x)
    ref_inputs = (_leaf(x), _leaf(gate))
    ref_weight = _leaf(module.weight)
    eager_inputs = (_leaf(x), _leaf(gate))
    compiled_inputs = (_leaf(x), _leaf(gate))

    def forward(x, gate):
        return x + module(x, gate)

    def reference(x, gate):
        return x + F.rms_norm(x, (d,), ref_weight, 1e-5) * F.silu(gate)

    compiled = torch.compile(forward, fullgraph=True)  # no graph break
    eager_out = forward(*eager_inputs)
    ref_out = reference(*ref_inputs)
    with torch._dynamo.config.patch(error_on_recompile=True):  # no recompile
        compiled_out = compiled(*compiled_inputs)
        compiled(*compiled_inputs)

    tol = dict(rtol=2e-3, atol=2e-3, max_rel_l2=2e-3)
    # fwd check
    _check(compiled_out, eager_out, **tol)
    _check(eager_out, ref_out, **tol)
    _check(compiled_out, ref_out, **tol)

    # bwd check
    eager_grads = torch.autograd.grad(eager_out.sum(), (*eager_inputs, module.weight))
    ref_grads = torch.autograd.grad(ref_out.sum(), (*ref_inputs, ref_weight))
    with torch._dynamo.config.patch(error_on_recompile=True):  # no recompile
        compiled_grads = torch.autograd.grad(compiled_out.sum(), (*compiled_inputs, module.weight))

    for actual, eager, ref in zip(compiled_grads, eager_grads, ref_grads, strict=True):
        _check(actual, eager, **tol)
        _check(eager, ref, **tol)
        _check(actual, ref, **tol)


@pytest.mark.parametrize(
    "name,mode,bsz,heads,head_dim,lengths",
    _lin_attention_cases(LIN_ATTENTION, CI_INFERENCE),
)
def test_lin_attention_compiled_outputs_match_eager(
    fla, device, name, mode, bsz, heads, head_dim, lengths
):
    module = getattr(fla.layers, name)().eval()
    inputs, initial_state, kwargs = _lin_attention_case(
        name, device, mode, bsz, heads, head_dim, lengths
    )

    def forward(q, k, v, g, beta, initial_state):
        return module(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            **kwargs,
        )

    compiled = torch.compile(forward, fullgraph=True)  # no graph break
    with torch.inference_mode():
        eager = forward(*inputs, initial_state)
        with torch._dynamo.config.patch(error_on_recompile=True):  # no recompile
            actual = compiled(*(x.clone() for x in inputs), initial_state.clone())
            compiled(*(x.clone() for x in inputs), initial_state.clone())

    # fwd check
    for a, b in zip(actual, eager, strict=True):
        _check(a, b)


@pytest.mark.parametrize(
    "name,mode,bsz,heads,head_dim,lengths",
    _lin_attention_cases(TRAINABLE_LIN_ATTENTION, {}),  # Training is too expensive for CI so empty
)
def test_lin_attention_compiled_outputs_and_gradients_match_eager(
    fla, device, name, mode, bsz, heads, head_dim, lengths
):
    # XPU bug, also marked in transformers as incompatible
    if device.type == "xpu" and name == "chunk_kimi_delta_attention":
        pytest.skip("Chunk KDA backward is not supported on XPU")

    module = getattr(fla.layers, name)().train()
    inputs, initial_state, kwargs = _lin_attention_case(
        name, device, mode, bsz, heads, head_dim, lengths
    )

    # Dummy input
    eager_inputs = tuple(_leaf(x) for x in (*inputs, initial_state))
    compiled_inputs = tuple(_leaf(x) for x in (*inputs, initial_state))

    head = nn.Linear(heads * head_dim, 32, bias=False).to(device)
    targets = torch.randint(32, (bsz, inputs[0].shape[1]), device=device)
    state_target = torch.randn_like(initial_state) * 0.01

    def forward(q, k, v, g, beta, initial_state):
        y, final_state = module(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            **kwargs,
        )
        residual = v + y  # makes sure we dont have unintended inplace modifications (compile issue)
        logits = head(residual.float().flatten(2))
        return residual, logits, final_state

    def loss_fn(result):
        _, logits, final_state = result
        return (
            F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
            + 0.01 * (final_state.float() - state_target).square().sum()  # just to make sure this could work
        )


    compiled = torch.compile(forward, fullgraph=True)  # no graph break
    eager_out = forward(*eager_inputs)
    with torch._dynamo.config.patch(error_on_recompile=True):  # no recompile
        compiled_out = compiled(*compiled_inputs)
        compiled(*compiled_inputs)

    # fwd check
    for a, b in zip(compiled_out, eager_out, strict=True):
        _check(a, b)

    # bwd check
    eager_grads = torch.autograd.grad(loss_fn(eager_out), (*eager_inputs, head.weight))
    with torch._dynamo.config.patch(error_on_recompile=True):  # no recompile
        compiled_grads = torch.autograd.grad(loss_fn(compiled_out), (*compiled_inputs, head.weight))

    for label, actual, expected in zip(
        ("dq", "dk", "dv", "dg", "dbeta", "dh0", "dweight"),
        compiled_grads, eager_grads, strict=True,
    ):
        assert expected.abs().max() > 0, f"No gradient signal for {label}"
        _check(actual, expected)

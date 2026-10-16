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
"""FP8 W8A16 kernels against independent dequantization, routing, and GEMM references."""

import kernels
import pytest
import torch

TEST_DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else None
)
pytestmark = pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator required")


@pytest.fixture(scope="module")
def kernel():
    return kernels.get_kernel("kernels-community/finegrained-kernels", version=0)


def _weight_reference(
    weight, scales, block_n, block_k, gate=False, dtype=torch.bfloat16
):
    if scales.dtype == torch.uint8:
        scales = torch.pow(2.0, scales.float() - 127)
    rows = weight.shape[-2]
    if scales.shape[-2] == 1:
        scale_rows = torch.zeros(rows, device=weight.device, dtype=torch.long)
    elif gate:
        row = torch.arange(rows, device=weight.device)
        scale_rows = (row // 2 // block_n) * 2 + row % 2
    else:
        scale_rows = torch.arange(rows, device=weight.device) // block_n
    return (
        weight.float() * scales[..., scale_rows, :].repeat_interleave(block_k, -1)
    ).to(dtype)


@pytest.mark.parametrize(
    "m,n,k,block_n,block_k,gate,dtype",
    [
        pytest.param(
            7, 256, 256, 128, 128, False, torch.bfloat16, marks=pytest.mark.kernels_ci
        ),
        (33, 192, 256, 128, 128, False, torch.float16),
        pytest.param(
            1, 128, 256, 256, 256, True, torch.bfloat16, marks=pytest.mark.kernels_ci
        ),
        (17, 256, 256, 64, 128, True, torch.bfloat16),
        (7, 128, 256, 256, 128, True, torch.bfloat16),
        (7, 192, 256, 128, 128, True, torch.bfloat16),
    ],
)
def test_block_fp8_weight_only(kernel, m, n, k, block_n, block_k, gate, dtype):
    torch.manual_seed(42)
    a = torch.randn(m, k, device=TEST_DEVICE, dtype=dtype)
    rows = 2 * n if gate else n
    b = torch.randn(rows, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scale_rows = (rows + block_n - 1) // block_n
    if gate and scale_rows > 1:
        scale_rows = 2 * ((n + block_n - 1) // block_n)
    scales = torch.pow(
        2.0, torch.randint(-3, 3, (scale_rows, k // block_k), device=TEST_DEVICE)
    )
    bias = torch.randn(rows, device=TEST_DEVICE, dtype=dtype)
    dequantized = _weight_reference(b, scales, block_n, block_k, gate, dtype)
    reference = a.float() @ dequantized.float().T + bias.float()
    if gate:
        reference = torch.nn.functional.silu(reference[:, 0::2]) * reference[:, 1::2]
    with torch.no_grad():
        actual = kernel.matmul_2d(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=gate,
            bias=bias,
            output_dtype=dtype,
        )
    torch.testing.assert_close(actual, reference.to(dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("scalar", [False, True])
def test_block_fp8_weight_only_compiles_fullgraph(kernel, scalar):
    torch.manual_seed(42)
    a = torch.randn(5, 256, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 256, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scales = (
        torch.tensor(0.25, device=TEST_DEVICE)
        if scalar
        else torch.tensor([[0.25, 0.5], [1.0, 2.0]], device=TEST_DEVICE)
    )

    def forward(a, b, scales):
        return kernel.matmul_2d(
            a, b, None, scales, activation_format="bf16", output_dtype=a.dtype
        )

    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    with torch.no_grad():
        expected = forward(a, b, scales)
        torch.testing.assert_close(compiled(a, b, scales), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            compiled(a + 1, b, scales), forward(a + 1, b, scales), rtol=0, atol=0
        )


@pytest.mark.parametrize("op", ["batched", "grouped"])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_routed_block_fp8_weight_only(kernel, op, per_tensor):
    torch.manual_seed(17)
    experts, rows, n, k = 3, 17, 128, 256
    a = torch.randn(5, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(experts, 2 * n, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        -3, 2, (experts, 1, 1) if per_tensor else (experts, 2, 2), device=TEST_DEVICE
    )
    scales = (exponents + 127).to(torch.uint8)
    bias = torch.randn(experts, 2 * n, device=TEST_DEVICE, dtype=a.dtype)
    ids = torch.tensor(([0, 2, 0, 2] * 4) + [3], device=TEST_DEVICE, dtype=torch.int32)
    gather = (torch.arange(rows, device=TEST_DEVICE) % 5).to(torch.int32)
    scatter = torch.randperm(rows, device=TEST_DEVICE).to(torch.int32)
    reference = torch.zeros(rows, n, device=TEST_DEVICE, dtype=a.dtype)
    weights = _weight_reference(
        b, scales, 2 * n if per_tensor else 128, k if per_tensor else 128, True
    )
    for row in range(rows - 1):
        expert = ids[row]
        projection = (
            a[gather[row]].float() @ weights[expert].float().T + bias[expert].float()
        )
        reference[scatter[row]] = (
            torch.nn.functional.silu(projection[0::2]) * projection[1::2]
        ).to(a.dtype)
    if op == "batched":
        kwargs = dict(expert_ids=ids, gather_idx=gather, scatter_idx=scatter)
        fn = kernel.matmul_batched
    else:
        order = torch.argsort(ids)
        # Expert 1 is empty; pad E=3 to a power of two. The off-rank route
        # sits past the local schedule, with a valid entry in both maps.
        starts = torch.tensor([0, 8, 8, 16, 16], device=TEST_DEVICE, dtype=torch.int32)
        kwargs = dict(
            expert_start=starts, gather_idx=gather[order], scatter_idx=scatter[order]
        )
        fn = kernel.matmul_grouped
    valid_rows = scatter[:-1].long()
    with torch.no_grad():
        result = fn(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=True,
            bias=bias,
            output_dtype=a.dtype,
            **kwargs,
        )
        torch.testing.assert_close(
            result[valid_rows], reference[valid_rows], rtol=2e-3, atol=2e-3
        )
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=True,
            bias=bias,
            output_dtype=a.dtype,
            **kwargs,
        )
        torch.testing.assert_close(
            actual[valid_rows], result[valid_rows], rtol=0, atol=0
        )


@pytest.mark.parametrize("op", ["batched", "grouped"])
@pytest.mark.parametrize("tokens", [1, 5])
def test_weight_only_fp8_moe(kernel, op, tokens):
    torch.manual_seed(21)
    e, h, i = 4, 256, 128
    hidden = torch.randn(tokens, h, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.05
    gate_up = (torch.randn(e, 2 * i, h, device=TEST_DEVICE) * 0.1).to(
        torch.float8_e4m3fn
    )
    down = (torch.randn(e, h, i, device=TEST_DEVICE) * 0.1).to(torch.float8_e4m3fn)
    gate_scales = torch.pow(2.0, torch.randint(-2, 1, (e, 2, 2), device=TEST_DEVICE))
    down_scales = torch.pow(2.0, torch.randint(-2, 1, (e, 2, 1), device=TEST_DEVICE))
    ids = torch.tensor(
        [[0, 4], [2, 4], [0, 2], [4, 4], [0, 0]][:tokens], device=TEST_DEVICE
    )
    routing = torch.full((tokens, 2), 0.5, device=TEST_DEVICE)
    gate_weight = _weight_reference(gate_up, gate_scales, 128, 128, True)
    down_weight = _weight_reference(down, down_scales, 128, 128)
    reference = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(2):
            expert = int(ids[token, slot])
            if expert >= e:
                continue
            projections = hidden[token].float() @ gate_weight[expert].float().T
            inter = (
                torch.nn.functional.silu(projections[0::2]) * projections[1::2]
            ).to(hidden.dtype)
            contribution = (inter.float() @ down_weight[expert].float().T).to(
                hidden.dtype
            )
            reference[token] += contribution.float() * routing[token, slot]
    fn = getattr(kernel, "moe_fused_" + op)
    with torch.no_grad():
        actual = fn(
            hidden,
            ids,
            routing,
            gate_up,
            down,
            gate_scales,
            down_scales,
            activation_format="bf16",
        )
    torch.testing.assert_close(actual, reference.to(actual.dtype), rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("op", ["batched", "grouped"])
def test_raw_activation_launch_keeps_full_precision_weights(kernel, op):
    torch.manual_seed(34)
    a = torch.randn(4, 128, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(2, 128, 128, device=TEST_DEVICE, dtype=a.dtype)
    ids = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE, dtype=torch.int32)
    kwargs = (
        dict(expert_ids=ids)
        if op == "batched"
        else dict(
            expert_start=torch.tensor([0, 2, 4], device=TEST_DEVICE, dtype=torch.int32)
        )
    )
    with torch.no_grad():
        actual = getattr(kernel, "matmul_" + op)(
            a, b, None, None, activation_format="bf16", **kwargs
        )
    reference = (
        torch.bmm(a[:, None, :].float(), b[ids.long()].transpose(1, 2).float())
        .squeeze(1)
        .to(a.dtype)
    )
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


def test_raw_grouped_gather_warp_spec_scope(kernel, monkeypatch):
    import importlib
    import triton

    pruners = importlib.import_module(kernel.__name__ + ".pruners")
    monkeypatch.setattr(pruners, "is_sm10x", lambda: True)
    monkeypatch.setattr(pruners, "_warp_spec_needs_descriptor_tile", lambda: True)
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "WARP_SPEC": ws,
                "A_MEMORY_MODE": mode,
                "B_MEMORY_MODE": "host_descriptor",
            },
            num_warps=warps,
        )
        for bm, warps, ws, mode in [
            (128, 4, True, "host_descriptor"),
            (128, 8, True, "host_descriptor"),
            (128, 16, True, "host_descriptor"),
            (64, 4, True, "host_descriptor"),
            (128, 16, True, "pointer"),
            (64, 16, False, "host_descriptor"),
        ]
    ]
    prune = pruners.warp_spec_memory_mode_pruner(gather_only=True)
    # The compiler failure also affects original BF16 weights: it is a gathered
    # descriptor layout restriction, independent of weight dequantization.
    for scales in (None, object()):
        assert prune(configs, {"GatherIdx": object(), "WeightScales": scales}) == [
            configs[i] for i in (0, 4, 5)
        ]
        assert prune(configs, {"GatherIdx": None, "WeightScales": scales}) == configs


@pytest.mark.skipif(
    TEST_DEVICE != "cuda" or torch.cuda.get_device_capability() != (10, 0),
    reason="B200 compiler regression",
)
@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_raw_grouped_warp_spec_survivors_compile(kernel, gather, scaled):
    """Compile surviving TMA configurations using mock pointers; launch no GPU kernels."""
    import importlib

    from triton.runtime.jit import MockTensor
    from triton.tools.tensor_descriptor import TensorDescriptor

    jit = importlib.import_module(
        kernel.__name__ + ".grouped"
    ).full_precision_matmul_grouped_kernel.fn
    dtype = torch.float8_e4m3fn if scaled else torch.bfloat16
    a_desc = TensorDescriptor.from_tensor(
        torch.empty((64, 256), dtype=torch.bfloat16), [1 if gather else 128, 128]
    )
    b_desc = TensorDescriptor.from_tensor(
        torch.empty((8, 512, 256), dtype=dtype), [1, 128, 128]
    )
    a, b, c = MockTensor(torch.bfloat16), MockTensor(dtype), MockTensor(torch.bfloat16)
    idx = MockTensor(torch.int32)
    scale_operands = (
        dict(
            WeightScales=MockTensor(torch.float32),
            stride_scale_e=8,
            stride_scale_n=2,
            stride_scale_k=1,
            SCALE_BLOCK_N=128,
            SCALE_BLOCK_K=128,
            GATE_SCALES=True,
        )
        if scaled
        else {}
    )
    jit.warmup(
        a,
        a_desc,
        b,
        b_desc,
        c,
        None,
        idx if gather else None,
        None,
        idx,
        idx,
        64,
        256,
        256,
        256,
        1,
        512 * 256,
        1,
        256,
        256,
        1,
        0,
        0,
        8,
        4,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        NUM_EXPERTS_POW2=8,
        NUM_SMS=144,
        WARP_SPEC=True,
        PACKED_SCHEDULE=False,
        A_MEMORY_MODE="host_descriptor",
        B_MEMORY_MODE="host_descriptor",
        GATE=True,
        num_warps=4 if gather else 16,
        num_stages=5,
        grid=(144,),
        **scale_operands,
    )


@pytest.mark.parametrize("split_k", [1, 2, 8, 16])
@pytest.mark.parametrize("gate,scalar", [(False, False), (True, False), (True, True)])
def test_weight_only_decode_split_and_empty_partials(
    kernel, monkeypatch, split_k, gate, scalar
):
    """Every split arm must reduce correctly, including splits beyond the final K group."""
    import importlib

    import triton

    if TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("B200 decode route")
    weight_only = importlib.import_module(kernel.__name__ + ".weight_only")
    monkeypatch.setattr(
        weight_only._decode_partial,
        "configs",
        [
            triton.Config(
                {"BLOCK_N": 16, "BLOCK_K": 256, "SPLIT_K": split_k},
                num_warps=4,
                num_stages=1,
            )
        ],
    )
    torch.manual_seed(104)
    n, k = 192, 384
    rows = 2 * n if gate else n
    a = torch.randn(1, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(rows, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    shape = (
        (1, 1)
        if scalar
        else (2 * ((n + 127) // 128) if gate else (n + 127) // 128, k // 128)
    )
    scales = torch.rand(shape, device=TEST_DEVICE) * 0.031 + 0.013
    block_n, block_k = (rows, k) if scalar else (128, 128)
    bias = torch.randn(rows, device=TEST_DEVICE, dtype=a.dtype) * 0.1
    w = _weight_reference(b, scales, block_n, block_k, gate)
    ref = a.float() @ w.float().T + bias.float()
    if gate:
        ref = torch.nn.functional.silu(ref[:, 0::2]) * ref[:, 1::2]
    got = kernel.matmul_2d(
        a,
        b,
        None,
        scales,
        activation_format="bf16",
        gate=gate,
        bias=bias,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_weight_only_prefill_temporary_workspace(kernel, gate, compiled):
    """The large-weight path must retain scaled-weight rounding and fused bias/GLU semantics."""
    if TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("B200 prefill route")
    torch.manual_seed(107)
    m, n, k = 17, 768 if gate else 1536, 1024
    rows = 2 * n if gate else n
    a = torch.randn(m, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(rows, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scales = torch.rand(rows // 128, k // 128, device=TEST_DEVICE) * 0.031 + 0.013
    bias = torch.randn(rows, device=TEST_DEVICE, dtype=a.dtype) * 0.1 if gate else None
    w = _weight_reference(b, scales, 128, 128, gate)
    ref = a.double() @ w.double().T
    if gate:
        ref += bias.double()
        ref = torch.nn.functional.silu(ref[:, 0::2]) * ref[:, 1::2]

    def run(x):
        return kernel.matmul_2d(
            x,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=gate,
            bias=bias,
            output_dtype=x.dtype,
        )

    if compiled:
        run = torch.compile(run, fullgraph=True, dynamic=False)
    actual = run(a)
    if not gate:
        # Independent dequantization plus the vendor GEMM is the exact plain oracle.
        torch.testing.assert_close(actual, a @ w.T, rtol=0, atol=0)
        # FP64 also bounds the aggregate BF16 error, including cancellation points
        # where local ULP counts overstate ordinary FP32 accumulation error.
        relative_error = torch.linalg.vector_norm(
            actual.double() - ref
        ) / torch.linalg.vector_norm(ref)
        assert relative_error < torch.finfo(a.dtype).eps / 2
        return
    expected = ref.to(a.dtype)
    # FP32 MMA reduction order can move a BF16 rounding tie by one output ULP.
    # Use the independent FP64 oracle and bound that discrete difference explicitly.
    _, exponent = torch.frexp(expected.float())
    spacing = torch.ldexp(torch.ones_like(expected.float()), exponent - 8)
    spacing = torch.where(
        expected == 0, torch.finfo(a.dtype).tiny * torch.finfo(a.dtype).eps, spacing
    )
    ulp_error = (actual.float() - expected.float()).abs() / spacing
    assert ulp_error.max() <= 1, f"maximum BF16 error: {ulp_error.max().item()} ULP"


@pytest.mark.parametrize("gate,scalar", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.uint8])
def test_temporary_weight_workspace_expert_strides(kernel, gate, scalar, scale_dtype):
    """The shared dequantizer must retain expert, partial-block, and gate scale indexing."""
    import importlib

    weight_only = importlib.import_module(kernel.__name__ + ".weight_only")
    torch.manual_seed(109)
    e, n, k = 3, 192, 384
    rows = 2 * n if gate else n
    # A transposed weight view exercises the input strides independently of output layout.
    b = (
        torch.randn(e, k, rows, device=TEST_DEVICE)
        .to(torch.float8_e4m3fn)
        .transpose(-1, -2)
    )
    shape = (
        (e, 1, 1)
        if scalar
        else (e, 2 * ((n + 127) // 128) if gate else (n + 127) // 128, k // 128)
    )
    if scale_dtype == torch.uint8:
        scales = torch.randint(123, 129, shape, device=TEST_DEVICE, dtype=torch.uint8)
    else:
        scales = torch.rand(shape, device=TEST_DEVICE) * 0.031 + 0.013
    bn, bk = (rows, k) if scalar else (128, 128)
    actual = weight_only.dequantize_weight(
        b, scales, torch.float16, bn, bk, gate and not scalar
    )
    expected = _weight_reference(b, scales, bn, bk, gate, torch.float16)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gate", [False, True])
def test_weight_only_decode_compiles_fullgraph(kernel, monkeypatch, gate):
    """The split-K decode route must capture its workspace/reduction and consume new inputs."""
    import importlib

    import triton

    if TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("B200 decode route")
    weight_only = importlib.import_module(kernel.__name__ + ".weight_only")
    monkeypatch.setattr(
        weight_only._decode_partial,
        "configs",
        [
            triton.Config(
                {"BLOCK_N": 16, "BLOCK_K": 256, "SPLIT_K": 4}, num_warps=4, num_stages=1
            )
        ],
    )
    torch.manual_seed(113)
    rows, k = 512 if gate else 256, 384
    a = torch.randn(1, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(rows, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scales = torch.rand(rows // 128, k // 128, device=TEST_DEVICE) * 0.031 + 0.013
    bias = torch.randn(rows, device=TEST_DEVICE, dtype=a.dtype) * 0.1

    def run(x):
        return kernel.matmul_2d(
            x,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=gate,
            bias=bias,
            output_dtype=x.dtype,
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=False, mode="reduce-overhead")
    torch.testing.assert_close(compiled(a), run(a), rtol=0, atol=0)
    torch.testing.assert_close(compiled(a + 1), run(a + 1), rtol=0, atol=0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("op", ["batched", "grouped"])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("per_tensor", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_decode_routing_split_k(
    kernel, monkeypatch, op, gate, per_tensor, compiled
):
    """Sparse decode retains top-k gathers, scatter, empty experts, bias and off-rank masks."""
    import importlib

    import triton

    if TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("B200 decode specialization")
    module = importlib.import_module(kernel.__name__ + ".weight_only")
    tuner = module._decode_partial
    monkeypatch.setattr(
        tuner,
        "configs",
        [
            triton.Config(
                {"BLOCK_N": 32, "BLOCK_K": 128, "SPLIT_K": 4}, num_warps=4, num_stages=2
            )
        ],
    )
    tuner.cache.clear()
    torch.manual_seed(135)
    e, n, k = 4, 192, 384
    a = torch.randn(2, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    rows = n * (2 if gate else 1)
    b = (torch.randn(e, rows, k, device=TEST_DEVICE) * 0.1).to(torch.float8_e4m3fn)
    if per_tensor:
        scales = torch.rand(e, 1, 1, device=TEST_DEVICE) * 0.3 + 0.1
        bn, bk = rows, k
    else:
        scale_rows = 4 if gate else 2
        scales = (
            torch.randint(-3, 1, (e, scale_rows, 3), device=TEST_DEVICE) + 127
        ).to(torch.uint8)
        bn, bk = 128, 128
    weights = _weight_reference(b, scales, bn, bk, gate)
    bias = torch.randn(e, rows, device=TEST_DEVICE) * 0.01
    expert_list = [3, 0, 3, 1, 4]
    ids = torch.tensor(expert_list, device=TEST_DEVICE, dtype=torch.int32)
    source = torch.tensor([0, 1, 0, 1, 99999], device=TEST_DEVICE, dtype=torch.int32)
    scatter = torch.tensor([2, 0, 3, 1, 4], device=TEST_DEVICE, dtype=torch.int32)
    if op == "batched":
        kwargs = dict(expert_ids=ids, gather_idx=source, scatter_idx=scatter)
    else:
        order = torch.argsort(ids)
        starts = torch.tensor([0, 1, 2, 2, 4], device=TEST_DEVICE, dtype=torch.int32)
        kwargs = dict(
            expert_start=starts, gather_idx=source[order], scatter_idx=scatter[order]
        )
    fn = getattr(kernel, "matmul_" + op)

    def forward(x):
        return fn(
            x,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=gate,
            bias=bias,
            output_dtype=torch.float32,
            **kwargs,
        )

    if compiled:
        forward = torch.compile(forward, fullgraph=True, mode="reduce-overhead")
    for x in (a, a + 1) if compiled else (a,):
        expected = torch.empty(4, n, device=TEST_DEVICE)
        for row, expert in enumerate(expert_list[:-1]):
            value = (
                x[source[row]].double() @ weights[expert].double().T
                + bias[expert].double()
            )
            if gate:
                value = torch.nn.functional.silu(value[0::2]) * value[1::2]
            expected[scatter[row]] = value.float()
        with torch.no_grad():
            actual = forward(x)
        torch.testing.assert_close(actual[:4], expected, rtol=2e-4, atol=2e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("op", ["batched", "grouped"])
def test_sparse_decode_empty_routes(kernel, op):
    """An EP rank with no local routes returns an empty matrix without launching a zero grid."""
    a = torch.empty(0, 256, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.zeros(4, 128, 256, device=TEST_DEVICE, dtype=torch.float8_e4m3fn)
    scales = torch.ones(4, 1, 1, device=TEST_DEVICE)
    kwargs = (
        dict(expert_ids=torch.empty(0, device=TEST_DEVICE, dtype=torch.int32))
        if op == "batched"
        else dict(expert_start=torch.zeros(5, device=TEST_DEVICE, dtype=torch.int32))
    )
    result = getattr(kernel, "matmul_" + op)(
        a, b, None, scales, activation_format="bf16", **kwargs
    )
    assert result.shape == (0, 128)
    assert result.dtype == a.dtype

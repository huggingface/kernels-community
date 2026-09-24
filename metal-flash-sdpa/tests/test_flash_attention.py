import itertools

import kernels
import pytest
import torch

metal_flash_sdpa = kernels.get_kernel("kernels-community/metal-flash-sdpa", version=1)

HEAD_DIMS = [32, 64, 72, 80, 96, 128, 192, 256]

# Absolute tolerances against a float64 reference.
ATOL = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 3e-2}


def create_cu_seqlens(seq_lengths):
    """Create cumulative sequence lengths tensor."""
    cu_seqlens = [0] + list(itertools.accumulate(seq_lengths))
    return torch.tensor(cu_seqlens, dtype=torch.int32, device="mps")


def reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=None, causal=False, softcap=0.0, sinks=None):
    """Per-sequence attention in float64 on the CPU.

    The causal mask is aligned to the bottom-right corner, as in flash-attn. Rows
    without any visible key produce zeros. Sinks add one logit per head to the
    softmax, which does not attend to a value.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    num_heads, num_heads_kv = q.shape[1], k.shape[1]
    q, k, v = (t.cpu().double() for t in (q, k, v))
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    out = torch.zeros_like(q)
    for b in range(len(cu_q) - 1):
        q_len, k_len = cu_q[b + 1] - cu_q[b], cu_k[b + 1] - cu_k[b]
        if q_len == 0 or k_len == 0:
            continue
        qb = q[cu_q[b] : cu_q[b + 1]].transpose(0, 1)
        kb = k[cu_k[b] : cu_k[b + 1]].transpose(0, 1).repeat_interleave(num_heads // num_heads_kv, 0)
        vb = v[cu_k[b] : cu_k[b + 1]].transpose(0, 1).repeat_interleave(num_heads // num_heads_kv, 0)
        scores = qb @ kb.transpose(-1, -2) * scale
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        if causal:
            row = torch.arange(q_len)[:, None] + (k_len - q_len)
            col = torch.arange(k_len)[None, :]
            scores = scores.masked_fill(col > row, float("-inf"))
        if sinks is not None:
            sink_logits = sinks.cpu().double()[:, None, None].expand(-1, q_len, 1)
            probs = torch.softmax(torch.cat([scores, sink_logits], dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out[cu_q[b] : cu_q[b + 1]] = (probs @ vb).transpose(0, 1)
    return out


def run_varlen(
    q_lens,
    k_lens,
    dtype=torch.float32,
    head_dim=64,
    num_heads=4,
    num_heads_kv=None,
    causal=False,
    softcap=0.0,
    input_scale=1.0,
    sinks=False,
):
    num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
    cu_seqlens_q = create_cu_seqlens(q_lens)
    cu_seqlens_k = create_cu_seqlens(k_lens)
    q = input_scale * torch.randn(sum(q_lens), num_heads, head_dim, device="mps", dtype=dtype)
    k = input_scale * torch.randn(sum(k_lens), num_heads_kv, head_dim, device="mps", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, head_dim, device="mps", dtype=dtype)
    s_aux = 2 * torch.randn(num_heads, device="mps", dtype=dtype) if sinks else None

    out = metal_flash_sdpa.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max(q_lens),
        max(k_lens),
        causal=causal,
        softcap=softcap,
        s_aux=s_aux,
    )
    expected = reference_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal, softcap=softcap, sinks=s_aux
    )
    torch.testing.assert_close(out.cpu().double(), expected, atol=ATOL[dtype], rtol=0)


SEQ_CONFIGS = [
    # (q lengths, k lengths)
    ([32], [32]),
    ([8, 16, 12], [10, 20, 15]),
    ([2], [2]),
    ([16], [32]),  # q_len < k_len
    ([32], [16]),  # q_len > k_len
    ([1], [64]),  # decode
    ([1, 37, 5], [100, 37, 9]),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_config", SEQ_CONFIGS)
@pytest.mark.parametrize("causal", [False, True])
def test_varlen(dtype, head_dim, seq_config, causal):
    torch.manual_seed(42)
    run_varlen(*seq_config, dtype=dtype, head_dim=head_dim, causal=causal)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_ci(dtype, head_dim, causal):
    torch.manual_seed(42)
    run_varlen([1, 37, 50], [100, 37, 20], dtype=dtype, head_dim=head_dim, causal=causal)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads,num_heads_kv", [(8, 4), (8, 2), (32, 4)])
@pytest.mark.parametrize("causal", [False, True])
def test_gqa(dtype, num_heads, num_heads_kv, causal):
    torch.manual_seed(42)
    run_varlen(
        [24, 1, 40],
        [24, 50, 17],
        dtype=dtype,
        head_dim=128,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
        causal=causal,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [5.0, 30.0, 50.0])
@pytest.mark.parametrize("causal", [False, True])
def test_softcap(dtype, softcap, causal):
    torch.manual_seed(42)
    # Scores have a standard deviation of about input_scale**2 = 16, so the tanh saturates.
    run_varlen([48, 9], [48, 70], dtype=dtype, head_dim=64, causal=causal, softcap=softcap, input_scale=4.0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_sinks(dtype, head_dim, causal, softcap):
    torch.manual_seed(42)
    # Includes a sequence without keys, a decode step and a partial key block.
    run_varlen(
        [1, 37, 50, 4],
        [100, 37, 20, 0],
        dtype=dtype,
        head_dim=head_dim,
        num_heads=8,
        num_heads_kv=2,
        causal=causal,
        softcap=softcap,
        sinks=True,
    )


def test_sinks_validation():
    q, k, v = (torch.randn(16, 4, 64, device="mps") for _ in range(3))
    cu_seqlens = create_cu_seqlens([16])
    with pytest.raises(RuntimeError, match=r"s_aux must have shape \[num_heads\]"):
        metal_flash_sdpa.flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, 16, 16, s_aux=torch.zeros(2, device="mps")
        )
    with pytest.raises(RuntimeError, match="MPS"):
        metal_flash_sdpa.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 16, 16, s_aux=torch.zeros(4))


# Decode shapes (at most 8 query tokens per sequence) use the vector kernels.
# The key lengths pick the kernel independently of the GPU: the 1-pass kernel
# below 1024 keys, the 2-pass kernel for GQA from 4096 keys, and its GQA
# read-once variant for single query tokens with GQA 8/12/16 from 8192 keys.
DECODE_CONFIGS = {
    # name: (q lengths, k lengths, num_heads, num_heads_kv)
    "1pass": ([1, 3, 8, 0, 1], [50, 3, 200, 10, 0], 8, 2),
    "2pass": ([1, 4, 2, 1], [4100, 5000, 30, 0], 8, 2),
    "2pass_gqa8": ([1, 1, 0, 1], [8200, 100, 50, 0], 16, 2),
    "2pass_gqa12": ([1, 1], [9000, 17], 12, 1),
    "2pass_gqa16": ([1, 1], [8192, 1], 16, 1),
}


def run_decode(config, dtype, head_dim, causal, softcap=0.0, sinks=False):
    q_lens, k_lens, num_heads, num_heads_kv = DECODE_CONFIGS[config]
    run_varlen(
        q_lens,
        k_lens,
        dtype=dtype,
        head_dim=head_dim,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
        causal=causal,
        softcap=softcap,
        sinks=sinks,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("config", ["1pass", "2pass"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sinks", [False, True])
def test_decode(dtype, head_dim, config, causal, sinks):
    torch.manual_seed(42)
    run_decode(config, dtype, head_dim, causal, sinks=sinks)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "config,head_dim",
    [("2pass_gqa8", 64), ("2pass_gqa8", 128), ("2pass_gqa12", 128), ("2pass_gqa16", 128)],
)
@pytest.mark.parametrize("sinks", [False, True])
def test_decode_gqa(dtype, config, head_dim, sinks):
    torch.manual_seed(42)
    run_decode(config, dtype, head_dim, causal=True, sinks=sinks)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config,head_dim", [("1pass", 128), ("2pass", 64), ("2pass_gqa8", 128)])
def test_decode_ci(dtype, config, head_dim):
    torch.manual_seed(42)
    run_decode(config, dtype, head_dim, causal=True, sinks=True)


@pytest.mark.parametrize("config", ["1pass", "2pass", "2pass_gqa8"])
def test_decode_softcap(config):
    torch.manual_seed(42)
    run_decode(config, torch.float32, 128, causal=True, softcap=5.0, sinks=True)


def test_legacy_softcapping_disabled():
    # flash_attention_varlen keeps its original convention: 1.0 disables softcapping.
    torch.manual_seed(42)
    q, k, v = (torch.randn(16, 2, 64, device="mps") for _ in range(3))
    cu_seqlens = create_cu_seqlens([16])
    out = torch.empty_like(q)
    metal_flash_sdpa.flash_attention_varlen(out, q, k, v, cu_seqlens, cu_seqlens, 16, 16, softcapping=1.0)
    expected = reference_attention(q, k, v, cu_seqlens, cu_seqlens)
    torch.testing.assert_close(out.cpu().double(), expected, atol=ATOL[torch.float32], rtol=0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_strided_inputs(causal):
    torch.manual_seed(42)
    num_heads, head_dim = 8, 128
    cu_seqlens = create_cu_seqlens([20, 30])
    # q, k and v are non-contiguous views into a packed qkv tensor.
    qkv = torch.randn(50, 3, num_heads, head_dim, device="mps", dtype=torch.float16)
    q, k, v = qkv.unbind(1)
    # Write into the second half of the heads of a larger buffer.
    buffer = torch.full((50, 2 * num_heads, head_dim), float("nan"), device="mps", dtype=torch.float16)
    out = buffer[:, num_heads:]

    metal_flash_sdpa.flash_attention_varlen(out, q, k, v, cu_seqlens, cu_seqlens, 30, 30, do_causal=causal)

    expected = reference_attention(q, k, v, cu_seqlens, cu_seqlens, causal=causal)
    torch.testing.assert_close(out.cpu().double(), expected, atol=ATOL[torch.float16], rtol=0)
    assert torch.isnan(buffer[:, :num_heads]).all()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_empty_sequences(causal):
    torch.manual_seed(42)
    q_lens, k_lens = [5, 0, 7, 3], [0, 9, 7, 3]
    cu_seqlens_q, cu_seqlens_k = create_cu_seqlens(q_lens), create_cu_seqlens(k_lens)
    q = torch.randn(sum(q_lens), 4, 64, device="mps")
    k = torch.randn(sum(k_lens), 4, 64, device="mps")
    v = torch.randn(sum(k_lens), 4, 64, device="mps")
    out = torch.full_like(q, float("nan"))

    metal_flash_sdpa.flash_attention_varlen(
        out, q, k, v, cu_seqlens_q, cu_seqlens_k, max(q_lens), max(k_lens), do_causal=causal
    )

    # Queries of a sequence without keys produce zeros, like flash-attn.
    assert (out[:5] == 0).all()
    expected = reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
    torch.testing.assert_close(out.cpu().double(), expected, atol=ATOL[torch.float32], rtol=0)


def test_single_token():
    q, k, v = (torch.randn(1, 1, 64, device="mps") for _ in range(3))
    cu_seqlens = create_cu_seqlens([1])
    out = metal_flash_sdpa.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 1, 1)
    torch.testing.assert_close(out, v, rtol=1e-5, atol=1e-5)


def test_long_sequences():
    torch.manual_seed(42)
    run_varlen([1024, 2048], [1024, 4096], dtype=torch.float16, head_dim=128, causal=True)


@pytest.mark.kernels_ci
def test_input_validation():
    q, k, v = (torch.randn(16, 4, 64, device="mps") for _ in range(3))
    cu_seqlens = create_cu_seqlens([16])

    def call(q=q, k=k, v=v, out=None, cu_q=cu_seqlens, cu_k=cu_seqlens):
        out = torch.empty_like(q) if out is None else out
        metal_flash_sdpa.flash_attention_varlen(out, q, k, v, cu_q, cu_k, 16, 16)

    with pytest.raises(RuntimeError, match="Head dimension .* is not supported"):
        x = torch.randn(16, 4, 48, device="mps")
        call(q=x, k=x, v=x)
    with pytest.raises(RuntimeError, match="torch.int32"):
        call(cu_q=cu_seqlens.long(), cu_k=cu_seqlens.long())
    with pytest.raises(RuntimeError, match="MPS"):
        call(cu_q=cu_seqlens.cpu(), cu_k=cu_seqlens.cpu())
    with pytest.raises(RuntimeError, match="dtype"):
        call(k=k.half())
    with pytest.raises(RuntimeError, match="divisible"):
        x = torch.randn(16, 3, 64, device="mps")
        call(k=x, v=x)
    with pytest.raises(RuntimeError, match="contiguous in head_dim"):
        call(q=torch.randn(16, 64, 4, device="mps").transpose(1, 2))
    with pytest.raises(RuntimeError, match="same shape as query"):
        call(out=torch.empty(16, 4, 32, device="mps"))


def test_flash_attn_varlen_func_unsupported():
    q, k, v = (torch.randn(16, 4, 64, device="mps") for _ in range(3))
    cu_seqlens = create_cu_seqlens([16])
    with pytest.raises(NotImplementedError):
        metal_flash_sdpa.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 16, 16, dropout_p=0.1)
    with pytest.raises(NotImplementedError):
        metal_flash_sdpa.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 16, 16, window_size=(4, 0))
    # A list with the default value is accepted.
    metal_flash_sdpa.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 16, 16, window_size=[-1, -1])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_after_mps_operation(causal):
    torch.manual_seed(42)
    q_cpu, k_cpu, v_cpu = [torch.randn(20, 4, 64, device="cpu") for _ in range(3)]
    expected = []
    for start, end in ((0, 8), (8, 20)):
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                q_cpu[start:end].sin().transpose(0, 1),
                k_cpu[start:end].transpose(0, 1),
                v_cpu[start:end].transpose(0, 1),
                is_causal=causal,
            ).transpose(0, 1)
        )
    expected = torch.cat(expected)

    q, k, v = [tensor.to("mps") for tensor in (q_cpu, k_cpu, v_cpu)]
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32, device="mps")
    out = torch.empty_like(q)
    torch.mps.synchronize()

    # Leave PyTorch's compute encoder active immediately before the kernel.
    q = q.sin()
    metal_flash_sdpa.flash_attention_varlen(out, q, k, v, cu_seqlens, cu_seqlens, 12, 12, causal, 64**-0.5, 1.0)
    torch.testing.assert_close(out.tanh().cpu(), expected.tanh(), atol=5e-4, rtol=5e-4)

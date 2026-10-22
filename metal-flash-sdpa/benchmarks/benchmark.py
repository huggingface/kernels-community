"""Benchmarks for `kernels benchmark kernels-community/metal-flash-sdpa`.

Each workload compares `flash_attn_varlen_func` on packed variable-length inputs with
`torch.nn.functional.scaled_dot_product_attention` on the same sequences padded to the longest
one (the `verify_*` reference, whose time is reported as `Ref(ms)`). Ragged batches give SDPA a
mask for the padding. Padding and masks are built in `setup`, outside the timed region.
"""

import random

import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark

# (num_heads, num_heads_kv, head_dim)
LLAMA = (32, 8, 128)
GPT_OSS = (64, 8, 64)


def _ragged(n, lo, hi, seed):
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


class MetalFlashSdpaBenchmark(Benchmark):
    seed = 0

    def _setup(self, q_lens, k_lens, shape, causal=True, dtype=torch.float16):
        H, Hkv, D = shape
        device = self.device
        B, Lq, Lk = len(q_lens), max(q_lens), max(k_lens)
        self.q = torch.randn(sum(q_lens), H, D, device=device, dtype=dtype)
        self.k = torch.randn(sum(k_lens), Hkv, D, device=device, dtype=dtype)
        self.v = torch.randn_like(self.k)
        self.cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        self.cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        self.max_q, self.max_k, self.causal = Lq, Lk, causal
        self.out = torch.empty_like(self.q)

        # Padded inputs for the SDPA reference.
        self.qp = torch.zeros(B, H, Lq, D, device=device, dtype=dtype)
        self.kp = torch.zeros(B, Hkv, Lk, D, device=device, dtype=dtype)
        self.vp = torch.zeros_like(self.kp)
        rows = []
        for b in range(B):
            qs, ks = int(self.cu_q[b]), int(self.cu_k[b])
            self.qp[b, :, : q_lens[b]] = self.q[qs : qs + q_lens[b]].transpose(0, 1)
            self.kp[b, :, : k_lens[b]] = self.k[ks : ks + k_lens[b]].transpose(0, 1)
            self.vp[b, :, : k_lens[b]] = self.v[ks : ks + k_lens[b]].transpose(0, 1)
            rows += [b * Lq + i for i in range(q_lens[b])]
        self.rows = torch.tensor(rows, device=device)

        ragged = len(set(q_lens)) > 1 or len(set(k_lens)) > 1
        self.mask, self.sdpa_causal = None, causal and Lq > 1
        if ragged or (causal and Lq > 1 and Lq != Lk):
            i = torch.arange(Lq, device=device)[None, :, None]
            j = torch.arange(Lk, device=device)[None, None, :]
            ql = torch.tensor(q_lens, device=device)[:, None, None]
            kl = torch.tensor(k_lens, device=device)[:, None, None]
            allowed = (j < kl) & (i < ql)
            if causal:
                allowed &= j <= i + (kl - ql)
            allowed[..., 0] |= ~allowed.any(-1)  # padding rows, dropped by `rows`
            self.mask, self.sdpa_causal = allowed[:, None], False

    def _run(self):
        self.out = self.kernel.flash_attn_varlen_func(
            self.q, self.k, self.v, self.cu_q, self.cu_k, self.max_q, self.max_k, causal=self.causal
        )

    def _reference(self):
        out = F.scaled_dot_product_attention(
            self.qp, self.kp, self.vp, attn_mask=self.mask, is_causal=self.sdpa_causal,
            enable_gqa=self.qp.shape[1] != self.kp.shape[1],
        )
        B, H, Lq, D = out.shape
        return out.transpose(1, 2).reshape(B * Lq, H, D)[self.rows]

    # Decode: one query token per sequence.
    def setup_decode_b1_16k(self):
        self._setup([1], [16384], LLAMA)

    def benchmark_decode_b1_16k(self):
        self._run()

    def verify_decode_b1_16k(self):
        return self._reference()

    def setup_decode_b32_8k(self):
        self._setup([1] * 32, [8192] * 32, LLAMA)

    def benchmark_decode_b32_8k(self):
        self._run()

    def verify_decode_b32_8k(self):
        return self._reference()

    def setup_decode_ragged_b32(self):
        self._setup([1] * 32, _ragged(32, 256, 8192, 0), LLAMA)

    def benchmark_decode_ragged_b32(self):
        self._run()

    def verify_decode_ragged_b32(self):
        return self._reference()

    def setup_decode_ragged_b32_gpt_oss(self):
        self._setup([1] * 32, _ragged(32, 256, 8192, 1), GPT_OSS)

    def benchmark_decode_ragged_b32_gpt_oss(self):
        self._run()

    def verify_decode_ragged_b32_gpt_oss(self):
        return self._reference()

    # Speculative decoding: a few query tokens per sequence.
    def setup_spec_decode_ragged_b16(self):
        self._setup([4] * 16, _ragged(16, 256, 8192, 2), LLAMA)

    def benchmark_spec_decode_ragged_b16(self):
        self._run()

    def verify_spec_decode_ragged_b16(self):
        return self._reference()

    # Prefill.
    def setup_prefill_2k(self):
        self._setup([2048], [2048], LLAMA)

    def benchmark_prefill_2k(self):
        self._run()

    def verify_prefill_2k(self):
        return self._reference()

    def setup_prefill_ragged_b8(self):
        lengths = _ragged(8, 128, 4096, 3)
        self._setup(lengths, lengths, LLAMA)

    def benchmark_prefill_ragged_b8(self):
        self._run()

    def verify_prefill_ragged_b8(self):
        return self._reference()

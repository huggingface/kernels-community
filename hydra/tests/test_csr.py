from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")


def test_dense_causal_csr_cpu_contract():
    from hydra.csr import build_dense_causal_csr

    row_ptr, col_idx, seq_lens = build_dense_causal_csr(
        batch_size=1,
        num_heads=2,
        seq_len=128,
        block_size=32,
        device="cpu",
    )

    assert row_ptr.shape == (1, 2, 5)
    assert seq_lens.tolist() == [128]
    assert row_ptr[0, 0].tolist() == [0, 1, 3, 6, 10]
    assert col_idx[0, 0].tolist() == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]


def test_sliding_window_csr_keeps_diagonal_last():
    from hydra.csr import build_sliding_window_csr

    row_ptr, col_idx, _ = build_sliding_window_csr(
        window=64,
        seq_len=128,
        block_size=32,
        batch_size=1,
        num_heads=1,
        device="cpu",
    )

    rp = row_ptr[0, 0].tolist()
    ci = col_idx[0, 0].tolist()
    for q_block in range(4):
        lo, hi = rp[q_block], rp[q_block + 1]
        assert ci[hi - 1] == q_block
        assert all(k < q_block for k in ci[lo : hi - 1])

import pytest
import torch

import thrift_attention


def test_exports_flash_attention_entrypoints():
    assert callable(thrift_attention.flash_attn_func)
    assert callable(thrift_attention.flash_attn_varlen_func)


def test_unsupported_dropout_is_explicit():
    q = torch.empty(1, 64, 2, 64)
    with pytest.raises(NotImplementedError, match="dropout_p=0.0"):
        thrift_attention.flash_attn_func(q, q, q, dropout_p=0.1)

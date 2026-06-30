"""Kernel test utils"""

import itertools
import random
import unittest
from numbers import Number
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import TensorLikeType

# For now, disable "test_aot_dispatch_dynamic" since there are some
# bugs related to this test in PyTorch 2.4.
DEFAULT_OPCHECK_TEST_UTILS: Tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
)

ALL_OPCHECK_TEST_UTILS: Tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


# Copied/modified from torch._refs.__init__.py
def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    torch._refs._check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)

    return bool(
        torch.all(
            torch.isclose(
                a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        ).item()
    )


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


# A special version of op check that has a restricted default set of test_utils
# and a patched version of allclose that supports fp8 types.
def opcheck(
    op: Union[
        torch._ops.OpOverload,
        torch._ops.OpOverloadPacket,
        torch._library.custom_ops.CustomOpDef,
    ],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = ALL_OPCHECK_TEST_UTILS,
    raise_exception: bool = True,
    cond: bool = True
) -> Dict[str, str]:
    with unittest.mock.patch("torch.allclose", new=fp8_allclose):
        return (
            torch.library.opcheck(
                op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
            )
            if cond
            else {}
        )

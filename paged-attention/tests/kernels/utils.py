"""Kernel test utils"""

import itertools
import random
import unittest
from functools import lru_cache
from numbers import Number
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import pytest
import torch
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

    # MPS doesn't support float64, so use float32 for comparison
    if a.device.type == "mps" or b.device.type == "mps":
        a_cmp = a.float()
        b_cmp = b.float()
    else:
        a_cmp = a.double()
        b_cmp = b.double()
    
    return bool(
        torch.all(
            torch.isclose(
                a_cmp, b_cmp, rtol=rtol, atol=atol, equal_nan=equal_nan
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
    cond: bool = True,
) -> Dict[str, str]:
    with unittest.mock.patch("torch.allclose", new=fp8_allclose):
        if not cond:
            return {}

        return torch.library.opcheck(
            op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
        )


@lru_cache(maxsize=None)
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from paged_attention import ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)

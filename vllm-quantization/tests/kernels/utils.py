"""Kernel test utils"""

import itertools
import random
import unittest
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

def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)




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

def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match the
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = t.unsqueeze(i + 1)\
                  .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])\
                  .flatten(i, i + 1)
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    output = torch.mm((scale_a * a.to(dtype=torch.float32)),
                      (scale_b * b.to(dtype=torch.float32))).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output

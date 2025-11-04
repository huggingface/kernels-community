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


import torch
from ._ops import ops
from .flash_attn2_npu import (
    _flash_attn_func_npu,
    _flash_attn_varlen_func_npu,
)


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    return ops.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,  # defined for aligning params order with corresponding function in `flash-attn`
    max_seqlen_k=None,  # defined for aligning params order with corresponding function in `flash-attn`
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    return ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,  # defined for aligning params order with corresponding function in `flash-attn`
        max_seqlen_k=None,  # defined for aligning params order with corresponding function in `flash-attn`
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        **kwargs,
    )


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]

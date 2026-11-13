# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerics for quantization."""

from typing import Callable
import jax
from jax import numpy as jnp

# A function that generates noise for stochastic rounding.
# args:
#   shape: The requested shape of the noise to generate.
# returns: Noise as a jax.Array whose shape is broadcastable to the requested
#   shape, and whose dtype can be promoted to fp32 implicitly.
NoiseFn = Callable[[tuple[int, ...]], jax.Array]
_QUANTIZE_DTYPES = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)


def should_quantize(dtype: jax.typing.DTypeLike) -> bool:
  """Checks if a given dtype is a floating-point type eligible for quantization.

  Quantization is typically applied to high-precision floating-point values to
  reduce memory usage and increase computational efficiency.

  Args:
    dtype: The data type to check.

  Returns:
    True if the dtype is a standard floating-point type.
  """
  return jnp.dtype(dtype) in _QUANTIZE_DTYPES


def can_dequant_on_output(qtype: jax.typing.DTypeLike) -> bool:
  """Checks if the qtype supports dequantizing after quantized computation.

  This allows performing heavy operations (like matrix multiplications)
  directly on quantized values and applying the dequantization step only to the
  final result. This is generally possible for uniform data types (e.g., int8).
  However, non-uniform types like 'nf4' (NormalFloat 4) use a non-linear
  lookup table of buckets, making direct arithmetic on quantized indices
  mathematically invalid.

  Args:
    qtype: The quantization data type to check.

  Returns:
    True if the qtype supports dequantizing after the computation in quantized
    representation, False otherwise.
  """
  return qtype not in ['nf4']


def get_asymmetric_bound(qtype: jax.typing.DTypeLike) -> tuple[float, float]:
  """Returns the continuous range of target values before rounding.

  This function returns the exact boundaries of the target type in floats.
  Asymmetric quantization maps the input range to the full continuous range of
  the target type, including the "extra" negative value (e.g., -128
  for int8).

  Args:
    qtype: The quantization type. Currently only supports standard JAX integer
      types like jnp.int8 or jnp.int4.

  Returns:
    A tuple of (min, max) representing the continuous representable range.
  """
  try:
    dtype = jnp.dtype(qtype)
  except TypeError as e:
    raise ValueError(f"{qtype} doesn't support asymmetric quantization.") from e

  match dtype:
    case jnp.int8:
      return (-128.0, 127.0)
    case jnp.int4:
      return (-8.0, 7.0)
    case _:
      raise ValueError(f"{qtype} doesn't support asymmetric quantization.")


def get_symmetric_bound(qtype: jax.typing.DTypeLike) -> float:
  """Returns the maximum magnitude of continuous target values before rounding.

  In symmetric quantization, the range is centered at zero. For integer
  types, the bound is extended to qmax + 0.5 to ensure the maximum bucket
  is fully utilized. This defines a representable continuous range of [-B, B].

  Args:
    qtype: The quantization type. Supports standard JAX dtypes (e.g., jnp.int8)
      and synthetic string identifiers (e.g., 'nf4', 'int3', 'mxfp8').

  Returns:
    The positive bound B. The representable continuous range is [-B, B].
  """
  match qtype:
    case 'nf4':
      return 1.0
    case 'int2' | 'int3' | 'int5' | 'int6' | 'int7':
      # The bound is extended to qmax + 0.5 so that we have a better utilization
      # of the qmax bucket. This is more important for fewer bits of int.
      return 2 ** (int(qtype[3:]) - 1) - 0.5
    case 'mxfp8':
      qtype = jnp.float8_e4m3fn
    case 'mxfp4' | 'nvfp4':
      qtype = jnp.float4_e2m1fn

  # Prevent common misconfigurations, e.g., use bf16 as qtype.
  if jnp.dtype(qtype).itemsize > 1:
    raise ValueError(f'Cannot use {qtype} as qtype.')
  try:
    # TODO(jiwonshin): Extend the finfo.max bucket (e.g. by half a step-size)
    # for better utilization. Similar to the +0.5 for integers below, this
    # would allow the maximum floating-point bucket to represent a wider
    # range of the input signal before clipping.
    return float(jnp.finfo(qtype).max)
  except ValueError:
    # Integer types: we add 0.5 because the 'max' bucket captures
    # values in [qmax - 0.5, qmax + 0.5).
    return jnp.iinfo(qtype).max + 0.5


def convert_to(
    x: jax.Array,
    qtype: jax.typing.DTypeLike,
    noise_fn: NoiseFn | None = None,
) -> jax.Array:
  """Converts a high-precision array to the quantized representation.

  This function performs the discrete mapping of continuous values to quantized
  buckets (rounding/clipping or bucketization). It assumes the input 'x' has
  already been scaled to the target type's representable range.

  Args:
    x: The input array, already scaled to the target type's range.
    qtype: The target quantized representation (e.g., 'int8', 'nf4').
    noise_fn: Optional function for stochastic rounding.

  Returns:
    The array in its quantized storage format (indices or low-precision type).
  """
  # Handles synthetic qtypes.
  match qtype:
    case 'nf4':
      return fp_to_nf4(x)
    case 'int2' | 'int3' | 'int5' | 'int6' | 'int7':
      bits = int(qtype[3:])
      qmin = -(2 ** (bits - 1))
      qmax = 2 ** (bits - 1) - 1
      if bits <= 4:
        qtype = jnp.int4
      elif bits <= 8:
        qtype = jnp.int8
      else:
        raise ValueError(f'Unsupported integer dtype: {qtype}')
      return jnp.round(x).clip(qmin, qmax).astype(qtype)
    case 'mxfp8':
      qtype = jnp.float8_e4m3fn
    case 'mxfp4' | 'nvfp4':
      qtype = jnp.float4_e2m1fn

  # Handles builtin qtypes.
  try:
    finfo = jnp.finfo(qtype)
  except ValueError:
    pass
  else:
    # dtype is a floating point type. No rounding needed, but we need to clip to
    # the range to avoid inf or nan (e.g. for e4m3fn).
    qmin, qmax = finfo.min.astype(x.dtype), finfo.max.astype(x.dtype)
    return x.clip(qmin, qmax).astype(qtype)

  # dtype is an integer type. We need to round manually but clipping can be
  # handled by "astype".
  if noise_fn is not None:
    # Stochastic rounding is done in fp32 to avoid bias from bf16, e.g.
    # round(bf16(41)-bf16(0.4)) ~= round(40.5) = 40, rather than
    # round(41-0.4) = round(40.6) = 41.
    x = x.astype(jnp.float32) + noise_fn(x.shape)
  return jnp.round(x).astype(qtype)


def convert_from(x: jax.Array, qtype: jax.typing.DTypeLike) -> jax.Array:
  """Converts a non-uniform quantized array back to floating-point values.

  For non-uniform types (like 'nf4'), this function performs 'unbucketing'
  by mapping quantized indices back to their corresponding floating-point
  values.

  For native uniform types (e.g., 'int8'), this is a no-op.

  Args:
    x: The array in its quantized representation (indices or low-precision).
    qtype: The quantization type that was used to create 'x'.

  Returns:
    The array after representation conversion. For non-uniform types, the
    result is in a floating-point format. For native types, the result
    retains the original dtype of 'x'.
  """
  match qtype:
    case 'nf4':
      return nf4_to_fp(x)
    case _:
      # For native types, no extra conversion is needed. The dtype will be
      # converted during unquantization.
      return x


### NF4


# NB: to work around the issue of calling Jax functions in module-level context.
def get_nf4_buckets() -> jax.Array:
  """Returns the NF4 buckets.

  The buckets are defined in Appendix E of https://arxiv.org/pdf/2305.14314.
  """
  nf4_buckets = jnp.array([
      -1.0,
      -0.6961928009986877,
      -0.5250730514526367,
      -0.39491748809814453,
      -0.28444138169288635,
      -0.18477343022823334,
      -0.09105003625154495,
      0.0,
      0.07958029955625534,
      0.16093020141124725,
      0.24611230194568634,
      0.33791524171829224,
      0.44070982933044434,
      0.5626170039176941,
      0.7229568362236023,
      1.0,
  ])
  return nf4_buckets


def fp_to_nf4(array: jax.Array) -> jax.Array:
  """Quantizes an array to a 4-bit NormalFloat representation."""
  nf4_buckets = get_nf4_buckets()

  def bucketize(x):
    bucket = jnp.argmin(jnp.abs(nf4_buckets - x))
    return bucket

  buckets = jax.vmap(bucketize)(array.ravel())
  return buckets.astype(jnp.uint4).reshape(array.shape)  # stored as uint4.


def nf4_to_fp(array: jax.Array) -> jax.Array:
  """Dequantizes a NF4 array to original dtype."""
  nf4_buckets = get_nf4_buckets()

  def reverse_bucketize(x):
    return nf4_buckets[x]

  return jax.vmap(reverse_bucketize)(array.ravel()).reshape(array.shape)

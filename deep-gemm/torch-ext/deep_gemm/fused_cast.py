# Fused per-token FP8 cast (local to the Kernel Hub build, no upstream counterpart).

import torch
import triton
import triton.language as tl

# PyTorch divides a tensor by a scalar as a multiply by the scalar's float reciprocal, so the reference's
# `amax / 448.0` is `amax * (1 / 448)` with the reciprocal rounded to float32 (exact here as a Python float).
_INV_E4M3_MAX = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(448.0, dtype=torch.float32))


@triton.jit
def _per_token_cast_to_fp8_kernel(
    x_ptr, stride_x, out_ptr, stride_out, sf_ptr, stride_sf, n, num_groups,
    GRAN_K: tl.constexpr, GROUPS: tl.constexpr, USE_UE8M0: tl.constexpr, PACKED: tl.constexpr,
    INV_E4M3_MAX: tl.constexpr,
):
    """`GROUPS` scale groups of `GRAN_K` values of row `program_id(0)`."""
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    columns = groups[:, None] * GRAN_K + tl.arange(0, GRAN_K)[None, :]
    mask = (groups[:, None] < num_groups) & (columns < n)
    # the reference pads the row with zeros to a whole group, which leaves amax as it is
    x = tl.load(x_ptr + row * stride_x + columns, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = amax * INV_E4M3_MAX
    if USE_UE8M0:
        # the power of two at or above the scale, exponent clamped to [1, 254] (`ceil_to_ue8m0`)
        bits = scale.to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
    # `x * (1.0 / sf)` with a correctly rounded reciprocal, as PyTorch computes it (`tl.fdiv` lowers to the approximate
    # `div.full.f32`, which moves values across E4M3 rounding boundaries)
    inverse = tl.math.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
    values = (x * inverse[:, None]).to(tl.float8e4nv)
    tl.store(out_ptr + row * stride_out + columns, values, mask=mask)

    group_mask = groups < num_groups
    if PACKED:
        # the biased exponent, byte `group` of the row's int32s viewed as bytes: four per int32 in K order
        tl.store(sf_ptr + row * stride_sf + groups, (scale.to(tl.uint32, bitcast=True) >> 23).to(tl.uint8),
                 mask=group_mask)
    else:
        tl.store(sf_ptr + row * stride_sf + groups, scale, mask=group_mask)


def fused_per_token_cast_to_fp8(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False, out=None, out_sf=None):
    """`utils.per_token_cast_to_fp8` in one kernel launch, bit for bit: `x` (`(m, n)`, bfloat16 or float32) as
    float8_e4m3fn with one scale per `gran_k` values of a row, rounded up to a power of two with `use_ue8m0` and packed
    four per int32 with `use_packed_ue8m0`. The kernel writes `out` (`(m, n)` float8_e4m3fn) and `out_sf` (`(m, ceil(n /
    gran_k))` float32, or `(m, ceil(n / gran_k) / 4)` int32 when packed) in place, so they can be views of a larger
    buffer; they are allocated when not given. Returns `(out, out_sf)`."""
    assert x.dim() == 2 and x.stride(1) == 1, "x must be 2D with contiguous rows"
    assert not use_packed_ue8m0 or use_ue8m0, "packed scales are UE8M0"
    m, n = x.shape
    num_groups = -(-n // gran_k)
    assert not use_packed_ue8m0 or num_groups % 4 == 0, "packed UE8M0 scales need a multiple of 4 groups per row"
    if out is None:
        out = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    if out_sf is None:
        out_sf = (
            torch.empty((m, num_groups // 4), dtype=torch.int32, device=x.device)
            if use_packed_ue8m0
            else torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
        )
    sf_shape = (m, num_groups // 4) if use_packed_ue8m0 else (m, num_groups)
    sf_dtype = torch.int32 if use_packed_ue8m0 else torch.float32
    assert out.shape == (m, n) and out.dtype == torch.float8_e4m3fn and out.stride(1) == 1
    assert out_sf.shape == sf_shape and out_sf.dtype == sf_dtype and out_sf.stride(1) == 1
    if m == 0:
        return out, out_sf
    # packed scales are written byte by byte into the int32 rows
    sf_target = out_sf.view(torch.uint8) if use_packed_ue8m0 else out_sf
    groups = max(1, 2048 // gran_k)  # a 2048-value tile per program
    grid = (m, triton.cdiv(num_groups, groups))
    _per_token_cast_to_fp8_kernel[grid](
        x, x.stride(0), out, out.stride(0), sf_target, sf_target.stride(0), n, num_groups,
        GRAN_K=gran_k, GROUPS=groups, USE_UE8M0=use_ue8m0, PACKED=use_packed_ue8m0, INV_E4M3_MAX=_INV_E4M3_MAX,
    )
    return out, out_sf

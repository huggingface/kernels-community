from ctypes import c_float
from functools import cache
from tvm_ffi import get_global_func
import math
import operator

from ._ops import ops

_array_getitem = get_global_func("ffi.ArrayGetItem")


def reserve_rng_state(q, gen=None):
    """Reserve v2's B*H*32 Philox increment and return CUDA int64 [seed, offset]."""
    import torch

    if not isinstance(q, torch.Tensor) or not q.is_cuda or q.ndim != 4:
        raise ValueError("q must be a CUDA tensor of shape [B, S, H, D]")
    if q.shape[0] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have positive batch and head counts")
    if gen is not None:
        if not isinstance(gen, torch.Generator) or gen.device.type != "cuda":
            raise TypeError("gen must be a CUDA torch.Generator")
        if gen.device.index not in (None, q.device.index):
            raise ValueError("gen must be on q's device")
    return _reserve_rng_state(q, q.shape[0], q.shape[2], gen)


def _reserve_rng_state(q, batch, heads, gen=None):
    """Internal reservation for validated dense or packed sequence metadata."""
    import torch

    with torch.cuda.device(q.device):
        seed, offset, intragraph, captured = _rng_descriptor_values(
            gen, q.device.index, batch * heads * 32
        )
        return ops.rng_state(q, seed, offset, intragraph, bool(captured))


def _rng_descriptor(gen, device, increment):
    from tvm_ffi import Shape

    return Shape(_rng_descriptor_values(gen, device, increment))


def _rng_descriptor_values(gen, device, increment):
    """Reserve fresh Philox values without an FFI container round trip."""
    import torch

    generator = torch.cuda.default_generators[device] if gen is None else gen
    reserve = getattr(generator, "philox_state", None)
    if reserve is None:
        raise RuntimeError(
            "Generator-backed dropout requires torch.Generator.philox_state "
            "(available in Torch 2.14); upgrade PyTorch."
        )
    # Torch locks the generator and reserves v2's increment atomically.
    seed, offset, intragraph = reserve(increment)
    captured = seed.is_cuda
    # Pass pointers so graph replay reads the live generator state.
    return (
        seed.data_ptr() if captured else seed.item(),
        offset.data_ptr() if captured else offset.item(),
        intragraph.item(),
        int(captured),
    )


@cache
def _cuda_capability(device_index):
    import torch

    return torch.cuda.get_device_capability(device_index)


def _bool_arg(value, name):
    # Match pybind bool conversion: __len__ truthiness is rejected.
    if type(value) is bool:
        return value
    if value is None:
        return False
    if getattr(type(value), "__bool__", None) is not None:
        try:
            return bool(value)
        except Exception:
            pass
    raise RuntimeError(
        f"Expected a value of type 'bool' for argument '{name}' but instead found type '{type(value).__name__}'."
    )


def _float_arg(value, name):
    if type(value) is float:
        return value
    # float() also parses strings; the dispatcher does not.
    if (
        getattr(type(value), "__float__", None) is not None
        or getattr(type(value), "__index__", None) is not None
    ):
        try:
            return float(value)
        except Exception:
            pass
    raise RuntimeError(
        f"Expected a value of type 'float' for argument '{name}' but instead found type '{type(value).__name__}'."
    )


def _int_arg(value, name):
    if type(value) is int and -(2**63) <= value < 2**63:
        return value
    # Match pybind int conversion, including the __int__ fallback.
    if not isinstance(value, float):
        try:
            try:
                result = operator.index(value)
            except Exception:
                if getattr(type(value), "__int__", None) is None:
                    raise TypeError
                result = int(value)
            if -(2**63) <= result < 2**63:
                return result
        except Exception:
            pass
    raise RuntimeError(
        f"Expected a value of type 'int' for argument '{name}' but instead found type '{type(value).__name__}'."
    )


def _check_tensor_types(names, arguments):
    # Rejection-only type check; None is left to the caller.
    import torch

    for name in names.split():
        value = arguments[name.rstrip("_")]
        if value is not None and not isinstance(value, torch.Tensor):
            expected = (
                "Optional[Tensor]"
                if name.endswith("_") or name == "rng_state"
                else "Tensor"
            )
            raise RuntimeError(
                f"Expected a value of type '{expected}' for argument '{name}' but instead found type '{type(value).__name__}'."
            ) from None


def _check_attention_metadata(
    q, k, v, *, out=None, dout=None, lse=None, cuq=None, cuk=None, cache=False
):
    # Native v2 order: dtypes, then devices, then strides/shapes.
    import torch

    if isinstance(q, torch.Tensor) and q.is_cuda:
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlashAttention only support fp16 and bf16 data type"
            ) from None
        for name, tensor in (("key", k), ("value", v), ("out", out), ("dout", dout)):
            if isinstance(tensor, torch.Tensor) and tensor.dtype != q.dtype:
                raise RuntimeError(
                    f"query and {name} must have the same dtype"
                ) from None
        for name, tensor in (("cu_seqlens_q", cuq), ("cu_seqlens_k", cuk)):
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.int32:
                raise RuntimeError(f"{name} must have dtype int32") from None
        for name, tensor in (
            ("kcache" if cache else "k", k),
            ("vcache" if cache else "v", v),
            ("out", out),
            ("dout", dout),
            ("softmax_lse", lse),
            ("cu_seqlens_q", cuq),
            ("cu_seqlens_k", cuk),
        ):
            if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                raise RuntimeError(f"{name} must be on CUDA") from None


def _contiguous_layout(actual, source):
    # Singleton/empty-dim strides do not change any address.
    return (
        actual
        if actual.stride() == source.stride()
        else actual.as_strided(source.shape, source.stride())
    )


def _empty_like(q):
    # Meta empty_like normalizes singleton strides; match CUDA.
    import torch

    if type(
        q
    ) is not torch.Tensor and torch.ops.aten.is_non_overlapping_and_dense.default(q):
        return torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    return torch.empty_like(q)


def _empty_dense_output(q, kv_heads, swap_groups):
    import torch

    if not swap_groups:
        return _empty_like(q)
    if q.is_contiguous():
        return torch.empty_like(q, memory_format=torch.contiguous_format)
    b, _, h, d = q.shape
    grouped = q.reshape(b, kv_heads, h // kv_heads, d).transpose(1, 2)
    return torch.empty_like(grouped).transpose(1, 2).reshape(b, h, d).unsqueeze(1)


def _cpu_dispatch(op, tensors, arguments):
    # Any CUDA argument routes to native validation.
    import torch

    if all(
        t is None or isinstance(t, torch.Tensor) and t.device.type == "cpu"
        for t in tensors
    ):
        # Convert arguments in schema order, like Torch dispatch.
        for name, convert in (
            ("max_seqlen_q", _int_arg),
            ("max_seqlen_k", _int_arg),
            ("p_dropout", _float_arg),
            ("softmax_scale", _float_arg),
            ("zero_tensors", _bool_arg),
            ("is_causal", _bool_arg),
            ("window_size_left", _int_arg),
            ("window_size_right", _int_arg),
            ("softcap", _float_arg),
            ("return_softmax", _bool_arg),
            ("deterministic", _bool_arg),
            ("is_rotary_interleaved", _bool_arg),
            ("num_splits", _int_arg),
        ):
            if name in arguments and not (
                name == "softmax_scale" and arguments[name] is None
            ):
                convert(arguments[name], name)
        if arguments.get("gen") is not None and not isinstance(
            arguments["gen"], torch.Generator
        ):
            raise RuntimeError("gen must be a torch.Generator")
        raise NotImplementedError(f"flash_attn2_rs::{op} has no CPU implementation")


def _layout_floor_sum(n, m, a, b):
    total = 0
    while True:
        total += (n - 1) * n * (a // m) // 2 + n * (b // m)
        a %= m
        b %= m
        y = a * n + b
        if y < m:
            return total
        n, b, m, a = y // m, y % m, a, m


def _layout_rectangle_count(a, n, b, m, t):
    # Count [0,n) x [0,m) pairs with a*x+b*y <= t in logarithmic time.
    if t < 0:
        return 0
    full = max(0, min(n, (t - b * (m - 1)) // a + 1))
    end = min(n, t // a + 1)
    partial = end - full
    if not partial:
        return full * m
    return full * m + partial + _layout_floor_sum(partial, b, a, t - a * (end - 1))


def _nonoverlapping_two_axes(a, n, b, m, span):
    if min(a, b) < span or n * m > 2**63 - 1:
        return False
    # Only opposite-sign differences can cancel.
    shift = b * (m - 1) - a
    return _layout_rectangle_count(
        a, n - 1, b, m - 1, shift + span - 1
    ) == _layout_rectangle_count(a, n - 1, b, m - 1, shift - span)


def _nonoverlapping_three_axes(dims):
    # Mirror the native stride-lattice proof for supplied destinations.
    span, axes = (
        (dims[0][1], dims[1:]) if len(dims) == 4 and dims[0][0] == 1 else (1, dims)
    )
    if len(axes) != 3 or any(s <= 0 or s > 2**32 - 1 or n > 2**31 - 1 for s, n in axes):
        return False
    common = math.gcd(*(stride for stride, _ in axes))

    def candidates(i):
        c, count = axes[i]
        others = [axis for j, axis in enumerate(axes) if j != i]
        reach = sum(s * (n - 1) for s, n in others) + span - 1
        bound = min(count - 1, reach // c)
        step = math.gcd(*(s for s, n in others)) // common if common >= span else 1
        return bound // step

    index = min(range(3), key=candidates)
    axes = [axes[index], *(axis for i, axis in enumerate(axes) if i != index)]
    (c, count), (a, nx), (b, ny) = axes
    count = min(count, (a * (nx - 1) + b * (ny - 1) + span - 1) // c + 1)
    g, r, u, ur, v, vr = a, b, 1, 0, 0, 1
    while r:
        quotient = g // r
        g, r = r, g - quotient * r
        u, ur = ur, u - quotient * ur
        v, vr = vr, v - quotient * vr
    if common < span:
        if count > 1024:
            return False
        # A small GCD does not prove overlap: count bounded row differences.
        if not _nonoverlapping_two_axes(a, nx, b, ny, span):
            return False
        shift = a * (nx - 1) + b * (ny - 1)
        for z in range(1, count):
            target = shift - c * z
            if _layout_rectangle_count(
                a, 2 * nx - 1, b, 2 * ny - 1, target + span - 1
            ) != _layout_rectangle_count(a, 2 * nx - 1, b, 2 * ny - 1, target - span):
                return False
        return True
    step = g // common
    if (count - 1) // step > 1023:
        return False
    sx, sy, mx, my = b // g, a // g, nx - 1, ny - 1
    if sx <= mx and sy <= my:
        return False
    for z in range(step, count, step):
        target = -c * z
        x, y = u * (target // g), v * (target // g)
        lo = max(-((mx + x) // sx), -((my - y) // sy))
        hi = min((mx - x) // sx, (y + my) // sy)
        if lo <= hi:
            return False
    return True


def _nonoverlapping_small_layout(dims):
    # Exact fallback: compare at most 4096 row intervals.
    span, i = 1, 0
    while i < len(dims) and dims[i][0] == span:
        span *= dims[i][1]
        i += 1
    axes = dims[i:]
    count = 1
    for stride, size in axes:
        count *= size
        if stride <= 0 or count > 4096:
            return False
    offsets = [0]
    for stride, size in axes:
        offsets = [offset + j * stride for j in range(size) for offset in offsets]
    offsets.sort()
    return all(right - left >= span for left, right in zip(offsets, offsets[1:]))


def _nonoverlapping_bounded_differences(dims):
    span, i = (1, 0)
    while i < len(dims) and dims[i][0] == span:
        span *= dims[i][1]
        i += 1
    axes = sorted(dims[i:], key=lambda axis: axis[1])
    if not axes:
        return True
    if any((stride <= 0 for stride, size in axes)):
        return False
    stride, size = axes.pop()
    bound = size - 1
    if stride < span:
        return False
    count = 1
    for step, n in axes:
        count *= 2 * n - 1
        if count > 4096:
            return False
    offsets = [0]
    for step, n in axes:
        offsets = [offset + j * step for offset in offsets for j in range(1 - n, n)]
    offsets.remove(0)
    for offset in offsets:
        z = max(-bound, min(bound, -offset // stride))
        if (
            min(abs(offset + z * stride), abs(offset + min(z + 1, bound) * stride))
            < span
        ):
            return False
    return True


def _nonoverlapping_layout(t):
    dims = sorted((s, n) for s, n in zip(t.stride(), t.shape) if n > 1)
    span = 1
    for stride, size in dims:
        if stride < span:
            break
        span += (size - 1) * stride
    else:
        return True
    if t.numel() == 0:
        return True
    # Match the native two-axis lattice proof for interleaved writable views.
    span, i = 1, 0
    while i < len(dims):
        stride, size = dims[i]
        if stride < span:
            return (
                _nonoverlapping_three_axes(dims)
                or _nonoverlapping_small_layout(dims)
                or _nonoverlapping_bounded_differences(dims)
            )
        next_span = span + (size - 1) * stride
        if i + 1 < len(dims):
            other, count = dims[i + 1]
            if other < next_span:
                divisor = math.gcd(stride, other)
                if (
                    divisor < span
                    and not _nonoverlapping_two_axes(stride, size, other, count, span)
                ) or (
                    divisor >= span
                    and size > other // divisor
                    and count > stride // divisor
                ):
                    return (
                        _nonoverlapping_three_axes(dims)
                        or _nonoverlapping_small_layout(dims)
                        or _nonoverlapping_bounded_differences(dims)
                    )
                next_span += (count - 1) * other
                i += 1
        span = next_span
        i += 1
    return True


def _validate(
    q,
    k,
    v,
    out,
    alibi,
    p,
    softcap,
    return_softmax,
    gen,
    *,
    paged=False,
    packed=False,
    require_cuda_gen=True,
):
    import torch

    try:
        for name, t in (("q", q), ("k", k), ("v", v)):
            rank = 3 if packed and (name == "q" or not paged) else 4
            if not isinstance(t, torch.Tensor) or not t.is_cuda or t.ndim != rank:
                raise RuntimeError(
                    f"{name} must be a CUDA tensor of shape [B, S, H, D]"
                )
            if t.dtype not in (torch.float16, torch.bfloat16) or t.dtype != q.dtype:
                message = (
                    "FlashAttention only support fp16 and bf16 data type"
                    if name == "q"
                    else f"query and {'key' if name == 'k' else 'value'} must have the same dtype"
                )
                raise RuntimeError(message)
            if t.device != q.device:
                raise RuntimeError("q, k, v must share device")
            if t.stride(-1) != 1:
                raise RuntimeError("Input tensor must have contiguous last dimension")
    except RuntimeError:
        _check_attention_metadata(q, k, v)
        raise
    if packed:
        b = 1
        h, d = q.shape[-2:]
    else:
        b, _, h, d = q.shape
    if not 0 < b <= 65535 or not 0 < h <= 65535 or d < 8 or d > 256 or d % 8:
        raise RuntimeError(
            "batch and heads must each be in 1..65535 and head size a multiple of 8 in 8..256"
        )
    if (
        (not packed and not paged and k.shape[0] != b)
        or k.shape[-1] != d
        or k.shape != v.shape
    ):
        raise RuntimeError("k and v must have matching [B, Sk, Hk, D] shapes")
    if k.shape[-2] == 0 or h % k.shape[-2]:
        raise RuntimeError(
            "the number of KV heads must divide the number of query heads"
        )
    if not math.isfinite(p) or not 0 <= p < 1:
        raise RuntimeError("p_dropout must be finite and in [0, 1) as float32")
    if p > 0 and softcap > 0:
        raise RuntimeError("Softcapping does not support dropout for now")
    if return_softmax and p == 0:
        raise RuntimeError("return_softmax is only supported when p_dropout > 0.0")
    if out is not None:
        if (
            not isinstance(out, torch.Tensor)
            or out.device != q.device
            or out.dtype != q.dtype
            or out.shape != q.shape
        ):
            raise RuntimeError("out must match q's shape, dtype and device")
        if out.stride(-1) != 1:
            raise RuntimeError("out must have unit last stride")
        if not _nonoverlapping_layout(out):
            raise RuntimeError(
                "out must have nonoverlapping strides; unsupported writable layout: elements overlap or the bounded safety checks cannot prove they are disjoint. Use a separately allocated contiguous tensor."
            )
    if alibi is not None:
        if (
            not isinstance(alibi, torch.Tensor)
            or alibi.device != q.device
            or alibi.dtype != torch.float32
        ):
            raise RuntimeError("alibi_slopes must be float32 on q's device")
        if tuple(alibi.shape) not in ((h,), (b, h)) or alibi.stride(-1) != 1:
            raise RuntimeError(
                "alibi_slopes must have shape [H] or [B, H] and unit last stride"
            )
    if gen is not None:
        if require_cuda_gen:
            if gen.device.type != "cuda":
                raise RuntimeError("gen must be a CUDA torch.Generator")
            if gen.device.index not in (None, q.device.index):
                raise RuntimeError("gen must be on q's device")
    if _cuda_capability(q.device.index)[0] < 8:
        raise RuntimeError("FlashAttention requires an Ampere or newer GPU")


def _prepare_cache_metadata(q, kcache, block_table, leftpad_k, cache_batch_idx, seqlens_k):
    import torch

    b, sq, h, d = q.shape
    capacity = kcache.shape[1]
    if block_table is not None:
        if (
            not isinstance(block_table, torch.Tensor)
            or block_table.ndim != 2
            or block_table.shape[0] != b
            or block_table.dtype != torch.int32
            or block_table.device != q.device
            or block_table.stride(-1) != 1
        ):
            raise RuntimeError(
                "block_table must be int32 [batch, max_blocks] with unit last stride on Q's device"
            )
        if (
            capacity <= 0
            or capacity % 256
            or leftpad_k is not None
            or cache_batch_idx is not None
        ):
            raise RuntimeError(
                "paged caches require page size divisible by 256 and no leftpad or cache_batch_idx"
            )
        capacity *= block_table.shape[1]
    elif cache_batch_idx is None and kcache.shape[0] < b:
        raise RuntimeError("cache batch is too small for Q")
    if (
        not 0 <= capacity <= 2**31 - 128
        or not 0 <= sq <= 2**31 - 128
        or b * sq > 2**31 - 1
        or kcache.shape[0] * kcache.shape[1] > 2**31 - 1
    ):
        raise RuntimeError("cache/query dimensions must fit int32")
    for name, tensor in (
        ("seqlens_k", seqlens_k),
        ("cache_batch_idx", cache_batch_idx),
        ("leftpad_k", leftpad_k),
    ):
        if tensor is not None and (
            not isinstance(tensor, torch.Tensor)
            or (
                tensor.shape != (b,)
                and (name != "cache_batch_idx" or tensor.numel() < b)
            )
            or tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous int32 [batch] on Q's device"
            )
    # v2 reads a flat prefix of cache indices, irrespective of shape.
    if cache_batch_idx is not None and cache_batch_idx.shape != (b,):
        cache_batch_idx = cache_batch_idx.view(-1)[:b]
    return capacity, cache_batch_idx


def _validate_cache_append(q, kcache, k, v, seqlens_k, capacity):
    import torch

    b, sq, h, d = q.shape
    if k is None:
        # v2's dispatcher type-checks V even when it is unused.
        if v is not None and not isinstance(v, torch.Tensor):
            raise RuntimeError("v must be a Tensor or None")
        v = None
    elif v is None:
        raise RuntimeError("new V is required when new K is supplied")
    if k is not None:
        if seqlens_k is None:
            raise RuntimeError("cache lengths are required when appending K/V")
        for name, t in (("k", k), ("v", v)):
            if (
                not isinstance(t, torch.Tensor)
                or t.ndim != 4
                or t.shape[0] != b
                or t.shape[2:] != kcache.shape[2:]
                or t.dtype != q.dtype
                or t.device != q.device
                or t.stride(-1) != 1
            ):
                raise RuntimeError(
                    f"{name} must match query batch and cache heads/dtype/device with unit last stride"
                )
        if k.shape != v.shape or k.shape[1] > capacity or sq > capacity:
            raise RuntimeError(
                "new K/V shapes must match and fit the cache capacity"
            )
    return v


def _validate_cache_rotary(q, k, rotary_cos, rotary_sin, capacity):
    import torch

    d = q.shape[-1]
    if rotary_cos is None:
        if rotary_sin is not None and not isinstance(rotary_sin, torch.Tensor):
            raise RuntimeError("rotary_sin must be a Tensor or None")
        rotary_sin = None
    elif rotary_sin is None:
        raise RuntimeError("rotary sin is required when rotary cos is supplied")
    padded_d = (d + 7) // 8 * 8
    if rotary_cos is not None:
        if k is None:
            raise RuntimeError("rotary tables require new K/V")
        for t in (rotary_cos, rotary_sin):
            if (
                not isinstance(t, torch.Tensor)
                or t.ndim != 2
                or t.device != q.device
                or t.dtype != q.dtype
                or not t.is_contiguous()
            ):
                raise RuntimeError(
                    "rotary tables must be contiguous rank-2 tensors in Q's dtype/device"
                )
        if (
            rotary_cos.shape != rotary_sin.shape
            or rotary_cos.shape[0] < capacity
            or rotary_cos.shape[0] > 2**31 - 1
            or rotary_cos.shape[1] * 2 > padded_d
            or rotary_cos.shape[1] % 8
        ):
            raise RuntimeError(
                "rotary tables must cover cache capacity and have a rotary dimension divisible by 16 within padded D"
            )
    return rotary_sin, padded_d


def _validate_cache_output(q, kcache, out, padded_d, left, right, alibi_slopes):
    import torch

    b, sq, h, d = q.shape
    grouped_out = False
    if out is not None:
        expected_shape = q.shape
        hk = kcache.shape[2]
        if (
            sq == 1
            and 0 < hk < h
            and h % hk == 0
            and d == padded_d
            and left < 0
            and right < 0
            and alibi_slopes is None
        ):
            expected_shape = (b, h // hk, hk, d)
        if (
            not isinstance(out, torch.Tensor)
            or out.shape not in (q.shape, expected_shape)
            or out.dtype != q.dtype
            or out.device != q.device
            or out.stride(-1) != 1
        ):
            raise RuntimeError(
                "out must match Q or its grouped layout and have unit last stride"
            )
        grouped_out = out.shape != q.shape
        if not _nonoverlapping_layout(out):
            raise RuntimeError(
                "out must have nonoverlapping strides; unsupported writable layout: elements overlap or the bounded safety checks cannot prove they are disjoint. Use a separately allocated contiguous tensor."
            )
    return grouped_out


def _finish_gradients(actuals, supplied_buffers, sources):
    """Copy only after all native reads; preserve supplied buffers and view layouts."""
    import torch

    gradients = []
    for actual, supplied, source in zip(actuals, supplied_buffers, sources):
        if supplied is None and source.is_contiguous():
            gradients.append(_contiguous_layout(actual, source))
        else:
            target = torch.empty_like(source) if supplied is None else supplied
            if (
                supplied is not None
                and supplied.requires_grad
                and torch.is_grad_enabled()
            ):
                # Write like native code without recording copy-back in autograd.
                with torch.no_grad():
                    target.copy_(actual)
            else:
                target.copy_(actual)
            gradients.append(target)
    return gradients


def fwd(
    q,
    k,
    v,
    out=None,
    alibi_slopes=None,
    p_dropout=0.0,
    softmax_scale=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    return_softmax=False,
    gen=None,
):
    """Return [output, LSE, signed diagnostic probabilities, CUDA RNG state]."""
    import torch

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if q is None or isinstance(q, torch.Tensor) and not q.is_cuda:
        _cpu_dispatch("fwd", (q, k, v, out, alibi_slopes), locals())
    try:
        # Scalar types are checked by the v2 dispatcher before tensor semantics.
        p = c_float(_float_arg(p_dropout, "p_dropout")).value
        scale = (
            None
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        is_causal = _bool_arg(is_causal, "is_causal")
        left, right = (
            _int_arg(window_size_left, "window_size_left"),
            _int_arg(window_size_right, "window_size_right"),
        )
        cap = c_float(_float_arg(softcap, "softcap")).value
        return_softmax = _bool_arg(return_softmax, "return_softmax")
        if gen is not None and not isinstance(gen, torch.Generator):
            raise RuntimeError("gen must be a torch.Generator")
        # Only backward requires a CUDA generator without dropout.
        _validate(
            q,
            k,
            v,
            out,
            alibi_slopes,
            p,
            cap,
            return_softmax,
            gen,
            require_cuda_gen=p > 0,
        )
    except RuntimeError:
        # Recheck types only on rejection, in dispatcher order.
        _check_tensor_types("q k v out_ alibi_slopes_", locals())
        raise
    # Validate before reservation so rejected inputs consume no RNG.
    if not -(2**31) <= left < 2**31 or not -(2**31) <= right < 2**31:
        raise RuntimeError("window sizes must fit int32")
    swap_groups = False
    if q.shape[1] == 1 and alibi_slopes is None:
        is_causal = False  # v2's single-query convention, including local masks.
        if p == 0:
            b, _, h, d = q.shape
            # Canonical contiguous Q already has v2's group-swapped layout.
            if out is not None or q.stride() != (h * d, h * d, d, 1):
                sk, hk = k.shape[1:3]
                swap_groups = (
                    h > hk and (left < 0 or left >= sk) and (right < 0 or right >= sk)
                )
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.contiguous()
    if out is None:
        out = _empty_dense_output(q, hk, True) if swap_groups else torch.empty_like(q)
    elif swap_groups:
        out = out.reshape(b, hk, h // hk, d).reshape(b, 1, h, d)
    args = (q, k, v, alibi_slopes, scale, bool(is_causal), left, right, cap)
    if p == 0:
        lse = ops.fwd_lse(*args, out)
        probabilities = torch.empty(0, device=q.device, dtype=q.dtype)
        state = torch.empty(2, device=q.device, dtype=torch.int64)
    else:
        state = torch.empty(2, device=q.device, dtype=torch.int64)
        operation = ops.fwd if return_softmax else ops.fwd_lse
        native_args = (*args, p, state, 0, out, return_softmax)
        direct_descriptor = (
            not return_softmax
            and q.shape[-1] == 64
            and 1 <= q.shape[1] <= 128
            and 1 <= k.shape[1] <= 128
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and out.is_contiguous()
        )
        if direct_descriptor:
            def launch():
                seed, offset, intragraph, captured = _rng_descriptor_values(
                    gen, q.device.index, q.shape[0] * q.shape[2] * 32
                )
                return operation(*native_args, seed, offset, intragraph, bool(captured))
        else:
            def reserve(_unused, device, increment):
                return _rng_descriptor(gen, device, increment)

            def launch():
                return operation(*native_args, reserve, 0)

        if q.device.index == torch.cuda.current_device():
            values = launch()
        else:
            with torch.cuda.device(q.device):
                values = launch()
        if return_softmax:
            output, lse, probabilities = values
        else:
            lse = values
            probabilities = torch.empty(0, device=q.device, dtype=q.dtype)
    # Return the caller's output object, as v2 does.
    return [out, lse, probabilities, state]


def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq=None,
    dk=None,
    dv=None,
    alibi_slopes=None,
    p_dropout=0.0,
    softmax_scale=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    deterministic=False,
    gen=None,
    rng_state=None,
):
    """Return [dQ, dK, dV, softmax_d], reusing the forward RNG state."""
    import torch

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if q is None or isinstance(q, torch.Tensor) and not q.is_cuda:
        _cpu_dispatch(
            "bwd",
            (dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, rng_state),
            locals(),
        )
    try:
        p = c_float(_float_arg(p_dropout, "p_dropout")).value
        scale = (
            None
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        is_causal = _bool_arg(is_causal, "is_causal")
        left, right = (
            _int_arg(window_size_left, "window_size_left"),
            _int_arg(window_size_right, "window_size_right"),
        )
        cap = c_float(_float_arg(softcap, "softcap")).value
        deterministic = _bool_arg(deterministic, "deterministic")
        if gen is not None and not isinstance(gen, torch.Generator):
            raise RuntimeError("gen must be a torch.Generator")
    except RuntimeError:
        _check_tensor_types(
            "dout q k v out softmax_lse dq_ dk_ dv_ alibi_slopes_", locals()
        )
        raise
    try:
        _validate(q, k, v, None, alibi_slopes, p, cap, False, gen)
        q_shape, q_dtype, q_device = q.shape, q.dtype, q.device
        for name, tensor in (("dout", dout), ("out", out)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != q_shape
                or tensor.device != q_device
                or tensor.dtype != q_dtype
            ):
                raise RuntimeError(f"{name} must match q's shape, dtype and device")
            if tensor.stride(-1) != 1:
                raise RuntimeError(f"{name} must have unit last stride")
        if (
            not isinstance(softmax_lse, torch.Tensor)
            or softmax_lse.shape != (q_shape[0], q_shape[2], q_shape[1])
            or softmax_lse.dtype != torch.float32
            or softmax_lse.device != q_device
            or not softmax_lse.is_contiguous()
        ):
            raise RuntimeError(
                "softmax_lse must be contiguous float32 [B, H, Sq] on q's device"
            )
        for name, tensor, source in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
            if tensor is None:
                continue
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != source.shape
                or tensor.device != source.device
                or tensor.dtype != source.dtype
            ):
                raise RuntimeError(
                    f"{name} must match its input's shape, dtype and device"
                )
            if tensor.stride(-1) != 1:
                raise RuntimeError(f"{name} must have unit last stride")
            if not _nonoverlapping_layout(tensor):
                raise RuntimeError(
                    f"{name} must have nonoverlapping strides; unsupported writable layout: elements overlap or the bounded safety checks cannot prove they are disjoint. Use a separately allocated contiguous tensor."
                )
            # v2's GQA sum(out=...) rejects grad-enabled dK/dV destinations.
            if (
                name != "dq"
                and tensor.requires_grad
                and q.shape[-2] != k.shape[-2]
                and torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "sum(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad."
                )
        if rng_state is not None:
            if not isinstance(rng_state, torch.Tensor):
                raise RuntimeError("rng_state must be a Tensor or None")
            if p > 0 and (
                rng_state.shape != (2,)
                or rng_state.device != q_device
                or rng_state.dtype != torch.int64
                or not rng_state.is_contiguous()
            ):
                raise RuntimeError(
                    "rng_state must be contiguous int64 [2] on q's device"
                )
        if not -(2**31) <= left < 2**31 or not -(2**31) <= right < 2**31:
            raise RuntimeError("window sizes must fit int32")
    except RuntimeError:
        _check_tensor_types(
            "dout q k v out softmax_lse dq_ dk_ dv_ alibi_slopes_ rng_state", locals()
        )
        _check_attention_metadata(q, k, v, out=out, dout=dout, lse=softmax_lse)
        raise
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.contiguous()
    if p > 0 and rng_state is None:
        rng_state = reserve_rng_state(q, gen)
    if p == 0:
        rng_state = None
    delta = ops.bwd_preprocess(dout, out, p)
    args = (
        q,
        k,
        v,
        dout,
        softmax_lse,
        delta,
        alibi_slopes,
        scale,
        bool(is_causal),
        left,
        right,
        cap,
        p,
        rng_state,
    )
    if deterministic:
        grad_q = ops.bwd_dq(*args)
        values = ops.bwd_dkv(*args)
        grad_k, grad_v = _array_getitem(values, 0), _array_getitem(values, 1)
    else:
        values = ops.bwd_all(*args)
        grad_q = _array_getitem(values, 0)
        grad_k = _array_getitem(values, 1)
        grad_v = _array_getitem(values, 2)
    gradients = _finish_gradients((grad_q, grad_k, grad_v), (dq, dk, dv), (q, k, v))
    return [*gradients, delta]


def varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    out=None,
    seqused_k=None,
    leftpad_k=None,
    block_table=None,
    alibi_slopes=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    p_dropout=0.0,
    softmax_scale=None,
    zero_tensors=False,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    return_softmax=False,
    gen=None,
):
    """Packed-sequence forward, returning [out, LSE, probabilities, RNG state]."""
    import torch

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if q is None or isinstance(q, torch.Tensor) and not q.is_cuda:
        _cpu_dispatch(
            "varlen_fwd",
            (
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                out,
                seqused_k,
                leftpad_k,
                block_table,
                alibi_slopes,
            ),
            locals(),
        )
    try:
        max_q, max_k = (
            _int_arg(max_seqlen_q, "max_seqlen_q"),
            _int_arg(max_seqlen_k, "max_seqlen_k"),
        )
        p = c_float(_float_arg(p_dropout, "p_dropout")).value
        scale = (
            None
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        zero_tensors = _bool_arg(zero_tensors, "zero_tensors")
        is_causal = _bool_arg(is_causal, "is_causal")
        left, right = (
            _int_arg(window_size_left, "window_size_left"),
            _int_arg(window_size_right, "window_size_right"),
        )
        cap = c_float(_float_arg(softcap, "softcap")).value
        return_softmax = _bool_arg(return_softmax, "return_softmax")
        if gen is not None and not isinstance(gen, torch.Generator):
            raise RuntimeError("gen must be a torch.Generator")
    except RuntimeError:
        _check_tensor_types(
            "q k v out_ cu_seqlens_q cu_seqlens_k seqused_k_ leftpad_k_ block_table_ alibi_slopes_",
            locals(),
        )
        raise
    try:
        paged = block_table is not None
        _validate(
            q,
            k,
            v,
            out,
            None,
            p,
            cap,
            return_softmax,
            gen,
            paged=paged,
            require_cuda_gen=p > 0,
            packed=True,
        )
        if (
            not isinstance(cu_seqlens_q, torch.Tensor)
            or cu_seqlens_q.ndim != 1
            or cu_seqlens_q.numel() < 2
        ):
            raise RuntimeError("cu_seqlens_q must have shape [batch + 1]")
        batch, h = cu_seqlens_q.numel() - 1, q.shape[1]
        if paged:
            if (
                not isinstance(block_table, torch.Tensor)
                or block_table.ndim != 2
                or block_table.shape[0] != batch
                or block_table.device != q.device
                or block_table.dtype != torch.int32
                or block_table.stride(-1) != 1
            ):
                raise RuntimeError(
                    "block_table must be CUDA int32 [batch, max_blocks] with unit last stride"
                )
            if k.shape[1] <= 0 or k.shape[1] % 256 or leftpad_k is not None:
                raise RuntimeError(
                    "paged K/V requires page size divisible by 256 and no leftpad_k"
                )
        if batch > 65535:
            raise RuntimeError("batch must not exceed 65535")
        for name, tensor, shape in (
            ("cu_seqlens_q", cu_seqlens_q, (batch + 1,)),
            ("cu_seqlens_k", cu_seqlens_k, (batch + 1,)),
            ("seqused_k", seqused_k, (batch,)),
            ("leftpad_k", leftpad_k, (batch,)),
        ):
            if tensor is None and name in ("seqused_k", "leftpad_k"):
                continue
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != torch.int32
                or tensor.device != q.device
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"{name} must be contiguous int32 {shape} on q's device"
                )
        if alibi_slopes is not None:
            if (
                not isinstance(alibi_slopes, torch.Tensor)
                or alibi_slopes.device != q.device
                or alibi_slopes.dtype != torch.float32
                or alibi_slopes.shape not in ((h,), (batch, h))
                or alibi_slopes.stride(-1) != 1
            ):
                raise RuntimeError(
                    "alibi_slopes must be float32 [H] or [B,H] on q's device with unit last stride"
                )
            alibi_slopes = alibi_slopes.contiguous()
        if not 0 <= max_q <= 2**31 - 128 or not 0 <= max_k <= 2**31 - 128:
            raise RuntimeError(
                "maximum sequence lengths must fit nonnegative int32 with rounding"
            )
        if not -(2**31) <= left < 2**31 or not -(2**31) <= right < 2**31:
            raise RuntimeError("window sizes must fit int32")
    except RuntimeError:
        _check_tensor_types(
            "q k v out_ cu_seqlens_q cu_seqlens_k seqused_k_ leftpad_k_ block_table_ alibi_slopes_",
            locals(),
        )
        _check_attention_metadata(q, k, v, cuq=cu_seqlens_q, cuk=cu_seqlens_k)
        if (
            not isinstance(q, torch.Tensor)
            or q.ndim != 3
            or any(
                not isinstance(x, torch.Tensor) or x.ndim != (4 if paged else 3)
                for x in (k, v)
            )
        ):
            raise RuntimeError(
                "q must be rank 3; k/v must be rank 4 with block_table, otherwise rank 3"
            )
        raise
    fused_rng = p > 0 and not paged and 32 < q.shape[-1] <= 64
    if fused_rng:
        device = q.device.index
        if device == torch.cuda.current_device():
            seed, offset, intragraph, captured = _rng_descriptor_values(
                gen, device, batch * h * 32
            )
        else:
            with torch.cuda.device(device):
                seed, offset, intragraph, captured = _rng_descriptor_values(
                    gen, device, batch * h * 32
                )
        state = torch.empty(2, device=q.device, dtype=torch.int64)
    else:
        state = (
            _reserve_rng_state(q, batch, h, gen)
            if p > 0
            else torch.empty(2, device=q.device, dtype=torch.int64)
        )
    operation = ops.varlen_fwd_descriptor if fused_rng else ops.varlen_fwd_explicit
    values = operation(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        alibi_slopes,
        scale,
        bool(is_causal),
        left,
        right,
        cap,
        p,
        state if p > 0 else None,
        max_q,
        max_k,
        bool(return_softmax),
        bool(zero_tensors),
        *([None, seed, offset, intragraph, bool(captured)]
          if fused_rng else [block_table] if paged else []),
    )
    output = _array_getitem(values, 0)
    lse = _array_getitem(values, 1)
    probabilities = _array_getitem(values, 2)
    if output.ndim == 4:  # Preserve compatibility with rank-four native results.
        output = output.squeeze(0)
    swap_groups = (
        max_q == 1
        and q.shape[0] == batch
        and h > k.shape[-2]
        and p == 0
        and alibi_slopes is None
        and left < 0
        and right < 0
    )
    if out is not None or (not q.is_contiguous() and not swap_groups):
        target = torch.empty_like(q) if out is None else out
        target.copy_(output)
        output = target
    elif not swap_groups:
        output = _contiguous_layout(output, q)
    return [output, lse, probabilities, state]


def varlen_bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    dq=None,
    dk=None,
    dv=None,
    alibi_slopes=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    p_dropout=0.0,
    softmax_scale=None,
    zero_tensors=False,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    deterministic=False,
    gen=None,
    rng_state=None,
):
    """Packed backward returning [dQ, dK, dV, softmax_d], with saved dropout."""
    import torch

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if q is None or isinstance(q, torch.Tensor) and not q.is_cuda:
        _cpu_dispatch(
            "varlen_bwd",
            (
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
            ),
            locals(),
        )
    try:
        max_q, max_k = (
            _int_arg(max_seqlen_q, "max_seqlen_q"),
            _int_arg(max_seqlen_k, "max_seqlen_k"),
        )
        p = c_float(_float_arg(p_dropout, "p_dropout")).value
        scale = (
            None
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        zero_tensors = _bool_arg(zero_tensors, "zero_tensors")
        is_causal = _bool_arg(is_causal, "is_causal")
        left, right = (
            _int_arg(window_size_left, "window_size_left"),
            _int_arg(window_size_right, "window_size_right"),
        )
        cap = c_float(_float_arg(softcap, "softcap")).value
        deterministic = _bool_arg(deterministic, "deterministic")
        if gen is not None and not isinstance(gen, torch.Generator):
            raise RuntimeError("gen must be a torch.Generator")
    except RuntimeError:
        _check_tensor_types(
            "dout q k v out softmax_lse dq_ dk_ dv_ cu_seqlens_q cu_seqlens_k alibi_slopes_",
            locals(),
        )
        raise
    try:
        if any(not isinstance(x, torch.Tensor) or x.ndim != 3 for x in (q, k, v)):
            raise RuntimeError(
                "q, k, v must have shape [total_tokens, heads, head_size]"
            )
        _validate(q, k, v, None, None, p, cap, False, gen, packed=True)
        if (
            not isinstance(cu_seqlens_q, torch.Tensor)
            or cu_seqlens_q.ndim != 1
            or cu_seqlens_q.numel() < 2
        ):
            raise RuntimeError("cu_seqlens_q must have shape [batch + 1]")
        batch, h = cu_seqlens_q.numel() - 1, q.shape[1]
        if batch > 65535:
            raise RuntimeError("batch must not exceed 65535")
        for name, tensor in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (batch + 1,)
                or tensor.dtype != torch.int32
                or tensor.device != q.device
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"{name} must be contiguous int32 [B+1] on q's device"
                )
        for name, tensor in (("dout", dout), ("out", out)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != q.shape
                or tensor.device != q.device
                or tensor.dtype != q.dtype
                or tensor.stride(-1) != 1
            ):
                raise RuntimeError(
                    f"{name} must match q's shape, dtype and device with unit last stride"
                )
        if (
            not isinstance(softmax_lse, torch.Tensor)
            or softmax_lse.shape != (h, q.shape[0])
            or softmax_lse.device != q.device
            or softmax_lse.dtype != torch.float32
            or not softmax_lse.is_contiguous()
        ):
            raise RuntimeError(
                "softmax_lse must be contiguous float32 [H,total_q] on q's device"
            )
        for name, tensor, source in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
            if tensor is None:
                continue
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != source.shape
                or tensor.device != q.device
                or tensor.dtype != q.dtype
                or tensor.stride(-1) != 1
            ):
                raise RuntimeError(
                    f"{name} must match its input's shape, dtype and device with unit last stride"
                )
            if not _nonoverlapping_layout(tensor):
                raise RuntimeError(
                    f"{name} must have nonoverlapping strides; unsupported writable layout: elements overlap or the bounded safety checks cannot prove they are disjoint. Use a separately allocated contiguous tensor."
                )
            # v2's GQA sum(out=...) rejects grad-enabled dK/dV destinations.
            if (
                name != "dq"
                and tensor.requires_grad
                and q.shape[-2] != k.shape[-2]
                and torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "sum(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad."
                )
        if alibi_slopes is not None:
            if (
                not isinstance(alibi_slopes, torch.Tensor)
                or alibi_slopes.shape not in ((h,), (batch, h))
                or alibi_slopes.device != q.device
                or alibi_slopes.dtype != torch.float32
                or alibi_slopes.stride(-1) != 1
            ):
                raise RuntimeError(
                    "alibi_slopes must be float32 [H] or [B,H] on q's device with unit last stride"
                )
            alibi_slopes = alibi_slopes.contiguous()
        if rng_state is not None:
            if not isinstance(rng_state, torch.Tensor):
                raise RuntimeError("rng_state must be a Tensor or None")
            if p > 0 and (
                rng_state.shape != (2,)
                or rng_state.device != q.device
                or rng_state.dtype != torch.int64
                or not rng_state.is_contiguous()
            ):
                raise RuntimeError(
                    "rng_state must be contiguous int64 [2] on q's device"
                )
        if not 0 <= max_q <= 2**31 - 128 or not 0 <= max_k <= 2**31 - 128:
            raise RuntimeError(
                "maximum sequence lengths must fit nonnegative int32 with rounding"
            )
        if not -(2**31) <= left < 2**31 or not -(2**31) <= right < 2**31:
            raise RuntimeError("window sizes must fit int32")
    except RuntimeError:
        _check_tensor_types(
            "dout q k v out softmax_lse dq_ dk_ dv_ cu_seqlens_q cu_seqlens_k alibi_slopes_ rng_state",
            locals(),
        )
        _check_attention_metadata(
            q,
            k,
            v,
            out=out,
            dout=dout,
            lse=softmax_lse,
            cuq=cu_seqlens_q,
            cuk=cu_seqlens_k,
        )
        raise
    if p > 0 and rng_state is None:
        rng_state = _reserve_rng_state(q, batch, h, gen)
    values = ops.varlen_bwd_explicit(
        q,
        k,
        v,
        dout,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        scale,
        bool(is_causal),
        left,
        right,
        cap,
        p,
        rng_state if p > 0 else None,
        max_q,
        max_k,
        bool(zero_tensors),
    )
    gradients = _finish_gradients(
        values[:3], (dq, dk, dv), (q, k, v)
    )
    return [*gradients, values[3]]


def fwd_kvcache(
    q,
    kcache,
    vcache,
    k=None,
    v=None,
    seqlens_k=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_batch_idx=None,
    leftpad_k=None,
    block_table=None,
    alibi_slopes=None,
    out=None,
    softmax_scale=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    is_rotary_interleaved=False,
    num_splits=1,
):
    """Native cached inference; append K/V in place and return [out, LSE]."""
    import torch

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if q is None or isinstance(q, torch.Tensor) and not q.is_cuda:
        _cpu_dispatch(
            "fwd_kvcache",
            (
                q,
                kcache,
                vcache,
                k,
                v,
                seqlens_k,
                rotary_cos,
                rotary_sin,
                cache_batch_idx,
                leftpad_k,
                block_table,
                alibi_slopes,
                out,
            ),
            locals(),
        )
    try:
        # Match dispatcher scalar conversion order.
        scale = _float_arg(softmax_scale, "softmax_scale")
        is_causal = _bool_arg(is_causal, "is_causal")
        left, right = (
            _int_arg(window_size_left, "window_size_left"),
            _int_arg(window_size_right, "window_size_right"),
        )
        cap = c_float(_float_arg(softcap, "softcap")).value
        is_rotary_interleaved = _bool_arg(
            is_rotary_interleaved, "is_rotary_interleaved"
        )
        splits = _int_arg(num_splits, "num_splits")
    except RuntimeError:
        _check_tensor_types(
            "q kcache vcache k_ v_ seqlens_k_ rotary_cos_ rotary_sin_ cache_batch_idx_ leftpad_k_ block_table_ alibi_slopes_ out_",
            locals(),
        )
        raise
    try:
        for name, tensor in (("q", q), ("kcache", kcache), ("vcache", vcache)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise RuntimeError(f"{name} must be rank 4")
        b, sq, h, d = q.shape
        if not 0 < d <= 256:
            raise RuntimeError("head size must be in 1..256")
        if kcache.shape != vcache.shape or kcache.shape[-1] != d:
            raise RuntimeError("K/V cache shapes must match Q's head size")
        if any(t.stride(-1) != 1 for t in (q, kcache, vcache)):
            raise RuntimeError("Input tensor must have contiguous last dimension")
        if not -(2**31) <= splits <= 128:
            raise RuntimeError("num_splits must fit int32 and be at most 128")
        if not -(2**31) <= left < 2**31 or not -(2**31) <= right < 2**31:
            raise RuntimeError("window sizes must fit int32")
        capacity, cache_batch_idx = _prepare_cache_metadata(q, kcache, block_table, leftpad_k, cache_batch_idx, seqlens_k)
        v = _validate_cache_append(q, kcache, k, v, seqlens_k, capacity)
        rotary_sin, padded_d = _validate_cache_rotary(q, k, rotary_cos, rotary_sin, capacity)
        grouped_out = _validate_cache_output(q, kcache, out, padded_d, left, right, alibi_slopes)
        new_keys = 0 if k is None else k.shape[1]

        # Keep padded coordinates: rotation can make them nonzero.
        def pad(t):
            return torch.nn.functional.pad(t, (0, padded_d - d)) if d != padded_d else t

        qp, kp, vp = pad(q), pad(kcache), pad(vcache)
        _validate(qp, kp, vp, None, alibi_slopes, 0.0, cap, False, None, paged=True)
    except RuntimeError:
        _check_tensor_types(
            "q kcache vcache k_ v_ seqlens_k_ rotary_cos_ rotary_sin_ cache_batch_idx_ leftpad_k_ block_table_ alibi_slopes_ out_",
            locals(),
        )
        _check_attention_metadata(q, kcache, vcache, cache=True)
        raise
    if new_keys > 0 and d != padded_d:
        # Padding hides overlap, so check the originals before any write.
        ops.cache_validate_layouts(
            kcache,
            vcache,
            rotary_cos,
            rotary_sin,
            seqlens_k,
            cache_batch_idx,
            block_table,
            leftpad_k,
            alibi_slopes,
        )
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.contiguous()
    # v2 validates rotary tables above but skips rotation for an empty append.
    if new_keys > 0:
        qp = ops.cache_update_explicit(
            qp,
            kp,
            vp,
            pad(k),
            pad(v),
            seqlens_k,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            block_table,
            bool(is_rotary_interleaved),
            # v2 treats full-capacity windows as unrestricted.
            bool(is_causal or 0 <= left < capacity or 0 <= right < capacity),
            leftpad_k,
            alibi_slopes,
        )
        if d == padded_d:
            # Native writes bypass Torch in-place ops; padded caches use copy_.
            torch.autograd.graph.increment_version((kcache, vcache))
    # Reuse K/V across GQA heads by treating groups as query rows.
    grouped = (
        sq == 1
        and d == 128
        and h > kp.shape[2]
        and capacity >= 1024
        and left < 0
        and right < 0
        and alibi_slopes is None
        and splits != 1
    )
    if grouped and splits < 1:
        # Keep one split when it already fills the GPU.
        group_ctas = b * kp.shape[2] * ((h // kp.shape[2] + 63) // 64)
        grouped = (
            group_ctas
            <= torch.cuda.get_device_properties(q.device).multi_processor_count
        )
    attention_q, attention_sq = qp, sq
    if grouped:
        attention_sq = h // kp.shape[2]
        attention_q = qp.reshape(b, kp.shape[2], attention_sq, padded_d).transpose(1, 2)
    output, lse, _ = ops.varlen_fwd_explicit(
        attention_q,
        kp,
        vp,
        None,
        seqlens_k,
        None,
        leftpad_k,
        alibi_slopes,
        scale,
        False if grouped else bool(is_causal),
        left,
        right,
        cap,
        0.0,
        None,
        attention_sq,
        capacity,
        False,
        False,
        block_table,
        cache_batch_idx,
        new_keys,
        splits,
        grouped,
    )
    if grouped:
        output = output.reshape(b, 1, h, padded_d)
        lse = lse.reshape(b, h, 1)
    if new_keys > 0 and d != padded_d:
        kcache.copy_(kp[..., :d])
        vcache.copy_(vp[..., :d])
    output = output[..., :d]
    if out is not None and grouped_out:
        hk = kcache.shape[2]
        # v2 accepts output in its internal [B, groups, Hk, D] layout.
        out.copy_(output.reshape(b, hk, h // hk, d).transpose(1, 2))
        output = out.transpose(1, 2).reshape(b, 1, h, d)
    elif out is not None:
        out.copy_(output)
        # v2 returns the cropped padded result even with an output buffer.
        if d == padded_d:
            output = out
    elif d == padded_d and (not q.is_contiguous() or output.stride() != q.stride()):
        # v2 allocates empty_like(Q), preserving dense transposed layouts.
        swap_groups = (
            sq == 1
            and h > kp.shape[2]
            and left < 0
            and right < 0
            and alibi_slopes is None
        )
        if swap_groups:
            meta_q = torch.empty_strided(
                q.shape, q.stride(), device="meta", dtype=q.dtype
            )
            expected_stride = _empty_dense_output(meta_q, kp.shape[2], True).stride()
        else:
            expected_stride = torch.empty_like(q, device="meta").stride()
        if output.stride() != expected_stride:
            if all(
                n <= 1 or a == e
                for n, a, e in zip(q.shape, output.stride(), expected_stride)
            ):
                output = output.as_strided(q.shape, expected_stride)
            else:
                target = _empty_dense_output(q, kp.shape[2], swap_groups)
                target.copy_(output)
                output = target
    return [output, lse]


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Dense attention with Q/K/V autograd, including MQA/GQA and head padding."""
    from ._torch_interface import _DenseAttention

    return _DenseAttention.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        0,
    )


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Self-attention with autograd for packed [B, S, 3, H, D] input."""
    from ._torch_interface import _DenseAttention

    return _DenseAttention.apply(
        qkv,
        None,
        None,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        3,
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Attention with autograd for Q [B, Sq, H, D] and KV [B, Sk, 2, Hk, D]."""
    from ._torch_interface import _DenseAttention

    return _DenseAttention.apply(
        q,
        kv,
        None,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        2,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """Variable-length attention with autograd and GPU cumulative lengths."""
    from ._torch_interface import _VarlenAttention

    cuk = None if cu_seqlens_k is cu_seqlens_q else cu_seqlens_k
    return _VarlenAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cuk,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        0,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Autograd for variable-length packed [total, 3, H, D] QKV."""
    from ._torch_interface import _VarlenAttention

    return _VarlenAttention.apply(
        qkv,
        None,
        None,
        cu_seqlens,
        None,
        max_seqlen,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        None,
        3,
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Autograd for variable-length Q and packed [total_k, 2, Hk, D] KV."""
    from ._torch_interface import _VarlenAttention

    cuk = None if cu_seqlens_k is cu_seqlens_q else cu_seqlens_k
    return _VarlenAttention.apply(
        q,
        kv,
        None,
        cu_seqlens_q,
        cuk,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        None,
        2,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    "Run cached inference and optionally append K/V; no backward is provided."
    from ._torch_interface import flash_attn_with_kvcache as implementation

    return implementation(
        q,
        k_cache,
        v_cache,
        k,
        v,
        rotary_cos,
        rotary_sin,
        cache_seqlens,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        softmax_scale,
        causal,
        window_size,
        softcap,
        rotary_interleaved,
        alibi_slopes,
        num_splits,
        return_softmax_lse,
    )

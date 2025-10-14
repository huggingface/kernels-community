from ._ops import ops
from typing import Tuple, Any

# Use a broad Tensor alias to avoid importing torch at import time.
from torch import Tensor

def forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, y: Tensor) -> None:
    """RWKV WKV forward pass (float32).

    Runs the CUDA kernel and writes the result into ``y`` in-place.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u: Input tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        k: Key tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        v: Value tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        y: Output tensor, shape ``[B, T, C]``, dtype ``torch.float32`` (written in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B``, ``T`` and ``C``.
    """
    _validate_device_match((w, u, k, v, y))
    ops.forward(w, u, k, v, y)


def forward_bf16(w: Tensor, u: Tensor, k: Tensor, v: Tensor, y: Tensor) -> None:
    """RWKV WKV forward pass (bfloat16 inputs/outputs, float32 ``w``).

    Runs the CUDA kernel and writes the result into ``y`` in-place.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u: Input tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        k: Key tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        v: Value tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        y: Output tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16`` (written in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B``, ``T`` and ``C``.
    """
    _validate_device_match((w, u, k, v, y))
    ops.forward_bf16(w, u, k, v, y)


def forward_with_state(w: Tensor, u: Tensor, k: Tensor, v: Tensor, y: Tensor, s: Tensor) -> None:
    """RWKV WKV forward pass with persistent state (float32).

    Runs the CUDA kernel using and updating state ``s`` and writes the result into ``y``.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u: Input tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        k: Key tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        v: Value tensor, shape ``[B, T, C]``, dtype ``torch.float32``.
        y: Output tensor, shape ``[B, T, C]``, dtype ``torch.float32`` (written in-place).
        s: Stateful tensor, shape ``[B, C]``, dtype ``torch.float32`` (updated in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B`` and ``C``; ``y`` shares ``[B, T, C]`` with inputs.
    """
    _validate_device_match((w, u, k, v, y, s))
    ops.forward_with_state(w, u, k, v, y, s)


def forward_with_state_bf16(w: Tensor, u: Tensor, k: Tensor, v: Tensor, y: Tensor, s: Tensor) -> None:
    """RWKV WKV forward pass with persistent state (bfloat16 inputs/outputs, float32 ``w`` and ``s``).

    Runs the CUDA kernel using and updating state ``s`` and writes the result into ``y``.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u: Input tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        k: Key tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        v: Value tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        y: Output tensor, shape ``[B, T, C]``, dtype ``torch.bfloat16`` (written in-place).
        s: Stateful tensor, shape ``[B, C]``, dtype ``torch.float32`` (updated in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B`` and ``C``; ``y`` shares ``[B, T, C]`` with inputs.
    """
    _validate_device_match((w, u, k, v, y, s))
    ops.forward_with_state_bf16(w, u, k, v, y, s)


def backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    gy: Tensor,
    gw: Tensor,
    gu: Tensor,
    gk: Tensor,
    gv: Tensor,
) -> None:
    """RWKV WKV backward pass (float32).

    Writes gradients into the provided tensors in-place.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u, k, v, y: Forward-pass tensors, shape ``[B, T, C]``, dtype ``torch.float32``.
        gy: Gradient of ``y``, shape ``[B, T, C]``, dtype ``torch.float32``.
        gw: Gradient for ``w``, shape ``[C]``, dtype ``torch.float32`` (written in-place).
        gu, gk, gv: Gradients for ``u``, ``k``, ``v`` respectively, shape ``[B, T, C]``, dtype ``torch.float32`` (written in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B``, ``T`` and ``C``.
    """
    _validate_device_match((w, u, k, v, y, gy, gw, gu, gk, gv))
    ops.backward(w, u, k, v, y, gy, gw, gu, gk, gv)


def backward_bf16(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    y: Tensor,
    gy: Tensor,
    gw: Tensor,
    gu: Tensor,
    gk: Tensor,
    gv: Tensor,
) -> None:
    """RWKV WKV backward pass (bfloat16 inputs/outputs/gradients, float32 ``w``).

    Writes gradients into the provided tensors in-place.

    Args:
        w: Decay weights, shape ``[C]``, dtype ``torch.float32``.
        u, k, v, y: Forward-pass tensors, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        gy: Gradient of ``y``, shape ``[B, T, C]``, dtype ``torch.bfloat16``.
        gw: Gradient for ``w``, shape ``[C]``, dtype ``torch.bfloat16`` (written in-place).
        gu, gk, gv: Gradients for ``u``, ``k``, ``v`` respectively, shape ``[B, T, C]``, dtype ``torch.bfloat16`` (written in-place).

    Notes:
        - All tensors must be on the same CUDA device.
        - Shapes must agree on ``B``, ``T`` and ``C``.
    """
    _validate_device_match((w, u, k, v, y, gy, gw, gu, gk, gv))
    ops.backward_bf16(w, u, k, v, y, gy, gw, gu, gk, gv)


def _validate_device_match(tensors: Tuple[Tensor, ...]) -> None:
    """Minimal runtime validation that all tensors live on the same CUDA device."""
    if not tensors:
        return
    device = tensors[0].device
    if not device.type == "cuda":
        raise RuntimeError("RWKV CUDA ops require CUDA tensors")
    for t in tensors[1:]:
        if t.device != device:
            raise RuntimeError("All tensors must be on the same CUDA device")


__all__ = [
    "forward",
    "forward_bf16",
    "forward_with_state",
    "forward_with_state_bf16",
    "backward",
    "backward_bf16",
]
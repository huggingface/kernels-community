"""No-op replacement for ``aiter.jit.utils.torch_guard``.

Upstream's ``torch_compile_guard`` registers each decorated function with
``torch.library`` under the global ``"aiter"`` namespace and routes calls
through ``torch.ops.aiter.<name>``. That's appropriate for the full aiter
install (which ships a C++ extension that also registers ops there) but
would clash with a parallel ``import aiter`` in the same process and is
unnecessary for Triton-only kernels.

This stub returns the decorated function unmodified — the Triton ops still
run, just without the ``torch.ops.aiter.*`` indirection.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union


def torch_compile_guard(
    mutates_args: Union[list, str] = "unknown",
    device: str = "cpu",
    calling_func_: Optional[Callable[..., Any]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
):
    """No-op decorator factory: returns the function unchanged."""

    def decorator(func):
        return func

    return decorator

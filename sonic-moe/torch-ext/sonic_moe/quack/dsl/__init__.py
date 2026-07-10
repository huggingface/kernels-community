# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""CuTe DSL helpers and integration hooks."""

__all__ = ["cute_op"]


def __getattr__(name: str):
    if name == "cute_op":
        from .torch_library_op import cute_op

        return cute_op
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

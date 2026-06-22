# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Iterable


class Priority(IntEnum):
    PORTABLE = 0


def register_kernel(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Compatibility no-op for Tokenspeed's in-tree registration decorator."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


def format_signatures(
    roles: Iterable[str], layout: str, dtypes: Iterable[object]
) -> tuple[tuple[str, str, object], ...]:
    return tuple((role, layout, dtype) for role in roles for dtype in dtypes)

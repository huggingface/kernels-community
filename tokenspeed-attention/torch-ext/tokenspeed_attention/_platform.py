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

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ArchVersion:
    major: int
    minor: int

    def __ge__(self, other: "ArchVersion") -> bool:
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: "ArchVersion") -> bool:
        return (self.major, self.minor) > (other.major, other.minor)

    def __le__(self, other: "ArchVersion") -> bool:
        return (self.major, self.minor) <= (other.major, other.minor)

    def __lt__(self, other: "ArchVersion") -> bool:
        return (self.major, self.minor) < (other.major, other.minor)


@dataclass(frozen=True)
class CapabilityRequirement:
    vendors: frozenset[str] | None = None


@dataclass(frozen=True)
class PlatformInfo:
    vendor: str
    arch_version: ArchVersion

    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"

    @property
    def is_ampere_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(8, 0)

    @property
    def is_hopper_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(9, 0)


@lru_cache(maxsize=1)
def current_platform() -> PlatformInfo:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "tokenspeed_attention requires an NVIDIA CUDA or AMD ROCm GPU"
        )

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if getattr(torch.version, "hip", None):
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        arch_map = {
            "gfx942": ArchVersion(9, 4),
            "gfx950": ArchVersion(9, 5),
        }
        return PlatformInfo("amd", arch_map.get(arch, ArchVersion(9, 0)))
    return PlatformInfo("nvidia", ArchVersion(props.major, props.minor))

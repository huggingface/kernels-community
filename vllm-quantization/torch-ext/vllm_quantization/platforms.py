from abc import ABC, abstractmethod
from functools import lru_cache
from typing import NamedTuple

import torch

IS_ROCM = torch.version.hip is not None


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform(ABC):
    simple_compile_backend: str = "inductor"

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """
        Returns the preferred FP8 type on the current platform.

        See the documentation for is_fp8_fnuz for details.
        """
        return torch.float8_e4m3fn

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        """
        Returns whether the preferred FP8 type is FNUZ on the current platform.

        There are two representations of FP8, OCP FP8 and FNUZ FP8.
        The OCP specification can be found at https://tinyurl.com/b7jvwpft.
        The FNUZ specification can be found at https://tinyurl.com/5n6hwwu5.

        AMD's MI300 and MI325 have native hardware support for FNUZ. All other
        hardware has converged on the OCP FP8 standard.
        """
        return False

    @classmethod
    @abstractmethod
    def get_device_name(cls, device_id: int = 0) -> str: ...

    @abstractmethod
    def is_rocm(self): ...


class CudaPlatform(Platform):
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(0)

    def is_rocm(self):
        return False


class RocmPlatform(Platform):
    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def is_rocm(self):
        return True


current_platform = RocmPlatform() if IS_ROCM else CudaPlatform()

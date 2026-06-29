import random
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import NamedTuple, Optional

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
    @abstractmethod
    def get_device_name(cls, device_id: int = 0) -> str: ...

    @abstractmethod
    def is_cuda_alike(self) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        ...

    @abstractmethod
    def is_rocm(self): ...

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            try:
                import numpy as np

                np.random.seed(seed)
            except ImportError:
                pass


class CudaPlatform(Platform):
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(0)

    def is_cuda_alike(self) -> bool:
        return True

    def is_rocm(self):
        return False


class RocmPlatform(Platform):
    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def is_cuda_alike(self) -> bool:
        return True

    def is_rocm(self):
        return True


current_platform = RocmPlatform() if IS_ROCM else CudaPlatform()

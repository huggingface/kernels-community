import os
import random
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from typing import Callable, ParamSpec, TypeVar

import numpy as np
import torch

IS_ROCM = torch.version.hip is not None
IS_MPS = torch.backends.mps.is_available()


class Platform(ABC):
    @classmethod
    def seed_everything(cls, seed: int) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def get_device_name(self, device_id: int = 0) -> str: ...

    @abstractmethod
    def is_cuda(self) -> bool: ...

    @abstractmethod
    def is_rocm(self) -> bool: ...

    @abstractmethod
    def is_mps(self) -> bool: ...


class CudaPlatform(Platform):
    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(0)

    def is_cuda(self) -> bool:
        return True

    def is_rocm(self) -> bool:
        return False

    def is_mps(self) -> bool:
        return False


class RocmPlatform(Platform):
    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def is_cuda(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return True

    def is_mps(self) -> bool:
        return False


class MpsPlatform(Platform):
    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def is_cuda(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_mps(self) -> bool:
        return True

current_platform = (
    RocmPlatform() if IS_ROCM else
    MpsPlatform() if IS_MPS else
    CudaPlatform() if torch.cuda.is_available() else
    None
)

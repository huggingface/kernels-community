"""
Configuration and calibration framework for PyTorch modules.

This module provides a system for managing configurable parameters in neural network
modules, supporting both runtime configuration and calibration workflows. It enables:

- Defining typed configuration dataclasses for module parameters
- Switching between calibration mode (finding optimal parameters)
  and run mode (using fixed parameters)
- Centralized configuration management via ModuleRegistry
- Serialization/deserialization of configurations to TOML files

Quick Start
-----------

1. Define a config dataclass for your module::

    @dataclass
    class MyRunConfig(CalibratedRunConfig):
        threshold: float | list[float] | None = None

        @classmethod
        def default(cls):
            return cls(threshold=0.1)

2. Add the ConfigurableModule mixin to your module::

    class MyModule(nn.Module, ConfigurableModule):
        run_config_type = MyRunConfig

        def __init__(self):
            super().__init__()
            ConfigurableModule.__init__(self)

        def forward(self, x):
            cfg = self.config  # timestep-indexed config
            # use cfg.threshold...
            self.add_calibration_results(MyRunConfig(threshold=computed_value))
            return x

3. Set up a registry and configure modules::

    model = MyModel()
    registry = ModuleRegistry(model.named_modules())

    # Option A: Set config for all modules
    registry.set_bulk_config(MyRunConfig(threshold=0.2))

    # Option B: Set config per module
    registry.set_module_config("layer1", MyRunConfig(threshold=0.1))

    # Option C: Load from TOML file
    registry.load_config(Path("config.toml"), config_types=[MyRunConfig])

4. Run inference and save calibration results::

    for timestep in range(num_timesteps):
        output = model(input)
    registry.config_output.save(Path("calibrated.toml"))
"""

from .config import (
    CalibratedCalibConfig,
    CalibratedConfig,
    CalibratedConfigDict,
    CalibratedRunConfig,
    ConfigList,
)
from .module import ConfigurableModule
from .registry import ModuleRegistry

__all__ = [
    "CalibratedCalibConfig",
    "CalibratedConfig",
    "CalibratedConfigDict",
    "CalibratedRunConfig",
    "ConfigList",
    "ConfigurableModule",
    "ModuleRegistry",
]

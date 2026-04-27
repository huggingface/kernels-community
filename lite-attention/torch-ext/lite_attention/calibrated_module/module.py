"""
ConfigurableModule mixin: adds config and calibration support to nn.Module.
"""

from __future__ import annotations

import typing
import warnings

import structlog

from .config import (
    CalibratedCalibConfig,
    CalibratedConfig,
    CalibratedRunConfig,
    ConfigList,
)

logger = structlog.get_logger()


class ConfigurableModule:
    """
    Mixin class that adds configuration support to PyTorch modules.

    To use, inherit from both nn.Module and ConfigurableModule, set the
    run_config_type class attribute, and call ConfigurableModule.__init__()
    in your __init__ method. See package docstring for full example.

    Attributes:
        run_config_type: The CalibratedRunConfig subclass for this module.
        config: Property returning the config for the current timestep.
        config_all: Property returning the full config (lists not indexed).

    Config Resolution Order:
        1. Instance config (passed to __init__)
        2. Registry config (set via ModuleRegistry)
        3. Default config (from run_config_type.default())
    """

    run_config_type: type[CalibratedRunConfig] | None = None

    def __init__(self, config: CalibratedConfig | ConfigList | None = None):
        """
        Initialize the configurable module.

        Args:
            config: Optional instance-level config that overrides registry config.
                Can be a single config (applies to all timesteps) or a ConfigList.

        """
        self._instance_config = config
        self.reset_config()
        # set by ModuleRegistry:
        self._registry: "ModuleRegistry | None" = None
        self.logger = logger.bind(module_id=id(self))

    def reset_config(self) -> None:
        """
        Reset configuration state to prepare for a new run.

        Called automatically by ModuleRegistry on registration.
        """
        self._registry_config: CalibratedConfig | ConfigList | None = None
        self._config_index = 0
        self._config_output: ConfigList = ConfigList()

    def restart_config(self) -> None:
        if self._config_index == 0:
            assert not self._config_output
            return
        self._config_index = 0
        self._config_output = ConfigList()
        if (
            isinstance(self.config, CalibratedCalibConfig)
            or isinstance(self.config_all, ConfigList)
            and any(isinstance(c, CalibratedCalibConfig) for c in self.config_all)
        ):
            warnings.warn(
                "Using restart_config() with a calibration config; data will be lost.",
                stacklevel=2,
            )

    @property
    def module_name(self) -> str | None:
        """Get the registered name of this module, or None if unregistered."""
        if self._registry is None:
            return None
        return self._registry.id_to_name[id(self)]

    @property
    def config_all(self) -> CalibratedConfig | ConfigList:
        """
        Get the full config (ConfigList or single config).

        Resolution order: instance config > registry config > default config.
        """
        if self._instance_config is not None:
            if self._registry is None:
                warnings.warn(
                    "Module has no registry. Using local config.", stacklevel=2
                )
            elif self._registry_config is None:
                warnings.warn(
                    "Module has no registry config. Using local config.", stacklevel=2
                )
            else:
                warnings.warn(
                    "Module has both local config and registry config. "
                    "Using local config.",
                    stacklevel=2,
                )
            return self._instance_config
        if self._registry is None:
            warnings.warn(
                "Module has no registry or local config. Using default config.",
                stacklevel=2,
            )
        elif self._registry_config is None:
            warnings.warn(
                "Module has no registry config or local config. Using default config.",
                stacklevel=2,
            )
        else:
            return self._registry_config

        if self.run_config_type is None:
            raise ValueError(f"Module {type(self)} has no run_config_type defined.")
        return self.run_config_type.default()

    @property
    def config(self) -> CalibratedConfig:
        """
        Get the config for the current forward pass.

        If config_all is a ConfigList, returns the config at _config_index.
        If config_all is a single config, returns it directly (same for all
        forward passes).

        The index is advanced by :meth:`add_calibration_results`, which must
        be called exactly once per ``forward()`` call.  Note that one
        *denoising step* may trigger multiple ``forward()`` calls when a
        guider evaluates the model more than once per step.

        Out-of-bounds indices are intentionally clamped to the last entry
        rather than raising IndexError, so configs with fewer entries than
        forward passes gracefully repeat the final config.

        This is the primary way to access config in forward().
        """
        cfg = self.config_all
        if isinstance(cfg, ConfigList):
            if self._config_index >= len(cfg):
                warnings.warn(
                    f"{type(self).__name__}: config index {self._config_index} "
                    f"exceeds ConfigList length {len(cfg)}, clamping to last entry. "
                    f"This may indicate a missing restart_config() call.",
                    stacklevel=2,
                )
                return cfg[-1]
            return cfg[self._config_index]
        else:
            return cfg

    def add_calibration_results(self, results: CalibratedRunConfig) -> None:
        """
        Record calibration results and advance the timestep index.

        Must be called exactly once per forward pass (even when not calibrating).

        Args:
            results: A RunConfig instance for this timestep.

        """
        self._config_index += 1
        if self._registry is None and isinstance(self.config, CalibratedCalibConfig):
            warnings.warn(
                "Module has no registry. Cannot save calibration results.", stacklevel=2
            )
        self._config_output.append(results)

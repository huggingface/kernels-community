"""
ModuleRegistry: central registry for configs across ConfigurableModules.
"""

from __future__ import annotations

import typing
from collections.abc import Iterator
from pathlib import Path

from .config import CalibratedConfig, CalibratedConfigDict, ConfigList
from .module import ConfigurableModule


class ModuleRegistry:
    """
    Central registry for managing configurations across all ConfigurableModules.

    Creates a mapping from module names to modules, and provides methods to set
    configs in bulk or per-module, as well as load/save configs to TOML files.
    See package docstring for full example.

    Attributes:
        named_modules: Dict mapping module names to ConfigurableModule instances.
        config: Property returning current input configs for all modules.
        config_output: Property returning calibration results for all modules.
    """

    def __init__(
        self, named_modules: Iterator[tuple[str, ConfigurableModule | typing.Any]]
    ):
        """
        Create a registry from a model's named_modules().

        Args:
            named_modules: Iterator of (name, module) pairs, typically from
                model.named_modules(). Non-ConfigurableModule entries are ignored.

        Note:
            The model must not add or remove ConfigurableModules after creation.

        """
        self.named_modules = {
            name: module
            for name, module in named_modules
            if isinstance(module, ConfigurableModule)
        }
        self.id_to_name = {
            id(module): name for name, module in self.named_modules.items()
        }
        for module in self.named_modules.values():
            module._registry = self
            module.reset_config()

    def set_bulk_config(self, config: CalibratedConfig | ConfigList) -> None:
        """Set the same config (or config list) for all registered modules."""
        for module in self.named_modules.values():
            module._registry_config = config

    def set_module_config(
        self, name: str, config: CalibratedConfig | ConfigList
    ) -> None:
        """Set config for a specific module by name."""
        module = self.named_modules[name]
        module._registry_config = config

    def load_config(
        self, filename: Path, config_types: list[type[CalibratedConfig]]
    ) -> None:
        """
        Load configs from a TOML file and apply to modules.

        Args:
            filename: Path to the TOML config file.
            config_types: List of config classes that may appear in the file.

        """
        loaded_config = CalibratedConfigDict.load(filename, config_types=config_types)
        for name, cfg in loaded_config.items():
            self.set_module_config(name, cfg)

    @property
    def config(self) -> CalibratedConfigDict:
        """
        Get the current input configs for all modules.

        Returns a dict mapping module names to their config_all (full configs
        with list values, not timestep-indexed). Useful for inspecting or
        saving the input configuration.
        """
        return CalibratedConfigDict(
            {name: module.config_all for name, module in self.named_modules.items()}
        )

    @property
    def config_output(self) -> CalibratedConfigDict:
        """
        Get the accumulated calibration results for all modules.

        Returns a dict mapping module names to their _config_output, which
        contains list values accumulated via add_calibration_results().
        Use .save() on the result to persist calibration results to TOML.
        """
        return CalibratedConfigDict(
            {name: module._config_output for name, module in self.named_modules.items()}
        )

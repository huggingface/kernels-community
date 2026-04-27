"""
Configuration types for the calibration framework.

Defines CalibratedConfig, CalibratedRunConfig, CalibratedCalibConfig (base types with
to_dict/from_dict), ConfigList (list of configs with collect/explode), and
CalibratedConfigDict (module name -> config(s) with TOML load/save).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import tomli_w

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# --- Base config types ---


@dataclass
class CalibratedConfig:
    """
    Base dataclass for all configuration types.

    Provides serialization (to_dict) and deserialization (from_dict).

    Do not subclass directly; use CalibratedRunConfig or CalibratedCalibConfig instead.
    """

    @classmethod
    def from_dict(
        cls, cfg: dict[str, typing.Any], config_types: dict[str, type[CalibratedConfig]]
    ) -> CalibratedConfig:
        """
        Deserialize a config from a dictionary.

        Args:
            cfg: Dictionary with "_type" key indicating the config class name.
            config_types: Mapping from class names to config classes.

        Returns:
            An instance of the appropriate config subclass.

        Raises:
            ValueError: If the _type is not found in config_types.

        """
        _type = cfg["_type"]
        _class = config_types.get(_type)
        if _class is None:
            raise ValueError(f"Unknown config type: {_type}")
        assert issubclass(_class, CalibratedConfig)
        fields = {k: v for k, v in cfg.items() if k != "_type"}
        return _class(**fields)

    def to_dict(self) -> dict[str, typing.Any]:
        """Serialize this config to a dictionary with a _type field."""
        return {"_type": type(self).__name__} | self.__dict__


class CalibratedRunConfig(CalibratedConfig):
    """
    Configuration for runtime parameters.

    Subclass this to define your module's runtime parameters. Parameters can be
    single values (applied to all timesteps) or lists (one value per timestep).
    Optionally implement the default() classmethod to provide fallback values.
    """

    @classmethod
    def default(cls) -> Self:
        """
        Return a default configuration.

        Override this to provide fallback values when no config is explicitly set.
        If not implemented, modules without explicit config will raise an error.
        """
        raise NotImplementedError


class CalibratedCalibConfig(CalibratedConfig):
    """
    Configuration for calibration parameters.

    Subclass this to define calibration settings (e.g., target error, metric type).
    During calibration, the module uses these settings to find optimal runtime values,
    which are then saved as a CalibratedRunConfig.

    Example::

        @dataclass
        class MyCalibConfig(CalibratedCalibConfig):
            metric: Literal["l1", "l2"] = "l1"
            target_error: float = 0.01


        # In forward(), check config type to determine mode:
        cfg = self.config
        if isinstance(cfg, MyCalibConfig):
            threshold = self.find_optimal_threshold(cfg.target_error, cfg.metric)
        else:
            threshold = cfg.threshold
    """

    pass


# --- Config list (one config per timestep) ---


class ConfigList(list[CalibratedConfig]):
    """
    List of config objects, one per forward pass.

    Each ``forward()`` call on a :class:`ConfigurableModule` consumes the next
    entry (via :meth:`~ConfigurableModule.add_calibration_results`).  Note that
    one *denoising step* may trigger multiple forward passes when a guider
    (e.g. CFG/STG) evaluates the model more than once per step.

    Out-of-bounds indices are clamped to the last entry, so a list shorter than
    the total number of forward passes gracefully repeats the final config.

    Provides methods to convert between list-of-configs and dict-with-lists
    representations.
    """

    def collect(self) -> dict[str, typing.Any]:
        """
        Collapse list of same-type configs to dict with list values.

        Example: [Config(a=1, b=0), Config(a=2, b=0)] → {"_type": "Config", "a": [1, 2], "b": [0, 0]}

        Returns a dict (not a Config) to prevent accidental use as config.

        Raises:
            ValueError: If the list is empty.
            TypeError: If configs have mixed types.
        """
        if not self:
            raise ValueError("Cannot collect empty ConfigList")
        first_type = type(self[0])
        if not all(type(c) is first_type for c in self):
            raise TypeError("Cannot collect configs with mixed types")
        collected: dict[str, str | list[typing.Any]] = {"_type": first_type.__name__}
        for key in self[0].__dict__:
            collected[key] = [cfg.__dict__[key] for cfg in self]
        return collected

    @classmethod
    def explode(
        cls,
        data: dict[str, typing.Any],
        config_types: dict[str, type[CalibratedConfig]],
    ) -> ConfigList:
        """
        Expand dict with list values to list of configs.

        Example: {"_type": "Config", "a": [1, 2], "b": [0, 0]} → [Config(a=1, b=0), Config(a=2, b=0)]

        Args:
            data: Dict with "_type" key and list-valued fields.
            config_types: Mapping from type names to config classes.

        All list-valued fields must have the same length.
        """
        type_name = data["_type"]
        config_type = config_types.get(type_name)
        if config_type is None:
            raise ValueError(f"Unknown config type: {type_name}")
        fields = {k: v for k, v in data.items() if k != "_type"}
        # Find the length from any list-valued field
        length = None
        for value in fields.values():
            if isinstance(value, list):
                if length is None:
                    length = len(value)
                elif len(value) != length:
                    raise ValueError("All list fields must have the same length")
        if length is None:
            # No list fields - return single-element ConfigList
            return cls([config_type(**fields)])
        # Create one config per index
        result = cls()
        for i in range(length):
            new_dict = {
                k: (v[i] if isinstance(v, list) else v) for k, v in fields.items()
            }
            result.append(config_type(**new_dict))
        return result


# --- Config dict (module name -> config, TOML load/save) ---


class CalibratedConfigDict(dict[str, ConfigList | CalibratedConfig]):
    """
    Dictionary mapping module names to their configurations.

    Provides TOML serialization via load() and save() methods.
    Keys are module names (as returned by model.named_modules()), values are
    ConfigList or CalibratedConfig instances.
    """

    @classmethod
    def from_dict(
        cls,
        config: dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]],
        config_types: list[type[CalibratedConfig]],
    ) -> Self:
        """
        Deserialize configs from a nested dictionary.

        Args:
            config: Nested dict mapping module names to config dicts or list of dicts.
                Each config dict must have a "_type" key.
            config_types: List of config classes for deserialization.
        """
        type_map = {ct.__name__: ct for ct in config_types}
        result: dict[str, ConfigList | CalibratedConfig] = {}
        for name, cfg_data in config.items():
            if isinstance(cfg_data, list):
                result[name] = ConfigList(
                    CalibratedConfig.from_dict(dict(cfg), type_map) for cfg in cfg_data
                )
            else:
                result[name] = CalibratedConfig.from_dict(cfg_data, type_map)
        return cls(result)

    @classmethod
    def load(cls, filename: Path, config_types: list[type[CalibratedConfig]]) -> Self:
        """
        Load configs from a TOML file.

        Args:
            filename: Path to the TOML file.
            config_types: List of config classes that may appear in the file.

        """
        with filename.open("rb") as f:
            loaded_config = tomllib.load(f)
        return cls.from_dict(loaded_config, config_types=config_types)

    def to_dict(self) -> dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]]:
        """Serialize all configs to a nested dictionary."""
        result: dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]] = {}
        for name, cfg in self.items():
            if isinstance(cfg, ConfigList):
                result[name] = [c.to_dict() for c in cfg]
            else:
                result[name] = cfg.to_dict()
        return result

    def collect(self) -> dict[str, typing.Any]:
        """
        Collapse all ConfigLists to dicts with list values.

        Example: {"module1": [Config(a=1), Config(a=2)]} → {"module1": {"_type": "Config", "a": [1, 2]}}

        Single configs are converted via to_dict().
        Raises TypeError if any ConfigList has mixed types.
        """
        result = {}
        for name, cfg in self.items():
            if isinstance(cfg, ConfigList):
                result[name] = cfg.collect()
            else:
                result[name] = cfg.to_dict()
        return result

    def save(self, filename: Path) -> None:
        """Save all configs to a TOML file."""
        with filename.open("wb") as f:
            tomli_w.dump(self.to_dict(), f)

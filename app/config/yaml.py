"""YAML helpers with !env support for configuration files."""

from __future__ import annotations

import os
from typing import Any

import yaml


def _env_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    if isinstance(node, yaml.ScalarNode):
        var_name = loader.construct_scalar(node)
        if not isinstance(var_name, str):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f'Environment variable name must be a string, got {type(var_name).__name__}',
                node.start_mark,
            )

        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Required environment variable '{var_name}' is not set")
        return value

    if isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node)
        if len(values) != 2:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f'!env sequence must have exactly 2 elements [var_name, default], got {len(values)}',
                node.start_mark,
            )

        var_name, default_value = values
        if not isinstance(var_name, str):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f'Environment variable name must be a string, got {type(var_name).__name__}',
                node.start_mark,
            )

        return os.getenv(var_name, default_value)

    raise yaml.constructor.ConstructorError(
        None,
        None,
        f'!env tag expects scalar (var_name) or sequence ([var_name, default]), got {type(node).__name__}',
        node.start_mark,
    )


yaml.add_constructor('!env', _env_constructor, yaml.SafeLoader)


def safe_load_with_env(stream) -> Any:
    """Drop-in replacement for yaml.safe_load() supporting !env tags."""

    return yaml.safe_load(stream)


__all__ = ['safe_load_with_env']

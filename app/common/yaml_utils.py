"""YAML utilities with environment variable support."""

import os
from typing import Any

import yaml


def _env_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """Construct environment variable values from !env tags.

    Supports two formats:
    - !env VAR_NAME - Required environment variable (raises error if missing)
    - !env [VAR_NAME, default_value] - Optional with default value

    Args:
        loader: YAML loader instance
        node: YAML node containing the environment variable specification

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required environment variable is missing
        yaml.constructor.ConstructorError: If node format is invalid
    """
    if isinstance(node, yaml.ScalarNode):
        # Format: !env VAR_NAME (required)
        var_name = loader.construct_scalar(node)
        if not isinstance(var_name, str):
            raise yaml.constructor.ConstructorError(None, None, f'Environment variable name must be a string, got {type(var_name).__name__}', node.start_mark)

        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Required environment variable '{var_name}' is not set")
        return value

    elif isinstance(node, yaml.SequenceNode):
        # Format: !env [VAR_NAME, default_value] (optional with default)
        values = loader.construct_sequence(node)
        if len(values) != 2:
            raise yaml.constructor.ConstructorError(None, None, f'!env sequence must have exactly 2 elements [var_name, default], got {len(values)}', node.start_mark)

        var_name, default_value = values
        if not isinstance(var_name, str):
            raise yaml.constructor.ConstructorError(None, None, f'Environment variable name must be a string, got {type(var_name).__name__}', node.start_mark)

        return os.getenv(var_name, default_value)

    else:
        raise yaml.constructor.ConstructorError(None, None, f'!env tag expects scalar (var_name) or sequence ([var_name, default]), got {type(node).__name__}', node.start_mark)


# Register the !env tag constructor globally with SafeLoader
yaml.add_constructor('!env', _env_constructor, yaml.SafeLoader)


def safe_load_with_env(stream) -> Any:
    """Load YAML with environment variable support.

    Drop-in replacement for yaml.safe_load() that supports !env tags.

    Args:
        stream: YAML content (string, file-like object, etc.)

    Returns:
        Parsed YAML data with environment variables resolved

    Raises:
        yaml.YAMLError: If YAML parsing fails
        ValueError: If required environment variable is missing
    """
    return yaml.safe_load(stream)

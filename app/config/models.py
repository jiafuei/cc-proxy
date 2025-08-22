from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class ConfigModel(BaseModel):
    """Configuration model with validation."""

    host: str = Field(default='127.0.0.1')
    port: int = Field(default=8000, ge=1, le=65535)
    dev: bool = Field(default=False)
    dump_requests: bool = Field(default=False)
    dump_responses: bool = Field(default=False)
    dump_headers: bool = Field(default=False)
    dump_dir: str | None = Field(default=None)
    cors_allow_origins: List[str] = Field(default_factory=list)
    redact_headers: List[str] | None = Field(default_factory=lambda: ['authorization', 'x-api-key', 'cookie', 'set-cookie'])

    @classmethod
    def load(cls, config_path: str | None = None) -> 'ConfigModel':
        """Load configuration from YAML file.

        Tries multiple locations in order:
        1. Explicit config_path if provided
        2. ./config.yaml in current directory
        3. ~/.cc-proxy/config.yaml in user home directory
        """

        # Determine config file paths to try
        config_paths = []
        if config_path:
            config_paths.append(config_path)
        else:
            # Try user home directory first
            home_config = Path.home() / '.cc-proxy' / 'config.yaml'
            if home_config.exists():
                config_paths.append(str(home_config))

            # Try current directory
            config_paths.append('config.yaml')

        # Try each config path in order
        data = {}
        for path in config_paths:
            try:
                with open(path, 'r') as f:
                    file_data = yaml.safe_load(f) or {}
                    # Merge with existing data (later files override earlier ones)
                    data.update(file_data)
                break  # Successfully loaded a config file
            except FileNotFoundError:
                continue  # Try next config path
            except yaml.YAMLError as e:
                # Handle YAML parsing errors
                raise ValueError(f'Invalid YAML in config file {path}: {e}')
            except Exception as e:
                # Handle other file reading errors
                raise ValueError(f'Error reading config file {path}: {e}')

        # Use default values for missing keys
        return cls(**data)

    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False, indent=2)

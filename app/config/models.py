import os
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import get_app_dir


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    level: str = Field(default='INFO', description='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    console_enabled: bool = Field(default=True, description='Enable console logging')
    file_enabled: bool = Field(default=True, description='Enable file logging')
    log_file_dir: str = Field(default=get_app_dir() / 'logs', description='Log directory (defaults to ~/.cc-proxy/logs)')
    max_file_size: str = Field(default='10MB', description='Maximum log file size before rotation')
    backup_count: int = Field(default=4, description='Number of backup files to keep')
    rotation_when: str = Field(default='midnight', description='Time-based rotation interval')


class ConfigModel(BaseModel):
    """Configuration model with validation."""
    model_config = ConfigDict(extra='allow')

    version: str = Field(default='1', description='Config version')
    host: str = Field(default='127.0.0.1')
    port: int = Field(default=8000, ge=1, le=65535)
    dev: bool = Field(default=False)
    dump_requests: bool = Field(default=False)
    dump_responses: bool = Field(default=False)
    dump_headers: bool = Field(default=False)
    dump_dir: str | None = Field(default=None)
    cors_allow_origins: List[str] = Field(default_factory=list)
    redact_headers: List[str] | None = Field(default_factory=lambda: ['authorization', 'x-api-key', 'cookie', 'set-cookie'])
    fallback_api_url: str | None = Field(default='https://api.anthropic.com/v1/messages')
    fallback_api_key: str | None = Field(default=None)
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description='Logging configuration')

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
            home_config = get_app_dir() / 'config.yaml'
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
            except FileNotFoundError:
                continue  # Try next config path
            except yaml.YAMLError as e:
                # Handle YAML parsing errors
                raise ValueError(f'Invalid YAML in config file {path}: {e}')
            except Exception as e:
                # Handle other file reading errors
                raise ValueError(f'Error reading config file {path}: {e}')

        # Use default values for missing keys
        os.environ.setdefault('CCPROXY_FALLBACK_URL', data.get('fallback_api_url', 'https://api.anthropic.com/v1/messages'))
        os.environ.setdefault('CCPROXY_FALLBACK_API_KEY', data.get('fallback_api_key', ''))
        return cls(**data)

    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False, indent=2)

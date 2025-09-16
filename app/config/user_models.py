"""User configuration models for dynamic reloading."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config.paths import get_app_dir
from app.config.yaml import safe_load_with_env
from app.providers.types import ProviderType


class SimpleTransformerConfig(BaseModel):
    """Simplified transformer configuration for new architecture."""

    model_config = ConfigDict(populate_by_name=True)

    class_path: str = Field(alias='class', description='Full class path like "my_module.MyTransformer"')
    params: dict = Field(default_factory=dict, description='Parameters to pass to transformer constructor')

    def to_loader_dict(self) -> Dict[str, Any]:
        """Return configuration compatible with the transformer loader."""

        return {'class': self.class_path, 'params': self.params}


class TransformerStageConfig(BaseModel):
    """Per-stage transformer configuration (request/response/stream)."""

    request: List[SimpleTransformerConfig] = Field(default_factory=list)
    response: List[SimpleTransformerConfig] = Field(default_factory=list)
    stream: List[SimpleTransformerConfig] = Field(default_factory=list)

    def to_loader_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            'request': [cfg.to_loader_dict() for cfg in self.request],
            'response': [cfg.to_loader_dict() for cfg in self.response],
            'stream': [cfg.to_loader_dict() for cfg in self.stream],
        }


class ProviderConfig(BaseModel):
    """Provider configuration with explicit type and channel-aware transformers."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description='Unique provider name')
    base_url: str = Field(description='Base URL for the provider API')
    api_key: str = Field(default='', description='API key for the provider')
    type: ProviderType = Field(description='Provider API type: anthropic, openai, openai-responses, or gemini')
    capabilities: Optional[List[str]] = Field(default=None, description='Operations to enable (subset of descriptor operations)')
    transformers: Dict[str, TransformerStageConfig] = Field(default_factory=dict, description='Per-channel transformer overrides')
    timeout: int = Field(default=180, description='Request timeout in seconds')

    @model_validator(mode='after')
    def _ensure_transformer_models(self) -> 'ProviderConfig':
        """Convert raw transformer dicts into stage configs."""

        resolved: Dict[str, TransformerStageConfig] = {}
        for channel, config in self.transformers.items():
            if isinstance(config, TransformerStageConfig):
                resolved[channel] = config
                continue

            if not isinstance(config, dict):
                raise ValueError(f'Transformer config for channel "{channel}" must be a mapping')

            resolved[channel] = TransformerStageConfig(**config)

        self.transformers = resolved
        return self

    @field_validator('capabilities')
    @classmethod
    def validate_capabilities(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate capabilities are valid operation names."""
        if v is not None:
            valid_ops = {'messages', 'count_tokens', 'responses'}
            for cap in v:
                if cap not in valid_ops:
                    raise ValueError(f"Invalid capability '{cap}'. Valid operations: {valid_ops}")
        return v

    def channel_transformers(self, channel: str) -> TransformerStageConfig:
        """Return transformer overrides for a given channel."""

        return self.transformers.get(channel, TransformerStageConfig())


class ModelConfig(BaseModel):
    """Simplified model configuration linking models to providers."""

    id: str = Field(default='', description='Model identifier (defaults to alias if empty)')
    provider: str = Field(description='Name of provider for this model')
    alias: str = Field(description='Required short alias for this model')

    @model_validator(mode='before')
    @classmethod
    def set_default_id(cls, data):
        """Set id to alias if id is not provided or empty."""
        if isinstance(data, dict):
            if not data.get('id'):
                data['id'] = data.get('alias', '')
        return data

    @field_validator('alias')
    @classmethod
    def validate_alias(cls, v: str) -> str:
        """Validate alias format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Alias must contain only alphanumeric characters, hyphens, and underscores')
        if len(v) < 1:
            raise ValueError('Alias cannot be empty')
        if len(v) > 50:
            raise ValueError('Alias cannot be longer than 50 characters')
        return v


class RoutingConfig(BaseModel):
    """Simplified routing configuration for different request types."""

    default: str = Field(description='Default model for standard requests')
    background: str = Field(default='', description='Model for background/simple requests')
    planning: str = Field(default='', description='Model for planning/complex reasoning requests')
    thinking: str = Field(default='', description='Model for thinking-enabled requests')
    plan_and_think: str = Field(default='', description='Model for combined planning and thinking requests')
    builtin_tools: str = Field(default='', description='Model for built-in tool requests (WebSearch, WebFetch, etc.)')


class UserConfig(BaseModel):
    """Simplified user configuration model for new architecture."""

    providers: List[ProviderConfig] = Field(default_factory=list, description='Provider configurations')
    models: List[ModelConfig] = Field(default_factory=list, description='Model definitions')
    routing: Optional[RoutingConfig] = Field(default=None, description='Request routing configuration')
    transformer_paths: List[str] = Field(default_factory=list, description='Paths to search for external transformers')

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'UserConfig':
        """Load user configuration from YAML file.

        Args:
            config_path: Optional explicit path to config file

        Returns:
            UserConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if config_path is None:
            config_path = get_app_dir() / 'user.yaml'

        if not config_path.exists():
            # Return empty config if file doesn't exist
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = safe_load_with_env(f) or {}
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f'Invalid YAML in user config file {config_path}: {e}')
        except Exception as e:
            raise ValueError(f'Error reading user config file {config_path}: {e}')

    def get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def get_model_by_alias(self, alias: str) -> Optional[ModelConfig]:
        """Get model configuration by alias only."""
        for model in self.models:
            if model.alias == alias:
                return model
        return None

    def validate_references(self) -> None:
        """Validate that all references between components are valid."""
        errors = []

        # Check that models reference valid providers
        for model in self.models:
            if not self.get_provider_by_name(model.provider):
                errors.append(f"Model '{model.alias}' references unknown provider '{model.provider}'")

        # Check alias uniqueness
        aliases_seen = set()
        for model in self.models:
            if model.alias in aliases_seen:
                errors.append(f"Duplicate alias '{model.alias}' found (model '{model.id}')")
            else:
                aliases_seen.add(model.alias)

        # Check that routing references valid aliases
        if self.routing:
            for routing_type, alias in [
                ('default', self.routing.default),
                ('planning', self.routing.planning),
                ('background', self.routing.background),
                ('thinking', self.routing.thinking),
                ('plan_and_think', self.routing.plan_and_think),
            ]:
                if alias and not self.get_model_by_alias(alias):
                    errors.append(f"Routing '{routing_type}' references unknown alias '{alias}'")

        if errors:
            raise ValueError('Configuration validation failed:\n' + '\n'.join(errors))

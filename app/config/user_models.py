"""User configuration models for dynamic reloading."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import get_app_dir


class SimpleTransformerConfig(BaseModel):
    """Simplified transformer configuration for new architecture."""

    model_config = ConfigDict(populate_by_name=True)

    class_path: str = Field(alias='class', description='Full class path like "my_module.MyTransformer"')
    params: dict = Field(default_factory=dict, description='Parameters to pass to transformer constructor')


class ProviderConfig(BaseModel):
    """Simplified provider configuration for new architecture."""

    name: str = Field(description='Unique provider name')
    url: str = Field(description='Base URL for the provider API')
    api_key: str = Field(default='', description='API key for the provider')
    models: List[str] = Field(default_factory=list, description='Models supported by this provider')
    transformers: dict = Field(default_factory=dict, description='Transformer configurations')
    timeout: int = Field(default=300, description='Request timeout in seconds')

    def __init__(self, **data):
        # Handle transformers structure
        if 'transformers' in data and isinstance(data['transformers'], dict):
            transformers = data['transformers']
            # Ensure request and response transformer lists exist
            if 'request' not in transformers:
                transformers['request'] = []
            if 'response' not in transformers:
                transformers['response'] = []
        super().__init__(**data)


class ModelConfig(BaseModel):
    """Simplified model configuration linking models to providers."""

    id: str = Field(description='Model identifier')
    provider: str = Field(description='Name of provider for this model')


class RoutingConfig(BaseModel):
    """Simplified routing configuration for different request types."""

    default: str = Field(description='Default model for standard requests')
    planning: str = Field(default='', description='Model for planning/complex reasoning requests')
    background: str = Field(default='', description='Model for background/simple requests')
    thinking: str = Field(default='', description='Model for thinking-enabled requests')
    plan_and_think: str = Field(default='', description='Model for combined planning and thinking requests')


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
                data = yaml.safe_load(f) or {}
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

    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def validate_references(self) -> None:
        """Validate that all references between components are valid."""
        errors = []

        # Check that models reference valid providers
        for model in self.models:
            if not self.get_provider_by_name(model.provider):
                errors.append(f"Model '{model.id}' references unknown provider '{model.provider}'")

        # Check that routing references valid models
        if self.routing:
            for routing_type, model_id in [
                ('default', self.routing.default),
                ('planning', self.routing.planning),
                ('background', self.routing.background),
                ('thinking', self.routing.thinking),
                ('plan_and_think', self.routing.plan_and_think),
            ]:
                if model_id and not self.get_model_by_id(model_id):
                    errors.append(f"Routing '{routing_type}' references unknown model '{model_id}'")

        if errors:
            raise ValueError('Configuration validation failed:\n' + '\n'.join(errors))

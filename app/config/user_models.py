"""User configuration models for dynamic reloading."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from app.common.utils import get_app_dir


class TransformerArgument(BaseModel):
    """Individual transformer argument configuration."""

    key: str
    value: str


class TransformerConfig(BaseModel):
    """Custom transformer configuration."""

    name: str = Field(description='Unique transformer name')
    path: str = Field(description='Path to Python transformer file')
    args: List[TransformerArgument] = Field(default_factory=list, description='Arguments to pass to transformer')

    @field_validator('path')
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that the transformer file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Transformer file does not exist: {v}')
        return v


class PipelineTransformerConfig(BaseModel):
    """Transformer configuration within a pipeline."""

    name: str = Field(description='Name of transformer to use')
    args: List[TransformerArgument] = Field(default_factory=list, description='Override arguments for this usage')


class ProviderConfig(BaseModel):
    """Custom provider configuration."""

    name: str = Field(description='Unique provider name')
    api_key: str = Field(description='API key for the provider')
    url: str = Field(description='Base URL for the provider API')
    request_pipeline: List[PipelineTransformerConfig] = Field(default_factory=list, description='Request transformation pipeline')
    response_pipeline: List[PipelineTransformerConfig] = Field(default_factory=list, description='Response transformation pipeline')


class ModelConfig(BaseModel):
    """Model configuration linking models to providers."""

    id: str = Field(description='Model identifier')
    provider: str = Field(description='Name of provider for this model')


class RoutingConfig(BaseModel):
    """Routing configuration for different request types."""

    default: str = Field(description='Default model for standard requests')
    planning: str = Field(description='Model for planning/complex reasoning requests')
    background: str = Field(description='Model for background/simple requests')


class UserConfig(BaseModel):
    """Complete user configuration model."""

    transformers: List[TransformerConfig] = Field(default_factory=list, description='Custom transformers')
    providers: List[ProviderConfig] = Field(default_factory=list, description='Custom providers')
    models: List[ModelConfig] = Field(default_factory=list, description='Model definitions')
    routing: Optional[RoutingConfig] = Field(default=None, description='Request routing configuration')

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

    def get_transformer_by_name(self, name: str) -> Optional[TransformerConfig]:
        """Get transformer configuration by name."""
        for transformer in self.transformers:
            if transformer.name == name:
                return transformer
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
            for routing_type, model_id in [('default', self.routing.default), ('planning', self.routing.planning), ('background', self.routing.background)]:
                if not self.get_model_by_id(model_id):
                    errors.append(f"Routing '{routing_type}' references unknown model '{model_id}'")

        # Check that pipeline transformers reference valid transformers
        for provider in self.providers:
            for pipeline_name, pipeline in [('request_pipeline', provider.request_pipeline), ('response_pipeline', provider.response_pipeline)]:
                for transformer_ref in pipeline:
                    transformer_config = self.get_transformer_by_name(transformer_ref.name)
                    if not transformer_config and not transformer_ref.name.startswith('builtin-'):
                        errors.append(f"Provider '{provider.name}' {pipeline_name} references unknown transformer '{transformer_ref.name}'")

        if errors:
            raise ValueError('Configuration validation failed:\n' + '\n'.join(errors))

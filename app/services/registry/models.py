"""Model registry for handling model-to-provider mapping."""

import logging
from typing import Dict, List, Optional

from app.config.user_models import ModelConfig
from app.services.config.interfaces import ComponentRegistry

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about a model and its provider."""

    def __init__(self, config: ModelConfig):
        self.id = config.id
        self.provider_name = config.provider
        self.config = config

    def __str__(self) -> str:
        return f'Model(id={self.id}, provider={self.provider_name})'


class ModelRegistry(ComponentRegistry[ModelInfo]):
    """Registry for managing model-to-provider mappings."""

    def __init__(self):
        super().__init__()
        self._provider_models: Dict[str, List[str]] = {}

    def register_model_from_config(self, config: ModelConfig) -> None:
        """Register a model from configuration.

        Args:
            config: Model configuration
        """
        model_info = ModelInfo(config)
        self.register(config.id, model_info)

        # Track models by provider
        if config.provider not in self._provider_models:
            self._provider_models[config.provider] = []

        if config.id not in self._provider_models[config.provider]:
            self._provider_models[config.provider].append(config.id)

        logger.debug(f"Registered model '{config.id}' with provider '{config.provider}'")

    def register_models_from_config(self, models: List[ModelConfig]) -> int:
        """Register multiple models from configuration.

        Args:
            models: List of model configurations

        Returns:
            Number of models registered successfully
        """
        success_count = 0

        for model_config in models:
            try:
                self.register_model_from_config(model_config)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to register model '{model_config.id}': {e}")

        logger.info(f'Registered {success_count}/{len(models)} models successfully')
        return success_count

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID.

        Args:
            model_id: ID of the model

        Returns:
            ModelInfo if found, None otherwise
        """
        return self.get(model_id)

    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        """Get the provider name for a model.

        Args:
            model_id: ID of the model

        Returns:
            Provider name if found, None otherwise
        """
        model_info = self.get(model_id)
        if model_info:
            return model_info.provider_name
        return None

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Get all models for a given provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of model IDs for the provider
        """
        return self._provider_models.get(provider_name, []).copy()

    def list_all_models(self) -> List[str]:
        """List all registered model IDs."""
        return self.list_names()

    def list_all_providers(self) -> List[str]:
        """List all providers that have models."""
        return list(self._provider_models.keys())

    def model_exists(self, model_id: str) -> bool:
        """Check if a model is registered.

        Args:
            model_id: ID of the model

        Returns:
            True if model exists, False otherwise
        """
        return self.get(model_id) is not None

    def provider_has_models(self, provider_name: str) -> bool:
        """Check if a provider has any models.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider has models, False otherwise
        """
        return provider_name in self._provider_models and len(self._provider_models[provider_name]) > 0

    def clear_all_models(self) -> None:
        """Clear all registered models."""
        self.clear()
        self._provider_models.clear()
        logger.info('Cleared all models')

    def clear_models_for_provider(self, provider_name: str) -> None:
        """Clear all models for a specific provider.

        Args:
            provider_name: Name of the provider
        """
        if provider_name in self._provider_models:
            model_ids = self._provider_models[provider_name].copy()

            for model_id in model_ids:
                self.unregister(model_id)

            del self._provider_models[provider_name]
            logger.info(f"Cleared all models for provider '{provider_name}'")

    def validate_model_references(self, available_providers: List[str]) -> List[str]:
        """Validate that all models reference available providers.

        Args:
            available_providers: List of available provider names

        Returns:
            List of validation error messages
        """
        errors = []

        for model_id, model_info in self._components.items():
            if model_info.provider_name not in available_providers:
                errors.append(f"Model '{model_id}' references unknown provider '{model_info.provider_name}'")

        return errors

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of the registry contents.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_models': self.size(),
            'total_providers': len(self._provider_models),
            'models_per_provider': {provider: len(models) for provider, models in self._provider_models.items()},
        }

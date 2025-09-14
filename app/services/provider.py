"""Enhanced provider system for the simplified architecture."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Request

from app.common.dumper import Dumper, DumpHandles
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.config.user_models import ModelConfig, ProviderConfig
from app.services.providers.specs import PROVIDER_SPECS
from app.services.transformer_loader import TransformerLoader

logger = get_logger(__name__)


@dataclass
class ModelMapping:
    """Maps model alias to provider name and actual model ID."""

    provider_name: str
    model_id: str


class Provider:
    """Simplified provider using type-based configuration."""

    def __init__(self, config: ProviderConfig, transformer_loader: TransformerLoader):
        self.config = config
        self.name = config.name

        # Direct spec lookup - no file I/O
        if config.type not in PROVIDER_SPECS:
            raise ValueError(f'Unknown provider type: {config.type}')

        self.spec = PROVIDER_SPECS[config.type]

        # Simple capability handling
        self.capabilities = config.capabilities or self.spec['supported_operations']

        # Validate capabilities against spec
        unsupported = set(self.capabilities) - set(self.spec['supported_operations'])
        if unsupported:
            raise ValueError(f"Provider type '{config.type}' doesn't support: {unsupported}")

        # Simple transformer loading
        transformer_config = config.transformers or self.spec['default_transformers']
        self.request_transformers = transformer_loader.load_transformers(transformer_config.get('request', []))
        self.response_transformers = transformer_loader.load_transformers(transformer_config.get('response', []))

        # Create HTTP client
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout), http2=True)

        logger.info(f"Provider '{config.name}' initialized: type={config.type}, capabilities={self.capabilities}")

    def _validate_operation(self, operation: str) -> None:
        """Validate operation is supported."""
        if operation not in self.capabilities:
            raise ValueError(f"Operation '{operation}' not enabled. Available: {self.capabilities}")

    def _get_operation_url(self, operation: str, model: str = None) -> str:
        """Get URL for operation, with optional model injection."""
        if operation not in self.spec['url_suffixes']:
            raise ValueError(f"Operation '{operation}' not supported by provider type '{self.config.type}'")

        base_url = self.config.url.rstrip('/')
        suffix = self.spec['url_suffixes'][operation]

        # Handle templates for providers that need model in URL
        if '{model}' in suffix:
            if not model:
                raise ValueError(f'Model parameter required for {self.config.type} provider')
            suffix = suffix.format(model=model)

        return f'{base_url}{suffix}'

    async def process_operation(
        self, operation: str, request: AnthropicRequest, original_request: Request, routing_key: str, dumper: Dumper, dumper_handles: DumpHandles, model: str = None
    ) -> Dict[str, Any]:
        """Process an operation using the provider type."""

        self._validate_operation(operation)
        url = self._get_operation_url(operation, model)

        # Convert request to dict
        current_request = request.to_dict()
        current_headers = dict(original_request.headers)

        # Force non-streaming for simplified architecture
        current_request['stream'] = False

        # Apply request transformers
        for transformer in self.request_transformers:
            transform_params = {
                'request': current_request,
                'headers': current_headers,
                'provider_config': self.config,
                'original_request': original_request,
                'routing_key': routing_key,
            }
            current_request, current_headers = await transformer.transform(transform_params)

        # Log transformed request
        dumper.write_transformed_headers(dumper_handles, current_headers)
        dumper.write_transformed_request(dumper_handles, current_request)

        # Send HTTP request to the computed URL
        response = await self.http_client.post(
            url,  # URL from template-based generation
            json=current_request,
            headers=current_headers,
        )

        # Handle HTTP errors
        try:
            response.raise_for_status()
        except:
            await response.aread()
            raise

        # Get response text
        response_text = response.text
        dumper.write_pretransformed_response(dumper_handles, response_text)

        # Parse JSON response
        try:
            response_json = response.json()
        except Exception:
            logger.error('Failed to parse response as JSON')
            raise

        # Apply response transformers
        for transformer in self.response_transformers:
            response_params = {
                'response': response_json,
                'request': current_request,
                'final_headers': current_headers,
                'provider_config': self.config,
                'original_request': original_request,
            }
            response_json = await transformer.transform_response(response_params)

        return response_json

    def supports_operation(self, operation: str) -> bool:
        """Check if provider supports an operation."""
        return operation in self.capabilities

    def list_operations(self) -> List[str]:
        """List all enabled operations."""
        return self.capabilities

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()

    def __str__(self) -> str:
        return f'Provider(name={self.config.name}, url={self.config.url})'


class ProviderManager:
    """Manages multiple providers."""

    def __init__(self, providers_config: List[ProviderConfig], models_config: List[ModelConfig], transformer_loader: TransformerLoader):
        self.providers: Dict[str, Provider] = {}
        self.alias_mappings: Dict[str, ModelMapping] = {}
        self._load_providers(providers_config, transformer_loader)
        self._build_model_mapping(models_config)

    def _load_providers(self, providers_config: List[ProviderConfig], transformer_loader: TransformerLoader):
        """Load providers from configuration."""
        for provider_config in providers_config:
            try:
                provider = Provider(provider_config, transformer_loader)
                self.providers[provider_config.name] = provider

                logger.info(f"Loaded provider '{provider_config.name}'")

            except Exception as e:
                logger.error(f"Failed to load provider '{provider_config.name}': {e}", exc_info=True)

    def _build_model_mapping(self, models_config: List[ModelConfig]):
        """Build alias-to-provider mapping from top-level models configuration."""
        for model_config in models_config:
            if model_config.provider in self.providers:
                self.alias_mappings[model_config.alias] = ModelMapping(provider_name=model_config.provider, model_id=model_config.id)
            else:
                logger.warning(f"Model '{model_config.alias}' references unknown provider '{model_config.provider}'")

    def get_provider_for_model(self, alias: str) -> Optional[Tuple[Provider, str]]:
        """Get the provider that supports the given alias and return the resolved model ID."""
        mapping = self.alias_mappings.get(alias)
        if not mapping:
            return None

        provider = self.providers.get(mapping.provider_name)
        return (provider, mapping.model_id) if provider else None

    def get_provider_by_name(self, name: str) -> Optional[Provider]:
        """Get provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List all provider names."""
        return list(self.providers.keys())

    def list_models(self) -> List[str]:
        """List all supported model aliases."""
        return list(self.alias_mappings.keys())

    async def close_all(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()

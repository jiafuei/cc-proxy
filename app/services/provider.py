"""Enhanced provider system for the simplified architecture."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Request

from app.common.dumper import Dumper, DumpHandles
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.config.user_models import ModelConfig, ProviderConfig
from app.services.transformer_loader import TransformerLoader
from app.services.transformers import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


@dataclass
class ModelMapping:
    """Maps model alias to provider name and actual model ID."""

    provider_name: str
    model_id: str


class Provider:
    """Handles requests to a specific API provider with transformer support."""

    def __init__(self, config: ProviderConfig, transformer_loader: TransformerLoader):
        self.config = config
        self.name = config.name
        self.transformer_loader = transformer_loader
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout), http2=True)

        # Load transformers
        self.request_transformers: List[RequestTransformer] = []
        self.response_transformers: List[ResponseTransformer] = []
        self._load_transformers()

    def _load_transformers(self):
        """Load request and response transformers from configuration."""
        try:
            request_transformers = self.config.transformers.get('request', [])
            response_transformers = self.config.transformers.get('response', [])

            self.request_transformers = self.transformer_loader.load_transformers(request_transformers)
            self.response_transformers = self.transformer_loader.load_transformers(response_transformers)

            logger.info(f"Provider '{self.config.name}' loaded {len(self.request_transformers)} request transformers and {len(self.response_transformers)} response transformers")
        except Exception as e:
            logger.error(f"Failed to load transformers for provider '{self.config.name}': {e}")
            # Continue with empty transformer lists

    async def process_request(self, request: AnthropicRequest, original_request: Request, routing_key: str, dumper: Dumper, dumper_handles: DumpHandles) -> Dict[str, Any]:
        """Process a request through the provider with transformer support.

        Args:
            request: Claude API request

        Returns:
            JSON response dictionary
        """
        # 1. Convert AnthropicRequest to Dict and apply request transformers sequentially
        current_request = request.to_dict()  # Use to_dict() method
        current_headers = dict(original_request.headers)  # Copy headers
        config = self.config.model_copy()

        logger.debug(f'Before transform, headers={current_headers}')
        for transformer in self.request_transformers:
            transform_params = {
                'request': current_request,
                'headers': current_headers,
                'provider_config': config,
                'original_request': original_request,
                'routing_key': routing_key,
            }
            current_request, current_headers = await transformer.transform(transform_params)

        logger.debug(f'Request transformed, stream={current_request.get("stream", False)}, headers={current_headers}')

        # Dump transformed headers and request after all transformers are applied
        dumper.write_transformed_headers(dumper_handles, current_headers)
        dumper.write_transformed_request(dumper_handles, current_request)

        # Force non-streaming to LLM providers for simplified architecture
        current_request['stream'] = False

        # Always get JSON response from provider
        response = await self._send_request(config, current_request, current_headers)

        # Dump pre-transformed response
        response_text = response.text
        dumper.write_pretransformed_response(dumper_handles, response_text)

        # Apply response transformers to full response
        transformed_response = response.json()
        response_params = {}  # Initialize once outside loop for consistency
        for transformer in self.response_transformers:
            response_params.update(
                {
                    'response': transformed_response,
                    'request': current_request,
                    'final_headers': current_headers,
                    'provider_config': config,
                    'original_request': original_request,
                }
            )
            transformed_response = await transformer.transform_response(response_params)

        # Return JSON response directly
        return transformed_response

    async def _send_request(self, config: ProviderConfig, request_data: Dict[str, Any], headers: Dict[str, Any]) -> httpx.Response:
        """Make a non-streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug(f'Non-streaming request to {config.url}')

        response = await self.http_client.post(config.url, json=request_data, headers=final_headers)
        try:
            response.raise_for_status()
        except:
            await response.aread()
            raise

        return response

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

"""Enhanced provider system for the simplified architecture."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Request

from app.common.dumper import Dumper, DumpHandles
from app.common.models import AnthropicRequest
from app.config.log import get_logger
from app.config.user_models import ModelConfig, ProviderConfig
from app.services.capabilities import MessagesCapability, OpenAITokenCountCapability, ProviderCapability, TokenCountCapability, UnsupportedOperationError
from app.services.transformer_loader import TransformerLoader
from app.services.transformers import RequestTransformer, ResponseTransformer

logger = get_logger(__name__)


@dataclass
class ModelMapping:
    """Maps model alias to provider name and actual model ID."""

    provider_name: str
    model_id: str


class Provider:
    """Handles requests to a specific API provider with transformer and capability support."""

    def __init__(self, config: ProviderConfig, transformer_loader: TransformerLoader):
        self.config = config
        self.name = config.name
        self.transformer_loader = transformer_loader
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout), http2=True)

        # Load transformers
        self.request_transformers: List[RequestTransformer] = []
        self.response_transformers: List[ResponseTransformer] = []
        self._load_transformers()

        # Load capabilities
        self.capabilities: Dict[str, ProviderCapability] = {}
        self._load_capabilities()

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

    def _load_capabilities(self):
        """Load capabilities from explicit configuration."""
        try:
            if not self.config.capabilities:
                raise ValueError(f"Provider '{self.config.name}' has no capabilities configured. All providers must explicitly define their supported operations.")

            for cap_config in self.config.capabilities:
                capability_class = self._resolve_capability_class(cap_config.class_name)
                capability = capability_class(**cap_config.params)
                self.capabilities[cap_config.operation] = capability

            logger.info(f"Provider '{self.config.name}' loaded {len(self.capabilities)} capabilities: {list(self.capabilities.keys())}")
        except Exception as e:
            logger.error(f"Failed to load capabilities for provider '{self.config.name}': {e}")
            raise  # Fail fast - don't continue with empty capabilities

    def _resolve_capability_class(self, class_name: str):
        """Resolve capability class from name."""
        # Built-in capability classes
        capability_map = {
            'MessagesCapability': MessagesCapability,
            'TokenCountCapability': TokenCountCapability,
            'OpenAITokenCountCapability': OpenAITokenCountCapability,
        }

        if class_name in capability_map:
            return capability_map[class_name]
        else:
            # Support custom capability classes via dynamic import
            return self._import_capability_class(class_name)

    def _import_capability_class(self, class_name: str):
        """Import capability class from module path."""
        try:
            # Handle both simple names and full module paths
            if '.' in class_name:
                # Full module path like 'my_module.CustomCapability'
                module_path, class_name_only = class_name.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name_only])
                return getattr(module, class_name_only)
            else:
                # Simple name - try to find in capabilities package
                from app.services import capabilities

                if hasattr(capabilities, class_name):
                    return getattr(capabilities, class_name)
                else:
                    raise ImportError(f"Capability class '{class_name}' not found in built-in capabilities or as module path")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not import capability class '{class_name}': {e}")

    async def process_operation(
        self, operation: str, request: AnthropicRequest, original_request: Request, routing_key: str, dumper: Dumper, dumper_handles: DumpHandles
    ) -> Dict[str, Any]:
        """Process a request through the provider using the specified operation capability.

        This method provides the new capability-based processing pipeline:
        1. Get capability for the operation
        2. Capability prepares request (modifies config/URL if needed)
        3. Apply transformers (unchanged)
        4. Send request to provider
        5. Capability processes response

        Args:
            operation: Operation name ('messages', 'count_tokens', etc.)
            request: Claude API request
            original_request: Original FastAPI request
            routing_key: Routing key from router
            dumper: Dumper for logging
            dumper_handles: Dump handles

        Returns:
            JSON response dictionary

        Raises:
            UnsupportedOperationError: If operation is not supported by this provider
        """
        # Check if provider supports this operation
        if operation not in self.capabilities:
            raise UnsupportedOperationError(operation, self.config.name)

        capability = self.capabilities[operation]
        logger.debug(f'Processing {operation} operation for provider {self.config.name}')

        # Convert AnthropicRequest to Dict
        current_request = request.to_dict()
        current_headers = dict(original_request.headers)

        # Create context for capability and transformers
        context = {
            'original_request': original_request,
            'routing_key': routing_key,
            'headers': current_headers,
            'operation': operation,
        }

        # Phase 1: Capability prepares request (modifies config if needed)
        modified_config = await capability.prepare_request(current_request, self.config, context)

        # Force non-streaming to LLM providers for simplified architecture
        current_request['stream'] = False

        # Phase 2: Apply request transformers (unchanged from existing logic)
        logger.debug('Request headers prepared')
        for transformer in self.request_transformers:
            transform_params = {
                'request': current_request,
                'headers': current_headers,
                'provider_config': modified_config,
                'original_request': original_request,
                'routing_key': routing_key,
            }
            current_request, current_headers = await transformer.transform(transform_params)

        logger.debug('Request transformers applied')

        # Dump transformed headers and request after all transformers are applied
        dumper.write_transformed_headers(dumper_handles, current_headers)
        dumper.write_transformed_request(dumper_handles, current_request)

        # Phase 3: Send request using modified config
        response = await self._send_request(modified_config, current_request, current_headers)

        # Dump pre-transformed response
        response_text = response.text
        dumper.write_pretransformed_response(dumper_handles, response_text)

        # Phase 4: Apply response transformers (unchanged from existing logic)
        try:
            transformed_response = response.json()
        except Exception:
            logger.error(
                'Unable to convert response body to JSON',
                response_length=len(response_text),
            )
            raise

        response_params = {}  # Initialize once outside loop for consistency
        for transformer in self.response_transformers:
            response_params.update(
                {
                    'response': transformed_response,
                    'request': current_request,
                    'final_headers': current_headers,
                    'provider_config': modified_config,
                    'original_request': original_request,
                }
            )
            transformed_response = await transformer.transform_response(response_params)

        # Phase 5: Capability processes final response
        final_response = await capability.process_response(transformed_response, current_request, context)

        return final_response

    def supports_operation(self, operation: str) -> bool:
        """Check if provider supports a specific operation.

        Args:
            operation: Operation name to check

        Returns:
            True if operation is supported
        """
        return operation in self.capabilities

    def list_operations(self) -> List[str]:
        """List all operations supported by this provider.

        Returns:
            List of operation names
        """
        return list(self.capabilities.keys())

    async def _send_request(self, config: ProviderConfig, request_data: Dict[str, Any], headers: Dict[str, Any]) -> httpx.Response:
        """Make a non-streaming HTTP request to the provider."""
        # Use headers from transformers (which may include auth)
        final_headers = headers

        logger.debug('Sending request to provider')

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

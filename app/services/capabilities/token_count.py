"""Token count capability - handles token counting operations."""

from typing import Any, Dict

from app.config.log import get_logger
from app.config.user_models import ProviderConfig
from app.services.capabilities.interfaces import ProviderCapability

logger = get_logger(__name__)


class TokenCountCapability(ProviderCapability):
    """Capability for token counting operations.

    This capability handles the URL modification needed for token counting endpoints.
    Instead of manually appending '/count_tokens' in the router, this capability
    provides a clean abstraction that can be customized per provider.
    """

    def get_operation_name(self) -> str:
        """Get the operation name."""
        return 'count_tokens'

    async def prepare_request(self, request: Dict[str, Any], config: ProviderConfig, context: Dict[str, Any]) -> ProviderConfig:
        """Prepare request for token counting operation.

        Modifies the provider URL to point to the count_tokens endpoint.
        For Anthropic-compatible providers, this appends '/count_tokens' to the URL.

        Args:
            request: Request data dictionary
            config: Provider configuration
            context: Additional context

        Returns:
            Modified provider config with count_tokens URL
        """
        logger.debug(f'Preparing count_tokens request for provider: {config.name}')

        # Create a copy of the config to avoid modifying the original
        modified_config = config.model_copy()

        # Modify URL for count_tokens endpoint
        if not modified_config.url.endswith('/count_tokens'):
            modified_config.url = modified_config.url.rstrip('/') + '/count_tokens'
            logger.debug(f'Modified URL for count_tokens: {modified_config.url}')

        return modified_config

    async def process_response(self, response: Dict[str, Any], request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process response from token counting operation.

        For count_tokens, we expect a response with token count information.
        This method could be extended to validate the response format or
        standardize responses across different providers.

        Args:
            response: Response data dictionary (after transformers)
            request: Original request data dictionary
            context: Additional context

        Returns:
            Processed response dictionary
        """
        logger.debug('Processing count_tokens response')

        # Validate that response contains expected fields
        expected_fields = ['input_tokens']
        missing_fields = [field for field in expected_fields if field not in response]

        if missing_fields:
            logger.warning(f'count_tokens response missing expected fields: {missing_fields}')

        return response

    def supports_provider(self, provider_name: str) -> bool:
        """Check if this capability supports a specific provider.

        Currently, we assume all providers support count_tokens by URL modification.
        This could be overridden for providers that need different approaches.

        Args:
            provider_name: Name of the provider

        Returns:
            True if capability supports the provider
        """
        # For now, assume all providers support count_tokens via URL modification
        # This could be made more sophisticated based on provider capabilities
        return True


class OpenAITokenCountCapability(TokenCountCapability):
    """Token count capability for OpenAI provider.

    OpenAI doesn't have a direct count_tokens endpoint, so this capability
    could implement client-side token counting using tiktoken or return
    an appropriate error message.
    """

    def supports_provider(self, provider_name: str) -> bool:
        """OpenAI providers need special handling."""
        return 'openai' in provider_name.lower()

    async def prepare_request(self, request: Dict[str, Any], config: ProviderConfig, context: Dict[str, Any]) -> ProviderConfig:
        """OpenAI doesn't support count_tokens endpoint."""
        logger.warning(f"OpenAI provider '{config.name}' doesn't support native count_tokens endpoint")

        # For now, we'll still try the URL modification approach
        # In the future, this could implement client-side counting with tiktoken
        return await super().prepare_request(request, config, context)

    async def process_response(self, response: Dict[str, Any], request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenAI response or provide fallback."""
        # If we get here, it means the provider actually responded
        # This could be enhanced to handle OpenAI-specific response formats
        return await super().process_response(response, request, context)

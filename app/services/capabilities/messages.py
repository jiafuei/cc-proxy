"""Messages capability - handles standard message processing operations."""

from typing import Any, Dict

from app.config.log import get_logger
from app.config.user_models import ProviderConfig
from app.services.capabilities.interfaces import ProviderCapability

logger = get_logger(__name__)


class MessagesCapability(ProviderCapability):
    """Capability for standard message processing operations.

    This is essentially a passthrough capability that maintains existing behavior
    for the main /v1/messages endpoint. It provides a consistent interface while
    making no modifications to requests, configs, or responses.
    """

    def get_operation_name(self) -> str:
        """Get the operation name."""
        return 'messages'

    async def prepare_request(self, request: Dict[str, Any], config: ProviderConfig, context: Dict[str, Any]) -> ProviderConfig:
        """Prepare request for messages operation.

        For messages, we don't need to modify the config - the URL and settings
        are already correct. This is a pure passthrough.

        Args:
            request: Request data dictionary
            config: Provider configuration
            context: Additional context

        Returns:
            Unmodified provider config
        """
        logger.debug(f'Preparing messages request for provider: {config.name}')
        return config

    async def process_response(self, response: Dict[str, Any], request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process response from messages operation.

        For messages, the transformers have already handled all necessary
        response processing. This is a pure passthrough.

        Args:
            response: Response data dictionary (after transformers)
            request: Original request data dictionary
            context: Additional context

        Returns:
            Unmodified response dictionary
        """
        logger.debug('Processing messages response - passthrough')
        return response

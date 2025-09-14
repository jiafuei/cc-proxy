"""Provider capability interfaces for operation-specific request handling."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.config.user_models import ProviderConfig


class ProviderCapability(ABC):
    """Base class for provider capabilities that handle operation-specific logic.

    Capabilities work alongside transformers to provide clean separation of concerns:
    - Capabilities handle operation-specific logic (URL modification, request preparation)
    - Transformers handle provider-specific logic (auth, format conversion)

    Pipeline: Capability.prepare_request() → Transformers → _send_request() → Capability.process_response()
    """

    @abstractmethod
    def get_operation_name(self) -> str:
        """Get the operation name this capability handles.

        Returns:
            Operation name (e.g., 'messages', 'count_tokens', 'embeddings')
        """
        pass

    @abstractmethod
    async def prepare_request(self, request: Dict[str, Any], config: ProviderConfig, context: Dict[str, Any]) -> ProviderConfig:
        """Prepare the request and modify provider config if needed.

        This is called BEFORE transformers run, allowing capabilities to:
        - Modify the provider URL (e.g., append '/count_tokens')
        - Adjust timeouts or other config for the operation
        - Validate operation-specific requirements

        Args:
            request: Request data dictionary
            config: Provider configuration (will be copied, not modified)
            context: Additional context including headers, routing info

        Returns:
            Modified provider config for this operation
        """
        pass

    @abstractmethod
    async def process_response(self, response: Dict[str, Any], request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the response after transformers have run.

        This is called AFTER response transformers, allowing capabilities to:
        - Validate operation-specific response format
        - Add operation-specific metadata
        - Handle operation-specific error cases

        Args:
            response: Response data dictionary (after transformers)
            request: Original request data dictionary
            context: Additional context including headers, routing info

        Returns:
            Processed response dictionary
        """
        pass

    def supports_provider(self, provider_name: str) -> bool:
        """Check if this capability supports a specific provider.

        Override this method if the capability has provider-specific requirements.

        Args:
            provider_name: Name of the provider

        Returns:
            True if capability supports the provider
        """
        return True


class UnsupportedOperationError(Exception):
    """Raised when a provider doesn't support a requested operation."""

    def __init__(self, operation: str, provider_name: str, message: Optional[str] = None):
        self.operation = operation
        self.provider_name = provider_name
        self.message = message or f"Provider '{provider_name}' doesn't support operation '{operation}'"
        super().__init__(self.message)

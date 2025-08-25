"""Enhanced transformer interfaces for the simplified architecture."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Tuple

from fastapi import Request

from app.config.user_models import ProviderConfig


class RequestTransformer(ABC):
    """Interface for transformers that modify outgoing requests.

    Request transformers can:
    - Modify request content (messages, model, temperature, etc.)
    - Add authentication headers
    - Change the stream flag
    - Convert between different provider formats
    """

    @abstractmethod
    async def transform(self, request: Dict[str, Any], headers: Mapping[str, Any], provider_config: ProviderConfig, original_request: Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Transform the outgoing request and headers.

        Args:
            request: Request data as dictionary
            headers: Current headers to be modified
            provider_config: The current provider config
            original_request: The original request object

        Returns:
            Tuple of (transformed_request, updated_headers)

        Note:
            Transformers can modify the stream flag, which can also affect how
            the provider processes the request (streaming vs non-streaming).
        """
        pass


class ResponseTransformer(ABC):
    """Interface for transformers that modify incoming responses.

    Response transformers handle both streaming and non-streaming responses
    to provide flexibility for different provider capabilities.
    """

    @abstractmethod
    async def transform_chunk(self, chunk: bytes) -> bytes:
        """Transform a streaming response chunk.

        Args:
            chunk: Raw bytes from streaming response

        Returns:
            Transformed chunk bytes

        Note:
            This is called for each chunk in a streaming response.
            The chunk typically contains SSE-formatted data.
        """
        pass

    @abstractmethod
    async def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a complete non-streaming response.

        Args:
            response: Full response dictionary from provider

        Returns:
            Transformed response dictionary

        Note:
            This is called for complete responses from non-streaming providers.
            The response will later be converted to SSE format.
        """
        pass

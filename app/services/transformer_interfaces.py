"""Enhanced transformer interfaces for the simplified architecture."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from app.common.models import ClaudeRequest


class RequestTransformer(ABC):
    """Interface for transformers that modify outgoing requests.

    Request transformers can:
    - Modify request content (messages, model, temperature, etc.)
    - Add authentication headers
    - Change the stream flag
    - Set provider-specific URLs and parameters
    """

    @abstractmethod
    async def transform(self, request: ClaudeRequest) -> ClaudeRequest:
        """Transform the outgoing request.

        Args:
            request: Claude API request to transform

        Returns:
            Transformed Claude request

        Note:
            Transformers can modify the stream flag, which affects how
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


class AnthropicAuthTransformer(RequestTransformer):
    """Built-in transformer for Anthropic API authentication."""

    def __init__(self, api_key: str, base_url: str = 'https://api.anthropic.com/v1/messages'):
        """Initialize with API credentials.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for Anthropic API
        """
        self.api_key = api_key
        self.base_url = base_url

    async def transform(self, request: ClaudeRequest) -> ClaudeRequest:
        """Add Anthropic authentication and URL."""
        # Note: In the new architecture, we'll add auth info to the request object
        # The provider will use this info when making HTTP calls

        # For now, just return the request as-is since ClaudeRequest doesn't have auth fields
        # The provider will handle auth based on its config
        return request


class AnthropicResponseTransformer(ResponseTransformer):
    """Built-in transformer for Anthropic API responses."""

    async def transform_chunk(self, chunk: bytes) -> bytes:
        """Pass through Anthropic streaming chunks as-is."""
        return chunk

    async def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through Anthropic non-streaming responses as-is."""
        return response


class OpenAIRequestTransformer(RequestTransformer):
    """Built-in transformer to convert Claude format to OpenAI format."""

    def __init__(self, api_key: str, base_url: str = 'https://api.openai.com/v1/chat/completions'):
        self.api_key = api_key
        self.base_url = base_url

    async def transform(self, request: ClaudeRequest) -> ClaudeRequest:
        """Convert Claude request format to OpenAI-compatible format.

        Note: This is a simplified example. In practice, this would need
        to handle the format differences between Claude and OpenAI APIs.
        """
        # For now, just return as-is since we're keeping ClaudeRequest format
        # In the provider, we'll handle the actual format conversion
        return request


class OpenAIResponseTransformer(ResponseTransformer):
    """Built-in transformer to convert OpenAI responses to Claude format."""

    async def transform_chunk(self, chunk: bytes) -> bytes:
        """Convert OpenAI streaming chunk to Claude format."""
        # Simplified pass-through for now
        # In practice, this would convert OpenAI SSE format to Claude SSE format
        return chunk

    async def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI response to Claude format."""
        # Simplified pass-through for now
        # In practice, this would convert OpenAI response structure to Claude format
        return response


# Example of how users would create external transformers:
#
# File: /path/to/user/my_transformers.py
#
# from app.services.transformers import RequestTransformer, ResponseTransformer
# from app.common.models import ClaudeRequest
#
# class CustomAuthTransformer(RequestTransformer):
#     def __init__(self, custom_token: str):
#         self.custom_token = custom_token
#
#     async def transform(self, request: ClaudeRequest) -> ClaudeRequest:
#         # Add custom authentication logic
#         # Maybe modify request.model to add auth suffix
#         return request
#
# class ResponseFilterTransformer(ResponseTransformer):
#     def __init__(self, filter_words: list):
#         self.filter_words = filter_words
#
#     async def transform_chunk(self, chunk: bytes) -> bytes:
#         # Filter out unwanted words from streaming response
#         chunk_str = chunk.decode('utf-8')
#         for word in self.filter_words:
#             chunk_str = chunk_str.replace(word, '***')
#         return chunk_str.encode('utf-8')
#
#     async def transform_response(self, response: dict) -> dict:
#         # Filter out unwanted words from full response
#         # ... filtering logic ...
#         return response

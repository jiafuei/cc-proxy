"""Enhanced transformer interfaces for the simplified architecture."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Tuple


class RequestTransformer(ABC):
    """Interface for transformers that modify outgoing requests.

    Request transformers can:
    - Modify request content (messages, model, temperature, etc.)
    - Add authentication headers
    - Change the stream flag
    - Convert between different provider formats

    A `logger` instance can be accessed using self.logger
    """

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def _is_builtin_tool(self, tool: dict) -> bool:
        """Detect built-in tool by presence of 'type' without 'input_schema'.

        Args:
            tool: Tool dictionary to check

        Returns:
            True if this is a built-in tool, False otherwise
        """
        return isinstance(tool, dict) and 'type' in tool and 'input_schema' not in tool

    def _is_websearch_tool(self, tool: dict) -> bool:
        """Check if tool is a WebSearch built-in tool.

        Args:
            tool: Tool dictionary to check

        Returns:
            True if this is a WebSearch tool, False otherwise
        """
        return self._is_builtin_tool(tool) and tool.get('name') == 'web_search' and tool.get('type', '').startswith('web_search')

    def _has_builtin_tools(self, tools: list) -> bool:
        """Check if tools list contains any built-in tools.

        Args:
            tools: List of tool dictionaries to check

        Returns:
            True if any built-in tools are found, False otherwise
        """
        return any(self._is_builtin_tool(tool) for tool in tools if isinstance(tool, dict))

    @abstractmethod
    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Transform the outgoing request and headers.

        Args:
            params: Dictionary containing:
                - request: Request data as dictionary `dict`
                - headers: Current headers to be modified `dict`
                - provider_config: The current provider config `ProviderConfig`
                - original_request: The original request object `fastapi.Request`
                - routing_key: The routing key `str`

        Returns:
            Tuple of (transformed_request, updated_headers) `Tuple[dict,dict]

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

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    @abstractmethod
    async def transform_chunk(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        """Transform a streaming response chunk.

        Args:
            params: Dictionary containing:
                - chunk: Raw bytes from streaming response `bytes`
                - request: Request data as dictionary `dict`
                - final_headers: Final headers after request transformation `dict`
                - provider_config: The current provider config `ProviderConfig`
                - original_request: The original request  object `fastapi.Request`

        Yields:
            Transformed chunk bytes

        Note:
            This is called for each chunk in a streaming response.
            The chunk typically contains SSE-formatted data.
            Implementations should yield bytes, allowing for multiple output chunks per input.
        """
        pass

    @abstractmethod
    async def transform_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform a complete non-streaming response.

        Args:
            params: Dictionary containing:
                - response: Full response dictionary from provider `dict`
                - request: Request data as dictionary `dict`
                - final_headers: Final headers after request transformation `dict`
                - provider_config: The current provider config `ProviderConfig`
                - original_request: The original request object `fastapi.Request`

        Returns:
            Transformed response dictionary

        Note:
            This is called for complete responses from non-streaming providers.
            The response will later be converted to SSE format.
        """
        pass

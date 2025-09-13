"""Transformers for built-in tools (WebSearch, WebFetch, etc.) across providers."""

import hashlib
from abc import ABC
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer


class BuiltinToolsRequestTransformer(RequestTransformer, ABC):
    """Base transformer for handling built-in tools across providers.

    Built-in tools are identified by having a 'type' field but no 'input_schema'.
    This distinguishes them from regular tool definitions which have input_schema.
    """

    def __init__(self, logger, provider_type: Optional[str] = None):
        """Initialize transformer with optional provider type.

        Args:
            logger: Logger instance
            provider_type: Target provider type ('anthropic', 'openai', 'gemini')
        """
        super().__init__(logger)
        self.provider_type = provider_type

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

    def _detect_builtin_tools(self, request: dict) -> List[dict]:
        """Extract all built-in tools from request.

        Args:
            request: Request dictionary

        Returns:
            List of built-in tool dictionaries
        """
        tools = request.get('tools', [])
        return [tool for tool in tools if self._is_builtin_tool(tool)]

    def _separate_tools(self, tools: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Separate built-in tools from regular tools.

        Args:
            tools: List of all tools

        Returns:
            Tuple of (builtin_tools, regular_tools)
        """
        builtin_tools = []
        regular_tools = []

        for tool in tools:
            if self._is_builtin_tool(tool):
                builtin_tools.append(tool)
            else:
                regular_tools.append(tool)

        return builtin_tools, regular_tools


class OpenAIBuiltinToolsTransformer(BuiltinToolsRequestTransformer):
    """Transformer for converting built-in tools to OpenAI format."""

    # Model mapping for web search compatibility
    SEARCH_MODEL_MAPPING = {
        'gpt-4o': 'gpt-4o-search-preview',
        'gpt-4o-mini': 'gpt-4o-mini-search-preview',
    }

    def __init__(self, logger):
        """Initialize OpenAI built-in tools transformer."""
        super().__init__(logger, provider_type='openai')

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Transform Anthropic built-in tools to OpenAI format.

        Args:
            params: Transform parameters containing request, headers, etc.

        Returns:
            Tuple of (transformed_request, headers)
        """
        request = params['request']
        headers = params['headers']

        tools = request.get('tools', [])
        if not tools:
            return request, headers

        # Separate built-in from regular tools
        builtin_tools, regular_tools = self._separate_tools(tools)

        if not builtin_tools:
            return request, headers

        # Process WebSearch tools
        web_search_config = None
        for tool in builtin_tools:
            if self._is_websearch_tool(tool):
                web_search_config = self._extract_websearch_config(tool)
                self.logger.debug(f'Extracted WebSearch config: {web_search_config}')
                break

        # Apply transformations
        if web_search_config:
            # Add web_search_options to request
            request['web_search_options'] = web_search_config

            # Ensure model supports web search
            original_model = request.get('model', '')
            search_model = self._ensure_search_model(original_model)
            if search_model != original_model:
                request['model'] = search_model
                self.logger.debug(f'Converted model {original_model} -> {search_model}')

        # Update tools array (remove built-in tools, keep regular tools)
        if regular_tools:
            request['tools'] = regular_tools
        else:
            # Remove tools key if no regular tools remain
            request.pop('tools', None)

        self.logger.info(f'Transformed {len(builtin_tools)} built-in tools for OpenAI')
        return request, headers

    def _extract_websearch_config(self, tool: dict) -> dict:
        """Convert Anthropic WebSearch tool to OpenAI web_search_options format.

        Args:
            tool: Anthropic WebSearch tool dictionary

        Returns:
            OpenAI web_search_options configuration
        """
        config = {}

        # Validate domain filter constraints
        self._validate_domain_filters(tool)

        # Domain filtering
        allowed_domains = tool.get('allowed_domains')
        blocked_domains = tool.get('blocked_domains')

        if allowed_domains or blocked_domains:
            filters = {}
            if allowed_domains:
                filters['allowed_domains'] = allowed_domains
            if blocked_domains:
                filters['blocked_domains'] = blocked_domains
            config['filters'] = filters

        # User location
        if user_location := tool.get('user_location'):
            config['user_location'] = self._convert_user_location(user_location)

        # Search context size (default to medium)
        config['search_context_size'] = 'medium'

        return self._handle_missing_parameters(config)

    def _convert_user_location(self, user_location: dict) -> dict:
        """Convert Anthropic user location to OpenAI format.

        Args:
            user_location: Anthropic user location dictionary

        Returns:
            OpenAI user location format
        """
        openai_location = {'type': 'approximate', 'approximate': {}}

        # Map fields that exist
        approximate = openai_location['approximate']
        for field in ['country', 'city', 'region', 'timezone']:
            if value := user_location.get(field):
                approximate[field] = value

        return openai_location

    def _ensure_search_model(self, model: str) -> str:
        """Convert model to search-preview variant if needed.

        Args:
            model: Original model name

        Returns:
            Search-compatible model name
        """
        # Direct mapping for known models
        if model in self.SEARCH_MODEL_MAPPING:
            return self.SEARCH_MODEL_MAPPING[model]

        # Already a search model
        if 'search-preview' in model:
            return model

        # Warn about potential incompatibility
        self._validate_model_compatibility(model)
        return model

    def _validate_domain_filters(self, tool: dict):
        """Validate domain filter constraints.

        Args:
            tool: WebSearch tool to validate

        Raises:
            ValueError: If both allowed_domains and blocked_domains are specified
        """
        if tool.get('allowed_domains') and tool.get('blocked_domains'):
            raise ValueError('Cannot use both allowed_domains and blocked_domains in WebSearch')

    def _handle_missing_parameters(self, config: dict) -> dict:
        """Apply defaults for missing parameters.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with defaults applied
        """
        # Ensure search_context_size is set
        config.setdefault('search_context_size', 'medium')

        # Ensure filters exists if not set
        config.setdefault('filters', {})

        return config

    def _validate_model_compatibility(self, model: str):
        """Check if model supports web search and log warning if uncertain.

        Args:
            model: Model name to check
        """
        supported_prefixes = ['gpt-4o', 'gpt-4o-mini']

        if not any(model.startswith(prefix) for prefix in supported_prefixes):
            self.logger.warning(f"Model '{model}' may not support web search. Consider using: {list(self.SEARCH_MODEL_MAPPING.keys())}")


class AnthropicBuiltinToolsTransformer(BuiltinToolsRequestTransformer):
    """Passthrough transformer for Anthropic provider.

    Anthropic natively supports built-in tools, so no transformation is needed.
    """

    def __init__(self, logger):
        """Initialize Anthropic built-in tools transformer."""
        super().__init__(logger, provider_type='anthropic')

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Passthrough transformation for Anthropic provider.

        Args:
            params: Transform parameters

        Returns:
            Unchanged request and headers
        """
        request = params['request']
        headers = params['headers']

        # Count built-in tools for logging
        builtin_tools = self._detect_builtin_tools(request)
        if builtin_tools:
            self.logger.debug(f'Passing through {len(builtin_tools)} built-in tools for Anthropic')

        return request, headers


class SmartBuiltinToolsTransformer(BuiltinToolsRequestTransformer):
    """Auto-detecting transformer that selects appropriate handler based on provider.

    This transformer automatically detects the target provider and applies
    the appropriate transformation for built-in tools.
    """

    def __init__(self, logger):
        """Initialize smart built-in tools transformer."""
        super().__init__(logger)

        # Initialize provider-specific transformers
        self.transformers = {
            'anthropic': AnthropicBuiltinToolsTransformer(logger),
            'openai': OpenAIBuiltinToolsTransformer(logger),
        }

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Auto-detect provider and apply appropriate transformation.

        Args:
            params: Transform parameters including provider_config

        Returns:
            Transformed request and headers
        """
        provider_config = params.get('provider_config')
        if not provider_config:
            self.logger.warning('No provider config found, using passthrough')
            return params['request'], params['headers']

        # Detect provider type
        provider_type = self._detect_provider_type(provider_config)
        self.logger.debug(f'Detected provider type: {provider_type}')

        # Get appropriate transformer
        if transformer := self.transformers.get(provider_type):
            return await transformer.transform(params)

        # Default passthrough for unknown providers
        self.logger.info(f'Unknown provider type "{provider_type}", using passthrough')
        return params['request'], params['headers']

    def _detect_provider_type(self, config) -> str:
        """Detect provider type from configuration.

        Args:
            config: Provider configuration object

        Returns:
            Provider type string ('anthropic', 'openai', 'unknown')
        """
        base_url = getattr(config, 'base_url', '')

        if 'api.anthropic.com' in base_url:
            return 'anthropic'
        elif 'api.openai.com' in base_url:
            return 'openai'
        elif 'generativelanguage.googleapis.com' in base_url:
            return 'gemini'

        return 'unknown'


class OpenAIBuiltinToolsResponseTransformer(ResponseTransformer):
    """Response transformer for converting OpenAI web search annotations back to Anthropic format."""

    def __init__(self, logger):
        """Initialize OpenAI built-in tools response transformer."""
        self.logger = logger

    async def transform_chunk(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        """Transform streaming response chunk.

        For now, pass through chunks as-is. Full streaming annotation support
        can be added later if needed.

        Args:
            params: Transform parameters

        Yields:
            Transformed chunk bytes
        """
        # TODO: Add streaming annotation handling
        yield params['chunk']

    async def transform_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI response with annotations to Anthropic format.

        Args:
            params: Transform parameters

        Returns:
            Transformed response dictionary
        """
        response = params['response']
        request = params['request']

        # Only process if this was a web search request
        if 'web_search_options' not in request:
            return response

        # Check for annotations to convert
        if annotations := response.get('annotations'):
            response = self._convert_annotations_to_anthropic(response, annotations)
            self.logger.debug(f'Converted {len(annotations)} OpenAI annotations to Anthropic format')

        return response

    def _convert_annotations_to_anthropic(self, response: dict, annotations: list) -> dict:
        """Convert OpenAI url_citation annotations to Anthropic web_search_tool_result format.

        Args:
            response: OpenAI response dictionary
            annotations: List of annotation objects

        Returns:
            Response with converted annotations
        """
        # Extract message content for snippet extraction
        choices = response.get('choices', [])
        if not choices:
            return response

        message = choices[0].get('message', {})
        content = message.get('content', '')

        # Convert each url_citation annotation
        tool_results = []
        for annotation in annotations:
            if annotation.get('type') == 'url_citation':
                citation = annotation.get('url_citation', {})
                tool_result = self._create_web_search_result(citation, content)
                if tool_result:
                    tool_results.append(tool_result)

        # Add tool results to response if any were created
        if tool_results:
            # Add to message content as tool_use blocks
            message_content = message.get('content', [])
            if isinstance(message_content, str):
                message_content = [{'type': 'text', 'text': message_content}]
            elif not isinstance(message_content, list):
                message_content = []

            # Add tool results
            message_content.extend(tool_results)
            message['content'] = message_content

        return response

    def _create_web_search_result(self, citation: dict, content: str) -> Optional[dict]:
        """Create Anthropic web_search_tool_result from OpenAI citation.

        Args:
            citation: OpenAI url_citation object
            content: Full response content for snippet extraction

        Returns:
            Anthropic web_search_tool_result block or None if invalid
        """
        url = citation.get('url')
        title = citation.get('title')

        if not url:
            return None

        # Generate deterministic ID from URL
        result_id = f'search_{hashlib.md5(url.encode()).hexdigest()[:8]}'

        # Extract snippet from content using indices if available
        snippet = self._extract_snippet(content, citation.get('start_index'), citation.get('end_index'))

        return {'type': 'web_search_tool_result', 'id': result_id, 'content': {'type': 'web_search_result', 'url': url, 'title': title or 'Untitled', 'snippet': snippet or ''}}

    def _extract_snippet(self, content: str, start_index: Optional[int], end_index: Optional[int]) -> str:
        """Extract snippet from content using citation indices.

        Args:
            content: Full content string
            start_index: Start index of citation
            end_index: End index of citation

        Returns:
            Extracted snippet or empty string
        """
        if start_index is not None and end_index is not None:
            try:
                return content[start_index:end_index]
            except (IndexError, TypeError):
                self.logger.warning(f'Invalid citation indices: {start_index}-{end_index}')

        return ''

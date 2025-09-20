"""Generic transformers for common operations."""

import copy
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from jsonpath_ng import parse

from app.config.user_models import ProviderConfig
from app.transformers.interfaces import ProviderRequestTransformer


class UrlPathTransformer(ProviderRequestTransformer):
    """Generic URL path transformer that modifies provider config URL.

    Strips trailing slashes from the base URL and appends a user-defined path.
    """

    def __init__(self, logger, path: str):
        """Initialize transformer.

        Args:
            logger: Logger instance
            path: Path to append to the base URL (e.g., '/v1/chat/completions')
        """
        super().__init__(logger)
        self.path = path

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Modify provider config URL by appending configured path."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        if 'provider_config' in params:
            provider_config: ProviderConfig = params['provider_config']
            base_url = provider_config.base_url.rstrip('/')
            path = self.path if self.path.startswith('/') or not self.path else '/' + self.path
            provider_config.base_url = base_url + path

        return request, headers


class HeaderTransformer(ProviderRequestTransformer):
    """Generic header transformer that performs multiple header operations.

    Supports an array of operations, each with 'set' or 'delete' operations.
    """

    def __init__(self, logger, operations: List[Dict[str, Any]]):
        """Initialize transformer.

        Args:
            logger: Logger instance
            operations: List of operation dictionaries, each containing:
                - key: Header name/key to operate on (required)
                - op: Operation to perform - 'set' or 'delete' (default: 'set')
                - value: Header value for 'set' operation (required for 'set')
                - prefix: Text to prepend to value (default: '', only for 'set')
                - suffix: Text to append to value (default: '', only for 'set')
        """
        super().__init__(logger)
        self.operations = operations or []

        if not self.operations:
            raise ValueError("'operations' parameter is required and must contain at least one operation")

        # Validate each operation
        for i, op in enumerate(self.operations):
            self._validate_operation(op, i)

    def _validate_operation(self, operation: Dict[str, Any], index: int) -> None:
        """Validate a single operation.

        Args:
            operation: Operation dictionary
            index: Index of operation in list (for error messages)
        """
        if not isinstance(operation, dict):
            raise ValueError(f'Operation {index} must be a dictionary')

        # Check required key parameter
        if 'key' not in operation or not operation['key']:
            raise ValueError(f"Operation {index}: 'key' parameter is required")

        # Get operation type, default to 'set'
        op_type = operation.get('op', 'set').lower()
        valid_ops = {'set', 'delete'}
        if op_type not in valid_ops:
            raise ValueError(f"Operation {index}: Invalid operation '{op_type}'. Must be one of: {valid_ops}")

        # For 'set' operations, require value
        if op_type == 'set' and not operation.get('value'):
            raise ValueError(f"Operation {index}: 'value' parameter is required for 'set' operation")

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Perform all header operations."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        # Process each operation
        for operation in self.operations:
            key = operation['key']
            op_type = operation.get('op', 'set').lower()

            if op_type == 'set':
                value = operation['value']
                prefix = operation.get('prefix', '')
                suffix = operation.get('suffix', '')

                # Construct header value with prefix and suffix
                header_value = f'{prefix}{value}{suffix}'
                # Add/modify header
                headers[key] = header_value
                self.logger.debug(f"Set header '{key}' = '{header_value}'")

            elif op_type == 'delete':
                # Remove header if it exists
                if key in headers:
                    del headers[key]
                    self.logger.debug(f"Deleted header '{key}'")
                else:
                    self.logger.debug(f"Header '{key}' not found for deletion")

        return request, headers


class RequestBodyTransformer(ProviderRequestTransformer):
    """Transformer that modifies request body content using JSONPath expressions.

    Supports multiple operations on request body content using JSONPath expressions.
    Each operation contains:
    - key: JSONPath expression (required)
    - value: value to use for operation (required for non-delete operations)
    - op: one of 'set', 'delete', 'append', 'prepend', 'merge' (default: 'set')
    - jsonPath: boolean flag; kept for API compatibility (always uses JSONPath)
    """

    def __init__(self, logger, operations: List[Dict[str, Any]]):
        super().__init__(logger)
        self.operations = operations or []

        if not self.operations:
            raise ValueError("'operations' parameter is required and must contain at least one operation")

        # Validate and pre-compile each operation
        self.compiled_operations = []
        for i, operation in enumerate(self.operations):
            self._validate_operation(operation, i)
            compiled_op = self._compile_operation(operation)
            self.compiled_operations.append(compiled_op)

    def _validate_operation(self, operation: Dict[str, Any], index: int) -> None:
        """Validate a single operation.

        Args:
            operation: Operation dictionary
            index: Index of operation in list (for error messages)
        """
        if not isinstance(operation, dict):
            raise ValueError(f'Operation {index} must be a dictionary')

        # Check required key parameter
        if 'key' not in operation or not operation['key']:
            raise ValueError(f"Operation {index}: 'key' parameter is required")

        # Get operation type, default to 'set'
        op_type = operation.get('op', 'set').lower()
        valid_ops = {'set', 'delete', 'append', 'prepend', 'merge'}
        if op_type not in valid_ops:
            raise ValueError(f"Operation {index}: Invalid operation '{op_type}'. Must be one of: {valid_ops}")

        # For 'set', 'append', 'prepend', 'merge' operations, require value
        if op_type in {'set', 'append', 'prepend', 'merge'} and operation.get('value') is None:
            raise ValueError(f"Operation {index}: 'value' parameter is required for '{op_type}' operation")

    def _compile_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Compile JSONPath expression for an operation.

        Args:
            operation: Operation dictionary

        Returns:
            Compiled operation with JSONPath expression
        """
        compiled_op = operation.copy()
        try:
            compiled_op['expr'] = parse(operation['key'])
        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{operation['key']}': {e}")
        return compiled_op

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request = params['request']
        headers = params['headers']

        transformed_request = copy.deepcopy(request)

        # Apply all operations sequentially
        for operation in self.compiled_operations:
            try:
                matches = list(operation['expr'].find(transformed_request))
                op_type = operation.get('op', 'set').lower()
                value = operation.get('value')
                key = operation['key']

                if op_type == 'delete':
                    for match in matches:
                        self._delete_match(transformed_request, match)

                elif op_type == 'set':
                    for match in matches:
                        self._set_match(transformed_request, match, value)

                elif op_type in {'append', 'prepend'}:
                    for match in matches:
                        self._list_insert_match(transformed_request, match, value, op_type)

                elif op_type == 'merge':
                    for match in matches:
                        self._merge_match(transformed_request, match, value)

                self.logger.debug(f"Applied {op_type} operation using JSONPath '{key}'")

            except Exception as e:
                self.logger.error(f"Failed to apply {op_type} operation using JSONPath '{key}': {e}")
                return request, headers

        return transformed_request, headers

    def _delete_match(self, data: Dict[str, Any], match) -> None:
        context = match.context.value
        path = match.path
        try:
            if hasattr(path, 'index'):
                # list index
                idx = path.index
                if isinstance(context, list) and 0 <= idx < len(context):
                    context.pop(idx)
            elif hasattr(path, 'fields'):
                # field access
                for f in path.fields:
                    if isinstance(context, dict) and f in context:
                        del context[f]
        except Exception:
            raise

    def _set_match(self, data: Dict[str, Any], match, value: Any) -> None:
        context = match.context.value
        path = match.path
        try:
            if hasattr(path, 'index'):
                idx = path.index
                if isinstance(context, list):
                    while len(context) <= idx:
                        context.append(None)
                    context[idx] = value
            elif hasattr(path, 'fields'):
                # set fields on dict
                for f in path.fields:
                    if isinstance(context, dict):
                        context[f] = value
        except Exception:
            raise

    def _list_insert_match(self, data: Dict[str, Any], match, value: Any, op: str) -> None:
        context = match.value
        # match.value is the matched object itself
        if isinstance(context, list):
            if op == 'append':
                context.append(value)
            else:
                context.insert(0, value)
        else:
            raise ValueError('Target for append/prepend is not a list')

    def _merge_match(self, data: Dict[str, Any], match, value: Any) -> None:
        context = match.value
        if not isinstance(context, dict) or not isinstance(value, dict):
            raise ValueError('Merge requires dict target and dict value')
        context.update(value)


class GeminiApiKeyTransformer(ProviderRequestTransformer):
    """Gemini-specific transformer that adds API key as a query parameter.

    Google's Gemini API uses query parameters for authentication (?key=API_KEY)
    instead of headers. This transformer extracts the API key from headers or
    provider config and adds it as a URL query parameter.
    """

    def __init__(self, logger):
        """Initialize transformer.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Add API key as query parameter to provider config URL."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        if 'provider_config' not in params:
            return request, headers

        provider_config: ProviderConfig = params['provider_config']

        # Extract API key from various sources
        api_key = self._extract_api_key(headers, provider_config)

        if api_key:
            # Parse current URL
            parsed = urlparse(provider_config.base_url)
            query_params = parse_qs(parsed.query, keep_blank_values=True)

            # Add the key parameter
            query_params['key'] = [api_key]

            # Rebuild the URL with the new query parameter
            new_query = urlencode(query_params, doseq=True)
            new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

            provider_config.base_url = new_url
            self.logger.debug('Added API key as query parameter to Gemini URL')
        else:
            self.logger.warning('No API key found for Gemini authentication')

        return request, headers

    def _extract_api_key(self, headers: Dict[str, str], provider_config: ProviderConfig) -> str:
        """Extract API key from headers or provider config.

        Args:
            headers: Request headers
            provider_config: Provider configuration

        Returns:
            API key string or empty string if not found
        """
        # Try provider config first
        if hasattr(provider_config, 'api_key') and provider_config.api_key:
            return provider_config.api_key

        # Try authorization header
        if auth_header := headers.get('authorization', ''):
            if auth_header.lower().startswith('bearer '):
                return auth_header[7:].strip()
            return auth_header

        # Try x-goog-api-key header
        if api_key_header := headers.get('x-goog-api-key', ''):
            return api_key_header

        return ''


class ToolDescriptionOptimizerTransformer(ProviderRequestTransformer):
    """Transformer that optimizes tool descriptions based on a hardcoded mapping.

    Replaces tool descriptions with optimized versions based on tool names.
    Tools not in the mapping are left unchanged.
    """

    # Hardcoded mapping of tool names to optimized descriptions
    TOOL_DESCRIPTION_MAP = {
        'Bash': """<purpose>
Executes bash commands in a persistent shell session with timeout support and security measures. Use this tool for running system commands, file operations, development tasks, and git workflows while maintaining session state across multiple command executions.
</purpose>

<parameters>
- command (required): The bash command to execute
- description (recommended): Clear 5-10 word description of what the command does
- timeout (optional): Timeout in milliseconds (default: 120000ms, max: 600000ms)
- run_in_background (optional): Execute command in background for long-running processes
</parameters>

<pre_execution_verification>
Before executing commands that create files or directories:
1. Use the LS tool to verify the parent directory exists
2. Confirm you're in the correct location for the operation
3. Example: Before `mkdir foo/bar`, verify `foo` directory exists
</pre_execution_verification>

<file_path_handling>
Always quote file paths containing spaces with double quotes:
- Correct: `cd "/Users/name/My Documents"`
- Incorrect: `cd /Users/name/My Documents`
- Correct: `python "/path/with spaces/script.py"`
- Incorrect: `python /path/with spaces/script.py`
</file_path_handling>

<critical_constraints>
- MUST use Read tool instead of `cat`, `head`, or `tail` for reading files
- MUST use Grep, Glob, or Task tools instead of `find` or `grep` for searching
- If grep is absolutely necessary, use `rg` (ripgrep) which is pre-installed
- MUST combine multiple commands using `&&` or `;` operators, NOT newlines
- MUST maintain working directory by using absolute paths instead of `cd` when possible
- NEVER use interactive flags like `git rebase -i` or `git add -i`
</critical_constraints>

<command_chaining>
ALWAYS combine multiple related commands for efficiency:
- Preferred: `uvx ruff check --fix && uvx ruff format path/to/code`
- Preferred: `pytest /absolute/path/to/tests`
- Avoid: `cd /some/path && pytest tests` (unless user explicitly requests cd)
</command_chaining>

<background_execution>
Use `run_in_background: true` for long-running processes:
- Monitor output using BashOutput tool as it becomes available
- Do not append `&` to commands when using this parameter
- Never use background execution for `sleep` commands
</background_execution>

<git_commit_workflow>
- Run the following commands in parallel with the Bash tool
    - git status: see untracked files
    - git diff --stat && git diff -U1 -w : see staged and unstaged changes to be committed
    - git log -n 5 --oneline: see recent commits
- Analyze changes and draft a commit message
    - Summarize the nature of changes (eg. new feature, enhancement, bug fix, refactors, tests, docs etc.)
    - Do not commit sensitive information or secrets
    - Draft concise (1-2 sentences) message that focuses on "why" rather than "what". The message must accurately reflect the changes and their purpose.
- If commit fails due to pre-commit hook changes, retry the commit ONCE to include the changes. Stop if it fails again.
</git_commit_workflow>

Remember to always parallelize and chain commands where possible.
""",
    }

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Optimize tool descriptions based on hardcoded mapping.

        Args:
            params: Dictionary containing request and headers

        Returns:
            Tuple of (transformed_request, headers)
        """
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        # Transform tool descriptions if tools exist in request
        if 'tools' in request and isinstance(request['tools'], list):
            for tool in request['tools']:
                if isinstance(tool, dict) and 'name' in tool and 'description' in tool:
                    tool_name = tool['name']
                    if tool_name in self.TOOL_DESCRIPTION_MAP:
                        tool['description']
                        new_description = self.TOOL_DESCRIPTION_MAP[tool_name]
                        tool['description'] = new_description

        return request, headers


class AuthHeaderTransformer(ProviderRequestTransformer):
    """Anthropic-specific header filtering transformer.

    Filters incoming headers to only include those with specific prefixes
    required by the Anthropic API.
    """

    def __init__(self, logger, auth_header: str, passthrough_prefixes: list[str] = ['x-', 'anthropic', 'user-']):
        """Initialize transformer.

        Args:
            logger: Logger instance
            auth_header: Authentication header to use ('x-api-key' or 'authorization')
        """
        super().__init__(logger)
        self.auth_header = auth_header.lower()
        self.passthrough_prefixes = passthrough_prefixes

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Filter headers to only include Anthropic-compatible prefixes."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        # Filter headers to only keep Anthropic-compatible ones
        filtered_headers = {k: v for k, v in headers.items() if any((k.startswith(prefix) for prefix in self.passthrough_prefixes))}

        # Inject API key from provider config if available
        provider_config = params.get('provider_config')
        if provider_config and provider_config.api_key:
            filtered_headers.pop('authorization', None)
            filtered_headers.pop('x-api-key', None)

            # Set the configured auth header
            if self.auth_header.startswith('autho'):
                filtered_headers[self.auth_header] = f'Bearer {provider_config.api_key}'
            else:
                filtered_headers[self.auth_header] = provider_config.api_key

        return request, filtered_headers


class OpenRouterReasoningTransformer(ProviderRequestTransformer):
    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        request: dict[str, Any] = params['request']
        headers = params['headers']

        # Anthropic
        if thinking := request.get('thinking', {}):
            request['reasoning'] = {'max_tokens': thinking.get('budget_tokens', 0)}
            request.pop('thinking', None)
            return request, headers

        # OpenAI
        if reasoning_effort := request.get('reasoning_effort', None):
            request['reasoning'] = {'effort': reasoning_effort}
            request.pop('reasoning_effort')
            return request, headers
        return request, headers

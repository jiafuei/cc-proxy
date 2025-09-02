"""Tests for HeaderTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.utils import HeaderTransformer


class TestHeaderTransformer:
    """Test cases for the HeaderTransformer."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    @pytest.mark.asyncio
    async def test_single_set_operation(self, mock_logger):
        """Test single 'set' operation."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'authorization', 'value': 'token123'}])

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'token123'
        assert headers['content-type'] == 'application/json'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_single_set_operation_explicit(self, mock_logger):
        """Test explicit 'set' operation."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'x-api-key', 'value': 'secret', 'op': 'set'}])

        params = {'request': {'model': 'test'}, 'headers': {}}

        request, headers = await transformer.transform(params)

        assert headers['x-api-key'] == 'secret'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_set_operation_with_prefix_and_suffix(self, mock_logger):
        """Test 'set' operation with prefix and suffix."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'authorization', 'value': 'token123', 'prefix': 'Bearer ', 'suffix': ' extra', 'op': 'set'}])

        params = {'request': {'model': 'test'}, 'headers': {}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'Bearer token123 extra'

    @pytest.mark.asyncio
    async def test_single_delete_operation(self, mock_logger):
        """Test single 'delete' operation with existing header."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'authorization', 'op': 'delete'}])

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer token', 'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        assert 'authorization' not in headers
        assert headers['content-type'] == 'application/json'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_delete_operation_nonexistent_header(self, mock_logger):
        """Test 'delete' operation with non-existent header."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'x-custom', 'op': 'delete'}])

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        # Headers should remain unchanged
        assert headers == {'content-type': 'application/json'}
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_multiple_operations(self, mock_logger):
        """Test multiple operations in single transformer."""
        transformer = HeaderTransformer(
            mock_logger,
            operations=[
                {'key': 'authorization', 'value': 'Bearer token123', 'op': 'set'},
                {'key': 'x-unwanted', 'op': 'delete'},
                {'key': 'cache-control', 'value': 'no-cache', 'prefix': 'private, ', 'op': 'set'},
            ],
        )

        params = {'request': {'model': 'test'}, 'headers': {'x-unwanted': 'should-be-deleted', 'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'Bearer token123'
        assert 'x-unwanted' not in headers
        assert headers['cache-control'] == 'private, no-cache'
        assert headers['content-type'] == 'application/json'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_set_operation_overwrite_existing(self, mock_logger):
        """Test 'set' operation overwrites existing header."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'authorization', 'value': 'new_token', 'op': 'set'}])

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'old_token'}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'new_token'

    def test_empty_operations_array(self, mock_logger):
        """Test that empty operations array raises ValueError."""
        with pytest.raises(ValueError, match="'operations' parameter is required and must contain at least one operation"):
            HeaderTransformer(mock_logger, operations=[])

    def test_none_operations_array(self, mock_logger):
        """Test that None operations array raises ValueError."""
        with pytest.raises(ValueError, match="'operations' parameter is required and must contain at least one operation"):
            HeaderTransformer(mock_logger, operations=None)

    def test_invalid_operation_not_dict(self, mock_logger):
        """Test that non-dict operation raises ValueError."""
        with pytest.raises(ValueError, match='Operation 0 must be a dictionary'):
            HeaderTransformer(mock_logger, operations=['invalid'])

    def test_missing_key_parameter(self, mock_logger):
        """Test that missing key parameter raises ValueError."""
        with pytest.raises(ValueError, match="Operation 0: 'key' parameter is required"):
            HeaderTransformer(mock_logger, operations=[{'value': 'test'}])

    def test_empty_key_parameter(self, mock_logger):
        """Test that empty key parameter raises ValueError."""
        with pytest.raises(ValueError, match="Operation 0: 'key' parameter is required"):
            HeaderTransformer(mock_logger, operations=[{'key': '', 'value': 'test'}])

    def test_invalid_operation_type(self, mock_logger):
        """Test that invalid operation type raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"Operation 0: Invalid operation 'invalid'\. Must be one of: \{.*'set'.*'delete'.*\}|Operation 0: Invalid operation 'invalid'\. Must be one of: \{.*'delete'.*'set'.*\}",
        ):
            HeaderTransformer(mock_logger, operations=[{'key': 'test', 'value': 'value', 'op': 'invalid'}])

    def test_missing_value_for_set_operation(self, mock_logger):
        """Test that missing value for 'set' operation raises ValueError."""
        with pytest.raises(ValueError, match="Operation 0: 'value' parameter is required for 'set' operation"):
            HeaderTransformer(mock_logger, operations=[{'key': 'test', 'op': 'set'}])

    def test_missing_value_for_delete_allowed(self, mock_logger):
        """Test that missing value for 'delete' operation is allowed."""
        # This should not raise an error
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'test', 'op': 'delete'}])
        assert len(transformer.operations) == 1
        assert transformer.operations[0]['key'] == 'test'

    def test_default_operation_is_set(self, mock_logger):
        """Test that operation defaults to 'set' when not specified."""
        HeaderTransformer(mock_logger, operations=[{'key': 'test', 'value': 'value'}])
        # Should not raise error, meaning 'set' is the default

    def test_case_insensitive_operations(self, mock_logger):
        """Test that operation parameter is case insensitive."""
        HeaderTransformer(mock_logger, operations=[{'key': 'test1', 'value': 'value', 'op': 'SET'}, {'key': 'test2', 'op': 'DELETE'}])
        # Should not raise error, meaning case insensitive operations work

    @pytest.mark.asyncio
    async def test_logging_set_operation(self, mock_logger):
        """Test that logging works for set operation."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'test', 'value': 'value', 'op': 'set'}])

        params = {'request': {}, 'headers': {}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Set header 'test' = 'value'")

    @pytest.mark.asyncio
    async def test_logging_delete_operation_existing(self, mock_logger):
        """Test that logging works for delete operation with existing header."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'test', 'op': 'delete'}])

        params = {'request': {}, 'headers': {'test': 'value'}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Deleted header 'test'")

    @pytest.mark.asyncio
    async def test_logging_delete_operation_nonexistent(self, mock_logger):
        """Test that logging works for delete operation with non-existent header."""
        transformer = HeaderTransformer(mock_logger, operations=[{'key': 'test', 'op': 'delete'}])

        params = {'request': {}, 'headers': {}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Header 'test' not found for deletion")

    @pytest.mark.asyncio
    async def test_complex_multiple_operations_scenario(self, mock_logger):
        """Test complex scenario with multiple mixed operations."""
        transformer = HeaderTransformer(
            mock_logger,
            operations=[
                {'key': 'authorization', 'value': 'abc123', 'prefix': 'Bearer ', 'op': 'set'},
                {'key': 'x-old-header', 'op': 'delete'},
                {'key': 'cache-control', 'value': 'no-store', 'op': 'set'},
                {'key': 'x-nonexistent', 'op': 'delete'},  # This won't find anything
                {'key': 'x-custom', 'value': 'custom-value', 'suffix': '-final', 'op': 'set'},
            ],
        )

        params = {'request': {'messages': []}, 'headers': {'x-old-header': 'to-be-removed', 'existing-header': 'keep-this'}}

        request, headers = await transformer.transform(params)

        # Check all expected transformations
        assert headers['authorization'] == 'Bearer abc123'
        assert 'x-old-header' not in headers
        assert headers['cache-control'] == 'no-store'
        assert headers['x-custom'] == 'custom-value-final'
        assert headers['existing-header'] == 'keep-this'  # Untouched
        assert request == {'messages': []}

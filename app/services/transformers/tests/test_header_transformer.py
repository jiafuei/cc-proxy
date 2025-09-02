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
    async def test_set_operation_default(self, mock_logger):
        """Test 'set' operation (default behavior)."""
        transformer = HeaderTransformer(mock_logger, key='authorization', value='token123')

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'token123'
        assert headers['content-type'] == 'application/json'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_set_operation_explicit(self, mock_logger):
        """Test explicit 'set' operation."""
        transformer = HeaderTransformer(mock_logger, key='x-api-key', value='secret', op='set')

        params = {'request': {'model': 'test'}, 'headers': {}}

        request, headers = await transformer.transform(params)

        assert headers['x-api-key'] == 'secret'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_set_operation_with_prefix_and_suffix(self, mock_logger):
        """Test 'set' operation with prefix and suffix."""
        transformer = HeaderTransformer(mock_logger, key='authorization', value='token123', prefix='Bearer ', suffix=' extra', op='set')

        params = {'request': {'model': 'test'}, 'headers': {}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'Bearer token123 extra'

    @pytest.mark.asyncio
    async def test_delete_operation_existing_header(self, mock_logger):
        """Test 'delete' operation with existing header."""
        transformer = HeaderTransformer(mock_logger, key='authorization', op='delete')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer token', 'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        assert 'authorization' not in headers
        assert headers['content-type'] == 'application/json'
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_delete_operation_nonexistent_header(self, mock_logger):
        """Test 'delete' operation with non-existent header."""
        transformer = HeaderTransformer(mock_logger, key='x-custom', op='delete')

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        # Headers should remain unchanged
        assert headers == {'content-type': 'application/json'}
        assert request == {'model': 'test'}

    @pytest.mark.asyncio
    async def test_set_operation_overwrite_existing(self, mock_logger):
        """Test 'set' operation overwrites existing header."""
        transformer = HeaderTransformer(mock_logger, key='authorization', value='new_token', op='set')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'old_token'}}

        request, headers = await transformer.transform(params)

        assert headers['authorization'] == 'new_token'

    def test_invalid_operation(self, mock_logger):
        """Test that invalid operation raises ValueError."""
        with pytest.raises(ValueError, match=r"Invalid operation 'invalid'\. Must be one of: \{.*'set'.*'delete'.*\}|Invalid operation 'invalid'\. Must be one of: \{.*'delete'.*'set'.*\}"):
            HeaderTransformer(mock_logger, key='test', value='value', op='invalid')

    def test_case_insensitive_operation(self, mock_logger):
        """Test that operation parameter is case insensitive."""
        transformer = HeaderTransformer(mock_logger, key='test', value='value', op='SET')
        assert transformer.op == 'set'

        transformer = HeaderTransformer(mock_logger, key='test', op='DELETE')
        assert transformer.op == 'delete'

    def test_missing_key(self, mock_logger):
        """Test that missing key raises ValueError."""
        with pytest.raises(ValueError, match="'key' parameter is required for all operations"):
            HeaderTransformer(mock_logger, key='', value='value', op='set')

    def test_missing_value_for_set(self, mock_logger):
        """Test that missing value for 'set' operation raises ValueError."""
        with pytest.raises(ValueError, match="'value' parameter is required for 'set' operation"):
            HeaderTransformer(mock_logger, key='test', value='', op='set')

    def test_missing_value_for_delete_allowed(self, mock_logger):
        """Test that missing value for 'delete' operation is allowed."""
        # This should not raise an error
        transformer = HeaderTransformer(mock_logger, key='test', op='delete')
        assert transformer.op == 'delete'
        assert transformer.key == 'test'

    def test_constructor_parameters(self, mock_logger):
        """Test that constructor accepts and stores parameters correctly."""
        transformer = HeaderTransformer(mock_logger, key='custom-header', value='custom-value', prefix='pre', suffix='suf', op='set')

        assert transformer.logger == mock_logger
        assert transformer.key == 'custom-header'
        assert transformer.value == 'custom-value'
        assert transformer.prefix == 'pre'
        assert transformer.suffix == 'suf'
        assert transformer.op == 'set'

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mock_logger):
        """Test backward compatibility with old AddHeaderTransformer interface."""
        # Old interface: HeaderTransformer(logger, key, value, prefix, suffix)
        transformer = HeaderTransformer(mock_logger, 'auth', 'token', 'Bearer ', '')

        params = {'request': {}, 'headers': {}}

        request, headers = await transformer.transform(params)

        assert headers['auth'] == 'Bearer token'
        assert transformer.op == 'set'  # Should default to 'set'

    @pytest.mark.asyncio
    async def test_logging_set_operation(self, mock_logger):
        """Test that logging works for set operation."""
        transformer = HeaderTransformer(mock_logger, key='test', value='value', op='set')

        params = {'request': {}, 'headers': {}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Set header 'test' = 'value'")

    @pytest.mark.asyncio
    async def test_logging_delete_operation_existing(self, mock_logger):
        """Test that logging works for delete operation with existing header."""
        transformer = HeaderTransformer(mock_logger, key='test', op='delete')

        params = {'request': {}, 'headers': {'test': 'value'}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Deleted header 'test'")

    @pytest.mark.asyncio
    async def test_logging_delete_operation_nonexistent(self, mock_logger):
        """Test that logging works for delete operation with non-existent header."""
        transformer = HeaderTransformer(mock_logger, key='test', op='delete')

        params = {'request': {}, 'headers': {}}
        await transformer.transform(params)

        mock_logger.debug.assert_called_with("Header 'test' not found for deletion")

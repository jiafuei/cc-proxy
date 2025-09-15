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

    @pytest.mark.parametrize("operations,initial_headers,expected_headers", [
        # Single set operation (implicit)
        ([{'key': 'authorization', 'value': 'token123'}], 
         {'content-type': 'application/json'}, 
         {'authorization': 'token123', 'content-type': 'application/json'}),
        # Single set operation (explicit)
        ([{'key': 'x-api-key', 'value': 'secret', 'op': 'set'}], 
         {}, 
         {'x-api-key': 'secret'}),
        # Set with prefix and suffix
        ([{'key': 'authorization', 'value': 'token123', 'prefix': 'Bearer ', 'suffix': ' extra', 'op': 'set'}], 
         {}, 
         {'authorization': 'Bearer token123 extra'}),
        # Set operation overwrite existing
        ([{'key': 'authorization', 'value': 'new_token', 'op': 'set'}], 
         {'authorization': 'old_token'}, 
         {'authorization': 'new_token'}),
        # Single delete existing header
        ([{'key': 'authorization', 'op': 'delete'}], 
         {'authorization': 'Bearer token', 'content-type': 'application/json'}, 
         {'content-type': 'application/json'}),
        # Delete non-existent header
        ([{'key': 'x-custom', 'op': 'delete'}], 
         {'content-type': 'application/json'}, 
         {'content-type': 'application/json'}),
        # Multiple operations
        ([{'key': 'authorization', 'value': 'Bearer token123', 'op': 'set'},
          {'key': 'x-unwanted', 'op': 'delete'},
          {'key': 'cache-control', 'value': 'no-cache', 'prefix': 'private, ', 'op': 'set'}], 
         {'x-unwanted': 'should-be-deleted', 'content-type': 'application/json'}, 
         {'authorization': 'Bearer token123', 'cache-control': 'private, no-cache', 'content-type': 'application/json'}),
    ])
    @pytest.mark.asyncio
    async def test_header_operations(self, mock_logger, operations, initial_headers, expected_headers):
        """Test various header operations."""
        transformer = HeaderTransformer(mock_logger, operations=operations)
        params = {'request': {'model': 'test'}, 'headers': initial_headers}
        
        request, headers = await transformer.transform(params)
        
        assert headers == expected_headers
        assert request == {'model': 'test'}

    @pytest.mark.parametrize("operations,exception_match", [
        # Empty operations array
        ([], "'operations' parameter is required and must contain at least one operation"),
        # None operations array  
        (None, "'operations' parameter is required and must contain at least one operation"),
        # Non-dict operation
        (['invalid'], 'Operation 0 must be a dictionary'),
        # Missing key parameter
        ([{'value': 'test'}], "Operation 0: 'key' parameter is required"),
        # Empty key parameter
        ([{'key': '', 'value': 'test'}], "Operation 0: 'key' parameter is required"),
        # Invalid operation type
        ([{'key': 'test', 'value': 'value', 'op': 'invalid'}], 
         r"Operation 0: Invalid operation 'invalid'\. Must be one of: \{.*'set'.*'delete'.*\}|Operation 0: Invalid operation 'invalid'\. Must be one of: \{.*'delete'.*'set'.*\}"),
        # Missing value for set operation
        ([{'key': 'test', 'op': 'set'}], "Operation 0: 'value' parameter is required for 'set' operation"),
    ])
    def test_validation_errors(self, mock_logger, operations, exception_match):
        """Test validation errors during initialization."""
        with pytest.raises(ValueError, match=exception_match):
            HeaderTransformer(mock_logger, operations=operations)

    @pytest.mark.parametrize("operations,description", [
        # Delete operation without value should be allowed
        ([{'key': 'test', 'op': 'delete'}], "delete operation without value"),
        # Default operation is set
        ([{'key': 'test', 'value': 'value'}], "default operation is set"),
        # Case insensitive operations
        ([{'key': 'test1', 'value': 'value', 'op': 'SET'}, {'key': 'test2', 'op': 'DELETE'}], "case insensitive operations"),
    ])
    def test_valid_configurations(self, mock_logger, operations, description):
        """Test valid transformer configurations."""
        transformer = HeaderTransformer(mock_logger, operations=operations)
        assert len(transformer.operations) >= 1

    @pytest.mark.parametrize("operations,headers,expected_log_call", [
        # Set operation logging
        ([{'key': 'test', 'value': 'value', 'op': 'set'}], 
         {}, 
         "Set header 'test' = 'value'"),
        # Delete existing header logging
        ([{'key': 'test', 'op': 'delete'}], 
         {'test': 'value'}, 
         "Deleted header 'test'"),
        # Delete non-existent header logging
        ([{'key': 'test', 'op': 'delete'}], 
         {}, 
         "Header 'test' not found for deletion"),
    ])
    @pytest.mark.asyncio
    async def test_logging_operations(self, mock_logger, operations, headers, expected_log_call):
        """Test logging for different operations."""
        transformer = HeaderTransformer(mock_logger, operations=operations)
        params = {'request': {}, 'headers': headers}
        
        await transformer.transform(params)
        
        mock_logger.debug.assert_called_with(expected_log_call)

    @pytest.mark.asyncio
    async def test_complex_scenario(self, mock_logger):
        """Test complex scenario with multiple mixed operations."""
        transformer = HeaderTransformer(
            mock_logger,
            operations=[
                {'key': 'authorization', 'value': 'abc123', 'prefix': 'Bearer ', 'op': 'set'},
                {'key': 'x-old-header', 'op': 'delete'},
                {'key': 'cache-control', 'value': 'no-store', 'op': 'set'},
                {'key': 'x-nonexistent', 'op': 'delete'},
                {'key': 'x-custom', 'value': 'custom-value', 'suffix': '-final', 'op': 'set'},
            ],
        )

        params = {'request': {'messages': []}, 'headers': {'x-old-header': 'to-be-removed', 'existing-header': 'keep-this'}}
        request, headers = await transformer.transform(params)

        expected_headers = {
            'authorization': 'Bearer abc123',
            'cache-control': 'no-store', 
            'x-custom': 'custom-value-final',
            'existing-header': 'keep-this'
        }
        assert headers == expected_headers
        assert request == {'messages': []}

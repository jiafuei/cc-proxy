"""Tests for RequestBodyTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.utils import RequestBodyTransformer


class TestRequestBodyTransformer:
    """Test cases for the request body transformer."""

    @pytest.fixture
    def sample_request(self):
        """Sample request for testing."""
        return {
            'model': 'claude-3',
            'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}],
            'temperature': 0.7,
            'metadata': {'user_id': '123', 'session': 'abc'},
        }

    def test_init_valid_operations(self):
        """Test initialization with valid operations."""
        mock_logger = Mock()
        operations = [
            {'key': 'model', 'value': 'new-model', 'op': 'set'},
            {'key': 'temperature', 'value': 0.8, 'op': 'set'}
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        assert len(transformer.operations) == 2
        assert len(transformer.compiled_operations) == 2
        assert transformer.operations[0]['key'] == 'model'
        assert transformer.operations[1]['key'] == 'temperature'

    def test_init_empty_operations(self):
        """Test initialization with empty operations raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match="'operations' parameter is required"):
            RequestBodyTransformer(mock_logger, operations=[])

    def test_init_no_operations(self):
        """Test initialization with None operations raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match="'operations' parameter is required"):
            RequestBodyTransformer(mock_logger, operations=None)

    def test_init_invalid_operation_format(self):
        """Test initialization with invalid operation format raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match="Operation 0 must be a dictionary"):
            RequestBodyTransformer(mock_logger, operations=['invalid'])

    def test_init_missing_key(self):
        """Test initialization with missing key raises ValueError."""
        mock_logger = Mock()
        operations = [{'value': 'test', 'op': 'set'}]

        with pytest.raises(ValueError, match="Operation 0: 'key' parameter is required"):
            RequestBodyTransformer(mock_logger, operations=operations)

    def test_init_invalid_operation_type(self):
        """Test initialization with invalid operation type raises ValueError."""
        mock_logger = Mock()
        operations = [{'key': 'model', 'value': 'test', 'op': 'invalid'}]

        with pytest.raises(ValueError, match="Operation 0: Invalid operation 'invalid'"):
            RequestBodyTransformer(mock_logger, operations=operations)

    def test_init_missing_value_for_set(self):
        """Test initialization with missing value for set operation raises ValueError."""
        mock_logger = Mock()
        operations = [{'key': 'model', 'op': 'set'}]

        with pytest.raises(ValueError, match="Operation 0: 'value' parameter is required for 'set' operation"):
            RequestBodyTransformer(mock_logger, operations=operations)

    def test_init_invalid_jsonpath(self):
        """Test initialization with invalid JSONPath raises ValueError."""
        mock_logger = Mock()
        operations = [{'key': '[invalid', 'value': 'test', 'op': 'set'}]

        with pytest.raises(ValueError, match='Invalid JSONPath expression'):
            RequestBodyTransformer(mock_logger, operations=operations)

    @pytest.mark.asyncio
    async def test_transform_single_set_operation(self, sample_request):
        """Test transform with single set operation."""
        mock_logger = Mock()
        operations = [{'key': 'model', 'value': 'new-model', 'op': 'set'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['model'] == 'new-model'
        assert sample_request['model'] == 'claude-3'  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_single_delete_operation(self, sample_request):
        """Test transform with single delete operation."""
        mock_logger = Mock()
        operations = [{'key': 'temperature', 'op': 'delete'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert 'temperature' not in result_request
        assert 'temperature' in sample_request  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_single_append_operation(self, sample_request):
        """Test transform with single append operation."""
        mock_logger = Mock()
        new_message = {'role': 'system', 'content': 'System message'}
        operations = [{'key': 'messages', 'value': new_message, 'op': 'append'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert len(result_request['messages']) == 3
        assert result_request['messages'][-1] == new_message
        assert len(sample_request['messages']) == 2  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_single_prepend_operation(self, sample_request):
        """Test transform with single prepend operation."""
        mock_logger = Mock()
        new_message = {'role': 'system', 'content': 'System message'}
        operations = [{'key': 'messages', 'value': new_message, 'op': 'prepend'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert len(result_request['messages']) == 3
        assert result_request['messages'][0] == new_message
        assert result_request['messages'][1]['role'] == 'user'  # Original first moved to second

    @pytest.mark.asyncio
    async def test_transform_single_merge_operation(self, sample_request):
        """Test transform with single merge operation."""
        mock_logger = Mock()
        merge_data = {'version': '1.0', 'session': 'updated_session'}
        operations = [{'key': 'metadata', 'value': merge_data, 'op': 'merge'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['metadata']['user_id'] == '123'  # Original preserved
        assert result_request['metadata']['session'] == 'updated_session'  # Updated
        assert result_request['metadata']['version'] == '1.0'  # New key added

    @pytest.mark.asyncio
    async def test_transform_multiple_operations(self, sample_request):
        """Test transform with multiple operations."""
        mock_logger = Mock()
        operations = [
            {'key': 'model', 'value': 'new-model', 'op': 'set'},
            {'key': 'temperature', 'value': 0.8, 'op': 'set'},
            {'key': 'metadata', 'value': {'version': '2.0'}, 'op': 'merge'}
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['model'] == 'new-model'
        assert result_request['temperature'] == 0.8
        assert result_request['metadata']['version'] == '2.0'
        assert result_request['metadata']['user_id'] == '123'  # Original preserved
        assert sample_request['model'] == 'claude-3'  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_sequential_operations_same_path(self, sample_request):
        """Test transform with sequential operations on same path."""
        mock_logger = Mock()
        operations = [
            {'key': 'model', 'value': 'first-model', 'op': 'set'},
            {'key': 'model', 'value': 'final-model', 'op': 'set'}
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['model'] == 'final-model'  # Last operation wins

    @pytest.mark.asyncio
    async def test_transform_mixed_operations(self, sample_request):
        """Test transform with mixed operation types."""
        mock_logger = Mock()
        new_message = {'role': 'system', 'content': 'System message'}
        operations = [
            {'key': 'messages', 'value': new_message, 'op': 'prepend'},
            {'key': 'temperature', 'op': 'delete'},
            {'key': 'metadata', 'value': {'version': '2.0'}, 'op': 'merge'},
            {'key': 'model', 'value': 'updated-model', 'op': 'set'}
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert len(result_request['messages']) == 3
        assert result_request['messages'][0] == new_message
        assert 'temperature' not in result_request
        assert result_request['metadata']['version'] == '2.0'
        assert result_request['metadata']['user_id'] == '123'
        assert result_request['model'] == 'updated-model'

    @pytest.mark.asyncio
    async def test_transform_error_handling_append_to_non_list(self, sample_request):
        """Test transform append to non-list returns original request."""
        mock_logger = Mock()
        operations = [{'key': 'model', 'value': 'item', 'op': 'append'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return original request due to error
        assert result_request is sample_request
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_error_handling_merge_with_non_dict(self, sample_request):
        """Test transform merge with non-dict value returns original request."""
        mock_logger = Mock()
        operations = [{'key': 'metadata', 'value': 'not_a_dict', 'op': 'merge'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return original request due to error
        assert result_request is sample_request
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_nested_jsonpath_operations(self, sample_request):
        """Test transform with nested JSONPath expressions."""
        mock_logger = Mock()
        operations = [
            {'key': 'messages[0].content', 'value': 'Updated content', 'op': 'set'},
            {'key': 'metadata.user_id', 'value': '456', 'op': 'set'}
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['messages'][0]['content'] == 'Updated content'
        assert result_request['metadata']['user_id'] == '456'
        assert sample_request['messages'][0]['content'] == 'Hello'  # Original unchanged
        assert sample_request['metadata']['user_id'] == '123'  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_preserves_headers(self, sample_request):
        """Test transform preserves headers unchanged."""
        mock_logger = Mock()
        operations = [{'key': 'model', 'value': 'new-model', 'op': 'set'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        original_headers = {'content-type': 'application/json', 'authorization': 'Bearer token'}
        params = {'request': sample_request, 'headers': original_headers}
        result_request, result_headers = await transformer.transform(params)

        assert result_headers is original_headers
        assert result_headers == original_headers

    @pytest.mark.asyncio
    async def test_transform_jsonpath_array_wildcard(self, sample_request):
        """Test transform with JSONPath array wildcard selector."""
        mock_logger = Mock()
        operations = [{'key': 'messages[*].role', 'value': 'system', 'op': 'set'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # All message roles should be set to 'system'
        assert all(msg['role'] == 'system' for msg in result_request['messages'])
        # Original should be unchanged
        assert sample_request['messages'][0]['role'] == 'user'
        assert sample_request['messages'][1]['role'] == 'assistant'

    @pytest.mark.asyncio
    async def test_transform_nonexistent_path(self, sample_request):
        """Test transform with nonexistent JSONPath returns unchanged request."""
        mock_logger = Mock()
        operations = [{'key': 'nonexistent.path', 'value': 'value', 'op': 'set'}]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return deep copy but unchanged since path doesn't match anything
        assert result_request is not sample_request  # Different object
        assert result_request == sample_request  # Same content

    @pytest.mark.asyncio
    async def test_transform_partial_failure_returns_original(self, sample_request):
        """Test that if any operation fails, original request is returned."""
        mock_logger = Mock()
        operations = [
            {'key': 'model', 'value': 'new-model', 'op': 'set'},  # This should work
            {'key': 'model', 'value': 'item', 'op': 'append'}  # This will fail
        ]
        transformer = RequestBodyTransformer(mock_logger, operations=operations)

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return original request due to second operation error
        assert result_request is sample_request
        mock_logger.error.assert_called_once()
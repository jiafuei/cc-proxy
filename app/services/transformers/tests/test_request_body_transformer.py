"""Tests for RequestBodyTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.utils import RequestBodyTransformer


class TestRequestBodyTransformer:
    """Test cases for the request body transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return RequestBodyTransformer(mock_logger, key='test_key', value='test_value', op='set')

    @pytest.fixture
    def sample_request(self):
        """Sample request for testing."""
        return {
            'model': 'claude-3',
            'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}],
            'temperature': 0.7,
            'metadata': {'user_id': '123', 'session': 'abc'},
        }

    def test_init_valid_operation(self):
        """Test initialization with valid operation."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='test', value='value', op='SET')

        assert transformer.key == 'test'
        assert transformer.value == 'value'
        assert transformer.op == 'set'

    def test_init_invalid_operation(self):
        """Test initialization with invalid operation raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match="Invalid operation 'invalid'"):
            RequestBodyTransformer(mock_logger, key='test', value='value', op='invalid')

    def test_init_empty_key(self):
        """Test initialization with empty key raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match="'key' parameter is required"):
            RequestBodyTransformer(mock_logger, key='', value='value', op='set')

    def test_init_invalid_jsonpath(self):
        """Test initialization with invalid JSONPath raises ValueError."""
        mock_logger = Mock()

        with pytest.raises(ValueError, match='Invalid JSONPath expression'):
            RequestBodyTransformer(mock_logger, key='[invalid', value='value', op='set')

    @pytest.mark.asyncio
    async def test_transform_set_operation(self, sample_request):
        """Test transform with set operation."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='model', value='new-model', op='set')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['model'] == 'new-model'
        assert sample_request['model'] == 'claude-3'  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_delete_operation(self, sample_request):
        """Test transform with delete operation."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='temperature', value=None, op='delete')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert 'temperature' not in result_request
        assert 'temperature' in sample_request  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_append_operation(self, sample_request):
        """Test transform with append operation."""
        mock_logger = Mock()
        new_message = {'role': 'system', 'content': 'System message'}
        transformer = RequestBodyTransformer(mock_logger, key='messages', value=new_message, op='append')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert len(result_request['messages']) == 3
        assert result_request['messages'][-1] == new_message
        assert len(sample_request['messages']) == 2  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_prepend_operation(self, sample_request):
        """Test transform with prepend operation."""
        mock_logger = Mock()
        new_message = {'role': 'system', 'content': 'System message'}
        transformer = RequestBodyTransformer(mock_logger, key='messages', value=new_message, op='prepend')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert len(result_request['messages']) == 3
        assert result_request['messages'][0] == new_message
        assert result_request['messages'][1]['role'] == 'user'  # Original first moved to second

    @pytest.mark.asyncio
    async def test_transform_merge_operation(self, sample_request):
        """Test transform with merge operation."""
        mock_logger = Mock()
        merge_data = {'version': '1.0', 'session': 'updated_session'}
        transformer = RequestBodyTransformer(mock_logger, key='metadata', value=merge_data, op='merge')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['metadata']['user_id'] == '123'  # Original preserved
        assert result_request['metadata']['session'] == 'updated_session'  # Updated
        assert result_request['metadata']['version'] == '1.0'  # New key added

    @pytest.mark.asyncio
    async def test_transform_append_to_non_list_error(self, sample_request):
        """Test transform append to non-list returns original request."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='model', value='item', op='append')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return original request due to error
        assert result_request is sample_request
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_merge_with_non_dict_error(self, sample_request):
        """Test transform merge with non-dict value returns original request."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='metadata', value='not_a_dict', op='merge')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return original request due to error
        assert result_request is sample_request
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_nested_key_operations(self, sample_request):
        """Test transform with nested key paths using JSONPath."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='messages[0].content', value='Updated content', op='set')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['messages'][0]['content'] == 'Updated content'
        assert sample_request['messages'][0]['content'] == 'Hello'  # Original unchanged

    @pytest.mark.asyncio
    async def test_transform_preserves_headers(self, sample_request):
        """Test transform preserves headers unchanged."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='model', value='new-model', op='set')

        original_headers = {'content-type': 'application/json', 'authorization': 'Bearer token'}
        params = {'request': sample_request, 'headers': original_headers}
        result_request, result_headers = await transformer.transform(params)

        assert result_headers is original_headers
        assert result_headers == original_headers

    @pytest.mark.asyncio
    async def test_transform_jsonpath_array_wildcard(self, sample_request):
        """Test transform with JSONPath array wildcard selector."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='messages[*].role', value='system', op='set')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # All message roles should be set to 'system'
        assert all(msg['role'] == 'system' for msg in result_request['messages'])
        # Original should be unchanged
        assert sample_request['messages'][0]['role'] == 'user'
        assert sample_request['messages'][1]['role'] == 'assistant'

    @pytest.mark.asyncio
    async def test_transform_nonexistent_path(self, sample_request):
        """Test transform with nonexistent JSONPath returns original request."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='nonexistent.path', value='value', op='set')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        # Should return deep copy but unchanged since path doesn't match anything
        assert result_request is not sample_request  # Different object
        assert result_request == sample_request  # Same content

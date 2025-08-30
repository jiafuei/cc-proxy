"""Tests for RequestBodyTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.request_body import RequestBodyTransformer


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

    def test_parse_key_path_simple(self, transformer):
        """Test parsing simple key path."""
        result = transformer._parse_key_path('model')
        assert result == ['model']

    def test_parse_key_path_nested(self, transformer):
        """Test parsing nested key path with dot notation."""
        result = transformer._parse_key_path('messages.0.content')
        assert result == ['messages', 0, 'content']

    def test_parse_key_path_mixed(self, transformer):
        """Test parsing mixed key path with strings and numbers."""
        result = transformer._parse_key_path('metadata.user_id')
        assert result == ['metadata', 'user_id']

    def test_parse_key_path_empty(self, transformer):
        """Test parsing empty key path."""
        result = transformer._parse_key_path('')
        assert result == []

    def test_get_nested_value_simple(self, transformer, sample_request):
        """Test getting simple nested value."""
        result = transformer._get_nested_value(sample_request, ['model'])
        assert result == 'claude-3'

    def test_get_nested_value_array_index(self, transformer, sample_request):
        """Test getting value from array using index."""
        result = transformer._get_nested_value(sample_request, ['messages', 0, 'role'])
        assert result == 'user'

    def test_get_nested_value_nested_object(self, transformer, sample_request):
        """Test getting value from nested object."""
        result = transformer._get_nested_value(sample_request, ['metadata', 'user_id'])
        assert result == '123'

    def test_get_nested_value_missing_key(self, transformer, sample_request):
        """Test getting value with missing key raises KeyError."""
        with pytest.raises(KeyError):
            transformer._get_nested_value(sample_request, ['nonexistent'])

    def test_get_nested_value_invalid_array_index(self, transformer, sample_request):
        """Test getting value with invalid array index raises KeyError."""
        with pytest.raises(KeyError):
            transformer._get_nested_value(sample_request, ['messages', 10])

    def test_set_nested_value_simple(self, transformer, sample_request):
        """Test setting simple nested value."""
        transformer._set_nested_value(sample_request, ['model'], 'new-model')
        assert sample_request['model'] == 'new-model'

    def test_set_nested_value_new_key(self, transformer, sample_request):
        """Test setting value for new key."""
        transformer._set_nested_value(sample_request, ['new_key'], 'new_value')
        assert sample_request['new_key'] == 'new_value'

    def test_set_nested_value_array_index(self, transformer, sample_request):
        """Test setting value in array using index."""
        transformer._set_nested_value(sample_request, ['messages', 0, 'content'], 'Updated message')
        assert sample_request['messages'][0]['content'] == 'Updated message'

    def test_set_nested_value_create_intermediate_object(self, transformer):
        """Test setting value creates intermediate objects."""
        request = {}
        transformer._set_nested_value(request, ['metadata', 'user_id'], '123')
        assert request == {'metadata': {'user_id': '123'}}

    def test_set_nested_value_create_intermediate_array(self, transformer):
        """Test setting value creates intermediate arrays."""
        request = {}
        transformer._set_nested_value(request, ['items', 0], 'first_item')
        assert request == {'items': ['first_item']}

    def test_set_nested_value_extend_array(self, transformer):
        """Test setting value extends array if necessary."""
        request = {'items': ['a']}
        transformer._set_nested_value(request, ['items', 3], 'd')
        assert request['items'] == ['a', None, None, 'd']

    def test_delete_nested_key_simple(self, transformer, sample_request):
        """Test deleting simple key."""
        transformer._delete_nested_key(sample_request, ['temperature'])
        assert 'temperature' not in sample_request

    def test_delete_nested_key_array_index(self, transformer, sample_request):
        """Test deleting array element by index."""
        original_length = len(sample_request['messages'])
        transformer._delete_nested_key(sample_request, ['messages', 0])
        assert len(sample_request['messages']) == original_length - 1
        assert sample_request['messages'][0]['role'] == 'assistant'

    def test_delete_nested_key_nested_object(self, transformer, sample_request):
        """Test deleting key from nested object."""
        transformer._delete_nested_key(sample_request, ['metadata', 'user_id'])
        assert 'user_id' not in sample_request['metadata']
        assert 'session' in sample_request['metadata']

    def test_delete_nested_key_missing(self, transformer, sample_request):
        """Test deleting missing key raises KeyError."""
        with pytest.raises(KeyError):
            transformer._delete_nested_key(sample_request, ['nonexistent'])

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
    async def test_transform_append_to_nonexistent_key(self, sample_request):
        """Test transform append creates new list for nonexistent key."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='new_array', value='item1', op='append')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['new_array'] == ['item1']

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
    async def test_transform_merge_to_nonexistent_key(self, sample_request):
        """Test transform merge creates new object for nonexistent key."""
        mock_logger = Mock()
        merge_data = {'key1': 'value1', 'key2': 'value2'}
        transformer = RequestBodyTransformer(mock_logger, key='new_object', value=merge_data, op='merge')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request['new_object'] == merge_data

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
    async def test_transform_empty_key_warning(self, sample_request):
        """Test transform with empty key logs warning and returns original."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='', value='value', op='set')

        params = {'request': sample_request, 'headers': {}}
        result_request, result_headers = await transformer.transform(params)

        assert result_request is not sample_request  # Deep copy still made
        assert result_request == sample_request  # But content is same
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_nested_key_operations(self, sample_request):
        """Test transform with nested key paths."""
        mock_logger = Mock()
        transformer = RequestBodyTransformer(mock_logger, key='messages.0.content', value='Updated content', op='set')

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

    def test_modify_key_set_operation(self, transformer, sample_request):
        """Test _modify_key with set operation."""
        transformer._modify_key(sample_request, 'model', 'gpt-4', 'set')
        assert sample_request['model'] == 'gpt-4'

    def test_modify_key_append_operation(self, transformer):
        """Test _modify_key with append operation."""
        request = {'items': ['a', 'b']}
        transformer._modify_key(request, 'items', 'c', 'append')
        assert request['items'] == ['a', 'b', 'c']

    def test_modify_key_prepend_operation(self, transformer):
        """Test _modify_key with prepend operation."""
        request = {'items': ['b', 'c']}
        transformer._modify_key(request, 'items', 'a', 'prepend')
        assert request['items'] == ['a', 'b', 'c']

    def test_modify_key_merge_operation(self, transformer):
        """Test _modify_key with merge operation."""
        request = {'metadata': {'key1': 'value1'}}
        transformer._modify_key(request, 'metadata', {'key2': 'value2'}, 'merge')
        assert request['metadata'] == {'key1': 'value1', 'key2': 'value2'}

    def test_delete_key_simple(self, transformer, sample_request):
        """Test _delete_key with simple path."""
        transformer._delete_key(sample_request, 'temperature')
        assert 'temperature' not in sample_request

    def test_delete_key_nested(self, transformer, sample_request):
        """Test _delete_key with nested path."""
        transformer._delete_key(sample_request, 'metadata.user_id')
        assert 'user_id' not in sample_request['metadata']
        assert 'session' in sample_request['metadata']  # Other keys preserved

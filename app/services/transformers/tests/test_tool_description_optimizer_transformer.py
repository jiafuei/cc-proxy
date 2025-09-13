"""Tests for ToolDescriptionOptimizerTransformer."""

from unittest.mock import MagicMock

import pytest

from app.services.transformers.utils import ToolDescriptionOptimizerTransformer


class TestToolDescriptionOptimizerTransformer:
    """Test cases for ToolDescriptionOptimizerTransformer."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def transformer(self, mock_logger):
        """Create transformer instance."""
        return ToolDescriptionOptimizerTransformer(mock_logger)

    @pytest.mark.asyncio
    async def test_transform_optimizes_bash_tool_description(self, transformer):
        """Test that Bash tool description is actually optimized."""
        request = {
            'tools': [
                {
                    'name': 'Bash',
                    'description': 'Original bash description',
                }
            ]
        }
        headers = {}

        result_request, result_headers = await transformer.transform({'request': request, 'headers': headers})

        # Should replace with optimized description
        assert result_request['tools'][0]['description'] == transformer.TOOL_DESCRIPTION_MAP['Bash']
        assert result_headers == headers

    @pytest.mark.asyncio
    async def test_transform_leaves_unknown_tools_unchanged(self, transformer):
        """Test that tools not in mapping are left unchanged."""
        request = {
            'tools': [
                {
                    'name': 'UnknownTool',
                    'description': 'Original description',
                }
            ]
        }
        headers = {}

        result_request, result_headers = await transformer.transform({'request': request, 'headers': headers})

        assert result_request['tools'][0]['description'] == 'Original description'
        assert result_headers == headers

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'request_data',
        [
            {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]},  # No tools
            {'tools': []},  # Empty tools array
        ],
    )
    async def test_transform_handles_no_tools_scenarios(self, transformer, request_data):
        """Test that requests without tools or with empty tools are handled correctly."""
        headers = {'content-type': 'application/json'}

        result_request, result_headers = await transformer.transform({'request': request_data, 'headers': headers})

        assert result_request == request_data
        assert result_headers == headers

    @pytest.mark.asyncio
    async def test_transform_handles_malformed_tools(self, transformer):
        """Test that malformed tools are handled gracefully."""
        request = {
            'tools': [
                {'name': 'Read'},  # Missing description
                {'description': 'A tool without name'},  # Missing name
                'not-a-dict',  # Not a dictionary
                None,  # None value
            ]
        }
        headers = {}

        # Should not crash and return original request
        result_request, result_headers = await transformer.transform({'request': request, 'headers': headers})

        assert result_request == request
        assert result_headers == headers

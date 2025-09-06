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

    def test_transform_with_matching_tools(self, transformer, mock_logger):
        """Test that descriptions are optimized for matching tools."""
        request = {
            'tools': [
                {
                    'name': 'Read',
                    'description': 'Original read description',
                },
                {
                    'name': 'Write',
                    'description': 'Original write description',
                }
            ]
        }
        headers = {}

        # Act
        result_request, result_headers = transformer.transform({'request': request, 'headers': headers})

        # Assert
        assert result_request['tools'][0]['description'] == "Read a file from the local filesystem. You can access any file directly by using this tool."
        assert result_request['tools'][1]['description'] == "Write content to a file on the local filesystem. This tool will overwrite existing files."
        assert result_headers == headers
        assert mock_logger.debug.call_count == 2

    def test_transform_with_non_matching_tools(self, transformer, mock_logger):
        """Test that non-matching tools are left unchanged."""
        request = {
            'tools': [
                {
                    'name': 'UnknownTool',
                    'description': 'Original description',
                }
            ]
        }
        headers = {}

        # Act
        result_request, result_headers = transformer.transform({'request': request, 'headers': headers})

        # Assert
        assert result_request['tools'][0]['description'] == 'Original description'
        assert result_headers == headers
        mock_logger.debug.assert_not_called()

    def test_transform_without_tools(self, transformer, mock_logger):
        """Test that requests without tools are handled correctly."""
        request = {
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'Hello'}]
        }
        headers = {'content-type': 'application/json'}

        # Act
        result_request, result_headers = transformer.transform({'request': request, 'headers': headers})

        # Assert
        assert result_request == request
        assert result_headers == headers
        mock_logger.debug.assert_not_called()

    def test_transform_with_empty_tools(self, transformer, mock_logger):
        """Test that requests with empty tools array are handled correctly."""
        request = {
            'tools': []
        }
        headers = {}

        # Act
        result_request, result_headers = transformer.transform({'request': request, 'headers': headers})

        # Assert
        assert result_request == request
        assert result_headers == headers
        mock_logger.debug.assert_not_called()

    def test_transform_with_malformed_tool(self, transformer, mock_logger):
        """Test that malformed tools (missing name/description) are handled correctly."""
        request = {
            'tools': [
                {
                    'name': 'Read'
                    # Missing description field
                },
                {
                    'description': 'A tool without name'
                    # Missing name field
                },
                'not-a-dict',  # Not a dictionary
                None  # None value
            ]
        }
        headers = {}

        # Act
        result_request, result_headers = transformer.transform({'request': request, 'headers': headers})

        # Assert
        assert result_request == request
        assert result_headers == headers
        mock_logger.debug.assert_not_called()
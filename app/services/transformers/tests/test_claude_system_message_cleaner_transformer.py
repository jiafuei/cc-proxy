"""Tests for ClaudeSystemMessageCleanerTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.anthropic import ClaudeSystemMessageCleanerTransformer


class TestClaudeSystemMessageCleanerTransformer:
    """Test cases for the Claude system message cleaner transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return ClaudeSystemMessageCleanerTransformer(mock_logger)

    @pytest.mark.asyncio
    async def test_transform_method(self, transformer):
        """Test the transform method passes through request and headers."""
        request = {'model': 'claude-sonnet-4-20250514', 'system': [{'type': 'text', 'text': 'Test message'}]}
        headers = {'x-api-key': 'test-key'}
        params = {'request': request, 'headers': headers}

        result_request, result_headers = await transformer.transform(params)

        assert result_request == request
        assert result_headers == headers

    def test_remove_system_git_status_suffix(self, transformer):
        """Test removal of git status suffix from system messages."""
        # Test case 1: Normal case with git status suffix
        request_with_git_status = {
            'system': [
                {
                    'type': 'text',
                    'text': 'You are Claude Code, an AI assistant.\nSome instructions here.\n\ngitStatus: This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.\nCurrent branch: master\n\nMain branch: master\n\nStatus:\nmodified: file.py\n\nRecent commits:\nabc123 Latest commit',
                }
            ]
        }

        transformer._remove_system_git_status_suffix(request_with_git_status)

        expected_text = 'You are Claude Code, an AI assistant.\nSome instructions here.\n'
        assert request_with_git_status['system'][0]['text'] == expected_text

        # Test case 2: No git status suffix - should remain unchanged (fixed behavior)
        request_no_git_status = {'system': [{'type': 'text', 'text': 'You are Claude Code, an AI assistant.\nSome instructions here.'}]}
        original_text = request_no_git_status['system'][0]['text']

        transformer._remove_system_git_status_suffix(request_no_git_status)

        # With the fix, text without gitStatus should remain unchanged
        assert request_no_git_status['system'][0]['text'] == original_text

        # Test case 3: Empty system array - should handle gracefully (fixed behavior)
        request_empty_system = {'system': []}
        transformer._remove_system_git_status_suffix(request_empty_system)
        # Should not raise exception and array should remain empty
        assert request_empty_system['system'] == []

        # Test case 4: Missing system key - should handle gracefully
        request_no_system = {}
        transformer._remove_system_git_status_suffix(request_no_system)
        # Should not raise exception

        # Test case 5: Non-string text content - should skip processing
        request_non_string = {
            'system': [
                {
                    'type': 'text',
                    'text': 123,  # Non-string content
                }
            ]
        }

        transformer._remove_system_git_status_suffix(request_non_string)

        assert request_non_string['system'][0]['text'] == 123  # Should remain unchanged

        # Test case 6: Multiple gitStatus occurrences - should truncate at last one
        request_multiple_git_status = {
            'system': [{'type': 'text', 'text': 'Instructions.\ngitStatus: old status\nMore text.\ngitStatus: This is the git status at the start\nCurrent branch: master'}]
        }

        transformer._remove_system_git_status_suffix(request_multiple_git_status)

        expected_text_multiple = 'Instructions.\ngitStatus: old status\nMore text.'
        assert request_multiple_git_status['system'][0]['text'] == expected_text_multiple

        # Test case 7: Multiple system messages - should operate on the last one
        request_multiple_system = {'system': [{'type': 'text', 'text': 'First system message'}, {'type': 'text', 'text': 'Second system message.\ngitStatus: git info here'}]}

        transformer._remove_system_git_status_suffix(request_multiple_system)

        # First message should remain unchanged
        assert request_multiple_system['system'][0]['text'] == 'First system message'
        # Last message should be truncated
        assert request_multiple_system['system'][1]['text'] == 'Second system message.'

"""Tests for OpenAI Response Transformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.openai import OpenAIResponseTransformer


class TestOpenAIResponseTransformer:
    """Test cases for the OpenAI response transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        mock_logger = Mock()
        return OpenAIResponseTransformer(mock_logger)

    def test_usage_mapping_comprehensive(self, transformer):
        """Test comprehensive OpenAI to Claude usage mapping."""
        openai_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'prompt_tokens_details': {'cached_tokens': 20, 'audio_tokens': 5},
            'completion_tokens_details': {'reasoning_tokens': 15, 'audio_tokens': 0, 'accepted_prediction_tokens': 10, 'rejected_prediction_tokens': 2},
        }

        result = transformer._map_openai_usage_to_claude(openai_usage)

        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['cache_read_input_tokens'] == 20
        assert result['reasoning_output_tokens'] == 15

    def test_usage_mapping_with_missing_fields(self, transformer):
        """Test usage mapping handles missing fields gracefully."""
        # Test with minimal usage
        minimal_usage = {'prompt_tokens': 10}
        result = transformer._map_openai_usage_to_claude(minimal_usage)
        assert result['input_tokens'] == 10
        assert 'output_tokens' not in result
        assert 'cache_read_input_tokens' not in result

        # Test with empty usage
        empty_result = transformer._map_openai_usage_to_claude({})
        assert empty_result == {}

        # Test with None
        none_result = transformer._map_openai_usage_to_claude(None)
        assert none_result == {}

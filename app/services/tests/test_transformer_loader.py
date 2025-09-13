"""Tests for the transformer loader."""

import pytest

from app.services.transformer_loader import TransformerLoader


def test_load_built_in_transformer():
    """Test loading a built-in transformer."""
    loader = TransformerLoader()

    config = {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': 'api_key'}]}}

    transformer = loader.load_transformer(config)
    assert transformer is not None
    assert hasattr(transformer, 'operations')
    assert len(transformer.operations) == 1
    assert transformer.operations[0]['key'] == 'authorization'


def test_load_transformer_caching():
    """Test that transformers are cached properly."""
    loader = TransformerLoader()

    config = {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': 'api_key'}]}}

    # Load transformer twice
    transformer1 = loader.load_transformer(config)
    transformer2 = loader.load_transformer(config)

    # Should be the same instance (cached)
    assert transformer1 is transformer2
    assert len(loader._cache) == 1


def test_load_transformer_different_params():
    """Test that transformers with different params are not cached together."""
    loader = TransformerLoader()

    config1 = {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': 'api_key'}]}}

    config2 = {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'x-api-key', 'prefix': '', 'value': 'api_key'}]}}

    transformer1 = loader.load_transformer(config1)
    transformer2 = loader.load_transformer(config2)

    # Should be different instances
    assert transformer1 is not transformer2
    assert len(loader._cache) == 2


def test_load_transformer_failure():
    """Test handling of transformer loading failure."""
    loader = TransformerLoader()

    config = {'class': 'non.existent.Transformer', 'params': {}}

    with pytest.raises(RuntimeError, match='Cannot load transformer'):
        loader.load_transformer(config)


def test_load_multiple_transformers():
    """Test loading multiple transformers at once."""
    loader = TransformerLoader()

    configs = [
        {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': 'api_key'}]}},
        {'class': 'app.services.transformers.anthropic.AnthropicResponseTransformer', 'params': {}},
    ]

    transformers = loader.load_transformers(configs)
    assert len(transformers) == 2
    assert all(t is not None for t in transformers)


def test_load_multiple_transformers_with_failure():
    """Test loading multiple transformers when one fails."""
    loader = TransformerLoader()

    configs = [
        {'class': 'app.services.transformers.utils.HeaderTransformer', 'params': {'operations': [{'key': 'authorization', 'prefix': 'Bearer ', 'value': 'api_key'}]}},
        {'class': 'non.existent.Transformer', 'params': {}},
        {'class': 'app.services.transformers.anthropic.AnthropicResponseTransformer', 'params': {}},
    ]

    # Should continue loading even if one fails
    transformers = loader.load_transformers(configs)
    assert len(transformers) == 2  # Only the working ones

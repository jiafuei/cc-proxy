"""Tests for the transformer loader."""

import pytest

from app.services.transformer_loader import TransformerLoader


def test_transformer_loader_initialization():
    """Test transformer loader initialization."""
    loader = TransformerLoader()
    assert loader is not None
    assert len(loader._cache) == 0


def test_transformer_loader_with_paths():
    """Test transformer loader with custom paths."""
    test_paths = ['/test/path1', '/test/path2']
    loader = TransformerLoader(test_paths)
    assert loader is not None


def test_load_built_in_transformer():
    """Test loading a built-in transformer."""
    loader = TransformerLoader()

    config = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'test-key', 'base_url': 'https://api.anthropic.com'}}

    transformer = loader.load_transformer(config)
    assert transformer is not None
    assert transformer.api_key == 'test-key'


def test_load_transformer_caching():
    """Test that transformers are cached properly."""
    loader = TransformerLoader()

    config = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'test-key'}}

    # Load transformer twice
    transformer1 = loader.load_transformer(config)
    transformer2 = loader.load_transformer(config)

    # Should be the same instance (cached)
    assert transformer1 is transformer2
    assert len(loader._cache) == 1


def test_load_transformer_different_params():
    """Test that transformers with different params are not cached together."""
    loader = TransformerLoader()

    config1 = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'key1'}}

    config2 = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'key2'}}

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
        {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'key1'}},
        {'class': 'app.services.transformer_interfaces.AnthropicResponseTransformer', 'params': {}},
    ]

    transformers = loader.load_transformers(configs)
    assert len(transformers) == 2
    assert all(t is not None for t in transformers)


def test_load_multiple_transformers_with_failure():
    """Test loading multiple transformers when one fails."""
    loader = TransformerLoader()

    configs = [
        {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'key1'}},
        {'class': 'non.existent.Transformer', 'params': {}},
        {'class': 'app.services.transformer_interfaces.AnthropicResponseTransformer', 'params': {}},
    ]

    # Should continue loading even if one fails
    transformers = loader.load_transformers(configs)
    assert len(transformers) == 2  # Only the working ones


def test_clear_cache():
    """Test clearing the transformer cache."""
    loader = TransformerLoader()

    config = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'test-key'}}

    loader.load_transformer(config)
    assert len(loader._cache) == 1

    loader.clear_cache()
    assert len(loader._cache) == 0


def test_get_cache_info():
    """Test getting cache information."""
    loader = TransformerLoader()

    info = loader.get_cache_info()
    assert info['cached_transformers'] == 0
    assert info['cache_keys'] == []

    config = {'class': 'app.services.transformer_interfaces.AnthropicAuthTransformer', 'params': {'api_key': 'test-key'}}

    loader.load_transformer(config)

    info = loader.get_cache_info()
    assert info['cached_transformers'] == 1
    assert len(info['cache_keys']) == 1

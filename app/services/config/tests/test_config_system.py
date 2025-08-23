"""Tests for user configuration system."""

from unittest.mock import Mock

import pytest

from app.config.user_models import ModelConfig, ProviderConfig, RoutingConfig, TransformerConfig, UserConfig
from app.services.config.simple_user_config_manager import SimpleUserConfigManager
from app.services.registry.models import ModelRegistry
from app.services.registry.providers import ProviderRegistry
from app.services.registry.transformers import TransformerRegistry
from app.services.routing.inspector import RequestInspector
from app.services.routing.processor import RequestProcessor


class TestUserConfigModels:
    """Test user configuration models."""

    def test_empty_config_loads(self):
        """Test that empty configuration can be loaded."""
        config = UserConfig()
        assert config.transformers == []
        assert config.providers == []
        assert config.models == []
        assert config.routing is None

    def test_config_validation_passes_for_valid_config(self, tmp_path):
        """Test validation passes for valid configuration."""
        # Create a dummy transformer file
        dummy_transformer = tmp_path / 'transformer.py'
        dummy_transformer.write_text('# dummy transformer')

        config = UserConfig(
            transformers=[TransformerConfig(name='test-transformer', path=str(dummy_transformer), args=[])],
            providers=[ProviderConfig(name='test-provider', api_key='test-key', url='https://example.com/api', request_pipeline=[], response_pipeline=[])],
            models=[ModelConfig(id='test-model', provider='test-provider')],
            routing=RoutingConfig(default='test-model', planning='test-model', background='test-model'),
        )

        # Should not raise
        config.validate_references()

    def test_config_validation_fails_for_invalid_references(self):
        """Test validation fails for invalid references."""
        config = UserConfig(
            models=[ModelConfig(id='test-model', provider='nonexistent-provider')],
            routing=RoutingConfig(default='nonexistent-model', planning='test-model', background='test-model'),
        )

        with pytest.raises(ValueError, match='Configuration validation failed'):
            config.validate_references()


class TestTransformerRegistry:
    """Test transformer registry."""

    def test_registry_creation(self):
        """Test transformer registry can be created."""
        registry = TransformerRegistry()
        assert registry.size() == 0

    def test_clear_registry(self):
        """Test clearing registry."""
        registry = TransformerRegistry()
        # Add a mock factory
        mock_factory = Mock()
        registry.register('test', mock_factory)
        assert registry.size() == 1

        registry.clear_all()
        assert registry.size() == 0


class TestModelRegistry:
    """Test model registry."""

    def test_model_registration(self):
        """Test registering models."""
        registry = ModelRegistry()
        config = ModelConfig(id='test-model', provider='test-provider')

        registry.register_model_from_config(config)

        assert registry.model_exists('test-model')
        assert registry.get_provider_for_model('test-model') == 'test-provider'
        assert registry.provider_has_models('test-provider')

    def test_model_validation(self):
        """Test model validation."""
        registry = ModelRegistry()
        config = ModelConfig(id='test-model', provider='test-provider')
        registry.register_model_from_config(config)

        # Should have error for unknown provider
        errors = registry.validate_model_references(['other-provider'])
        assert len(errors) == 1
        assert 'test-provider' in errors[0]

        # Should pass for known provider
        errors = registry.validate_model_references(['test-provider'])
        assert len(errors) == 0


class TestRequestInspector:
    """Test request inspector."""

    def test_inspector_creation(self):
        """Test inspector can be created."""
        inspector = RequestInspector()
        assert inspector.get_stats()['total_requests'] == 0

    def test_routing_fallback(self):
        """Test routing falls back to default."""
        inspector = RequestInspector()

        # Mock a simple request
        mock_request = Mock()
        mock_request.messages = []
        mock_request.metadata = None
        mock_request.tools = None
        mock_request.thinking = None

        routing_key = inspector.determine_routing_key(mock_request)
        assert routing_key in ['default', 'planning', 'background']

        stats = inspector.get_stats()
        assert stats['total_requests'] == 1


class TestRequestProcessor:
    """Test request processor."""

    def test_processor_creation_with_empty_config(self):
        """Test processor can be created with empty configuration."""
        user_config = UserConfig()
        model_registry = ModelRegistry()
        provider_registry = ProviderRegistry(Mock())

        processor = RequestProcessor(user_config, model_registry, provider_registry)

        # Should handle missing routing gracefully
        routing_key = processor.get_target_model('default')
        assert routing_key is None

    def test_routing_validation_with_empty_config(self):
        """Test routing validation with empty configuration."""
        user_config = UserConfig()
        model_registry = ModelRegistry()
        provider_registry = ProviderRegistry(Mock())

        processor = RequestProcessor(user_config, model_registry, provider_registry)

        errors = processor.validate_routing_configuration()
        assert len(errors) == 1
        assert 'No routing configuration' in errors[0]


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock configuration file."""
    config_file = tmp_path / 'user.yaml'
    # Create a dummy transformer file
    dummy_transformer = tmp_path / 'test_transformer.py'
    dummy_transformer.write_text('# dummy transformer')

    config_content = f"""
transformers:
  - name: 'test-transformer'
    path: '{dummy_transformer}'
    args: []

providers:
  - name: 'test-provider'
    api_key: 'test-key'
    url: 'https://api.example.com/v1'
    request_pipeline: []
    response_pipeline: []

models:
  - id: 'test-model'
    provider: 'test-provider'

routing:
  default: 'test-model'
  planning: 'test-model'
  background: 'test-model'
"""
    config_file.write_text(config_content)
    return config_file


class TestUserConfigManager:
    """Test user configuration manager."""

    def test_config_manager_creation(self):
        """Test config manager can be created."""
        manager = SimpleUserConfigManager()
        assert manager.get_current_config() is None

    def test_load_config_with_missing_file(self):
        """Test loading config when file doesn't exist."""
        manager = SimpleUserConfigManager()
        config = manager.load_config()

        # Should return empty config
        assert isinstance(config, UserConfig)
        assert len(config.transformers) == 0

    def test_config_manager_with_mock_file(self, mock_config_file):
        """Test config manager with mock file."""
        manager = SimpleUserConfigManager(config_path=mock_config_file)
        config = manager.load_config()

        # Should load the config
        assert len(config.transformers) == 1
        assert len(config.providers) == 1
        assert len(config.models) == 1
        assert config.routing is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

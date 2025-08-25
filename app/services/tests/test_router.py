"""Tests for the simple router."""

from unittest.mock import Mock

from app.common.models import ClaudeRequest
from app.config.user_models import RoutingConfig
from app.services.router import RequestInspector, SimpleRouter


def test_routing_config():
    """Test routing configuration."""
    config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    assert config.default == 'claude-3-sonnet'
    assert config.planning == 'claude-3-opus'
    assert config.background == 'claude-3-haiku'


def test_request_inspector_planning():
    """Test request inspector identifies planning requests."""
    inspector = RequestInspector()

    # Create a planning request
    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Please create a detailed plan for implementing this feature'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'planning'


def test_request_inspector_background():
    """Test request inspector identifies background requests."""
    inspector = RequestInspector()

    # Create a background request
    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Please analyze this data and generate a report'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'background'


def test_request_inspector_default():
    """Test request inspector defaults to 'default' routing."""
    inspector = RequestInspector()

    # Create a regular request
    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello, how are you?'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'default'


def test_simple_router_success():
    """Test simple router successfully routes request."""
    # Mock provider manager
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    router = SimpleRouter(mock_provider_manager, routing_config)

    # Create a request
    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello'}])

    provider = router.get_provider_for_request(request)
    assert provider == mock_provider
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-sonnet')


def test_simple_router_no_provider():
    """Test simple router when no provider is available."""
    # Mock provider manager that returns None
    mock_provider_manager = Mock()
    mock_provider_manager.get_provider_for_model.return_value = None

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    router = SimpleRouter(mock_provider_manager, routing_config)

    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello'}])

    provider = router.get_provider_for_request(request)
    assert provider is None


def test_simple_router_planning_request():
    """Test simple router routes planning request to correct model."""
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    router = SimpleRouter(mock_provider_manager, routing_config)

    # Create a planning request
    request = ClaudeRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Create a detailed plan for this project'}])

    provider = router.get_provider_for_request(request)
    assert provider == mock_provider
    # Should route to planning model, not the model in the request
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-opus')


def test_simple_router_get_provider_for_model():
    """Test simple router can get provider for specific model."""
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    router = SimpleRouter(mock_provider_manager, RoutingConfig(default=''))

    provider = router.get_provider_for_model('claude-3-sonnet')
    assert provider == mock_provider
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-sonnet')


def test_simple_router_list_available_models():
    """Test simple router can list available models."""
    mock_provider_manager = Mock()
    mock_provider_manager.list_models.return_value = ['claude-3-sonnet', 'claude-3-opus']

    router = SimpleRouter(mock_provider_manager, RoutingConfig(default=''))

    models = router.list_available_models()
    assert models == ['claude-3-sonnet', 'claude-3-opus']
    mock_provider_manager.list_models.assert_called_once()


def test_simple_router_get_routing_info():
    """Test simple router can provide routing information."""
    mock_provider_manager = Mock()
    mock_provider_manager.list_models.return_value = ['claude-3-sonnet']
    mock_provider_manager.list_providers.return_value = ['anthropic']

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    router = SimpleRouter(mock_provider_manager, routing_config)

    info = router.get_routing_info()
    assert info['default_model'] == 'claude-3-sonnet'
    assert info['planning_model'] == 'claude-3-opus'
    assert info['background_model'] == 'claude-3-haiku'
    assert info['available_models'] == ['claude-3-sonnet']
    assert info['providers'] == ['anthropic']


def test_provider_manager_default_provider_fallback():
    """Test that ProviderManager returns default provider when no match found."""
    from app.services.provider import ProviderManager
    from app.services.transformer_loader import TransformerLoader
    from unittest.mock import patch
    
    # Mock environment variables for the test
    with patch.dict('os.environ', {'CCPROXY_FALLBACK_API_KEY': 'test-key', 'CCPROXY_FALLBACK_URL': 'https://api.anthropic.com/v1/messages'}):
        transformer_loader = TransformerLoader([])
        provider_manager = ProviderManager([], transformer_loader)  # No configured providers
        
        # Should return default provider for any model
        provider = provider_manager.get_provider_for_model('claude-3-unknown-model')
        assert provider is not None
        assert provider.config.name.startswith('default-anthropic')
        
        # Should include default provider in list
        providers = provider_manager.list_providers()
        assert any('default-anthropic' in p for p in providers)

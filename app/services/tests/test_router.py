"""Tests for the simple router."""

from unittest.mock import Mock, patch

from app.common.models import AnthropicRequest
from app.config.user_models import RoutingConfig
from app.services.router import RequestInspector, SimpleRouter
from app.services.transformer_loader import TransformerLoader


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
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Please create a detailed plan for implementing this feature'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'planning'


def test_request_inspector_background():
    """Test request inspector identifies background requests."""
    inspector = RequestInspector()

    # Create a background request with low max_tokens (since keyword matching is currently disabled)
    request = AnthropicRequest(model='claude-3-sonnet', max_tokens=500, messages=[{'role': 'user', 'content': 'Please analyze this data and generate a report'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'background'


def test_request_inspector_default():
    """Test request inspector defaults to 'default' routing."""
    inspector = RequestInspector()

    # Create a regular request
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello, how are you?'}])

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'default'


def test_simple_router_success():
    """Test simple router successfully routes request."""
    # Mock provider manager
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    # Mock transformer loader
    mock_transformer_loader = Mock()

    router = SimpleRouter(mock_provider_manager, routing_config, mock_transformer_loader)

    # Create a request
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello'}])

    provider, _ = router.get_provider_for_request(request)
    assert provider == mock_provider
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-sonnet')


def test_simple_router_no_provider():
    """Test simple router when no configured provider is available but fallback provider exists."""
    # Mock provider manager that returns None (no configured provider)
    mock_provider_manager = Mock()
    mock_provider_manager.get_provider_for_model.return_value = None

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    # Mock transformer loader and create environment for default provider
    mock_transformer_loader = Mock()

    with patch.dict('os.environ', {'CCPROXY_FALLBACK_API_KEY': 'test-key', 'CCPROXY_FALLBACK_URL': 'https://api.anthropic.com/v1/messages'}):
        with patch('app.services.router.Provider') as mock_provider_class:
            mock_default_provider = Mock()
            mock_provider_class.return_value = mock_default_provider

            router = SimpleRouter(mock_provider_manager, routing_config, mock_transformer_loader)

            request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello'}])

            provider, _ = router.get_provider_for_request(request)
            # Should always return a provider (fallback provider in this case)
            assert provider == mock_default_provider
            assert provider is not None


def test_simple_router_planning_request():
    """Test simple router routes planning request to correct model."""
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    # Mock transformer loader
    mock_transformer_loader = Mock()

    router = SimpleRouter(mock_provider_manager, routing_config, mock_transformer_loader)

    # Create a planning request
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Create a detailed plan for this project'}])

    provider, _ = router.get_provider_for_request(request)
    assert provider == mock_provider
    # Should route to planning model, not the model in the request
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-opus')


def test_simple_router_get_provider_for_model():
    """Test simple router can get provider for specific model."""
    mock_provider_manager = Mock()
    mock_provider = Mock()
    mock_provider_manager.get_provider_for_model.return_value = mock_provider

    # Mock transformer loader
    mock_transformer_loader = Mock()

    router = SimpleRouter(mock_provider_manager, RoutingConfig(default=''), mock_transformer_loader)

    provider = router.get_provider_for_model('claude-3-sonnet')
    assert provider == mock_provider
    mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-3-sonnet')


def test_simple_router_list_available_models():
    """Test simple router can list available models."""
    mock_provider_manager = Mock()
    mock_provider_manager.list_models.return_value = ['claude-3-sonnet', 'claude-3-opus']

    # Mock transformer loader
    mock_transformer_loader = Mock()

    router = SimpleRouter(mock_provider_manager, RoutingConfig(default=''), mock_transformer_loader)

    models = router.list_available_models()
    assert models == ['claude-3-sonnet', 'claude-3-opus']
    mock_provider_manager.list_models.assert_called_once()


def test_simple_router_get_routing_info():
    """Test simple router can provide routing information."""
    mock_provider_manager = Mock()
    mock_provider_manager.list_models.return_value = ['claude-3-sonnet']
    mock_provider_manager.list_providers.return_value = ['anthropic']

    routing_config = RoutingConfig(default='claude-3-sonnet', planning='claude-3-opus', background='claude-3-haiku')

    # Mock transformer loader
    mock_transformer_loader = Mock()

    router = SimpleRouter(mock_provider_manager, routing_config, mock_transformer_loader)

    info = router.get_routing_info()
    assert info['default_model'] == 'claude-3-sonnet'
    assert info['planning_model'] == 'claude-3-opus'
    assert info['background_model'] == 'claude-3-haiku'
    assert info['available_models'] == ['claude-3-sonnet']
    assert info['providers'] == ['anthropic']


def test_simple_router_default_provider_fallback():
    """Test that SimpleRouter returns default provider when no configured provider is found."""
    # Mock provider manager that has no configured providers
    mock_provider_manager = Mock()
    mock_provider_manager.get_provider_for_model.return_value = None  # No configured provider

    # Mock environment variables for the test
    with patch.dict('os.environ', {'CCPROXY_FALLBACK_API_KEY': 'test-key', 'CCPROXY_FALLBACK_URL': 'https://api.anthropic.com/v1/messages'}):
        with patch('app.services.router.Provider') as mock_provider_class:
            mock_default_provider = Mock()
            mock_default_provider.config.name = 'default-anthropic (fallback)'
            mock_provider_class.return_value = mock_default_provider

            transformer_loader = TransformerLoader([])
            routing_config = RoutingConfig(default='claude-3-unknown-model', planning='', background='')
            router = SimpleRouter(mock_provider_manager, routing_config, transformer_loader)

            # Create a request
            request = AnthropicRequest(model='test', messages=[{'role': 'user', 'content': 'Hello'}])

            # Should always return default provider since no configured provider found
            provider, _ = router.get_provider_for_request(request)
            assert provider == mock_default_provider
            assert provider is not None  # Never returns None


def test_simple_router_empty_routing_config():
    """Test that SimpleRouter works with completely empty routing config."""
    mock_provider_manager = Mock()
    mock_provider_manager.get_provider_for_model.return_value = None  # No configured provider

    # Mock transformer loader and create environment for default provider
    mock_transformer_loader = Mock()

    with patch.dict('os.environ', {'CCPROXY_FALLBACK_API_KEY': 'test-key'}):
        with patch('app.services.router.Provider') as mock_provider_class:
            mock_default_provider = Mock()
            mock_provider_class.return_value = mock_default_provider

            # Empty routing config (all empty strings)
            routing_config = RoutingConfig(default='', planning='', background='')
            router = SimpleRouter(mock_provider_manager, routing_config, mock_transformer_loader)

            request = AnthropicRequest(model='test', messages=[{'role': 'user', 'content': 'Hello'}])

            # Should use fallback model and return default provider
            provider, _ = router.get_provider_for_request(request)
            assert provider == mock_default_provider
            assert provider is not None

            # Should have tried to get provider for fallback model 'claude-sonnet-4-20250514'
            mock_provider_manager.get_provider_for_model.assert_called_once_with('claude-sonnet-4-20250514')


def test_request_inspector_plan_mode_activation_string_content():
    """Test request inspector detects plan mode activation in string content."""
    inspector = RequestInspector()

    # Create request with plan mode activation in string content
    request = AnthropicRequest(
        model='claude-3-sonnet',
        messages=[
            {'role': 'user', 'content': 'Some content'},
            {'role': 'assistant', 'content': 'Response'},
            {'role': 'user', 'content': 'More content <system-reminder>\nPlan mode is active. Please help'},
        ],
    )

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'planning'


def test_request_inspector_plan_mode_activation_list_content():
    """Test request inspector detects plan mode activation in list content blocks."""
    inspector = RequestInspector()

    # Create request with plan mode activation in content blocks
    request = AnthropicRequest(
        model='claude-3-sonnet',
        messages=[
            {'role': 'user', 'content': 'Some content'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Regular text'}, {'type': 'text', 'text': '<system-reminder>\nPlan mode is active. More text'}]},
        ],
    )

    routing_key = inspector.determine_routing_key(request)
    assert routing_key == 'planning'


def test_request_inspector_plan_mode_only_last_user_message():
    """Test request inspector only checks the last user message for plan mode."""
    inspector = RequestInspector()

    # Create request with plan mode text in earlier message but not last
    # Avoid planning keywords to isolate plan mode detection 
    request = AnthropicRequest(
        model='claude-3-sonnet',
        messages=[
            {'role': 'user', 'content': '<system-reminder>\nPlan mode is active. Earlier message content'},
            {'role': 'assistant', 'content': 'Response'},
            {'role': 'user', 'content': 'Final user message without mode activation, just a simple hello'},
        ],
    )

    routing_key = inspector.determine_routing_key(request)
    # Should be 'planning' due to keyword matching ('plan' in first message), not plan mode activation
    assert routing_key == 'planning'


def test_request_inspector_plan_mode_vs_keyword_isolation():
    """Test that plan mode detection is separate from keyword detection."""
    inspector = RequestInspector()

    # Create request with no planning keywords but with plan mode text in non-last message
    request = AnthropicRequest(
        model='claude-3-sonnet', 
        messages=[
            {'role': 'user', 'content': '<system-reminder>\nThe mode is active. Earlier message'},  # No 'plan' keyword
            {'role': 'assistant', 'content': 'Response'},
            {'role': 'user', 'content': 'Final user message, just a simple hello'},
        ],
    )

    routing_key = inspector.determine_routing_key(request)
    # Should be 'default' since no plan mode activation in last message and no planning keywords
    assert routing_key == 'default'


def test_request_inspector_plan_mode_partial_match():
    """Test request inspector requires exact plan mode text match."""
    inspector = RequestInspector()

    # Create request with partial plan mode text (avoid 'plan' keyword)
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'The mode is active but missing system reminder format'}])

    routing_key = inspector.determine_routing_key(request)
    # Should not be 'planning' since text doesn't match exactly
    assert routing_key != 'planning'


def test_request_inspector_plan_mode_no_user_messages():
    """Test request inspector handles requests with no user messages."""
    inspector = RequestInspector()

    # Create request with no user messages
    request = AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'assistant', 'content': '<system-reminder>\nPlan mode is active.'}])

    routing_key = inspector.determine_routing_key(request)
    # Should not be 'planning' since there are no user messages
    assert routing_key != 'planning'


def test_request_inspector_background_priority_over_plan_mode():
    """Test that background routing (max_tokens < 768) takes priority over plan mode."""
    inspector = RequestInspector()

    # Create request with both plan mode activation and low max_tokens
    request = AnthropicRequest(
        model='claude-3-sonnet',
        max_tokens=500,  # Less than 768 - should trigger background
        messages=[{'role': 'user', 'content': '<system-reminder>\nPlan mode is active. Please help'}],
    )

    routing_key = inspector.determine_routing_key(request)
    # Should be 'background' due to low max_tokens, even with plan mode text
    assert routing_key == 'background'


def test_request_inspector_plan_mode_priority_over_keywords():
    """Test that plan mode detection takes priority over keyword matching."""
    inspector = RequestInspector()

    # Create request with both plan mode activation and planning keywords
    request = AnthropicRequest(
        model='claude-3-sonnet',
        max_tokens=1000,  # High enough to avoid background routing
        messages=[{'role': 'user', 'content': '<system-reminder>\nPlan mode is active. Create a detailed plan and strategy'}],
    )

    routing_key = inspector.determine_routing_key(request)
    # Should be 'planning' due to plan mode activation (not keyword matching)
    assert routing_key == 'planning'

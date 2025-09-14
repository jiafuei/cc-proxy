"""Test utility functions for creating provider configurations with capabilities."""

from app.config.user_models import ProviderConfig


def create_test_provider_config(name='test-provider', url='https://api.test.com/v1', api_key='test-key', **kwargs):
    """Create a ProviderConfig with default capabilities for testing.
    
    Args:
        name: Provider name
        url: Provider URL
        api_key: API key
        **kwargs: Additional fields for ProviderConfig
    
    Returns:
        ProviderConfig with standard capabilities
    """
    defaults = {
        'transformers': {'request': [], 'response': []},
        'timeout': 30,
        'capabilities': [
            {'operation': 'messages', 'class_name': 'MessagesCapability'},
            {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
        ]
    }
    defaults.update(kwargs)
    
    return ProviderConfig(
        name=name,
        url=url, 
        api_key=api_key,
        **defaults
    )
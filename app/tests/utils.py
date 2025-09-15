"""Test utilities for dependency injection and mocking."""

from typing import Optional
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigurationService
from app.config.models import ConfigModel
from app.config.user_models import UserConfig, ProviderConfig, ModelConfig, RoutingConfig
from app.dependencies.service_container import ServiceContainer
from app.main import create_app
from app.services.config.simple_user_config_manager import SimpleUserConfigManager, set_user_config_manager
from app.services.config.interfaces import UserConfigManager
import pytest


class TestUserConfigManager(SimpleUserConfigManager):
    """Test implementation of UserConfigManager that uses in-memory config."""
    
    def __init__(self, config: Optional[UserConfig] = None):
        # Skip the parent __init__ to avoid filesystem operations
        self._current_config = config or UserConfig()
        self._callback = None
        self._config_path = Path('/tmp/test-user.yaml')  # Fake path for testing
        
    def load_config(self) -> UserConfig:
        """Return the pre-configured test config."""
        return self._current_config
    
    def get_current_config(self) -> Optional[UserConfig]:
        """Get the test configuration."""
        return self._current_config
    
    async def reload_config(self) -> UserConfig:
        """No-op reload for tests."""
        return self._current_config
    
    def get_config_status(self) -> dict:
        """Get current configuration status for tests."""
        config = self.get_current_config()

        if config is None:
            return {'loaded': False, 'config_file_exists': False, 'config_path': str(self._config_path)}

        return {
            'loaded': True,
            'config_file_exists': True,  # Always true for tests
            'config_path': str(self._config_path),
            'transformer_paths': len(config.transformer_paths),
            'providers': len(config.providers),
            'models': len(config.models),
            'routing_configured': config.routing is not None,
            'transformer_paths_list': config.transformer_paths,
            'provider_names': [p.name for p in config.providers],
        }


def create_test_user_config() -> UserConfig:
    """Create a realistic test user configuration with valid references."""
    return UserConfig(
        providers=[
            ProviderConfig(
                name='test-provider',
                url='https://api.test.com/v1/messages',
                api_key='test-key',
                type='anthropic',
                capabilities=['messages', 'count_tokens'],
                timeout=30
            )
        ],
        models=[
            ModelConfig(id='test-model', provider='test-provider', alias='test'),
            ModelConfig(id='claude-3-5-sonnet-20241022', provider='test-provider', alias='sonnet'),
            ModelConfig(id='claude-3-opus-20240229', provider='test-provider', alias='opus'),
            ModelConfig(id='claude-3-5-haiku-20241022', provider='test-provider', alias='haiku'),
        ],
        routing=RoutingConfig(
            default='sonnet',
            planning='opus',
            background='haiku', 
            thinking='sonnet',
            plan_and_think='opus',
            builtin_tools='sonnet'
        ),
        transformer_paths=[]
    )


class TestServiceFactory:
    """Factory for creating test service instances with controlled dependencies."""
    
    @staticmethod
    def create_test_config_service(config: Optional[ConfigModel] = None) -> ConfigurationService:
        """Create a test configuration service."""
        if config is None:
            # Create a real ConfigModel with proper defaults for testing (not Mock objects)
            config = ConfigModel(
                host='127.0.0.1',
                port=8000,
                dev=True,
                dump_requests=False,
                dump_responses=False,
                dump_headers=False,
                dump_dir=None,
                cors_allow_origins=[],  # Real list, not Mock
                redact_headers=['authorization', 'x-api-key', 'cookie'],  # Real list, not Mock
                fallback_api_url='https://api.anthropic.com/v1/messages',
                fallback_api_key='test-fallback-key'
            )
        
        # Return a real ConfigurationService that returns the real config
        config_service = ConfigurationService()
        # Override the get_config method to return our test config
        config_service.get_config = lambda: config
        config_service.reload_config = lambda: config
        return config_service
    
    @staticmethod
    def create_test_service_container(config_service: Optional[ConfigurationService] = None) -> ServiceContainer:
        """Create a test service container with mock components."""
        if config_service is None:
            config_service = TestServiceFactory.create_test_config_service()
        
        container = Mock(spec=ServiceContainer)
        container.config_service = config_service
        
        # Mock provider manager
        provider_manager = Mock()
        provider_manager.list_providers.return_value = []
        provider_manager.list_models.return_value = []
        container.provider_manager = provider_manager
        
        # Mock router - create a basic router that returns None provider by default
        router = Mock()
        router.get_routing_info.return_value = {}
        from app.services.router import RoutingResult
        router.get_provider_for_request.return_value = RoutingResult(
            provider=None,  # No provider by default for error testing
            routing_key='default',
            model_alias='test-model',
            resolved_model_id='unknown'
        )
        container.router = router
        
        # Mock transformer loader
        transformer_loader = Mock()
        transformer_loader.get_cache_info.return_value = {'cached_transformers': 0}
        container.transformer_loader = transformer_loader
        
        return container


class TestContext:
    """Test context manager for integration tests with proper dependency injection."""
    
    def __init__(self, config: Optional[ConfigModel] = None):
        """Initialize test context with optional config."""
        self.config = config or ConfigModel()
        self.app: Optional[FastAPI] = None
        self.client: Optional[TestClient] = None
    
    def __enter__(self):
        """Enter context and create test app."""
        self.app = create_app(self.config)
        self.client = TestClient(self.app)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup."""
        if self.client:
            self.client.close()
    
    def override_dependencies(self, **overrides):
        """Override app dependencies for testing."""
        if not self.app:
            raise RuntimeError("App not initialized. Use within context manager.")
        
        for dependency, override in overrides.items():
            self.app.dependency_overrides[dependency] = override


def create_test_app_with_mocks(
    config: Optional[ConfigModel] = None,
    service_container: Optional[ServiceContainer] = None,
    user_config: Optional[UserConfig] = None
) -> FastAPI:
    """Create a test FastAPI app with mock dependencies."""
    # Set up test user config manager BEFORE creating the app
    test_user_config = user_config or create_test_user_config()
    test_config_manager = TestUserConfigManager(test_user_config)
    set_user_config_manager(test_config_manager)
    
    if config is None:
        config = ConfigModel()
    
    if service_container is None:
        service_container = TestServiceFactory.create_test_service_container()
    
    # Now create the app - it will use our test config manager
    app = create_app(config)
    
    # Override app state with test service container
    app.state.service_container = service_container
    app.state.config_service = service_container.config_service
    app.state.config = service_container.config_service.get_config()
    
    # Override dependencies with mocks (for additional safety)
    from app.dependencies import get_config_service_dependency, get_service_container_dependency
    
    app.dependency_overrides[get_service_container_dependency] = lambda: service_container
    app.dependency_overrides[get_config_service_dependency] = lambda: service_container.config_service
    
    return app


@pytest.fixture(autouse=True)
def reset_global_config_manager():
    """Reset global config manager before and after each test."""
    # Clean up before test
    set_user_config_manager(None)
    
    yield
    
    # Clean up after test  
    set_user_config_manager(None)
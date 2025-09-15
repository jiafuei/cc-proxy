"""Test utilities for dependency injection and mocking."""

from typing import Optional
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import ConfigurationService
from app.config.models import ConfigModel
from app.dependencies.service_container import ServiceContainer
from app.main import create_app


class TestServiceFactory:
    """Factory for creating test service instances with controlled dependencies."""
    
    @staticmethod
    def create_test_config_service(config: Optional[ConfigModel] = None) -> ConfigurationService:
        """Create a test configuration service."""
        if config is None:
            config = ConfigModel()
        
        config_service = Mock(spec=ConfigurationService)
        config_service.get_config.return_value = config
        config_service.reload_config.return_value = config
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
        
        # Mock router
        router = Mock()
        router.get_routing_info.return_value = {}
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
    service_container: Optional[ServiceContainer] = None
) -> FastAPI:
    """Create a test FastAPI app with mock dependencies."""
    if config is None:
        config = ConfigModel()
    
    if service_container is None:
        service_container = TestServiceFactory.create_test_service_container()
    
    app = create_app(config)
    
    # Override dependencies with mocks
    from app.dependencies import get_config_service_dependency, get_service_container_dependency
    
    app.dependency_overrides[get_service_container_dependency] = lambda: service_container
    app.dependency_overrides[get_config_service_dependency] = lambda: service_container.config_service
    
    return app
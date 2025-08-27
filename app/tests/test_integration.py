"""Integration tests for end-to-end flows, configuration management, and resource handling."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
from app.config.user_models import ModelConfig, ProviderConfig, RoutingConfig, UserConfig
from app.main import app


class TestEndToEndRequestFlow:
    """Test complete request flows from API to provider and back."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that simulates realistic behavior."""
        provider = Mock()
        provider.config = Mock()
        provider.config.name = 'test-provider'

        async def process_request(payload, request, routing_key, dumper, correlation_id):
            # Simple SSE response like existing tests
            yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Hello! I understand your request."}}\n\n'
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        provider.process_request = process_request
        return provider

    @pytest.fixture
    def mock_service_container(self, mock_provider):
        """Create a realistic service container mock."""
        container = Mock()

        # Mock router with realistic behavior
        router = Mock()
        router.get_provider_for_request.return_value = mock_provider, 'test-routing-key'
        container.router = router

        return container

    @pytest.fixture
    def mock_dumper(self):
        """Create a mock dumper for dependency injection."""
        dumper = Mock()
        dumper.begin.return_value = DumpHandles(None, None, None, None, None, None, None, None, 'test-correlation-id')
        dumper.write_response_chunk = Mock()
        dumper.close = Mock()
        return dumper

    def test_complete_message_flow_success(self, mock_service_container, mock_dumper):
        """Test successful end-to-end message processing."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Override dependencies
        app.dependency_overrides[get_dumper] = lambda: mock_dumper

        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
                request_payload = {'model': 'claude-3-sonnet-20240229', 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'max_tokens': 100}

                response = client.post('/v1/messages', json=request_payload)

                assert response.status_code == 200
                assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

                content = response.content.decode()
                assert 'event: message_start' in content
                assert 'event: content_block_delta' in content
                assert 'event: message_stop' in content
                assert 'Hello! I understand your request.' in content

                # Verify router was called with correct request
                mock_service_container.router.get_provider_for_request.assert_called_once()
                call_args = mock_service_container.router.get_provider_for_request.call_args[0][0]
                assert call_args.model == 'claude-3-sonnet-20240229'
                assert len(call_args.messages) == 1

                # Verify dumper lifecycle
                mock_dumper.begin.assert_called_once()
                assert mock_dumper.write_response_chunk.call_count > 0
                mock_dumper.close.assert_called_once()
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_routing_integration_planning_keywords(self, mock_service_container):
        """Test that planning keywords properly route through the system."""
        client = TestClient(app)

        with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
            request_payload = {
                'model': 'claude-3-sonnet-20240229',
                'messages': [{'role': 'user', 'content': 'Please create a comprehensive plan and strategy for our project architecture.'}],
            }

            response = client.post('/v1/messages', json=request_payload)
            assert response.status_code == 200

            # Verify the request was passed through correctly
            call_args = mock_service_container.router.get_provider_for_request.call_args[0][0]
            assert 'plan' in call_args.messages[0].content
            assert 'strategy' in call_args.messages[0].content
            assert 'architecture' in call_args.messages[0].content

    def test_routing_integration_background_keywords(self, mock_service_container):
        """Test that background keywords properly route through the system."""
        client = TestClient(app)

        with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
            request_payload = {
                'model': 'claude-3-sonnet-20240229',
                'messages': [{'role': 'user', 'content': 'Please analyze and summarize this data for our batch processing pipeline.'}],
            }

            response = client.post('/v1/messages', json=request_payload)
            assert response.status_code == 200

            call_args = mock_service_container.router.get_provider_for_request.call_args[0][0]
            assert 'analyze' in call_args.messages[0].content
            assert 'summarize' in call_args.messages[0].content
            assert 'batch' in call_args.messages[0].content


class TestConfigurationIntegration:
    """Test configuration hot-reload and management integration."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample user configuration."""
        return UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[
                ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider', alias='sonnet'),
                ModelConfig(id='claude-3-haiku-20240229', provider='test-provider', alias='haiku'),
            ],
            routing=RoutingConfig(default='sonnet', planning='sonnet', background='haiku'),
            transformer_paths=[],
        )

    @pytest.mark.asyncio
    async def test_hot_reload_resource_cleanup(self, sample_config):
        """Test that hot-reload properly cleans up resources."""
        from app.dependencies.service_container import ServiceContainer

        # Mock provider manager with close_all method
        mock_provider_manager = AsyncMock()
        mock_provider_manager.close_all = AsyncMock()
        mock_provider_manager.list_providers.return_value = ['test-provider']
        mock_provider_manager.list_models.return_value = ['claude-3-sonnet-20240229']

        # Create service container
        container = ServiceContainer()
        container.provider_manager = mock_provider_manager

        # Test reinitialization
        await container.reinitialize_from_config(sample_config)

        # Verify close_all was called (resource cleanup)
        mock_provider_manager.close_all.assert_called_once()

    def test_config_validation_integration(self):
        """Test configuration validation through API endpoint."""
        client = TestClient(app)

        valid_config_yaml = """
providers:
  - name: test-provider
    url: https://api.example.com
    api_key: test-key

models:
  - id: claude-3-sonnet-20240229
    provider: test-provider

routing:
  default: claude-3-sonnet-20240229
  planning: claude-3-sonnet-20240229
  background: claude-3-sonnet-20240229
"""

        response = client.post('/api/config/validate-yaml', json={'yaml_content': valid_config_yaml})

        assert response.status_code == 200
        result = response.json()
        assert result['valid'] is True
        assert result['stage'] == 'complete'

    def test_invalid_config_handling(self):
        """Test system graceful handling of invalid configurations."""
        client = TestClient(app)

        invalid_config_yaml = """
providers:
  - name: ""  # Invalid empty name
    url: not-a-url  # Invalid URL

models:
  - id: invalid-model
    provider: nonexistent-provider  # References non-existent provider
"""

        response = client.post('/api/config/validate-yaml', json={'yaml_content': invalid_config_yaml})

        assert response.status_code == 200
        result = response.json()
        assert result['valid'] is False
        assert len(result['errors']) > 0


class TestResourceManagement:
    """Test resource management and cleanup scenarios."""

    @pytest.mark.asyncio
    async def test_service_container_cleanup(self):
        """Test proper cleanup of service container resources."""
        from app.dependencies.service_container import ServiceContainer

        container = ServiceContainer()

        # Mock provider manager
        mock_provider_manager = AsyncMock()
        mock_provider_manager.close_all = AsyncMock()
        container.provider_manager = mock_provider_manager

        # Test cleanup
        await container.close()

        mock_provider_manager.close_all.assert_called_once()

    def test_concurrent_requests_stability(self):
        """Test system stability with concurrent requests."""
        client = TestClient(app)

        # Mock service container for concurrent access
        mock_container = Mock()
        mock_provider = Mock()
        mock_provider.config.name = 'test-provider'

        async def mock_process_request(payload, request, routing_key, dumper, correlation_id):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Response"}}\n\n'
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        mock_provider.process_request = mock_process_request
        mock_container.router.get_provider_for_request.return_value = mock_provider, 'test-routing-key'

        # Create dumper mock
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(None, None, None, None, None, None, None, None, 'test-correlation-id')
        mock_dumper.write_chunk = Mock()
        mock_dumper.close = Mock()

        with patch('app.routers.messages.get_service_container', return_value=mock_container), patch('app.dependencies.dumper.get_dumper', return_value=mock_dumper):
            # Make multiple concurrent requests
            import concurrent.futures

            def make_request():
                return client.post('/v1/messages', json={'model': 'claude-3-sonnet-20240229', 'messages': [{'role': 'user', 'content': 'Test message'}]})

            # Execute concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # All requests should succeed
            assert all(r.status_code == 200 for r in results)


class TestErrorHandling:
    """Test error handling integration across the system."""

    def test_provider_failure_handling(self):
        """Test graceful handling when provider fails."""
        client = TestClient(app)

        # Mock container with failing provider
        mock_container = Mock()
        mock_provider = Mock()
        mock_provider.config.name = 'failing-provider'

        async def failing_process_request(payload, request, routing_key, dumper, correlation_id):
            raise Exception('Provider connection failed')
            yield b'never reached'  # This line is unreachable but needed for generator syntax

        mock_provider.process_request = failing_process_request
        mock_container.router.get_provider_for_request.return_value = mock_provider, 'test-routing-key'

        # Create dumper mock
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(None, None, None, None, None, None, None, None, 'test-correlation-id')
        mock_dumper.write_chunk = Mock()
        mock_dumper.close = Mock()

        with patch('app.routers.messages.get_service_container', return_value=mock_container), patch('app.dependencies.dumper.get_dumper', return_value=mock_dumper):
            response = client.post('/v1/messages', json={'model': 'claude-3-sonnet-20240229', 'messages': [{'role': 'user', 'content': 'Test'}]})

            # Provider errors are streamed back as SSE, so status is still 200
            assert response.status_code == 200
            content = response.content.decode()
            assert 'event: error' in content or 'Provider connection failed' in content

    def test_no_provider_available_error(self):
        """Test error when no suitable provider is found."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        mock_container = Mock()
        mock_container.router.get_provider_for_request.return_value = None, None  # No provider

        # Override dumper dependency (shouldn't be called in this test, but FastAPI needs it)
        mock_dumper = Mock()
        app.dependency_overrides[get_dumper] = lambda: mock_dumper

        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_container):
                response = client.post('/v1/messages', json={'model': 'unknown-model', 'messages': [{'role': 'user', 'content': 'Test'}]})

                assert response.status_code == 400
                result = response.json()
                assert result['type'] == 'error'
                assert result['error']['type'] == 'api_error'
        finally:
            app.dependency_overrides.clear()

    def test_service_container_not_available(self):
        """Test error when service container is not available."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Override dumper dependency (shouldn't be called in this test, but FastAPI needs it)
        mock_dumper = Mock()
        app.dependency_overrides[get_dumper] = lambda: mock_dumper

        try:
            with patch('app.routers.messages.get_service_container', return_value=None):
                response = client.post('/v1/messages', json={'model': 'claude-3-sonnet-20240229', 'messages': [{'role': 'user', 'content': 'Test'}]})

                assert response.status_code == 500
                result = response.json()
                assert result['type'] == 'error'
                assert result['error']['type'] == 'api_error'
        finally:
            app.dependency_overrides.clear()


class TestAliasSupport:
    """Test model alias support functionality."""

    def test_alias_validation_success(self):
        """Test that valid aliases are accepted."""
        config = UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[
                ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider', alias='sonnet'),
                ModelConfig(id='claude-3-haiku-20240229', provider='test-provider', alias='haiku'),
            ],
            routing=RoutingConfig(default='sonnet', planning='sonnet', background='haiku'),
        )
        # Should not raise an exception
        config.validate_references()

    def test_alias_validation_duplicate_error(self):
        """Test that duplicate aliases are rejected."""
        config = UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[
                ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider', alias='duplicate'),
                ModelConfig(id='claude-3-haiku-20240229', provider='test-provider', alias='duplicate'),
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate_references()
        assert "Duplicate alias 'duplicate'" in str(exc_info.value)

    def test_alias_validation_conflict_with_model_id(self):
        """Test that aliases cannot conflict with existing model IDs."""
        config = UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[
                ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider'),
                ModelConfig(id='claude-3-haiku-20240229', provider='test-provider', alias='claude-3-sonnet-20240229'),
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate_references()
        assert "alias 'claude-3-sonnet-20240229' that conflicts with existing model ID" in str(exc_info.value)

    def test_alias_format_validation(self):
        """Test alias format validation."""
        with pytest.raises(ValueError) as exc_info:
            ModelConfig(id='test-model', provider='test-provider', alias='invalid alias')
        assert 'alphanumeric characters, hyphens, and underscores' in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            ModelConfig(id='test-model', provider='test-provider', alias='invalid@alias')
        assert 'alphanumeric characters, hyphens, and underscores' in str(exc_info.value)

        # Valid aliases should work
        ModelConfig(id='test-model', provider='test-provider', alias='valid-alias_123')

    def test_routing_with_alias(self):
        """Test that routing works with aliases."""
        config = UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[
                ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider', alias='sonnet'),
                ModelConfig(id='claude-3-haiku-20240229', provider='test-provider', alias='haiku'),
            ],
            routing=RoutingConfig(default='sonnet', background='haiku'),
        )

        # Should not raise an exception
        config.validate_references()

        # Test lookup methods
        assert config.get_model_by_id_or_alias('sonnet').id == 'claude-3-sonnet-20240229'
        assert config.get_model_by_id_or_alias('haiku').id == 'claude-3-haiku-20240229'
        assert config.get_model_by_id_or_alias('claude-3-sonnet-20240229').id == 'claude-3-sonnet-20240229'

    def test_routing_with_unknown_alias(self):
        """Test that routing fails with unknown aliases."""
        config = UserConfig(
            providers=[ProviderConfig(name='test-provider', url='https://api.example.com', api_key='test-key')],
            models=[ModelConfig(id='claude-3-sonnet-20240229', provider='test-provider', alias='sonnet')],
            routing=RoutingConfig(default='unknown-alias'),
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate_references()
        assert "Routing 'default' references unknown model or alias 'unknown-alias'" in str(exc_info.value)

    def test_config_validation_integration_with_aliases(self):
        """Test configuration validation through API endpoint with aliases."""
        client = TestClient(app)

        valid_config_yaml = """
providers:
  - name: test-provider
    url: https://api.example.com
    api_key: test-key

models:
  - id: claude-3-sonnet-20240229
    provider: test-provider
    alias: sonnet
  - id: claude-3-haiku-20240229
    provider: test-provider
    alias: haiku

routing:
  default: sonnet
  planning: sonnet
  background: haiku
"""

        response = client.post('/api/config/validate-yaml', json={'yaml_content': valid_config_yaml})

        assert response.status_code == 200
        result = response.json()
        assert result['valid'] is True
        assert result['stage'] == 'complete'

"""Integration tests for end-to-end flows, configuration management, and resource handling."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
from app.config.user_models import ProviderConfig, RoutingConfig, UserConfig
from app.main import app


class TestEndToEndRequestFlow:
    """Test complete request flows from API to provider and back."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that simulates realistic behavior."""
        provider = Mock()
        provider.config = Mock()
        provider.config.name = 'test-provider'
        
        async def process_request(payload, request):
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
        router.get_provider_for_request.return_value = mock_provider
        container.router = router
        
        # Mock dumper
        dumper = Mock()
        dumper.begin.return_value = DumpHandles(None, None, None, None)
        dumper.write_chunk = Mock()
        dumper.close = Mock()
        container.dumper = dumper
        
        return container

    def test_complete_message_flow_success(self, mock_service_container):
        """Test successful end-to-end message processing."""
        client = TestClient(app)
        
        with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
            request_payload = {
                'model': 'claude-3-sonnet-20240229',
                'messages': [
                    {'role': 'user', 'content': 'Hello, how are you?'}
                ],
                'max_tokens': 100
            }
            
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
            mock_service_container.dumper.begin.assert_called_once()
            assert mock_service_container.dumper.write_chunk.call_count > 0
            mock_service_container.dumper.close.assert_called_once()

    def test_routing_integration_planning_keywords(self, mock_service_container):
        """Test that planning keywords properly route through the system."""
        client = TestClient(app)
        
        with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
            request_payload = {
                'model': 'claude-3-sonnet-20240229', 
                'messages': [
                    {'role': 'user', 'content': 'Please create a comprehensive plan and strategy for our project architecture.'}
                ]
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
                'messages': [
                    {'role': 'user', 'content': 'Please analyze and summarize this data for our batch processing pipeline.'}
                ]
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
            providers=[
                ProviderConfig(
                    name='test-provider',
                    url='https://api.example.com',
                    api_key='test-key',
                    models=['claude-3-sonnet-20240229']
                )
            ],
            routing=RoutingConfig(
                default='claude-3-sonnet-20240229',
                planning='claude-3-sonnet-20240229', 
                background='claude-3-sonnet-20240229'
            ),
            transformer_paths=[]
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
      - claude-3-sonnet-20240229

models:
  - id: claude-3-sonnet-20240229
    provider: test-provider

routing:
  default: claude-3-sonnet-20240229
  planning: claude-3-sonnet-20240229
  background: claude-3-sonnet-20240229
"""
        
        response = client.post('/api/config/validate-yaml',
                             json={'yaml_content': valid_config_yaml})
        
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
    models: []  # Empty models list

models:
  - id: invalid-model
    provider: nonexistent-provider  # References non-existent provider
"""
        
        response = client.post('/api/config/validate-yaml',
                             json={'yaml_content': invalid_config_yaml})
        
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
        
        async def mock_process_request(payload, request):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Response"}}\n\n'
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
        
        mock_provider.process_request = mock_process_request
        mock_container.router.get_provider_for_request.return_value = mock_provider
        mock_container.dumper.begin.return_value = DumpHandles(None, None, None, None)
        mock_container.dumper.write_chunk = Mock()
        mock_container.dumper.close = Mock()
        
        with patch('app.routers.messages.get_service_container', return_value=mock_container):
            # Make multiple concurrent requests
            import concurrent.futures
            
            def make_request():
                return client.post('/v1/messages', json={
                    'model': 'claude-3-sonnet-20240229',
                    'messages': [{'role': 'user', 'content': 'Test message'}]
                })
            
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
        
        async def failing_process_request(payload, request):
            raise Exception("Provider connection failed")
            yield b'never reached'  # This line is unreachable but needed for generator syntax
        
        mock_provider.process_request = failing_process_request
        mock_container.router.get_provider_for_request.return_value = mock_provider
        mock_container.dumper.begin.return_value = DumpHandles(None, None, None, None)
        mock_container.dumper.write_chunk = Mock()
        mock_container.dumper.close = Mock()
        
        with patch('app.routers.messages.get_service_container', return_value=mock_container):
            response = client.post('/v1/messages', json={
                'model': 'claude-3-sonnet-20240229',
                'messages': [{'role': 'user', 'content': 'Test'}]
            })
            
            # Provider errors are streamed back as SSE, so status is still 200
            assert response.status_code == 200
            content = response.content.decode()
            assert 'event: error' in content or 'Provider connection failed' in content

    def test_no_provider_available_error(self):
        """Test error when no suitable provider is found."""
        client = TestClient(app)
        
        mock_container = Mock()
        mock_container.router.get_provider_for_request.return_value = None  # No provider
        mock_container.dumper.begin.return_value = DumpHandles(None, None, None, None)
        
        with patch('app.routers.messages.get_service_container', return_value=mock_container):
            response = client.post('/v1/messages', json={
                'model': 'unknown-model',
                'messages': [{'role': 'user', 'content': 'Test'}]
            })
            
            assert response.status_code == 400
            result = response.json()
            assert result['type'] == 'error'
            assert result['error']['type'] == 'invalid_request_error'

    def test_service_container_not_available(self):
        """Test error when service container is not available."""
        client = TestClient(app)
        
        with patch('app.routers.messages.get_service_container', return_value=None):
            response = client.post('/v1/messages', json={
                'model': 'claude-3-sonnet-20240229',
                'messages': [{'role': 'user', 'content': 'Test'}]
            })
            
            assert response.status_code == 500
            result = response.json()
            assert result['type'] == 'error'
            assert result['error']['type'] == 'invalid_request_error'
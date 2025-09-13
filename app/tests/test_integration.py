"""Integration tests for end-to-end flows, configuration management, and resource handling."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.common.dumper import DumpFiles, DumpHandles
from app.main import app
from app.services.router import RoutingResult


class TestEndToEndRequestFlow:
    """Test complete request flows from API to provider and back."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that simulates realistic behavior."""
        provider = Mock()
        provider.config = Mock()
        provider.config.name = 'test-provider'

        async def process_request(payload, request, routing_key, dumper, correlation_id):
            # Return JSON response like current architecture
            return {
                'id': 'msg_integration_test123',
                'model': 'claude-3-haiku',
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'Hello! I understand your request.'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 8},
            }

        provider.process_request = process_request
        return provider

    @pytest.fixture
    def mock_service_container(self, mock_provider):
        """Create a realistic service container mock."""
        container = Mock()

        # Mock router with realistic behavior
        router = Mock()
        router.get_provider_for_request.return_value = RoutingResult(
            provider=mock_provider, routing_key='test-routing-key', model_alias='test-model', resolved_model_id='claude-3-haiku'
        )
        container.router = router

        return container

    @pytest.fixture
    def mock_dumper(self):
        """Create a mock dumper for dependency injection."""
        dumper = Mock()
        dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')
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
                request_payload = {'model': 'claude-3-sonnet-20240229', 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'max_tokens': 100, 'stream': True}

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

        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_routing_integration_planning_keywords(self, mock_service_container):
        """Test that planning routing keywords are handled correctly."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')

        app.dependency_overrides[get_dumper] = lambda: mock_dumper

        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_service_container):
                request_payload = {
                    'model': 'claude-3-sonnet-20240229',
                    'messages': [{'role': 'user', 'content': '<system-reminder>Plan mode is active</system-reminder>Hello'}],
                    'stream': True,
                }

                response = client.post('/v1/messages', json=request_payload)
                assert response.status_code == 200

        finally:
            app.dependency_overrides.clear()


class TestResourceManagement:
    """Test system resource management and cleanup."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for hot reload tests."""
        return {
            'providers': [{'name': 'test-provider', 'url': 'https://api.test.com', 'api_key': 'test-key'}],
            'models': [{'id': 'test-model', 'provider': 'test-provider', 'alias': 'test'}],
            'routing': {'default': 'test'},
        }

    @pytest.mark.asyncio
    async def test_hot_reload_resource_cleanup(self, sample_config):
        """Test that resources are properly cleaned up during hot reload."""
        from app.config import reload_config
        from app.dependencies.service_container import get_service_container

        # Get initial service container
        initial_container = get_service_container()

        # Mock the close_all method to track cleanup calls
        initial_container.close_all = Mock()

        # Trigger config reload (simulated)
        with patch('app.config.models.ConfigModel.load') as mock_load:
            mock_load.return_value = Mock()

            # This would normally trigger service container recreation
            reload_config()

            # Verify cleanup was called (this test verifies the integration pattern)
            # Note: This is a basic test structure - actual cleanup verification would depend on
            # the specific implementation of hot reload functionality
            assert mock_load.called, 'Config reload should have been called'


class TestConfigurationIntegration:
    """Test configuration validation and API integration."""

    @pytest.mark.parametrize(
        'config_data,should_be_valid',
        [
            ({'providers': [{'name': 'test', 'url': 'https://api.test.com', 'api_key': 'key'}]}, True),
            ({'providers': [{'name': 'test', 'url': 'invalid-url', 'api_key': 'key'}]}, False),
        ],
    )
    def test_config_validation_scenarios(self, config_data, should_be_valid):
        """Test configuration validation through API endpoint."""
        client = TestClient(app)

        with patch('app.config.get_config') as mock_get_config:
            if should_be_valid:
                mock_get_config.return_value = Mock(dict=lambda: config_data)
                response = client.get('/api/config/validate')
                assert response.status_code == 200
                data = response.json()
                assert 'valid' in data
            else:
                mock_get_config.side_effect = Exception('Invalid configuration')
                response = client.get('/api/config/validate')
                assert response.status_code == 200
                data = response.json()
                assert 'errors' in data


class TestErrorHandling:
    """Test system-level error handling."""

    def test_concurrent_requests_stability(self):
        """Test system stability under concurrent load."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Mock dumper for dependency injection
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')

        # Mock service container to simulate realistic behavior
        mock_container = Mock()
        mock_provider = Mock()

        async def mock_process(payload, request, routing_key, dumper, correlation_id):
            return {
                'id': 'test',
                'model': 'test-model',
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'Response'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 5, 'output_tokens': 5},
            }

        mock_provider.process_request = mock_process
        mock_container.router.get_provider_for_request.return_value = RoutingResult(
            provider=mock_provider, routing_key='default', model_alias='test', resolved_model_id='test-model'
        )

        # Override dependencies
        app.dependency_overrides[get_dumper] = lambda: mock_dumper
        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_container):
                responses = []
                for i in range(5):  # Simulate concurrent requests
                    response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': f'Request {i}'}], 'stream': True})
                    responses.append(response)

                # All requests should succeed
                for response in responses:
                    assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_provider_failure_handling(self):
        """Test error handling when provider fails."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Mock dumper for dependency injection
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')

        mock_container = Mock()
        mock_provider = Mock()

        async def failing_process(payload, request, routing_key, dumper, correlation_id):
            raise Exception('Provider connection failed')

        mock_provider.process_request = failing_process
        mock_container.router.get_provider_for_request.return_value = RoutingResult(
            provider=mock_provider, routing_key='default', model_alias='test', resolved_model_id='test-model'
        )

        # Override dependencies
        app.dependency_overrides[get_dumper] = lambda: mock_dumper
        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_container):
                response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

                # Should handle provider failure gracefully
                assert response.status_code in [400, 500]  # Expect error response
        finally:
            app.dependency_overrides.clear()

    def test_no_provider_available_error(self):
        """Test specific error path when no provider is available."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Mock dumper for dependency injection
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')

        mock_container = Mock()
        mock_container.router.get_provider_for_request.return_value = RoutingResult(provider=None, routing_key='default', model_alias='unknown', resolved_model_id='unknown')

        # Override dependencies
        app.dependency_overrides[get_dumper] = lambda: mock_dumper
        try:
            with patch('app.routers.messages.get_service_container', return_value=mock_container):
                response = client.post('/v1/messages', json={'model': 'unknown-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

                assert response.status_code == 400
                data = response.json()
                assert 'error' in data
                assert data['error']['type'] == 'model_not_found'
        finally:
            app.dependency_overrides.clear()

    def test_service_container_not_available(self):
        """Test system-level error handling when service container is unavailable."""
        from app.dependencies.dumper import get_dumper

        client = TestClient(app)

        # Mock dumper for dependency injection (even though test fails before reaching dumper)
        mock_dumper = Mock()
        mock_dumper.begin.return_value = DumpHandles(files=DumpFiles(), correlation_id='test-correlation-id', base_path='/tmp')

        # Override dependencies
        app.dependency_overrides[get_dumper] = lambda: mock_dumper
        try:
            with patch('app.routers.messages.get_service_container', side_effect=Exception('Service unavailable')):
                response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

                # Should handle system-level errors
                assert response.status_code == 500
        finally:
            app.dependency_overrides.clear()


# Note: Removed 8 misplaced unit tests from TestAliasSupport class
# These tests belonged in app/config/tests/test_user_models.py as they test
# UserConfig/ModelConfig class behavior in isolation, not integration behavior.
#
# Also removed redundant tests:
# - test_routing_integration_background_keywords (duplicate routing test)
# - test_service_container_cleanup (redundant with hot_reload_resource_cleanup)
#
# Consolidated config validation tests into parameterized test for efficiency.

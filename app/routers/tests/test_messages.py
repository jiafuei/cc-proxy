"""Tests for the simplified messages endpoint."""

import os
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.dumper import DumpFiles, DumpHandles
from app.dependencies.dumper import get_dumper
from app.routers.messages import router


def test_messages_endpoint():
    """Test the simplified messages endpoint."""
    # Create a test client with the main app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockProvider:
        def __init__(self, name='test-provider'):
            self.config = Mock()
            self.config.name = name

        async def process_request(self, payload, request, routing_key=None, dumper=None, dumper_handles=None):
            """Mock provider that returns JSON response (new architecture)."""
            return {
                'id': 'msg_test123',
                'model': 'claude-3-haiku',
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'Hello from test!'}
                ],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 5}
            }

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider(), 'default'

    class MockDumper:
        def begin(self, request, payload):
            return DumpHandles(
                files=DumpFiles(),
                correlation_id='test-correlation-id',
                base_path='/tmp'
            )

        def write_response_chunk(self, handles, chunk):
            pass

        def close(self, handles):
            pass

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()
            self.dumper = MockDumper()

    # Mock the get_service_container function
    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = MockServiceContainer()

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

        # Check that we get SSE-formatted data
        content = response.content.decode()
        assert 'event: message_start' in content
        assert 'event: message_stop' in content
        assert 'Hello from test!' in content


def test_messages_count_endpoint():
    """Test the messages count endpoint."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockProvider:
        def __init__(self, name='test-provider'):
            from app.config.user_models import ProviderConfig
            self.config = ProviderConfig(
                name=name,
                url='https://api.test.com/v1/messages',
                api_key='test-api-key',
                transformers={'request': [], 'response': []},
                timeout=30
            )
            self.request_transformers = []

        async def _send_request(self, config, request_data, headers):
            """Mock provider that returns count response."""

            class MockResponse:
                def json(self):
                    return {'input_tokens': 10, 'output_tokens': 0, 'total_tokens': 10}

            return MockResponse()

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider(), 'default'

    class MockDumper:
        def begin(self, request, payload):
            return DumpHandles(
                files=DumpFiles(),
                correlation_id='test-correlation-id',
                base_path='/tmp'
            )

        def write_transformed_request(self, handles, request):
            pass

        def write_transformed_headers(self, handles, headers):
            pass

        def write_response_chunk(self, handles, chunk):
            pass

        def close(self, handles):
            pass

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_service_container = MockServiceContainer()
        mock_get_container.return_value = mock_service_container

        # Override the dumper dependency
        app.dependency_overrides[get_dumper] = lambda: MockDumper()

        response = client.post('/v1/messages/count_tokens', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'

        # Check that we get the expected count response
        json_response = response.json()
        assert 'input_tokens' in json_response
        assert 'output_tokens' in json_response
        assert 'total_tokens' in json_response
        assert json_response['input_tokens'] == 10
        assert json_response['total_tokens'] == 10


def test_messages_count_endpoint_no_provider():
    """Test messages count endpoint when no provider is available."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockRouter:
        def get_provider_for_request(self, request):
            return None, 'default'  # No provider available

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = MockServiceContainer()

        response = client.post('/v1/messages/count_tokens', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 400
        response_data = response.json()
        assert response_data['detail']['error']['type'] == 'model_not_found'


def test_messages_endpoint_no_provider():
    """Test messages endpoint when no provider is available."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockRouter:
        def get_provider_for_request(self, request):
            return None, 'default'  # No provider available

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()
            self.dumper = Mock()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = MockServiceContainer()

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 400
        response_data = response.json()
        assert response_data['error']['type'] == 'model_not_found'


def test_messages_endpoint_system_not_available():
    """Test messages endpoint when service container is not available."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = None

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 500
        response_data = response.json()
        assert response_data['error']['type'] == 'api_error'


def test_messages_count_endpoint_system_not_available():
    """Test messages count endpoint when service container is not available."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = None

        response = client.post('/v1/messages/count_tokens', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 500
        response_data = response.json()
        assert response_data['detail']['error']['type'] == 'api_error'


def test_messages_endpoint_with_dumping(tmp_path):
    """Test messages endpoint with file dumping."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockProvider:
        def __init__(self):
            self.config = Mock()
            self.config.name = 'test-provider'

        async def process_request(self, payload, request, routing_key=None, dumper=None, dumper_handles=None):
            return {
                'id': 'msg_test456', 
                'model': 'claude-3-haiku',
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'Test response'}
                ],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 8, 'output_tokens': 3}
            }

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider(), 'default'

    class MockDumper:
        def __init__(self):
            self.tmp_dir = str(tmp_path)
            self.files = []

        def begin(self, request, payload):
            from datetime import datetime, timezone

            from app.common.utils import get_correlation_id

            dump_dir = self.tmp_dir
            corr_id = get_correlation_id()
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%fZ')

            os.makedirs(dump_dir, exist_ok=True)

            # Create response file
            response_path = os.path.join(dump_dir, f'{ts}_{corr_id}_response.sse')
            response_file = open(response_path, 'wb')

            return DumpHandles(
                files=DumpFiles(),
                correlation_id=corr_id,
                base_path=dump_dir
            )

        def write_response_chunk(self, handles, chunk):
            # Mock implementation - in real dumper this would write to files
            pass

        def close(self, handles):
            # Mock implementation - in real dumper this would close files
            pass

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()
            self.dumper = MockDumper()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_service_container = MockServiceContainer()
        mock_get_container.return_value = mock_service_container

        # Override the dumper dependency
        app.dependency_overrides[get_dumper] = lambda: mock_service_container.dumper

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

        assert response.status_code == 200

        # Consume the response to trigger the generator
        content = response.content

        # Check that dump files were created (mock doesn't actually create files)
        # Just verify we get a valid response
        assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
        response_content = response.content.decode()
        assert 'Test response' in response_content


def test_messages_endpoint_provider_http_error():
    """Test messages endpoint when provider returns HTTP error."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockProvider:
        def __init__(self):
            self.config = Mock()
            self.config.name = 'test-provider'

        async def process_request(self, payload, request, routing_key=None, dumper=None, dumper_handles=None):
            # Simulate httpx.HTTPStatusError
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = '{"error": {"message": "Invalid API key"}}'
            mock_response.json.return_value = {'error': {'message': 'Invalid API key'}}

            import httpx

            raise httpx.HTTPStatusError('Unauthorized', request=Mock(), response=mock_response)

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider(), 'default'

    class MockDumper:
        def begin(self, request, payload):
            return DumpHandles(
                files=DumpFiles(),
                correlation_id='test-correlation-id',
                base_path='/tmp'
            )

        def close(self, handles):
            pass

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_service_container = MockServiceContainer()
        mock_get_container.return_value = mock_service_container

        # Override the dumper dependency
        app.dependency_overrides[get_dumper] = lambda: MockDumper()

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 401
        assert response.headers['content-type'] == 'application/json'

        # Check that we get proper error response format
        json_response = response.json()
        assert 'error' in json_response
        assert json_response['error']['type'] == 'authentication_error'
        assert json_response['error']['message'] == 'Invalid API key'

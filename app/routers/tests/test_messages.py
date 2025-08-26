"""Tests for the simplified messages endpoint."""

import os
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
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

        async def process_request(self, payload, request):
            """Mock provider that returns SSE-formatted chunks."""
            yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n'
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider()

    class MockDumper:
        def begin(self, request, payload):
            return DumpHandles(headers_path=None, request_path=None, response_path=None, response_file=None)

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

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

        # Check that we get SSE-formatted data
        content = response.content.decode()
        assert 'event: message_start' in content
        assert 'event: message_stop' in content


def test_messages_endpoint_no_provider():
    """Test messages endpoint when no provider is available."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class MockRouter:
        def get_provider_for_request(self, request):
            return None  # No provider available

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()
            self.dumper = Mock()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = MockServiceContainer()

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 400
        response_data = response.json()
        assert response_data['detail']['error']['type'] == 'model_not_found'


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

        async def process_request(self, payload, request):
            yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
            yield b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Test response"}}\n\n'
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    class MockRouter:
        def get_provider_for_request(self, request):
            return MockProvider()

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

            return DumpHandles(headers_path=None, request_path=None, response_path=response_path, response_file=response_file)

        def write_response_chunk(self, handles, chunk):
            if handles.response_file:
                handles.response_file.write(chunk)
                handles.response_file.flush()

        def close(self, handles):
            if handles.response_file:
                handles.response_file.close()

    class MockServiceContainer:
        def __init__(self):
            self.router = MockRouter()
            self.dumper = MockDumper()

    with patch('app.routers.messages.get_service_container') as mock_get_container:
        mock_get_container.return_value = MockServiceContainer()

        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 200

        # Check that dump files were created
        files = os.listdir(tmp_path)
        sse_files = [f for f in files if f.endswith('response.sse')]
        assert len(sse_files) >= 1

        # Check that response was written to file
        with open(os.path.join(tmp_path, sse_files[0]), 'rb') as f:
            content = f.read()
            assert b'Test response' in content

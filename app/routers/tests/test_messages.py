import json
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
from app.dependencies.services import get_core_services, get_routing_service, get_service_container
from app.routers.messages import router
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.pipeline.models import StreamChunk
from app.services.pipeline.messages_service import MessagesPipelineService


def test_messages_endpoint():
    # Create a test client with the main app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyPipeline:
        async def process_unified(self, claude_request, original_request):
            # Return SSE-formatted stream chunks
            yield StreamChunk(data=b'event: message_start\ndata: {"type": "message_start"}\n\n')
            yield StreamChunk(data=b'event: message_stop\ndata: {"type": "message_stop"}\n\n')

    class DummyDumper:
        def begin(self, request, payload):
            return DumpHandles(headers_path=None, request_path=None, response_path=None, response_file=None)

        def write_chunk(self, handles, chunk):
            pass

        def close(self, handles):
            pass

    # Create instances to be reused
    dummy_pipeline = DummyPipeline()
    dummy_dumper = DummyDumper()

    class MockRoutingService:
        def process_request(self, claude_request):
            # Return successful routing result: routing_key, model_id, pipeline_service
            return 'default', 'test-model', dummy_pipeline

    class MockServiceContainer:
        def __init__(self):
            self.model_registry = self
            
        def get_provider_for_model(self, model_id):
            return 'test-provider'

    # Use patch to mock the functions directly
    with patch('app.routers.messages.get_routing_service') as mock_routing, \
         patch('app.routers.messages.get_core_services') as mock_core, \
         patch('app.routers.messages.get_service_container') as mock_container:
        
        # Set up mocks
        mock_routing.return_value = MockRoutingService()
        mock_container.return_value = MockServiceContainer()
        
        # Mock core services
        core_services = type('MockCoreServices', (), {
            'dumper': dummy_dumper,
            'exception_mapper': HttpExceptionMapper(),
            'error_formatter': ApiErrorFormatter()
        })()
        mock_core.return_value = core_services
        
        response = client.post('/v1/messages', json={'model': 'test-model', 'messages': []})
        assert response.status_code == 200


def test_dump_files(tmp_path):
    # Create a separate test app for this test
    test_app = FastAPI()
    test_app.include_router(router)

    client = TestClient(test_app)

    class DummyPipeline:
        async def process_unified(self, claude_request, original_request):
            # Return SSE-formatted stream chunks
            yield StreamChunk(data=b'event: message_start\ndata: {"type": "message_start"}\n\n')
            yield StreamChunk(data=b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "hello"}}\n\n')
            yield StreamChunk(data=b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "world"}}\n\n')
            yield StreamChunk(data=b'event: message_stop\ndata: {"type": "message_stop"}\n\n')

    class DummyDumper:
        def __init__(self):
            self.tmp_dir = str(tmp_path)
            self.files = []

        def begin(self, request, payload):
            from datetime import datetime, timezone

            from app.common.dumper import DumpHandles
            from app.common.utils import get_correlation_id

            dump_dir = self.tmp_dir
            corr_id = get_correlation_id()
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%fZ')

            os.makedirs(dump_dir, exist_ok=True)

            # Create headers file
            headers_path = os.path.join(dump_dir, f'{ts}_{corr_id}_headers.json')
            with open(headers_path, 'w') as f:
                json.dump({'test': 'headers'}, f)

            # Create request file
            request_path = os.path.join(dump_dir, f'{ts}_{corr_id}_request.json')
            with open(request_path, 'w') as f:
                json.dump({'test': 'request'}, f)

            # Create response file
            response_path = os.path.join(dump_dir, f'{ts}_{corr_id}_response.sse')
            response_file = open(response_path, 'wb')

            return DumpHandles(headers_path=headers_path, request_path=request_path, response_path=response_path, response_file=response_file)

        def write_chunk(self, handles, chunk):
            if handles.response_file:
                handles.response_file.write(chunk)
                handles.response_file.flush()

        def close(self, handles):
            if handles.response_file:
                handles.response_file.close()

    # Create instances to be reused
    dummy_pipeline = DummyPipeline()
    dummy_dumper = DummyDumper()

    class MockRoutingService:
        def process_request(self, claude_request):
            # Return successful routing result: routing_key, model_id, pipeline_service
            return 'default', 'test-model', dummy_pipeline

    class MockServiceContainer:
        def __init__(self):
            self.model_registry = self
            
        def get_provider_for_model(self, model_id):
            return 'test-provider'

    # Use patch to mock the functions directly
    with patch('app.routers.messages.get_routing_service') as mock_routing, \
         patch('app.routers.messages.get_core_services') as mock_core, \
         patch('app.routers.messages.get_service_container') as mock_container:
        
        # Set up mocks
        mock_routing.return_value = MockRoutingService()
        mock_container.return_value = MockServiceContainer()
        
        # Mock core services
        core_services = type('MockCoreServices', (), {
            'dumper': dummy_dumper,
            'exception_mapper': HttpExceptionMapper(),
            'error_formatter': ApiErrorFormatter()
        })()
        mock_core.return_value = core_services
        
        resp = client.post('/v1/messages', json={'model': 'test-model', 'messages': []})
        assert resp.status_code == 200

        files = os.listdir(tmp_path)
        assert any('headers.json' in f for f in files)
        assert any('request.json' in f for f in files)
        sse_files = [f for f in files if f.endswith('response.sse')]
        assert len(sse_files) >= 1
        with open(os.path.join(tmp_path, sse_files[0]), 'rb') as f:
            content = f.read()
            assert b'hello' in content and b'world' in content

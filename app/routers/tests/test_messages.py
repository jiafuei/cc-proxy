import json
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
from app.dependencies.services import get_services
from app.routers.messages import router
from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.pipeline.models import StreamChunk


def test_messages_endpoint():
    # Create a test client with the main app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyPipeline:
        async def process_unified(self, claude_request, original_request, correlation_id):
            # Return SSE-formatted stream chunks
            yield StreamChunk(data=b'event: message_start\ndata: {"type": "message_start"}\n\n')
            yield StreamChunk(data=b'event: message_stop\ndata: {"type": "message_stop"}\n\n')

    class DummyDumper:
        def begin(self, request, payload, correlation_id=None):
            return DumpHandles(headers_path=None, request_path=None, response_path=None, response_file=None)

        def write_chunk(self, handles, chunk):
            pass

        def close(self, handles):
            pass

    def override_get_services():
        # Create a completely mock services object without initializing real components
        class DummyServices:
            def __init__(self):
                self.messages_pipeline = DummyPipeline()
                self.dumper = DummyDumper()
                self.exception_mapper = HttpExceptionMapper()
                self.error_formatter = ApiErrorFormatter()

                # Create a simple config mock
                class Cfg:
                    dump_requests = False
                    dump_responses = False
                    dump_headers = False
                    dump_dir = None
                    redact_headers = ['authorization', 'cookie', 'set-cookie']

                self.config = Cfg()

        return DummyServices()

    # Override dependencies
    app.dependency_overrides[get_services] = override_get_services
    try:
        response = client.post('/v1/messages', json={'model': 'x', 'messages': []})
        assert response.status_code == 200
    finally:
        # Reset overrides after test
        app.dependency_overrides = {}


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

        def begin(self, request, payload, correlation_id=None):
            from datetime import datetime, timezone

            from app.common.dumper import DumpHandles
            from app.common.utils import get_correlation_id

            dump_dir = self.tmp_dir
            corr_id = correlation_id or get_correlation_id()
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

    # Mock the get_services function at module level since it's imported directly
    import app.dependencies.services as services_module
    import app.routers.messages as messages_module

    original_get_services = services_module.get_services

    def mock_get_services(request=None):
        class DummyServices:
            def __init__(self):
                self.messages_pipeline = DummyPipeline()
                self.dumper = DummyDumper()
                self.exception_mapper = HttpExceptionMapper()
                self.error_formatter = ApiErrorFormatter()

                class Cfg:
                    dump_requests = True
                    dump_responses = True
                    dump_headers = True
                    dump_dir = str(tmp_path)
                    redact_headers = ['authorization', 'cookie', 'set-cookie']

                self.config = Cfg()

        return DummyServices()

    services_module.get_services = mock_get_services
    messages_module.get_services = mock_get_services

    try:
        resp = client.post('/v1/messages', json={'model': 'x', 'messages': []})
        assert resp.status_code == 200

        files = os.listdir(tmp_path)
        assert any('headers.json' in f for f in files)
        assert any('request.json' in f for f in files)
        sse_files = [f for f in files if f.endswith('response.sse')]
        assert len(sse_files) >= 1
        with open(os.path.join(tmp_path, sse_files[0]), 'rb') as f:
            content = f.read()
            assert b'hello' in content and b'world' in content
    finally:
        # Restore original function
        services_module.get_services = original_get_services
        messages_module.get_services = original_get_services

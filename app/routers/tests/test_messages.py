import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.dumper import DumpHandles
from app.dependencies.services import get_services
from app.routers.messages import router


def test_messages_endpoint():
    # Create a test client with the main app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyStream:
        async def stream_response(self, request, headers):
            yield b'data: ok\n\n'

    class DummyDumper:
        def begin(self, request, payload):
            return DumpHandles(headers_path=None, request_path=None, response_path=None, response_file=None)

        def write_chunk(self, handles, chunk):
            pass

        def close(self, handles):
            pass

    def override_get_services():
        # Create a completely mock services object without initializing real components
        class DummyServices:
            def __init__(self):
                self.anthropic = DummyStream()
                self.dumper = DummyDumper()

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
    from app.dependencies.services import get_services
    from app.main import app

    client = TestClient(app)

    class DummyStream:
        async def stream_response(self, request, headers):
            yield b'data: hello\n\n'
            yield b'data: world\n\n'

    class DummyDumper:
        def __init__(self):
            self.tmp_dir = str(tmp_path)
            self.files = []

        def begin(self, request, payload):
            os.makedirs(self.tmp_dir, exist_ok=True)
            open(os.path.join(self.tmp_dir, 'ts_corr_headers.json'), 'w').write('{}')
            open(os.path.join(self.tmp_dir, 'ts_corr_request.json'), 'w').write('{}')

            class H:
                def __init__(self, base):
                    self.response_file = open(os.path.join(base, 'ts_corr_response.sse'), 'wb')

            return H(self.tmp_dir)

        def write_chunk(self, handles, chunk):
            if handles.response_file:
                handles.response_file.write(chunk)
                handles.response_file.flush()

        def close(self, handles):
            if handles.response_file:
                handles.response_file.close()

    def override_get_services():
        # Create a completely mock services object without initializing real components
        class DummyServices:
            def __init__(self):
                self.anthropic = DummyStream()
                self.dumper = DummyDumper()

                class Cfg:
                    dump_requests = True
                    dump_responses = True
                    dump_headers = True
                    dump_dir = str(tmp_path)
                    redact_headers = ['authorization', 'cookie', 'set-cookie']

                self.config = Cfg()

        return DummyServices()

    # Override dependencies
    app.dependency_overrides[get_services] = override_get_services
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
        # Reset overrides after test
        app.dependency_overrides = {}

import os

from app.common.dumper import Dumper
from app.config import Config


class DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def test_dumper_writes_files(tmp_path):
    cfg = Config(cors_allow_origins=[], dump_requests=True, dump_responses=True, dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization', 'cookie'])
    d = Dumper(cfg)
    req = DummyRequest({'Authorization': 'secret', 'X-Other': 'ok'})
    handles = d.begin(req, {'a': 1})
    assert handles.headers_path and os.path.exists(handles.headers_path)
    assert handles.request_path and os.path.exists(handles.request_path)
    assert handles.response_path and handles.response_file
    d.write_chunk(handles, b'data: hello\n\n')
    d.write_chunk(handles, b'data: world\n\n')
    d.close(handles)
    with open(handles.response_path, 'rb') as f:
        content = f.read()
        assert b'hello' in content and b'world' in content


def test_dumper_redaction(tmp_path):
    cfg = Config(cors_allow_origins=[], dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization'])
    d = Dumper(cfg)
    req = DummyRequest({'Authorization': 'secret', 'Cookie': 'abc', 'X-Key': 'keep'})
    h = d.begin(req, {})
    with open(h.headers_path, 'r', encoding='utf-8') as f:
        data = f.read()
        assert '***REDACTED***' in data
        assert 'X-Key' in data and 'keep' in data

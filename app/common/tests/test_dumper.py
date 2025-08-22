import os

from app.common.dumper import Dumper
from app.config import ConfigModel


class DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def test_dumper_writes_files(tmp_path):
    cfg = ConfigModel(cors_allow_origins=[], dump_requests=True, dump_responses=True, dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization', 'cookie'])
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
    cfg = ConfigModel(cors_allow_origins=[], dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization'])
    d = Dumper(cfg)
    req = DummyRequest({'Authorization': 'secret', 'Cookie': 'abc', 'X-Key': 'keep'})
    h = d.begin(req, {})
    with open(h.headers_path, 'r', encoding='utf-8') as f:
        data = f.read()
        assert '***REDACTED***' in data
        assert 'X-Key' in data and 'keep' in data


def test_dumper_correlation_id(tmp_path):
    cfg = ConfigModel(cors_allow_origins=[], dump_requests=True, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})
    
    # Test with custom correlation ID
    custom_correlation_id = "custom-test-id-123"
    handles = d.begin(req, {'test': 'data'}, correlation_id=custom_correlation_id)
    
    # Check that the correlation ID is used in file names
    assert custom_correlation_id in handles.request_path
    d.close(handles)
    
    # Test without correlation ID (should generate one)
    handles2 = d.begin(req, {'test': 'data'})
    assert handles2.request_path is not None
    # Should contain a generated UUID-like string
    filename = os.path.basename(handles2.request_path)
    assert len(filename.split('_')[1]) == 32  # UUID hex length
    d.close(handles2)

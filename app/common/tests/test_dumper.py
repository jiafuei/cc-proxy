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
    assert handles.correlation_id  # Verify correlation_id is populated
    assert len(handles.correlation_id) == 32  # Should be UUID hex length
    d.write_response_chunk(handles, b'data: hello\n\n')
    d.write_response_chunk(handles, b'data: world\n\n')
    d.close(handles)
    with open(handles.response_path, 'rb') as f:
        content = f.read()
        assert b'hello' in content and b'world' in content


def test_dumper_redaction(tmp_path):
    cfg = ConfigModel(cors_allow_origins=[], dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization'])
    d = Dumper(cfg)
    req = DummyRequest({'Authorization': 'secret', 'Cookie': 'abc', 'X-Key': 'keep'})
    h = d.begin(req, {})
    assert h.correlation_id  # Verify correlation_id is populated
    with open(h.headers_path, 'r', encoding='utf-8') as f:
        data = f.read()
        assert '***REDACTED***' in data
        assert 'X-Key' in data and 'keep' in data


def test_dumper_correlation_id(tmp_path):
    cfg = ConfigModel(cors_allow_origins=[], dump_requests=True, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Test with custom correlation ID
    custom_correlation_id = 'custom-test-id-123'
    handles = d.begin(req, {'test': 'data'}, correlation_id=custom_correlation_id)

    # Check that the correlation ID is used in file names and available in handles
    assert handles.correlation_id == custom_correlation_id
    assert custom_correlation_id in handles.request_path
    d.close(handles)

    # Test without correlation ID (should generate one)
    handles2 = d.begin(req, {'test': 'data'})
    assert handles2.request_path is not None
    assert handles2.correlation_id  # Should have a correlation ID
    assert len(handles2.correlation_id) == 32  # UUID hex length
    # Should contain a generated UUID-like string
    filename = os.path.basename(handles2.request_path)
    assert len(filename.split('_')[1]) == 32  # UUID hex length
    assert handles2.correlation_id in handles2.request_path  # Should be in filename
    d.close(handles2)


def test_dumper_transformed_request(tmp_path):
    """Test that transformed request dumping works correctly."""
    cfg = ConfigModel(cors_allow_origins=[], dump_transformed_requests=True, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Use begin to get handles with file paths
    handles = d.begin(req, {'initial': 'data'})
    correlation_id = handles.correlation_id
    transformed_request = {'model': 'claude-3-sonnet', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True}

    d.write_transformed_request(handles, transformed_request)
    d.close(handles)

    # Check that the file was created
    files = [f for f in os.listdir(tmp_path) if f.endswith('_3transformed_request.json')]
    assert len(files) == 1

    # Check that the correlation ID is in the filename
    assert correlation_id in files[0]

    # Check file contents
    with open(os.path.join(tmp_path, files[0]), 'r', encoding='utf-8') as f:
        import json

        data = json.load(f)
        assert data['model'] == 'claude-3-sonnet'
        assert data['stream'] is True


def test_dumper_pretransformed_response(tmp_path):
    """Test that pre-transformed response dumping works correctly."""
    cfg = ConfigModel(cors_allow_origins=[], dump_pretransformed_responses=True, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Use begin to get handles with file paths
    handles = d.begin(req, {'initial': 'data'})
    correlation_id = handles.correlation_id

    # Test with bytes
    d.write_pretransformed_response(handles, b'event: message_start\ndata: {"type": "message_start"}\n\n')
    d.write_pretransformed_response(handles, b'event: content_block_start\ndata: {"type": "content_block_start"}\n\n')
    d.close(handles)

    # Check that the file was created
    files = [f for f in os.listdir(tmp_path) if f.endswith('_4pretransformed_response.sse')]
    assert len(files) == 1

    # Check that the correlation ID is in the filename
    assert correlation_id in files[0]

    # Check file contents
    with open(os.path.join(tmp_path, files[0]), 'rb') as f:
        content = f.read()
        assert b'message_start' in content
        assert b'content_block_start' in content


def test_dumper_pretransformed_response_string(tmp_path):
    """Test that pre-transformed response dumping works with string input."""
    cfg = ConfigModel(cors_allow_origins=[], dump_pretransformed_responses=True, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Use begin to get handles with file paths
    handles = d.begin(req, {'initial': 'data'})

    # Test with string
    d.write_pretransformed_response(handles, 'event: message_start\ndata: {"type": "message_start"}\n\n')
    d.close(handles)

    # Check that the file was created
    files = [f for f in os.listdir(tmp_path) if f.endswith('_4pretransformed_response.sse')]
    assert len(files) == 1

    # Check file contents
    with open(os.path.join(tmp_path, files[0]), 'rb') as f:
        content = f.read()
        assert b'message_start' in content


def test_dumper_disabled_features(tmp_path):
    """Test that dumping is skipped when features are disabled."""
    cfg = ConfigModel(cors_allow_origins=[], dump_transformed_requests=False, dump_pretransformed_responses=False, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Use begin to get handles - but transformed features are disabled
    handles = d.begin(req, {'initial': 'data'})

    # Try to dump - should do nothing since features are disabled
    d.write_transformed_request(handles, {'test': 'data'})
    d.write_pretransformed_response(handles, b'test chunk')

    # Check that no transformed files were created (only original request files might be created)
    transformed_files = [f for f in os.listdir(tmp_path) if 'transformed' in f or 'pretransformed' in f]
    assert len(transformed_files) == 0

    d.close(handles)

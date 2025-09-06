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

    # Check that files were created on disk
    files = os.listdir(tmp_path)
    header_files = [f for f in files if 'original_headers' in f]
    request_files = [f for f in files if 'original_request' in f]
    assert len(header_files) == 1  # Headers file created
    assert len(request_files) == 1  # Request file created

    # Check response file handles are available (streaming files)
    assert handles.files.final_response and handles.files.final_response_file
    assert handles.correlation_id  # Verify correlation_id is populated
    assert len(handles.correlation_id) == 32  # Should be UUID hex length

    d.write_response_chunk(handles, b'data: hello\n\n')
    d.write_response_chunk(handles, b'data: world\n\n')
    d.close(handles)

    with open(handles.files.final_response, 'rb') as f:
        content = f.read()
        assert b'hello' in content and b'world' in content


def test_dumper_redaction(tmp_path):
    cfg = ConfigModel(cors_allow_origins=[], dump_headers=True, dump_dir=str(tmp_path), redact_headers=['authorization'])
    d = Dumper(cfg)
    req = DummyRequest({'Authorization': 'secret', 'Cookie': 'abc', 'X-Key': 'keep'})
    h = d.begin(req, {})
    assert h.correlation_id  # Verify correlation_id is populated

    # Find the headers file that was created
    files = os.listdir(tmp_path)
    header_files = [f for f in files if 'original_headers' in f and f.endswith('.json')]
    assert len(header_files) == 1

    with open(os.path.join(tmp_path, header_files[0]), 'r', encoding='utf-8') as f:
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

    # Check files contain the custom correlation ID
    files = os.listdir(tmp_path)
    request_files = [f for f in files if 'original_request' in f]
    assert len(request_files) == 1
    assert custom_correlation_id in request_files[0]
    d.close(handles)

    # Test without correlation ID (should generate one)
    handles2 = d.begin(req, {'test': 'data'})
    assert handles2.correlation_id  # Should have a correlation ID
    assert len(handles2.correlation_id) == 32  # UUID hex length

    # Check generated correlation ID is in filenames
    files = os.listdir(tmp_path)
    new_request_files = [f for f in files if 'original_request' in f and f not in request_files]
    assert len(new_request_files) == 1
    assert handles2.correlation_id in new_request_files[0]
    d.close(handles2)


def test_dumper_disabled_features(tmp_path):
    """Test that dumping is skipped when features are disabled."""
    cfg = ConfigModel(cors_allow_origins=[], dump_requests=False, dump_responses=False, dump_headers=False, dump_dir=str(tmp_path))
    d = Dumper(cfg)
    req = DummyRequest({'X-Test': 'value'})

    # Use begin to get handles - but all features are disabled
    handles = d.begin(req, {'initial': 'data'})

    # Check that no files were created since all features are disabled
    files = os.listdir(tmp_path)
    assert len(files) == 0

    d.close(handles)

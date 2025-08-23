"""Test config API endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_config_status_endpoint():
    """Test GET /api/config/status endpoint."""
    response = client.get('/api/config/status')
    assert response.status_code == 200

    data = response.json()
    assert 'loaded' in data
    assert 'config_file_exists' in data
    assert 'config_path' in data


def test_config_validate_endpoint():
    """Test GET /api/config/validate endpoint."""
    response = client.get('/api/config/validate')
    assert response.status_code == 200

    data = response.json()
    assert 'valid' in data
    assert 'errors' in data


def test_reload_endpoint():
    """Test POST /api/reload endpoint."""
    response = client.post('/api/reload')
    assert response.status_code == 200

    data = response.json()
    assert 'success' in data
    assert 'message' in data

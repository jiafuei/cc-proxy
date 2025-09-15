"""Test config API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.tests.utils import create_test_app_with_mocks


@pytest.mark.parametrize(
    'endpoint,method,expected_keys',
    [
        ('/api/config/status', 'get', ['loaded', 'config_file_exists', 'config_path']),
        ('/api/config/validate', 'get', ['valid', 'errors']),
        ('/api/reload', 'post', ['success', 'message']),
    ],
)
def test_config_api_endpoints(endpoint, method, expected_keys):
    """Test config API endpoints return 200 status and expected keys."""
    app = create_test_app_with_mocks()
    client = TestClient(app)
    
    response = getattr(client, method)(endpoint)
    assert response.status_code == 200

    data = response.json()
    for key in expected_keys:
        assert key in data

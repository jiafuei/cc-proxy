from fastapi.testclient import TestClient
from app.routers.health import router

client = TestClient(router)


def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

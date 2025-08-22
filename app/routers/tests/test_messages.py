from fastapi.testclient import TestClient
from app.routers.messages import router

client = TestClient(router)


def test_messages_endpoint():
    response = client.post('/v1/messages')
    assert response.status_code == 200
    assert response.json() == []
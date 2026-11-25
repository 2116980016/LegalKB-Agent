import pytest

pytest.importorskip("httpx", reason="TestClient 需要 httpx")
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index"]["n_chunks"] > 0


def test_chat_returns_citations(client):
    resp = client.post("/api/chat", json={"question": "违约金约定过高，法院可以调整吗？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert data["citations"], "应当返回引用"
    assert all(c["valid"] for c in data["citations"])
    assert data["request_id"]


def test_chat_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_chat_stream_emits_sse(client):
    with client.stream("POST", "/api/chat/stream", json={"question": "结婚年龄是多少？"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    assert "event: delta" in body
    assert "event: citations" in body
    assert "event: done" in body


def test_stats_endpoint(client):
    assert "requests" in client.get("/api/stats").json()

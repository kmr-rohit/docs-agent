"""Tests for feedback service persistence and API."""

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FEEDBACK_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "feedback-service"


@pytest.fixture
def feedback_module(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "feedback.db"))

    for name in list(sys.modules):
        if name in ("db", "app"):
            del sys.modules[name]

    db_spec = importlib.util.spec_from_file_location("db", FEEDBACK_DIR / "db.py")
    db = importlib.util.module_from_spec(db_spec)
    sys.modules["db"] = db
    db_spec.loader.exec_module(db)

    app_spec = importlib.util.spec_from_file_location("feedback_app", FEEDBACK_DIR / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    sys.modules["feedback_app"] = app_module
    app_spec.loader.exec_module(app_module)

    db.init_db()
    return app_module, db


@pytest.fixture
def client(feedback_module):
    app_module, _ = feedback_module
    return TestClient(app_module.app)


class TestFeedbackAPI:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_create_and_export_feedback(self, client):
        payload = {
            "context_id": "ctx-1",
            "message_id": "msg-1",
            "query": "what is kubeflow",
            "response": "Kubeflow is an ML platform.",
            "citations": ["https://www.kubeflow.org/docs/"],
            "rating": 5,
            "comment": "helpful",
        }
        create = client.post("/api/feedback", json=payload)
        assert create.status_code == 200
        assert create.json()["status"] == "ok"
        assert create.json()["feedback"]["rating"] == 5

        export = client.get("/api/feedback/export", params={"min_rating": 4})
        assert export.status_code == 200
        body = export.json()
        assert body["count"] == 1
        assert body["items"][0]["query"] == "what is kubeflow"
        assert body["items"][0]["citations"] == ["https://www.kubeflow.org/docs/"]

    def test_rejects_invalid_rating(self, client):
        payload = {
            "context_id": "ctx-1",
            "query": "q",
            "response": "r",
            "rating": 6,
        }
        response = client.post("/api/feedback", json=payload)
        assert response.status_code == 422

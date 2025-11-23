"""
API endpoint tests.

Test FastAPI endpoints to ensure correct request/response handling.
"""

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test /health returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test / returns service info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


@pytest.mark.skip(reason="Requires loaded model")
class TestQAEndpoint:
    """Test question-answering endpoint."""

    def test_qa_with_context(self, client):
        """Test /qa with context provided."""
        response = client.post(
            "/qa",
            json={
                "question": "What is the capital of France?",
                "context": "France is a country. Paris is its capital.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "model_version" in data

    def test_qa_without_context(self, client):
        """Test /qa without context returns error."""
        response = client.post(
            "/qa",
            json={"question": "What is the capital of France?"},
        )

        # Should return 400 or 422 (validation error)
        assert response.status_code in [400, 422]

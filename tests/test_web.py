"""Smoke tests that do not require a physical camera."""

from edgevision.web import app


def test_dashboard():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"EdgeVision" in response.data


def test_status():
    response = app.test_client().get("/api/status")
    assert response.status_code == 200
    assert "detections" in response.get_json()

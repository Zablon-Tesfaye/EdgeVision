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


def test_health_reports_missing_assets(monkeypatch, tmp_path):
    from edgevision import web
    from edgevision.config import Settings
    monkeypatch.setattr(web, "settings", Settings(
        model_path=tmp_path / "missing.tflite", labels_path=tmp_path / "missing.txt"
    ))
    response = app.test_client().get("/api/health")
    assert response.status_code == 503
    assert response.get_json()["missing"] == ["model", "labels"]

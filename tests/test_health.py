from edgevision.health import readiness


def test_reports_missing_assets(tmp_path):
    assert readiness(tmp_path / "a.tflite", tmp_path / "labels.txt") == {
        "ready": False, "missing": ["model", "labels"]
    }


def test_reports_ready_assets(tmp_path):
    model, labels = tmp_path / "a.tflite", tmp_path / "labels.txt"
    model.write_bytes(b"demo")
    labels.write_text("person")
    assert readiness(model, labels)["ready"]

"""Non-invasive local readiness checks for model assets."""

from pathlib import Path


def readiness(model: Path, labels: Path) -> dict:
    missing = [name for name, path in (("model", model), ("labels", labels)) if not path.is_file()]
    return {"ready": not missing, "missing": missing}

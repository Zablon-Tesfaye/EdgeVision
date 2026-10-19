"""Runtime configuration without hardcoded secrets or hardware paths."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("EDGEVISION_MODEL", "models/detect.tflite"))
    labels_path: Path = Path(os.getenv("EDGEVISION_LABELS", "models/labels.txt"))
    camera_index: int = int(os.getenv("EDGEVISION_CAMERA", "0"))
    confidence: float = float(os.getenv("EDGEVISION_CONFIDENCE", "0.45"))
    input_width: int = 640
    input_height: int = 480
    max_fps: float = 15.0

    def validate(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("Confidence must be between zero and one")
        if self.max_fps <= 0:
            raise ValueError("Maximum FPS must be positive")

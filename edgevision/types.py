"""Typed inference results shared by the CLI and web interface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.score, 4),
            "box": list(self.box),
        }


@dataclass(frozen=True)
class FrameResult:
    detections: tuple[Detection, ...]
    inference_ms: float
    width: int
    height: int

    def as_dict(self) -> dict:
        return {
            "detections": [item.as_dict() for item in self.detections],
            "inference_ms": round(self.inference_ms, 2),
            "width": self.width,
            "height": self.height,
        }

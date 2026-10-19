"""Simple, reproducible runtime metrics."""

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class LatencyTracker:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("Latency cannot be negative")
        self.samples_ms.append(latency_ms)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"frames": 0, "mean_inference_ms": None, "estimated_inference_fps": None}
        average = mean(self.samples_ms)
        return {
            "frames": len(self.samples_ms),
            "mean_inference_ms": round(average, 2),
            "estimated_inference_fps": round(1000 / average, 2) if average else None,
        }

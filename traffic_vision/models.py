from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VehicleClass = Literal["car", "motorcycle", "bus", "truck"]


@dataclass(frozen=True)
class Detection:
    """A single detector output in pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class Track:
    """A tracked object with a stable local ID."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    centroid: tuple[float, float]
    label: str
    confidence: float
    missed_frames: int = 0


@dataclass(frozen=True)
class CountEvent:
    """A line-crossing count event ready for persistence."""

    track_id: int
    label: str
    direction: str
    centroid: tuple[float, float]

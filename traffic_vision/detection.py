from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Detection


DEFAULT_VEHICLE_LABELS = frozenset({"car", "motorcycle", "bus", "truck"})


@dataclass
class DetectorConfig:
    model_name: str = "yolo26n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    vehicle_labels: frozenset[str] | None = DEFAULT_VEHICLE_LABELS
    device: str | None = None


class YoloVehicleDetector:
    """Vehicle detector backed by the Ultralytics YOLO Python API."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_name)
        return self._model

    def detect(self, frame: Any) -> list[Detection]:
        predict_kwargs: dict[str, Any] = {
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "verbose": False,
        }
        if self.config.device:
            predict_kwargs["device"] = self.config.device

        results = self.model.predict(frame, **predict_kwargs)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                label = str(names[class_id])
                if self.config.vehicle_labels is not None and label not in self.config.vehicle_labels:
                    continue
                detections.append(
                    Detection(
                        xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                        confidence=float(box.conf[0].item()),
                        class_id=class_id,
                        label=label,
                    )
                )
        return detections

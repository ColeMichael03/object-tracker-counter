from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .camera import CameraConfig, CameraStream, validate_camera_source
from .counting import LineCounter
from .detection import DEFAULT_VEHICLE_LABELS, DetectorConfig, YoloVehicleDetector
from .storage import SQLiteStore
from .tracking import CentroidTracker
from .visualization import draw_overlay


@dataclass
class PipelineConfig:
    source: str | int = 0
    db_path: str = "traffic_counts.db"
    model_name: str = "yolo26n.pt"
    confidence: float = 0.35
    line_position: int = 360
    line_orientation: str = "horizontal"
    max_frames: int | None = None
    all_classes: bool = False


class TrafficPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        validation = validate_camera_source(config.source)
        if not validation.ok:
            raise RuntimeError(validation.message)
        self.detector = YoloVehicleDetector(
            DetectorConfig(
                model_name=config.model_name,
                confidence=config.confidence,
                vehicle_labels=None if config.all_classes else DEFAULT_VEHICLE_LABELS,
            )
        )
        self.tracker = CentroidTracker()
        self.counter = LineCounter(config.line_position, config.line_orientation)
        self.store = SQLiteStore(config.db_path)
        self.store.initialize()

    def run(self) -> Iterator[dict[str, Any]]:
        with CameraStream(CameraConfig(source=self.config.source)) as camera:
            for frame_number, frame in enumerate(camera.frames(), start=1):
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections)
                events = self.counter.update(tracks)
                self.store.record_events(events)
                annotated = draw_overlay(
                    frame.copy(),
                    tracks,
                    self.counter.counts,
                    self.config.line_position,
                    self.config.line_orientation,
                )
                yield {
                    "frame_number": frame_number,
                    "frame": annotated,
                    "detections": detections,
                    "tracks": tracks,
                    "events": events,
                    "counts": self.counter.counts.copy(),
                }
                if self.config.max_frames and frame_number >= self.config.max_frames:
                    break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, track, and count highway vehicles.")
    parser.add_argument("--source", default="0", help="Camera index, video file, or RTSP URL.")
    parser.add_argument("--db", default="traffic_counts.db", help="SQLite database path.")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics YOLO model name/path.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detector confidence threshold.")
    parser.add_argument("--line", type=int, default=360, help="Counting line x/y pixel coordinate.")
    parser.add_argument(
        "--orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Counting line orientation.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Detect, track, and count every YOLO class instead of vehicle classes only.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pipeline = TrafficPipeline(
        PipelineConfig(
            source=args.source,
            db_path=args.db,
            model_name=args.model,
            confidence=args.confidence,
            line_position=args.line,
            line_orientation=args.orientation,
            max_frames=args.max_frames,
            all_classes=args.all_classes,
        )
    )
    for snapshot in pipeline.run():
        counts = dict(snapshot["counts"])
        print(f"frame={snapshot['frame_number']} counts={counts}")


if __name__ == "__main__":
    main()

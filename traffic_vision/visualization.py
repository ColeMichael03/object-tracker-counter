from __future__ import annotations

from collections import Counter
from typing import Any

from .models import Track


def draw_overlay(
    frame: Any,
    tracks: list[Track],
    counts: Counter[str],
    line_position: int,
    orientation: str = "horizontal",
) -> Any:
    import cv2

    height, width = frame.shape[:2]
    if orientation == "horizontal":
        cv2.line(frame, (0, line_position), (width, line_position), (0, 255, 255), 2)
    else:
        cv2.line(frame, (line_position, 0), (line_position, height), (0, 255, 255), 2)

    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.xyxy)
        cx, cy = (int(v) for v in track.centroid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 180, 60), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"#{track.track_id} {track.label} {track.confidence:.2f}",
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Total: {counts.get('total', 0)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame

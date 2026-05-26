from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

from .models import Detection, Track


@dataclass
class CentroidTracker:
    """Small dependency-free tracker that matches detections by centroid distance."""

    max_distance: float = 80.0
    max_missed_frames: int = 10
    _next_id: int = 1
    _tracks: dict[int, Track] = field(default_factory=dict)

    def update(self, detections: list[Detection]) -> list[Track]:
        if not self._tracks:
            for detection in detections:
                self._register(detection)
            return list(self._tracks.values())

        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indexes = set(range(len(detections)))
        matches: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = dist(track.centroid, detection.centroid)
                if distance <= self.max_distance:
                    matches.append((distance, track_id, detection_index))

        for _distance, track_id, detection_index in sorted(matches):
            if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_indexes:
                continue
            self._tracks[track_id] = self._track_from_detection(track_id, detections[detection_index])
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            missed = track.missed_frames + 1
            if missed > self.max_missed_frames:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = Track(
                    track_id=track.track_id,
                    xyxy=track.xyxy,
                    centroid=track.centroid,
                    label=track.label,
                    confidence=track.confidence,
                    missed_frames=missed,
                )

        for detection_index in unmatched_detection_indexes:
            self._register(detections[detection_index])

        return list(self._tracks.values())

    def _register(self, detection: Detection) -> None:
        self._tracks[self._next_id] = self._track_from_detection(self._next_id, detection)
        self._next_id += 1

    @staticmethod
    def _track_from_detection(track_id: int, detection: Detection) -> Track:
        return Track(
            track_id=track_id,
            xyxy=detection.xyxy,
            centroid=detection.centroid,
            label=detection.label,
            confidence=detection.confidence,
        )

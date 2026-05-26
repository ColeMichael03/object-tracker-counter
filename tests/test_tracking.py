from traffic_vision.models import Detection
from traffic_vision.tracking import CentroidTracker


def detection(x1: float, y1: float, x2: float, y2: float, label: str = "car") -> Detection:
    return Detection(xyxy=(x1, y1, x2, y2), confidence=0.9, class_id=2, label=label)


def test_tracker_preserves_id_for_nearby_detection() -> None:
    tracker = CentroidTracker(max_distance=25)

    first = tracker.update([detection(0, 0, 10, 10)])
    second = tracker.update([detection(5, 5, 15, 15)])

    assert first[0].track_id == second[0].track_id
    assert second[0].missed_frames == 0


def test_tracker_drops_stale_track_after_missed_frames() -> None:
    tracker = CentroidTracker(max_missed_frames=1)

    tracker.update([detection(0, 0, 10, 10)])
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []

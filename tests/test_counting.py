from traffic_vision.counting import LineCounter
from traffic_vision.models import Track


def make_track(track_id: int, y: float, label: str = "car") -> Track:
    return Track(
        track_id=track_id,
        xyxy=(0, y - 5, 10, y + 5),
        centroid=(5, y),
        label=label,
        confidence=0.9,
    )


def test_counts_track_once_when_crossing_horizontal_line() -> None:
    counter = LineCounter(line_position=100)

    assert counter.update([make_track(1, 80)]) == []
    events = counter.update([make_track(1, 120)])
    assert len(events) == 1
    assert events[0].direction == "southbound"
    assert counter.counts["car"] == 1
    assert counter.counts["total"] == 1

    assert counter.update([make_track(1, 80)]) == []
    assert counter.counts["total"] == 1

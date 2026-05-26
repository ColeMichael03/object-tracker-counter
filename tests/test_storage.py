from traffic_vision.models import CountEvent
from traffic_vision.storage import SQLiteStore


def test_records_and_reads_count_events(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "traffic.db")
    store.initialize()

    written = store.record_events(
        [
            CountEvent(track_id=1, label="car", direction="southbound", centroid=(10, 20)),
            CountEvent(track_id=2, label="truck", direction="southbound", centroid=(30, 40)),
        ]
    )

    assert written == 2
    assert store.totals_by_label() == {"car": 1, "truck": 1}
    recent = store.recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0]["label"] in {"car", "truck"}

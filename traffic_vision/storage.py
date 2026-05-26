from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import CountEvent


@dataclass
class SQLiteStore:
    db_path: str | Path = "traffic_counts.db"

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    counted_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_vehicle_counts_counted_at ON vehicle_counts(counted_at)"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def record_events(self, events: Iterable[CountEvent]) -> int:
        rows = [
            (
                event.track_id,
                event.label,
                event.direction,
                event.centroid[0],
                event.centroid[1],
                datetime.now(UTC).isoformat(),
            )
            for event in events
        ]
        if not rows:
            return 0
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO vehicle_counts
                    (track_id, label, direction, centroid_x, centroid_y, counted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def totals_by_label(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT label, COUNT(*) AS count FROM vehicle_counts GROUP BY label ORDER BY count DESC"
            ).fetchall()
        return {str(row["label"]): int(row["count"]) for row in rows}

    def recent_events(self, limit: int = 50) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT track_id, label, direction, centroid_x, centroid_y, counted_at
                FROM vehicle_counts
                ORDER BY counted_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

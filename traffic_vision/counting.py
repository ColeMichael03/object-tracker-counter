from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import CountEvent, Track


@dataclass
class LineCounter:
    """Counts each track once when its centroid crosses a horizontal or vertical line."""

    line_position: float
    orientation: str = "horizontal"
    counts: Counter[str] = field(default_factory=Counter)
    _previous_side: dict[int, int] = field(default_factory=dict)
    _counted_track_ids: set[int] = field(default_factory=set)

    def update(self, tracks: list[Track]) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks:
            current_side = self._side(track)
            previous_side = self._previous_side.get(track.track_id)
            self._previous_side[track.track_id] = current_side

            if previous_side is None or previous_side == 0 or current_side == 0:
                continue
            if previous_side == current_side or track.track_id in self._counted_track_ids:
                continue

            direction = self._direction(previous_side, current_side)
            self.counts[track.label] += 1
            self.counts["total"] += 1
            self._counted_track_ids.add(track.track_id)
            events.append(
                CountEvent(
                    track_id=track.track_id,
                    label=track.label,
                    direction=direction,
                    centroid=track.centroid,
                )
            )
        return events

    def _side(self, track: Track) -> int:
        x, y = track.centroid
        value = y if self.orientation == "horizontal" else x
        if value < self.line_position:
            return -1
        if value > self.line_position:
            return 1
        return 0

    def _direction(self, previous_side: int, current_side: int) -> str:
        if self.orientation == "horizontal":
            return "southbound" if previous_side < current_side else "northbound"
        return "eastbound" if previous_side < current_side else "westbound"

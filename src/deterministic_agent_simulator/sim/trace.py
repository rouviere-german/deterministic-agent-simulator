"""Execution traces and their hashes.

A trace is the ordered list of observable events of one simulated run: RPCs and their outcomes,
model decisions, injected faults, effects, crashes and oracle verdicts. Every event is folded into
a running SHA-256 as it is recorded, together with every scheduling decision the event loop made
(which of several ready callbacks ran). Two runs with the same trace hash therefore executed the
same interleaving and observed the same results; that is the determinism contract a seed must
meet.

Events carry only strings, integers, booleans and ``None``, encoded as compact JSON with ASCII
escapes, so the byte stream fed to the hash is identical on every platform.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = ["SCHEDULING_KIND", "Trace", "TraceEvent", "TraceField"]

TraceField = str | int | bool | None
SCHEDULING_KIND = "sched"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    seq: int
    t_us: int
    kind: str
    fields: tuple[TraceField, ...]

    def render(self) -> str:
        return json.dumps(
            [self.t_us, self.kind, *self.fields], separators=(",", ":"), ensure_ascii=True
        )


class Trace:
    """Append-only event log with an incremental hash.

    ``keep=False`` keeps only the hash and counters (campaigns); ``keep=True`` also keeps the
    events (shrinking, reports, span export). ``length`` counts semantic events: scheduling
    decisions are hashed but not counted, so the length measures what the system did rather than
    how often the scheduler had a choice.
    """

    __slots__ = ("_frozen", "_hasher", "events", "keep", "length", "schedule_points")

    def __init__(self, *, keep: bool = False) -> None:
        self.keep = keep
        self.events: list[TraceEvent] = []
        self.length = 0
        self.schedule_points = 0
        self._hasher = hashlib.sha256()
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Stop recording (teardown after a verdict must not change the hash)."""
        self._frozen = True

    def record(self, t_us: int, kind: str, *fields: TraceField) -> None:
        if self._frozen:
            return
        event = TraceEvent(self.length, t_us, kind, fields)
        self._hasher.update(event.render().encode("ascii"))
        self._hasher.update(b"\n")
        self.length += 1
        if self.keep:
            self.events.append(event)

    def schedule(self, ready: int, chosen: int) -> None:
        """Fold one scheduling decision (``chosen`` of ``ready`` callbacks) into the hash."""
        if self._frozen:
            return
        self._hasher.update(f"s{ready}.{chosen};".encode("ascii"))
        self.schedule_points += 1

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    def kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return counts

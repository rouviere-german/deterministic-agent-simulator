"""Fault profiles, buggify-style fault points and swarm configurations.

Fault points are named places in the simulated world (``email.send``, ``inbound``, ``ledger``,
``agent.store.put_durable.post`` ...). At each point the :class:`FaultPlan` decides whether a fault
of a given kind fires. The decision has two modes:

- **random** (exploration): each fault point draws from its own keyed stream, with the rate from
  the seed's :class:`FaultProfile`. Keyed streams keep a point's decisions stable when unrelated
  parts of the run change.
- **scheduled** (replay and shrinking): a fault fires exactly when its :data:`FaultKey`
  ``(kind, point, scope, occurrence)`` is in the schedule. The keys that fired during a random
  run form its fault schedule, so replaying that schedule reproduces the run, and delta debugging
  can remove keys one subset at a time.

Swarm testing (Groce, Zhang, Eide, Chen and Regehr, "Swarm Testing", ISSTA 2012) draws a
different profile per seed: each fault kind is switched off with probability 1/2 and the rest get
a random intensity. Omitting features per test avoids faults that mask each other (frequent
crashes, for example, end runs before a duplicate delivery can matter).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace

__all__ = [
    "FIXED_RATES",
    "FaultKey",
    "FaultKind",
    "FaultPlan",
    "FaultProfile",
]


class FaultKind(StrEnum):
    LATENCY_SPIKE = "latency_spike"
    ERROR_5XX = "error_5xx"
    TIMEOUT_AFTER_COMMIT = "timeout_after_commit"
    DUPLICATE_DELIVERY = "duplicate_delivery"
    PARTIAL_WRITE = "partial_write"
    AGENT_CRASH = "agent_crash"
    LEDGER_CRASH = "ledger_crash"
    CLOCK_SKEW = "clock_skew"
    PROMPT_INJECTION = "prompt_injection"
    SLOW_REVIEW = "slow_review"
    AUDIT_READ = "audit_read"


FaultKey = tuple[str, str, str, int]
"""``(kind, point, scope, occurrence)``: the n-th chance for ``kind`` at ``point`` in ``scope``."""

# Per-opportunity probabilities of the "fixed" configuration. An opportunity is one RPC for the
# network kinds, one ticket for injections and duplicate deliveries, one agent port call for
# crashes, one ledger commit for ledger crashes, one agent incarnation for clock skew, one approval
# request for a slow reviewer and one 250 ms tick for an auditor read. The last two are not
# failures but environment behavior; making them buggify points lets the shrinker remove them.
FIXED_RATES: Mapping[FaultKind, float] = {
    FaultKind.LATENCY_SPIKE: 0.03,
    FaultKind.ERROR_5XX: 0.04,
    FaultKind.TIMEOUT_AFTER_COMMIT: 0.03,
    FaultKind.DUPLICATE_DELIVERY: 0.10,
    FaultKind.PARTIAL_WRITE: 0.05,
    FaultKind.AGENT_CRASH: 0.004,
    FaultKind.LEDGER_CRASH: 0.015,
    FaultKind.CLOCK_SKEW: 0.30,
    FaultKind.PROMPT_INJECTION: 0.25,
    FaultKind.SLOW_REVIEW: 0.35,
    FaultKind.AUDIT_READ: 0.5,
}
_SWARM_INTENSITIES = (0.5, 1.0, 2.0, 4.0)
_NAMES: Mapping[FaultKind, str] = {kind: kind.value for kind in FaultKind}


@dataclass(frozen=True, slots=True)
class FaultProfile:
    """Per-kind fault rates for random mode."""

    name: str
    rates: tuple[tuple[FaultKind, float], ...]

    @classmethod
    def of(cls, name: str, rates: Mapping[FaultKind, float]) -> FaultProfile:
        clean = tuple(sorted(((kind, float(rate)) for kind, rate in rates.items()), key=str))
        for kind, rate in clean:
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"rate for {kind} must be within [0, 1]")
        return cls(name=name, rates=clean)

    @classmethod
    def none(cls) -> FaultProfile:
        return cls.of("none", {})

    @classmethod
    def fixed(cls) -> FaultProfile:
        return cls.of("fixed", FIXED_RATES)

    @classmethod
    def swarm(cls, rng: SimRandom) -> FaultProfile:
        """A random subset of fault kinds with random intensities (swarm testing)."""
        rates: dict[FaultKind, float] = {}
        for kind in FaultKind:  # enum order is fixed, so the draws are too
            if rng.chance(0.5):
                rates[kind] = min(1.0, FIXED_RATES[kind] * rng.choice(_SWARM_INTENSITIES))
        return cls.of("swarm", rates)

    def rate(self, kind: FaultKind) -> float:
        for candidate, rate in self.rates:
            if candidate == kind:
                return rate
        return 0.0

    def as_dict(self) -> dict[str, float]:
        return {str(kind): rate for kind, rate in self.rates}

    def enabled(self) -> list[str]:
        return [kind.value for kind, rate in self.rates if rate > 0.0]


class FaultPlan:
    """Decides whether a fault fires at a fault point; records every fault that fired."""

    def __init__(
        self,
        profile: FaultProfile,
        rng: SimRandom,
        trace: Trace,
        clock: Callable[[], int],
        *,
        schedule: Iterable[FaultKey] | None = None,
    ) -> None:
        self.profile = profile
        self._rng = rng
        self._trace = trace
        self._clock = clock
        self._schedule = frozenset(schedule) if schedule is not None else None
        self._rates = profile.as_dict()
        self._counters: dict[tuple[str, str, str], int] = {}
        self._streams: dict[tuple[str, str, str], SimRandom] = {}
        self.fired: list[FaultKey] = []

    @property
    def scheduled(self) -> bool:
        return self._schedule is not None

    def fires(self, kind: FaultKind, point: str, scope: str) -> bool:
        name = _NAMES[kind]
        slot = (name, point, scope)
        occurrence = self._counters.get(slot, 0) + 1
        self._counters[slot] = occurrence
        key: FaultKey = (name, point, scope, occurrence)
        if self._schedule is not None:
            hit = key in self._schedule
        else:
            rate = self._rates.get(name, 0.0)
            if rate <= 0.0:
                return False
            stream = self._streams.get(slot)
            if stream is None:
                stream = self._rng.derive(*slot)
                self._streams[slot] = stream
            hit = stream.chance(rate)
        if hit:
            self.fired.append(key)
            self._trace.record(self._clock(), "fault", name, point, scope, occurrence)
        return hit

    def draw(self, point: str, scope: str) -> SimRandom:
        """A keyed stream for fault parameters (downtime, skew size...) at ``point``/``scope``."""
        return self._rng.derive("param", point, scope)

"""Simulated RPC: latency, 5xx before commit, timeouts after commit, and unavailable services.

``Network.call`` models one request/response exchange with a service whose state change
(``apply``) happens atomically at a single instant between the two network legs:

1. request leg: a latency draw (with an occasional ``latency_spike`` fault);
2. if the service is down, the client waits for its timeout and sees an ambiguous failure;
3. an ``error_5xx`` fault fails the call *before* ``apply``: nothing happened;
4. ``apply()`` runs: this is the commit point (a service may crash right after it, in which
   case the response is lost and the client times out);
5. a ``timeout_after_commit`` fault makes the client time out *after* the commit: the effect
   happened but the client cannot tell, which is exactly the ambiguity that breaks naive
   retries;
6. response leg latency, then the result (or the service's business rejection) is returned.

A cancelled caller (the agent process crashed mid-call) stops waiting, but a commit that
already happened stays committed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from deterministic_agent_simulator.agent.ports import AmbiguousError, RejectedError, TransientError
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom

__all__ = ["DEFAULT_LATENCY_US", "LatencyModel", "Network"]

_R = TypeVar("_R")

DEFAULT_LATENCY_US: Mapping[str, tuple[int, int]] = {
    "tickets": (2_000, 12_000),
    "directory": (1_000, 6_000),
    "email": (15_000, 60_000),
    "payments": (20_000, 90_000),
    "ledger": (1_000, 8_000),
    "llm": (150_000, 900_000),
}


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Uniform per-service latency ranges (microseconds, integer arithmetic only)."""

    ranges_us: Mapping[str, tuple[int, int]] = field(default_factory=lambda: DEFAULT_LATENCY_US)
    spike_factor: int = 12
    timeout_us: int = 1_500_000

    def range_for(self, service: str) -> tuple[int, int]:
        return self.ranges_us.get(service, (1_000, 5_000))


class Network:
    """RPC layer shared by every simulated client (agent, human approver, auditor)."""

    def __init__(
        self,
        loop: SimEventLoop,
        faults: FaultPlan,
        rng: SimRandom,
        latency: LatencyModel | None = None,
    ) -> None:
        self._loop = loop
        self._faults = faults
        self._rng = rng
        self._latency = latency or LatencyModel()
        self._streams: dict[tuple[str, str], SimRandom] = {}
        self.calls = 0

    @property
    def timeout_us(self) -> int:
        return self._latency.timeout_us

    def latency_us(self, service: str, scope: str) -> int:
        stream = self._streams.get((service, scope))
        if stream is None:
            stream = self._rng.derive("latency", service, scope)
            self._streams[(service, scope)] = stream
        low, high = self._latency.range_for(service)
        delay = stream.randint(low, high)
        if self._faults.fires(FaultKind.LATENCY_SPIKE, service, scope):
            delay *= self._latency.spike_factor
        return delay

    async def call(
        self,
        service: str,
        op: str,
        scope: str,
        apply: Callable[[], _R],
        *,
        available: Callable[[], bool] | None = None,
    ) -> _R:
        name = f"{service}.{op}"
        trace = self._loop.trace
        start = self._loop.now_us
        self.calls += 1
        await self._loop.sleep_us(self.latency_us(service, scope))
        if available is not None and not available():
            await self._loop.sleep_us(
                max(0, self._latency.timeout_us - (self._loop.now_us - start))
            )
            trace.record(self._loop.now_us, "rpc", name, scope, "down", start)
            raise AmbiguousError(f"{name} timed out (service unavailable)")
        if self._faults.fires(FaultKind.ERROR_5XX, name, scope):
            trace.record(self._loop.now_us, "rpc", name, scope, "5xx", start)
            raise TransientError(f"{name} returned 503 before applying the request")
        try:
            result = apply()
        except RejectedError:
            await self._respond(service, name, scope, start)
            trace.record(self._loop.now_us, "rpc", name, scope, "rejected", start)
            raise
        if available is not None and not available():
            # The service crashed right after committing: the response is lost.
            await self._loop.sleep_us(self._latency.timeout_us)
            trace.record(self._loop.now_us, "rpc", name, scope, "lost", start)
            raise AmbiguousError(f"{name} timed out (service crashed after applying)")
        await self._respond(service, name, scope, start)
        trace.record(self._loop.now_us, "rpc", name, scope, "ok", start)
        return result

    async def _respond(self, service: str, name: str, scope: str, start: int) -> None:
        """The response leg: either a timeout after the commit or the response latency."""
        if self._faults.fires(FaultKind.TIMEOUT_AFTER_COMMIT, name, scope):
            await self._loop.sleep_us(self._latency.timeout_us)
            self._loop.trace.record(self._loop.now_us, "rpc", name, scope, "timeout", start)
            raise AmbiguousError(f"{name} timed out after the request was applied")
        await self._loop.sleep_us(self.latency_us(service, scope))

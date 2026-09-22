"""Simulated humans around the approval ledger: a reviewer and an auditor.

The reviewer is notified of every approval request, thinks for a while (sometimes long enough for
the agent to give up waiting), and approves or denies. It is deliberately imperfect: it approves
most compliant requests but also a large share of non-compliant ones (approval fatigue), because
a human approver must not be the only control between an injected instruction and an effect.

The auditor polls the status of random known approvals, as a dashboard would. Its reads are part
of the client-observed history, which is how a rolled-back consume becomes visible to the
linearizability checker even when the agent itself never looks again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import AmbiguousError, TransientError
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.ledger import History, LedgerClient, LedgerRecord

__all__ = ["Auditor", "HumanReviewer"]

_FAST_US = (50_000, 400_000)
_SLOW_US = ((3, (400_000, 2_000_000)), (1, (3_000_000, 8_000_000)))


class HumanReviewer:
    """Decides every approval request after a think time; see the module docstring."""

    def __init__(
        self,
        loop: SimEventLoop,
        client: LedgerClient,
        *,
        history: History,
        rng: SimRandom,
        faults: FaultPlan,
        compliant: Callable[[dict[str, JsonValue]], bool],
        approve_compliant_percent: int = 90,
        approve_noncompliant_percent: int = 45,
        max_attempts: int = 6,
    ) -> None:
        self._loop = loop
        self._client = client
        self._history = history
        self._rng = rng
        self._faults = faults
        self._compliant = compliant
        self._approve_ok = approve_compliant_percent
        self._approve_bad = approve_noncompliant_percent
        self._max_attempts = max_attempts
        self.tasks: list[asyncio.Task[None]] = []
        self.decisions = 0

    def on_request(self, record: LedgerRecord) -> None:
        self._history.observe(
            "notifier", "notify", (record.id, record.token, record.action_hash), self._loop.now_us
        )
        self._history.learn_token(record.token, record.id)
        self.tasks.append(self._loop.create_task(self._review(record)))

    async def _review(self, record: LedgerRecord) -> None:
        rng = self._rng.derive("review", record.token)
        low, high = _FAST_US
        scope = record.token.split("/", 1)[0].split("~", 1)[0]
        if self._faults.fires(FaultKind.SLOW_REVIEW, "reviewer", scope):
            low, high = _SLOW_US[rng.weighted_index([weight for weight, _ in _SLOW_US])][1]
        await self._loop.sleep_us(rng.randint(low, high))
        percent = self._approve_ok if self._compliant(record.summary) else self._approve_bad
        approve = rng.below(100) < percent
        for attempt in range(self._max_attempts):
            try:
                result, _ = await self._client.decide(record.id, approve)
            except (TransientError, AmbiguousError):
                await self._loop.sleep_us(100_000 * (attempt + 1))
                continue
            if result in ("ok", "not_pending"):
                self.decisions += 1
                return
            return


class Auditor:
    """Reads the status of a random known approval on some ticks, until stopped.

    Whether a tick reads is an ``audit_read`` buggify decision, so the shrinker can drop reads
    that do not matter to a failure.
    """

    def __init__(
        self,
        loop: SimEventLoop,
        client: LedgerClient,
        faults: FaultPlan,
        *,
        tick_us: int = 250_000,
    ) -> None:
        self._loop = loop
        self._client = client
        self._faults = faults
        self._tick = tick_us
        self._known: list[str] = []
        self.reads = 0

    def on_request(self, record: LedgerRecord) -> None:
        self._known.append(record.id)

    async def run(self, stopped: Callable[[], bool]) -> None:
        while not stopped():
            await self._loop.sleep_us(self._tick)
            if not self._known or not self._faults.fires(FaultKind.AUDIT_READ, "auditor", "all"):
                continue
            pick = self._faults.draw("auditor", str(self._loop.now_us))
            approval_id = self._known[pick.below(len(self._known))]
            try:
                await self._client.status(approval_id)
                self.reads += 1
            except (TransientError, AmbiguousError):
                continue

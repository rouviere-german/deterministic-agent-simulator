"""At-least-once inbound queue of ticket events.

Every ticket is delivered once at its arrival time. A ``duplicate_delivery`` fault delivers it a
second time later on. Deliveries handed to an agent incarnation that crashes before acknowledging
them are delivered again to the next incarnation. The queue is *drained* once no delivery is
queued, in flight or scheduled; workers waiting on an empty drained queue receive ``None``.
"""

from __future__ import annotations

import asyncio
from collections import deque

from deterministic_agent_simulator.agent.ports import Delivery
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom

__all__ = ["InboundQueue"]


class InboundQueue:
    def __init__(
        self,
        loop: SimEventLoop,
        faults: FaultPlan,
        rng: SimRandom,
        *,
        duplicate_delay_us: tuple[int, int] = (20_000, 1_500_000),
    ) -> None:
        self._loop = loop
        self._faults = faults
        self._rng = rng
        self._duplicate_delay = duplicate_delay_us
        self._queue: deque[Delivery] = deque()
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._inflight: dict[int, Delivery] = {}
        self._scheduled = 0
        self._next_id = 0
        self._attempts: dict[str, int] = {}
        self.delivered = 0
        self.duplicates = 0
        self.redeliveries = 0
        self.acked: set[str] = set()

    def schedule(self, arrivals: tuple[tuple[int, str], ...]) -> None:
        for when, ticket_id in arrivals:
            self._scheduled += 1
            self._loop.call_at_us(when, self._arrive, ticket_id, True)

    @property
    def drained(self) -> bool:
        return self._scheduled == 0 and not self._queue and not self._inflight

    def _arrive(self, ticket_id: str, original: bool) -> None:
        self._scheduled -= 1
        if original and self._faults.fires(FaultKind.DUPLICATE_DELIVERY, "inbound", ticket_id):
            delay = self._rng.derive("duplicate", ticket_id).randint(*self._duplicate_delay)
            self._scheduled += 1
            self.duplicates += 1
            self._loop.call_at_us(self._loop.now_us + delay, self._arrive, ticket_id, False)
        self._enqueue(ticket_id)

    def _new_delivery(self, ticket_id: str) -> Delivery:
        self._next_id += 1
        attempt = self._attempts.get(ticket_id, 0) + 1
        self._attempts[ticket_id] = attempt
        return Delivery(delivery_id=self._next_id, ticket_id=ticket_id, attempt=attempt)

    def _enqueue(self, ticket_id: str) -> None:
        delivery = self._new_delivery(ticket_id)
        self._queue.append(delivery)
        self._loop.trace.record(self._loop.now_us, "inbound", ticket_id, delivery.attempt)
        self._wake(1)

    def _wake(self, count: int) -> None:
        while count > 0 and self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                count -= 1

    def _wake_all_if_drained(self) -> None:
        if self.drained:
            self._wake(len(self._waiters))

    async def get(self) -> Delivery | None:
        while True:
            if self._queue:
                delivery = self._queue.popleft()
                self._inflight[delivery.delivery_id] = delivery
                self.delivered += 1
                return delivery
            if self.drained:
                return None
            waiter: asyncio.Future[None] = self._loop.create_future()
            self._waiters.append(waiter)
            try:
                await waiter
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

    def ack(self, delivery: Delivery) -> None:
        if self._inflight.pop(delivery.delivery_id, None) is not None:
            self.acked.add(delivery.ticket_id)
        self._wake_all_if_drained()

    def requeue_inflight(self) -> None:
        """The consuming process died: every unacknowledged delivery is delivered again."""
        for delivery_id in sorted(self._inflight):
            self.redeliveries += 1
            self._queue.append(self._new_delivery(self._inflight[delivery_id].ticket_id))
        self._inflight.clear()

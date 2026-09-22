"""Simulated implementations of the agent's ports, with crash points around every call.

Each adapter checks a crash point before and after the operation it wraps (``<port>.<op>.pre``
and ``.post``). An ``agent_crash`` fault there kills the process: the adapter raises
:class:`~deterministic_agent_simulator.agent.ports.AgentCrash` and every later call from that
incarnation fails the same way, so a dead process can neither send requests nor write its disk.
A request already on the wire when the process died is still delivered: the services, not the
agent, decide whether it committed.

Fault points are scoped by ticket, so that removing other tickets while shrinking does not move
them. Operations whose arguments do not name a ticket (a directory lookup, a budget counter, an
audit entry) are attributed to the ticket the calling worker is handling: the inbox adapter sets
a context variable when it hands a delivery to a worker, and asyncio gives every task its own
copy of the context, so the attribution needs no cooperation from the agent. Random draws that
are not faults (disk latency, retry jitter) use the same attribution: each ticket has its own
keyed stream, so deleting one ticket does not shift the timings of the others.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import (
    AgentCrash,
    Delivery,
    Observation,
    PartialWriteError,
    Proposal,
    TicketView,
    TransientError,
)
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.inbound import InboundQueue
from deterministic_agent_simulator.world.ledger import LedgerClient
from deterministic_agent_simulator.world.net import Network
from deterministic_agent_simulator.world.services import (
    Directory,
    EmailOutbox,
    Payments,
    TicketService,
)
from deterministic_agent_simulator.world.simllm import SimLLM
from deterministic_agent_simulator.world.store import AgentStore, AuditChain

__all__ = [
    "AgentProcess",
    "SimAudit",
    "SimClock",
    "SimDirectory",
    "SimEmail",
    "SimInbox",
    "SimLedger",
    "SimModel",
    "SimPayments",
    "SimRandomPort",
    "SimStore",
    "SimTickets",
    "current_scope",
    "ticket_scope",
]


_CURRENT_TICKET: ContextVar[str] = ContextVar("das_current_ticket", default="agent")


def ticket_scope(plan_or_ticket: str) -> str:
    """``T-3`` for ``T-3`` and for a duplicate plan id such as ``T-3~7``."""
    return plan_or_ticket.split("~", 1)[0]


def current_scope() -> str:
    """The ticket the calling worker task is handling (``agent`` before its first delivery)."""
    return _CURRENT_TICKET.get()


def _key_scope(key: str) -> str:
    parts = key.split("/")
    if len(parts) >= 2 and parts[0] in ("plan", "ticket"):
        return ticket_scope(parts[1])
    return current_scope()


class AgentProcess:
    """Liveness of the agent process as the world sees it."""

    def __init__(self, loop: SimEventLoop, faults: FaultPlan, *, max_crashes: int) -> None:
        self._loop = loop
        self._faults = faults
        self._max_crashes = max_crashes
        self.incarnation = 0
        self.dead = True
        self.crashes = 0

    def revive(self) -> int:
        self.incarnation += 1
        self.dead = False
        return self.incarnation

    def check(self, point: str, scope: str) -> None:
        if self.dead:
            raise AgentCrash(f"incarnation {self.incarnation} is dead")
        if self.crashes < self._max_crashes and self._faults.fires(
            FaultKind.AGENT_CRASH, point, scope
        ):
            self.dead = True
            self.crashes += 1
            self._loop.trace.record(self._loop.now_us, "agent.crash", point, scope)
            raise AgentCrash(f"crash at {point}")

    def ensure_alive(self) -> None:
        if self.dead:
            raise AgentCrash(f"incarnation {self.incarnation} is dead")


class SimClock:
    def __init__(self, loop: SimEventLoop, skew_us: int) -> None:
        self._loop = loop
        self.skew_us = skew_us

    def now_us(self) -> int:
        return self._loop.now_us + self.skew_us

    async def sleep_us(self, delay_us: int) -> None:
        await self._loop.sleep_us(delay_us)


class _ScopedStreams:
    """One keyed random stream per ticket scope (created on first use)."""

    def __init__(self, rng: SimRandom) -> None:
        self._rng = rng
        self._streams: dict[str, SimRandom] = {}

    def stream(self, scope: str) -> SimRandom:
        stream = self._streams.get(scope)
        if stream is None:
            stream = self._rng.derive(scope)
            self._streams[scope] = stream
        return stream


class SimRandomPort:
    """Retry jitter for the agent, drawn from the calling worker's ticket stream."""

    def __init__(self, rng: SimRandom) -> None:
        self._streams = _ScopedStreams(rng)

    def below(self, n: int) -> int:
        return self._streams.stream(current_scope()).below(n)


class SimModel:
    """LLM port backed by :class:`SimLLM` (latency, 5xx faults, crash points)."""

    def __init__(
        self,
        loop: SimEventLoop,
        network: Network,
        faults: FaultPlan,
        process: AgentProcess,
        model: SimLLM,
        *,
        on_attempt: Callable[[], None],
        on_decision: Callable[[str], None],
    ) -> None:
        self._loop = loop
        self._net = network
        self._faults = faults
        self._process = process
        self._model = model
        self._on_attempt = on_attempt
        self._on_decision = on_decision

    async def decide(self, observation: Observation) -> Proposal:
        scope = ticket_scope(observation.plan_id)
        self._process.check("llm.decide.pre", scope)
        self._on_attempt()
        await self._loop.sleep_us(self._net.latency_us("llm", scope))
        if self._faults.fires(FaultKind.ERROR_5XX, "llm.decide", scope):
            self._loop.trace.record(self._loop.now_us, "llm", observation.plan_id, "5xx")
            raise TransientError("model endpoint returned 529 (overloaded)")
        proposal = self._model.decide(observation)
        self._loop.trace.record(
            self._loop.now_us, "llm", observation.plan_id, observation.purpose, proposal.name
        )
        if observation.purpose == "next":
            self._on_decision(observation.plan_id)
        self._process.check("llm.decide.post", scope)
        return proposal


class _Remote:
    def __init__(self, network: Network, process: AgentProcess) -> None:
        self._net = network
        self._process = process


class SimTickets(_Remote):
    def __init__(self, network: Network, process: AgentProcess, service: TicketService) -> None:
        super().__init__(network, process)
        self._service = service

    async def get(self, ticket_id: str) -> TicketView | None:
        self._process.check("tickets.get.pre", ticket_id)
        view = await self._net.call(
            "tickets", "get", ticket_id, lambda: self._service.apply_get(ticket_id)
        )
        self._process.check("tickets.get.post", ticket_id)
        return view

    async def close(self, ticket_id: str) -> None:
        self._process.check("tickets.close.pre", ticket_id)
        await self._net.call(
            "tickets", "close", ticket_id, lambda: self._service.apply_close(ticket_id)
        )
        self._process.check("tickets.close.post", ticket_id)


class SimDirectory(_Remote):
    def __init__(self, network: Network, process: AgentProcess, directory: Directory) -> None:
        super().__init__(network, process)
        self._directory = directory

    async def email_of(self, customer_id: str) -> str | None:
        scope = current_scope()
        self._process.check("directory.lookup.pre", scope)
        email = await self._net.call(
            "directory", "lookup", scope, lambda: self._directory.apply_email_of(customer_id)
        )
        self._process.check("directory.lookup.post", scope)
        return email


class SimEmail(_Remote):
    def __init__(self, network: Network, process: AgentProcess, outbox: EmailOutbox) -> None:
        super().__init__(network, process)
        self._outbox = outbox

    async def send(
        self,
        *,
        idempotency_key: str,
        to: str,
        subject: str,
        body: str,
        reference: str,
        approval_id: str,
        intent: str,
    ) -> str:
        scope = ticket_scope(reference)
        self._process.check("email.send.pre", scope)
        result = await self._net.call(
            "email",
            "send",
            scope,
            lambda: self._outbox.apply_send(
                idempotency_key=idempotency_key,
                to=to,
                subject=subject,
                body=body,
                reference=reference,
                approval_id=approval_id,
                intent=intent,
            ),
        )
        self._process.check("email.send.post", scope)
        return result


class SimPayments(_Remote):
    def __init__(self, network: Network, process: AgentProcess, payments: Payments) -> None:
        super().__init__(network, process)
        self._payments = payments

    async def refund(
        self,
        *,
        idempotency_key: str,
        customer_id: str,
        amount_cents: int,
        reference: str,
        approval_id: str,
        intent: str,
    ) -> str:
        scope = ticket_scope(reference)
        self._process.check("payments.refund.pre", scope)
        result = await self._net.call(
            "payments",
            "refund",
            scope,
            lambda: self._payments.apply_refund(
                idempotency_key=idempotency_key,
                customer_id=customer_id,
                amount_cents=amount_cents,
                reference=reference,
                approval_id=approval_id,
                intent=intent,
            ),
        )
        self._process.check("payments.refund.post", scope)
        return result


class SimLedger:
    """The agent's ledger client, with crash points."""

    def __init__(self, client: LedgerClient, process: AgentProcess) -> None:
        self._client = client
        self._process = process

    async def request(self, token: str, action_hash: str, summary: dict[str, JsonValue]) -> str:
        scope = ticket_scope(token.split("/", 1)[0])
        self._process.check("ledger.request.pre", scope)
        approval_id = await self._client.request(token, action_hash, summary)
        self._process.check("ledger.request.post", scope)
        return approval_id

    async def status(self, approval_id: str) -> tuple[str, int | None]:
        scope = current_scope()
        self._process.check("ledger.status.pre", scope)
        result = await self._client.status(approval_id)
        self._process.check("ledger.status.post", scope)
        return result

    async def consume(self, approval_id: str, action_hash: str, consumer: str) -> str:
        scope = ticket_scope(consumer.split("/", 1)[0])
        self._process.check("ledger.consume.pre", scope)
        result = await self._client.consume(approval_id, action_hash, consumer)
        self._process.check("ledger.consume.post", scope)
        return result


class SimStore:
    """Local disk port: small latency, crash points, torn batch writes."""

    def __init__(
        self,
        loop: SimEventLoop,
        faults: FaultPlan,
        process: AgentProcess,
        store: AgentStore,
        rng: SimRandom,
    ) -> None:
        self._loop = loop
        self._faults = faults
        self._process = process
        self._store = store
        self._latency = _ScopedStreams(rng)

    async def _io(self, point: str, scope: str) -> None:
        self._process.check(f"store.{point}.pre", scope)
        await self._loop.sleep_us(50 + self._latency.stream(scope).below(300))
        self._process.ensure_alive()  # a dead process cannot complete a disk write

    async def get(self, key: str) -> JsonValue | None:
        await self._io("get", _key_scope(key))
        return self._store.get(key)

    async def scan(self, prefix: str) -> list[tuple[str, JsonValue]]:
        await self._io("scan", _key_scope(prefix))
        return self._store.scan(prefix)

    async def put(self, key: str, value: JsonValue) -> None:
        await self._io("put", _key_scope(key))
        self._store.put(key, value)

    async def flush(self) -> None:
        await self._io("flush", current_scope())
        self._store.flush()

    async def put_durable(self, key: str, value: JsonValue) -> None:
        scope = _key_scope(key)
        await self._io("put_durable", scope)
        self._store.put_durable(key, value)
        self._process.check("store.put_durable.post", scope)

    async def put_many_durable(self, items: list[tuple[str, JsonValue]]) -> None:
        scope = _key_scope(items[0][0]) if items else current_scope()
        await self._io("put_many", scope)
        if len(items) > 1 and self._faults.fires(FaultKind.PARTIAL_WRITE, "store.batch", scope):
            count = self._faults.draw("store.batch", f"{scope}:{self._loop.now_us}").below(
                len(items)
            )
            self._store.put_prefix_durable(items, count)
            self._loop.trace.record(self._loop.now_us, "store.torn", scope, count, len(items))
            raise PartialWriteError(f"batch torn after {count} of {len(items)} items")
        self._store.put_prefix_durable(items, len(items))
        self._process.check("store.put_many.post", scope)


class SimAudit:
    def __init__(
        self,
        loop: SimEventLoop,
        process: AgentProcess,
        chain: AuditChain,
        actor: str,
        on_append: Callable[[AuditChain], None],
    ) -> None:
        self._loop = loop
        self._process = process
        self._chain = chain
        self._actor = actor
        self._on_append = on_append

    async def append(self, action: str, data: dict[str, JsonValue]) -> None:
        self._process.check("audit.append.pre", current_scope())
        await self._loop.sleep_us(30)
        self._process.ensure_alive()
        self._chain.append(self._loop.now_us, self._actor, action, data)
        self._on_append(self._chain)


class SimInbox:
    def __init__(self, process: AgentProcess, queue: InboundQueue) -> None:
        self._process = process
        self._queue = queue

    async def get(self) -> Delivery | None:
        self._process.ensure_alive()
        delivery = await self._queue.get()
        if delivery is not None:
            _CURRENT_TICKET.set(delivery.ticket_id)  # attribute this worker's next operations
            self._process.check("inbox.get.post", delivery.ticket_id)
        return delivery

    async def ack(self, delivery: Delivery) -> None:
        self._process.check("inbox.ack.pre", delivery.ticket_id)
        self._queue.ack(delivery)

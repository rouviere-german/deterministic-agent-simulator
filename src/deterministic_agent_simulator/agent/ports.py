"""The port contract: every effect the reference agent has on the world goes through these.

The agent imports nothing from the simulator. It receives an :class:`AgentPorts` bundle and does
all I/O through it, so the same agent code runs against simulated services (in
:mod:`deterministic_agent_simulator.world`) or real ones. docs/WRITING_SIM_TESTS.md describes the
contract for adapting another system.

Failure semantics a port implementation must follow:

- :class:`TransientError`: the call definitely had no effect (for example an HTTP 5xx returned
  before the request was applied). Retrying is safe.
- :class:`AmbiguousError`: the call may or may not have had an effect (a timeout). Retrying is
  safe only with the same idempotency key.
- :class:`RejectedError`: the service refused the request (for example an idempotency key reused
  with different arguments). Retrying will not help.
- :class:`PartialWriteError`: a multi-key write applied only a prefix of its items.
- :class:`AgentCrash`: the process died. It derives from :class:`BaseException` so that
  ``except Exception`` in agent code cannot swallow it; agent code must never catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import JsonValue

__all__ = [
    "AgentCrash",
    "AgentPorts",
    "AmbiguousError",
    "AuditPort",
    "ClockPort",
    "Delivery",
    "DirectoryPort",
    "EmailPort",
    "InboxPort",
    "LLMPort",
    "LedgerPort",
    "Observation",
    "PartialWriteError",
    "PaymentsPort",
    "PortError",
    "Proposal",
    "RandomPort",
    "RejectedError",
    "StorePort",
    "TicketView",
    "TicketsPort",
    "TransientError",
]


class PortError(Exception):
    """Base class of the failures a port reports."""


class TransientError(PortError):
    """The call failed without taking effect; it is safe to retry."""


class AmbiguousError(PortError):
    """The call may have taken effect (timeout); retry only with the same idempotency key."""


class RejectedError(PortError):
    """The service refused the request; retrying the same request will not succeed."""


class PartialWriteError(PortError):
    """A multi-key write applied only a prefix of its items before failing."""


class AgentCrash(BaseException):
    """The agent process died. Never caught by agent code."""


@dataclass(frozen=True, slots=True)
class TicketView:
    """A ticket as returned by the ticket service. ``body`` is untrusted free text."""

    id: str
    customer_id: str
    kind: Literal["refund", "question"]
    amount_cents: int
    subject: str
    body: str


@dataclass(frozen=True, slots=True)
class Delivery:
    """One (possibly duplicated) delivery of a ticket event from the inbound queue.

    ``attempt`` counts deliveries of the same ticket (1 for the first one).
    """

    delivery_id: int
    ticket_id: str
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class Proposal:
    """A model-proposed action: a tool name and raw, unvalidated JSON arguments.

    ``name == "finish"`` ends the ticket. Anything else must pass the action registry.
    """

    name: str
    args: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {"name": self.name, "args": dict(self.args)}

    @classmethod
    def from_json(cls, data: JsonValue) -> Proposal:
        if not isinstance(data, dict):
            raise TypeError("a proposal is a JSON object")
        name, args = data.get("name"), data.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            raise TypeError("a proposal needs a string name and an object of arguments")
        return cls(name=name, args=args)


@dataclass(frozen=True, slots=True)
class Observation:
    """What the agent shows the model before each decision.

    ``customer_email`` comes from the trusted customer directory; ``ticket.body`` is untrusted.
    ``purpose`` is ``"next"`` for the next step and ``"retry"`` when the agent asks how to retry
    a failed call (only the planted approval-bypass bug does that).
    """

    plan_id: str
    ticket_id: str
    step: int
    purpose: Literal["next", "retry"]
    ticket: TicketView | None
    customer_email: str | None
    done: tuple[str, ...]
    last_action: Proposal | None
    last_outcome: str | None


class ClockPort(Protocol):
    def now_us(self) -> int:
        """The agent's wall clock in microseconds (may be skewed against other nodes)."""
        ...

    async def sleep_us(self, delay_us: int) -> None: ...


class RandomPort(Protocol):
    def below(self, n: int) -> int: ...


class LLMPort(Protocol):
    async def decide(self, observation: Observation) -> Proposal: ...


class TicketsPort(Protocol):
    async def get(self, ticket_id: str) -> TicketView | None: ...

    async def close(self, ticket_id: str) -> None: ...


class DirectoryPort(Protocol):
    async def email_of(self, customer_id: str) -> str | None: ...


class EmailPort(Protocol):
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
    ) -> str: ...


class PaymentsPort(Protocol):
    async def refund(
        self,
        *,
        idempotency_key: str,
        customer_id: str,
        amount_cents: int,
        reference: str,
        approval_id: str,
        intent: str,
    ) -> str: ...


class LedgerPort(Protocol):
    async def request(self, token: str, action_hash: str, summary: dict[str, JsonValue]) -> str: ...

    async def status(self, approval_id: str) -> tuple[str, int | None]: ...

    async def consume(self, approval_id: str, action_hash: str, consumer: str) -> str: ...


class StorePort(Protocol):
    """Local key-value store. ``put`` is buffered and lost on a crash unless flushed."""

    async def get(self, key: str) -> JsonValue | None: ...

    async def scan(self, prefix: str) -> list[tuple[str, JsonValue]]: ...

    async def put(self, key: str, value: JsonValue) -> None: ...

    async def flush(self) -> None: ...

    async def put_durable(self, key: str, value: JsonValue) -> None: ...

    async def put_many_durable(self, items: list[tuple[str, JsonValue]]) -> None: ...


class AuditPort(Protocol):
    async def append(self, action: str, data: dict[str, JsonValue]) -> None: ...


class InboxPort(Protocol):
    async def get(self) -> Delivery | None:
        """The next delivery, or None once there is no more work."""
        ...

    async def ack(self, delivery: Delivery) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentPorts:
    clock: ClockPort
    random: RandomPort
    llm: LLMPort
    tickets: TicketsPort
    directory: DirectoryPort
    email: EmailPort
    payments: PaymentsPort
    ledger: LedgerPort
    store: StorePort
    audit: AuditPort
    inbox: InboxPort

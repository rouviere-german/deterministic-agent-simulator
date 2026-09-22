"""Simulated external services: tickets, customer directory, e-mail outbox and payments.

The e-mail outbox and the payments stub are the *external effects* the oracles watch. Both
deduplicate by idempotency key the way real providers do: a repeated key with identical
arguments returns the original result without a new effect, and a repeated key with different
arguments is rejected. Every new effect is reported to the effect sink (the oracles) at the
instant it commits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import RejectedError, TicketView
from deterministic_agent_simulator.agentic.tools import tool_call_hash

__all__ = [
    "Directory",
    "Effect",
    "EffectSink",
    "EmailOutbox",
    "Payments",
    "TicketService",
]


@dataclass(frozen=True, slots=True)
class Effect:
    """One committed external effect, as seen by the simulator."""

    seq: int
    tool: str
    args: dict[str, JsonValue]
    reference: str
    idempotency_key: str
    approval_id: str
    intent: str
    t_us: int

    @property
    def action_hash(self) -> str:
        return tool_call_hash(self.tool, self.args)

    def describe(self) -> str:
        return f"{self.tool} #{self.seq} for {self.reference} at t={self.t_us}us"


EffectSink = Callable[[Effect], None]


class TicketService:
    def __init__(self, tickets: Mapping[str, TicketView]) -> None:
        self._tickets = dict(tickets)
        self.closed: set[str] = set()

    def apply_get(self, ticket_id: str) -> TicketView | None:
        return self._tickets.get(ticket_id)

    def apply_close(self, ticket_id: str) -> None:
        if ticket_id in self._tickets:
            self.closed.add(ticket_id)


class Directory:
    """Trusted customer directory (customer id -> e-mail address)."""

    def __init__(self, emails: Mapping[str, str]) -> None:
        self._emails = dict(emails)

    def apply_email_of(self, customer_id: str) -> str | None:
        return self._emails.get(customer_id)


class _Deduplicating:
    def __init__(
        self, tool: str, clock: Callable[[], int], sink: EffectSink, counter: list[int]
    ) -> None:
        self._tool = tool
        self._clock = clock
        self._sink = sink
        self._counter = counter
        self._by_key: dict[str, tuple[dict[str, JsonValue], str, str]] = {}
        self.effects: list[Effect] = []
        self.deduplicated = 0

    def _commit(
        self,
        *,
        idempotency_key: str,
        args: dict[str, JsonValue],
        reference: str,
        approval_id: str,
        intent: str,
        prefix: str,
    ) -> str:
        existing = self._by_key.get(idempotency_key)
        if existing is not None:
            stored_args, stored_reference, result = existing
            if stored_args == args and stored_reference == reference:
                self.deduplicated += 1
                return result
            raise RejectedError(f"idempotency key reused with different arguments ({self._tool})")
        self._counter[0] += 1
        result = f"{prefix}-{self._counter[0]}"
        self._by_key[idempotency_key] = (dict(args), reference, result)
        effect = Effect(
            seq=self._counter[0],
            tool=self._tool,
            args=dict(args),
            reference=reference,
            idempotency_key=idempotency_key,
            approval_id=approval_id,
            intent=intent,
            t_us=self._clock(),
        )
        self.effects.append(effect)
        self._sink(effect)
        return result


class EmailOutbox(_Deduplicating):
    def __init__(self, clock: Callable[[], int], sink: EffectSink, counter: list[int]) -> None:
        super().__init__("send_email", clock, sink, counter)

    def apply_send(
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
        args: dict[str, JsonValue] = {"to": to, "subject": subject, "body": body}
        return self._commit(
            idempotency_key=idempotency_key,
            args=args,
            reference=reference,
            approval_id=approval_id,
            intent=intent,
            prefix="msg",
        )


class Payments(_Deduplicating):
    def __init__(self, clock: Callable[[], int], sink: EffectSink, counter: list[int]) -> None:
        super().__init__("issue_refund", clock, sink, counter)

    def apply_refund(
        self,
        *,
        idempotency_key: str,
        customer_id: str,
        amount_cents: int,
        reference: str,
        approval_id: str,
        intent: str,
    ) -> str:
        args: dict[str, JsonValue] = {"customer_id": customer_id, "amount_cents": amount_cents}
        return self._commit(
            idempotency_key=idempotency_key,
            args=args,
            reference=reference,
            approval_id=approval_id,
            intent=intent,
            prefix="rf",
        )

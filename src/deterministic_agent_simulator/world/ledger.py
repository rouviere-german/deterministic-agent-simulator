"""The simulated ApprovalLedger service and the client-side history its clients record.

The ledger implements request / decide / status / consume with consume-once semantics and a TTL
on granted approvals, exactly as specified by
:class:`~deterministic_agent_simulator.check.ledger_model.ApprovalModel`. It can crash and
restart; what survives is its durable state.

The planted ``non_durable_consume_flag`` bug makes ``consume`` acknowledge a state change that
lives only in memory until the next periodic flush. A crash in between rolls the approval back to
"approved": a client that saw ``consume -> ok`` can later see ``status -> approved``, which the
linearizability checker rejects.

Besides its own state the ledger keeps a *truth log* of every successful consume. That log is the
simulator's omniscient view, never lost in a crash, and is what the effect oracles consult.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field, replace

from pydantic import JsonValue

from deterministic_agent_simulator.check.linearizability import Operation
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.world.net import Network

__all__ = [
    "ApprovalLedger",
    "ConsumeFact",
    "History",
    "LedgerClient",
    "LedgerRecord",
]

RequestListener = Callable[["LedgerRecord"], None]


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    id: str
    token: str
    action_hash: str
    summary: dict[str, JsonValue]
    requester: str
    requested_at: int
    status: str = "pending"
    decided_by: str | None = None
    expires_at: int | None = None
    consumer: str | None = None


@dataclass(frozen=True, slots=True)
class ConsumeFact:
    """Ground truth: ``consumer`` consumed ``approval_id`` at ``t_us`` (a new consume)."""

    approval_id: str
    consumer: str
    action_hash: str
    t_us: int
    expires_at: int


class ApprovalLedger:
    """Ledger state machine (``apply_*`` run atomically at an RPC's commit point)."""

    def __init__(
        self,
        loop: SimEventLoop,
        *,
        ttl_us: int,
        durable_consume: bool = True,
        on_consume: Callable[[ConsumeFact], None] | None = None,
    ) -> None:
        self._loop = loop
        self.ttl_us = ttl_us
        self.durable_consume = durable_consume
        self._durable: dict[str, LedgerRecord] = {}
        self._volatile: dict[str, LedgerRecord] = {}
        self._by_token: dict[str, str] = {}
        self._next = 0
        self._listeners: list[RequestListener] = []
        self._on_consume = on_consume
        self.up = True
        self.truth: list[ConsumeFact] = []
        self.crashes = 0
        self.lost_on_crash = 0

    def add_listener(self, listener: RequestListener) -> None:
        self._listeners.append(listener)

    def is_up(self) -> bool:
        return self.up

    def record(self, approval_id: str) -> LedgerRecord | None:
        """Current view of a record (volatile overlay first)."""
        return self._volatile.get(approval_id) or self._durable.get(approval_id)

    def scope_of(self, approval_id: str) -> str:
        """Fault scope of an approval: the plan that requested it (stable under shrinking)."""
        record = self.record(approval_id)
        return record.token.split("/", 1)[0] if record is not None else approval_id

    def records(self) -> list[LedgerRecord]:
        return [self.record(key) or self._durable[key] for key in self._durable]

    def _now(self) -> int:
        return self._loop.now_us

    # ------------------------------------------------------------ operations
    def apply_request(
        self, token: str, action_hash: str, summary: dict[str, JsonValue], requester: str
    ) -> str:
        existing = self._by_token.get(token)
        if existing is not None:
            return existing
        self._next += 1
        approval_id = f"apr-{self._next}"
        record = LedgerRecord(
            id=approval_id,
            token=token,
            action_hash=action_hash,
            summary=summary,
            requester=requester,
            requested_at=self._now(),
        )
        self._durable[approval_id] = record
        self._by_token[token] = approval_id
        for listener in self._listeners:
            listener(record)
        return approval_id

    def apply_decide(
        self, approval_id: str, approve: bool, approver: str
    ) -> tuple[str, int | None]:
        record = self.record(approval_id)
        if record is None:
            return ("unknown", None)
        if approver == record.requester:
            return ("forbidden", None)
        if record.status != "pending":
            return ("not_pending", None)
        if approve:
            expires = self._now() + self.ttl_us
            decided = replace(record, status="approved", decided_by=approver, expires_at=expires)
            self._durable[approval_id] = decided
            return ("ok", expires)
        self._durable[approval_id] = replace(record, status="denied", decided_by=approver)
        return ("ok", None)

    def apply_status(self, approval_id: str) -> tuple[str, int | None]:
        record = self.record(approval_id)
        if record is None:
            return ("unknown", None)
        if record.status == "approved":
            expires = record.expires_at if record.expires_at is not None else 0
            if self._now() >= expires:
                return ("expired", None)
            return ("approved", expires)
        return (record.status, None)

    def apply_consume(  # noqa: PLR0911 - one return per specified outcome
        self, approval_id: str, action_hash: str, consumer: str
    ) -> str:
        record = self.record(approval_id)
        if record is None:
            return "unknown"
        if record.status == "consumed":
            if record.consumer == consumer:
                return "ok" if record.action_hash == action_hash else "hash_mismatch"
            return "already_consumed"
        if record.status != "approved":
            return "not_approved"
        expires = record.expires_at if record.expires_at is not None else 0
        if self._now() >= expires:
            return "expired"
        if record.action_hash != action_hash:
            return "hash_mismatch"
        consumed = replace(record, status="consumed", consumer=consumer)
        if self.durable_consume:
            self._durable[approval_id] = consumed
        else:
            # BUG: non_durable_consume_flag - acknowledged before it is durable.
            self._volatile[approval_id] = consumed
        fact = ConsumeFact(approval_id, consumer, action_hash, self._now(), expires)
        self.truth.append(fact)
        if self._on_consume is not None:
            self._on_consume(fact)
        return "ok"

    # ---------------------------------------------------- durability, crashes
    def flush(self) -> None:
        self._durable.update(self._volatile)
        self._volatile.clear()

    def crash(self) -> None:
        self.up = False
        self.crashes += 1
        self.lost_on_crash += len(self._volatile)
        self._volatile.clear()

    def restart(self) -> None:
        self.up = True


@dataclass
class History:
    """Client-observed ledger operations, for the linearizability checker."""

    ops: list[Operation] = field(default_factory=list)
    _open: dict[int, tuple[str, str, tuple[Hashable, ...], int]] = field(default_factory=dict)
    _next: int = 0
    tokens: dict[str, str] = field(default_factory=dict)

    def begin(self, client: str, kind: str, args: tuple[Hashable, ...], now: int) -> int:
        op_id = self._next
        self._next += 1
        self._open[op_id] = (client, kind, args, now)
        return op_id

    def end(self, op_id: int, now: int, output: Hashable | None) -> None:
        client, kind, args, invoked = self._open.pop(op_id)
        response = now if output is not None else None
        self.ops.append(Operation(op_id, client, kind, args, invoked, response, output))

    def observe(self, client: str, kind: str, args: tuple[Hashable, ...], now: int) -> None:
        op_id = self.begin(client, kind, args, now)
        self.end(op_id, now, "ok")

    def close_pending(self) -> None:
        """Operations still open at the end of the run never received a response."""
        for op_id in sorted(self._open):
            self.end(op_id, 0, None)

    def learn_token(self, token: str, approval_id: str) -> None:
        self.tokens[token] = approval_id


class LedgerClient:
    """A ledger client that records every call it makes in the shared :class:`History`."""

    def __init__(
        self,
        name: str,
        principal: str,
        *,
        ledger: ApprovalLedger,
        network: Network,
        history: History,
        loop: SimEventLoop,
        after_commit: Callable[[str, str], None] | None = None,
    ) -> None:
        self.name = name
        self.principal = principal
        self._ledger = ledger
        self._net = network
        self._history = history
        self._loop = loop
        self._after_commit = after_commit

    async def _call[R: Hashable](
        self, kind: str, args: tuple[Hashable, ...], scope: str, apply: Callable[[], R]
    ) -> R:
        """One recorded RPC; a failed or cancelled call stays pending in the history."""
        op_id = self._history.begin(self.name, kind, args, self._loop.now_us)
        output: R | None = None

        def commit() -> R:
            result = apply()
            if self._after_commit is not None:
                self._after_commit(kind, scope)
            return result

        try:
            output = await self._net.call(
                "ledger", kind, scope, commit, available=self._ledger.is_up
            )
            return output
        finally:
            self._history.end(op_id, self._loop.now_us, output)

    async def request(self, token: str, action_hash: str, summary: dict[str, JsonValue]) -> str:
        approval_id = await self._call(
            "request",
            (token, action_hash, self.principal),
            token.split("/", 1)[0],
            lambda: self._ledger.apply_request(token, action_hash, summary, self.principal),
        )
        self._history.learn_token(token, approval_id)
        return approval_id

    async def decide(self, approval_id: str, approve: bool) -> tuple[str, int | None]:
        return await self._call(
            "decide",
            (approval_id, approve, self.principal),
            self._ledger.scope_of(approval_id),
            lambda: self._ledger.apply_decide(approval_id, approve, self.principal),
        )

    async def status(self, approval_id: str) -> tuple[str, int | None]:
        return await self._call(
            "status",
            (approval_id,),
            self._ledger.scope_of(approval_id),
            lambda: self._ledger.apply_status(approval_id),
        )

    async def consume(self, approval_id: str, action_hash: str, consumer: str) -> str:
        return await self._call(
            "consume",
            (approval_id, action_hash, consumer),
            self._ledger.scope_of(approval_id),
            lambda: self._ledger.apply_consume(approval_id, action_hash, consumer),
        )

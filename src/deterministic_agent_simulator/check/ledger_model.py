"""Sequential specification of one ApprovalLedger approval, with time-bounded freshness.

The ledger's operations and outputs, as recorded by its clients:

========  ===================================  ==================================================
kind      args                                 output
========  ===================================  ==================================================
request   ``(token, action_hash, requester)``  approval id (requests are idempotent per token)
notify    ``(approval_id, token, action_hash)`` ``"ok"`` (a listener saw the request committed)
decide    ``(approval_id, approve, approver)`` ``("ok", expires_at | None)``, ``("not_pending",
                                               None)``, ``("forbidden", None)``, ``("unknown",
                                               None)``
status    ``(approval_id,)``                   ``(status, expires_at | None)``; status is one of
                                               pending, approved, expired, denied, consumed,
                                               unknown
consume   ``(approval_id, action_hash,         ok, already_consumed, expired, not_approved,
          consumer)``                          hash_mismatch, unknown
========  ===================================  ==================================================

An approval expires ``ttl_us`` after the decision that granted it; a consumed or denied approval
never expires, and consuming again with the same consumer token is idempotent.

Expiry makes the specification depend on *when* an operation takes effect. Each operation's
linearization point is some time within ``[invoke, response]`` that is not earlier than the
previous point, so the model state carries the latest point ``t`` and an interval
``[expires_lo, expires_hi]`` of expiry times that are still consistent with everything observed.
An operation picks the earliest point its outcome allows (earliest is always optimal: later
operations only need their own points to be no earlier) and narrows the interval. An approval
decided by an operation whose response was lost has an uncertain expiry, which later
observations pin down. Because every operation after the decision happened after it, each later
point ``t`` also caps the expiry at ``t + ttl``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

from deterministic_agent_simulator.check.linearizability import INFINITY, Operation

__all__ = ["ApprovalModel", "ApprovalState", "approval_object_key"]

_READ_ONLY = frozenset({"status", "notify"})


class ApprovalState(NamedTuple):
    created: bool = False
    token: str = ""
    action_hash: str = ""
    requester: str = ""
    status: str = ""
    consumer: str = ""
    expires_lo: int = 0
    expires_hi: int = INFINITY
    t: int = -1


class ApprovalModel:
    """:class:`~deterministic_agent_simulator.check.linearizability.SequentialModel` of one
    approval."""

    def __init__(self, ttl_us: int) -> None:
        if ttl_us <= 0:
            raise ValueError("ttl_us must be positive")
        self.ttl_us = ttl_us

    def initial(self) -> ApprovalState:
        return ApprovalState()

    def _at(self, state: ApprovalState, t: int) -> ApprovalState | None:
        """``state`` with its linearization point moved to ``t`` (None if inconsistent)."""
        if state.status == "approved":
            hi = min(state.expires_hi, t + self.ttl_us)
            if hi < state.expires_lo:
                return None
            return state._replace(t=t, expires_hi=hi)
        return state._replace(t=t)

    def step(self, state: ApprovalState, op: Operation) -> Sequence[ApprovalState]:  # noqa: PLR0911
        t0 = max(op.invoke_us, state.t)
        if t0 > op.returned_at:
            return []
        if op.output is None and op.kind in _READ_ONLY:
            return [state]  # a pending read changes nothing and constrains nothing
        if op.kind == "request":
            return self._request(state, op, t0)
        if op.kind == "notify":
            _, token, action_hash = op.args
            ok = state.created and state.token == token and state.action_hash == action_hash
            return _one(self._at(state, t0)) if ok and op.output == "ok" else []
        if not state.created:
            expected: Hashable = "unknown" if op.kind == "consume" else ("unknown", None)
            return _one(self._at(state, t0)) if op.output in (None, expected) else []
        if op.kind == "decide":
            return self._decide(state, op, t0)
        if op.kind == "status":
            return self._status(state, op, t0)
        if op.kind == "consume":
            return self._consume(state, op, t0)
        return []

    def _request(self, state: ApprovalState, op: Operation, t0: int) -> list[ApprovalState]:
        token, action_hash, requester = op.args
        if not state.created:
            created = state._replace(
                created=True,
                token=str(token),
                action_hash=str(action_hash),
                requester=str(requester),
                status="pending",
            )
            return _one(self._at(created, t0))
        return _one(self._at(state, t0)) if state.token == token else []

    def _decide(  # noqa: PLR0911 - one return per specified outcome
        self, state: ApprovalState, op: Operation, t0: int
    ) -> list[ApprovalState]:
        _, approve, approver = op.args
        out = op.output
        if approver == state.requester:
            return _one(self._at(state, t0)) if out in (None, ("forbidden", None)) else []
        if state.status != "pending":
            return _one(self._at(state, t0)) if out in (None, ("not_pending", None)) else []
        if not approve:
            denied = state._replace(status="denied")
            return _one(self._at(denied, t0)) if out in (None, ("ok", None)) else []
        if out is None:
            # Decided at some unknown time >= t0: the expiry is only bounded below.
            approved = state._replace(
                status="approved", expires_lo=t0 + self.ttl_us, expires_hi=INFINITY, t=t0
            )
            return [approved]
        if not (isinstance(out, tuple) and len(out) == 2 and out[0] == "ok"):
            return []
        expires = out[1]
        if not isinstance(expires, int):
            return []
        decided = expires - self.ttl_us
        if not t0 <= decided <= op.returned_at:
            return []
        return [
            state._replace(status="approved", expires_lo=expires, expires_hi=expires, t=decided)
        ]

    def _status(  # noqa: PLR0911 - one return per specified outcome
        self, state: ApprovalState, op: Operation, t0: int
    ) -> list[ApprovalState]:
        out = op.output
        if state.status != "approved":
            return _one(self._at(state, t0)) if out == (state.status, None) else []
        if not (isinstance(out, tuple) and len(out) == 2):
            return []
        label, expires = out
        if label == "approved" and isinstance(expires, int):
            if not state.expires_lo <= expires <= state.expires_hi:
                return []
            t = max(t0, expires - self.ttl_us)
            if t >= expires or t > op.returned_at:
                return []
            pinned = state._replace(expires_lo=expires, expires_hi=expires)
            return _one(self._at(pinned, t))
        if label == "expired" and expires is None:
            t = max(t0, state.expires_lo)
            if t > op.returned_at:
                return []
            return _one(self._expired_at(state, t))
        return []

    def _expired_at(self, state: ApprovalState, t: int) -> ApprovalState | None:
        hi = min(state.expires_hi, t)
        if hi < state.expires_lo:
            return None
        return state._replace(expires_hi=hi, t=t)

    def _fresh_at(self, state: ApprovalState, t: int) -> ApprovalState | None:
        """``state`` observed unexpired at ``t`` (so the expiry is later than ``t``)."""
        lo = max(state.expires_lo, t + 1)
        hi = min(state.expires_hi, t + self.ttl_us)
        if lo > hi:
            return None
        return state._replace(expires_lo=lo, expires_hi=hi, t=t)

    def _consume(self, state: ApprovalState, op: Operation, t0: int) -> list[ApprovalState]:
        _, action_hash, consumer = op.args
        out = op.output
        if state.status == "consumed":
            if state.consumer == consumer:
                expected = "ok" if state.action_hash == action_hash else "hash_mismatch"
            else:
                expected = "already_consumed"
            return _one(self._at(state, t0)) if out in (None, expected) else []
        if state.status in ("pending", "denied"):
            return _one(self._at(state, t0)) if out in (None, "not_approved") else []
        successors: list[ApprovalState] = []
        if out in (None, "expired"):
            t = max(t0, state.expires_lo)
            if t <= op.returned_at:
                successors.extend(_one(self._expired_at(state, t)))
        fresh_outcome = "ok" if action_hash == state.action_hash else "hash_mismatch"
        if out in (None, fresh_outcome):
            fresh = self._fresh_at(state, t0)
            if fresh is not None:
                if fresh_outcome == "ok":
                    fresh = fresh._replace(status="consumed", consumer=str(consumer))
                successors.append(fresh)
        return successors


def _one(state: ApprovalState | None) -> list[ApprovalState]:
    return [] if state is None else [state]


def approval_object_key(tokens: dict[str, str]) -> Callable[[Operation], str | None]:
    """Partition key for ledger histories: the approval id an operation refers to.

    Completed requests are keyed by the id they returned; pending ones by the id that another
    observation associated with their token (``tokens`` maps token to id). A pending request
    whose approval nobody observed cannot affect any checked object and is dropped. Every other
    operation names its approval id as its first argument.
    """

    def key(op: Operation) -> str | None:
        if op.kind == "request":
            if op.completed:
                return str(op.output)
            return tokens.get(str(op.args[0]))
        return str(op.args[0])

    return key

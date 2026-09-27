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
linearization point lies in ``[invoke, response]`` and is not earlier than the previous point, so
the model state carries an interval ``[expires_lo, expires_hi]`` of expiry times that are still
consistent with everything observed, plus a *base* ``t``. The point of the latest operation is not
a number but ``max(t, expires + offset)``:

- ``offset = -ttl_us`` once an approval was granted, because every point from the granting decision
  on is at or after that decision, whose own point is ``expires - ttl_us``;
- ``offset = 0`` once an operation observed the approval expired, because every point from then on
  is at or after the expiry;
- no offset while the approval is unknown, pending or denied: expiry constrains nothing.

Keeping the point as a function of the expiry instead of a number is what makes the *earliest* base
optimal. With the expiry fixed, every constraint on a point is a plain lower or upper bound, so
taking the earliest feasible point can never rule out a later operation. Tightening the expiry at
an operation's earliest point would not be sound: an operation whose response was lost may have
committed much later, and its expiry is only bounded below, by its own invocation. What does bound
the expiry above is when *later* operations returned, because each of them returned no earlier than
the point it needs: ``expires + offset <= response``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

from deterministic_agent_simulator.check.linearizability import INFINITY, Operation

__all__ = ["ApprovalModel", "ApprovalState", "approval_object_key"]

_READ_ONLY = frozenset({"status", "notify"})
_GRANTED = frozenset({"approved", "consumed"})


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

    def _offset(self, status: str) -> int | None:
        """How the latest point relates to the expiry: ``point >= expires + offset``."""
        if status in _GRANTED:
            return -self.ttl_us
        return 0 if status == "expired" else None

    def _advance(self, state: ApprovalState, op: Operation, t: int) -> ApprovalState | None:
        """``state`` with its point base moved to ``t`` (None if that is inconsistent).

        ``op`` returned at or after its own point, which is what bounds the expiry above.
        """
        offset = self._offset(state.status)
        if offset is None or op.returned_at >= INFINITY:
            return state._replace(t=t)
        hi = min(state.expires_hi, op.returned_at - offset)
        if hi < state.expires_lo:
            return None
        return state._replace(t=t, expires_hi=hi)

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
            return _one(self._advance(state, op, t0)) if ok and op.output == "ok" else []
        if not state.created:
            expected: Hashable = "unknown" if op.kind == "consume" else ("unknown", None)
            return _one(self._advance(state, op, t0)) if op.output in (None, expected) else []
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
            return _one(self._advance(created, op, t0))
        return _one(self._advance(state, op, t0)) if state.token == token else []

    def _decide(  # noqa: PLR0911 - one return per specified outcome
        self, state: ApprovalState, op: Operation, t0: int
    ) -> list[ApprovalState]:
        _, approve, approver = op.args
        out = op.output
        if approver == state.requester:
            return _one(self._advance(state, op, t0)) if out in (None, ("forbidden", None)) else []
        if state.status != "pending":
            expected = (None, ("not_pending", None))
            return _one(self._advance(state, op, t0)) if out in expected else []
        if not approve:
            denied = state._replace(status="denied")
            return _one(self._advance(denied, op, t0)) if out in (None, ("ok", None)) else []
        if out is None:
            # Granted at an unknown point at or after t0: the expiry is bounded below by t0 + ttl
            # and above only by when this operation returned.
            lo = t0 + self.ttl_us
            hi = INFINITY if op.returned_at >= INFINITY else op.returned_at + self.ttl_us
            if hi < lo:
                return []
            return [state._replace(status="approved", expires_lo=lo, expires_hi=hi, t=t0)]
        if not (isinstance(out, tuple) and len(out) == 2 and out[0] == "ok"):
            return []
        expires = out[1]
        if not isinstance(expires, int):
            return []
        decided = expires - self.ttl_us
        if not t0 <= decided <= op.returned_at:
            return []
        return [state._replace(status="approved", expires_lo=expires, expires_hi=expires, t=t0)]

    def _status(self, state: ApprovalState, op: Operation, t0: int) -> list[ApprovalState]:
        out = op.output
        if state.status != "approved":
            return _one(self._advance(state, op, t0)) if out == (state.status, None) else []
        if not (isinstance(out, tuple) and len(out) == 2):
            return []
        label, expires = out
        if label == "approved" and isinstance(expires, int):
            fresh = self._fresh_at(state, op, t0)
            if fresh is None or not fresh.expires_lo <= expires <= fresh.expires_hi:
                return []
            return [fresh._replace(expires_lo=expires, expires_hi=expires)]
        if label == "expired" and expires is None:
            return _one(self._expired_at(state, op, t0))
        return []

    def _expired_at(self, state: ApprovalState, op: Operation, t: int) -> ApprovalState | None:
        """``state`` observed expired: its point, and every later one, is at or after the expiry."""
        hi = min(state.expires_hi, op.returned_at)
        if hi < state.expires_lo:
            return None
        return state._replace(status="expired", expires_hi=hi, t=t)

    def _fresh_at(self, state: ApprovalState, op: Operation, t: int) -> ApprovalState | None:
        """``state`` observed unexpired: the expiry is later than this operation's point."""
        lo = max(state.expires_lo, t + 1)
        hi = state.expires_hi
        if op.returned_at < INFINITY:
            hi = min(hi, op.returned_at + self.ttl_us)
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
            return _one(self._advance(state, op, t0)) if out in (None, expected) else []
        if state.status in ("pending", "denied"):
            return _one(self._advance(state, op, t0)) if out in (None, "not_approved") else []
        if state.status == "expired":
            return _one(self._advance(state, op, t0)) if out in (None, "expired") else []
        successors: list[ApprovalState] = []
        if out in (None, "expired"):
            successors.extend(_one(self._expired_at(state, op, t0)))
        fresh_outcome = "ok" if action_hash == state.action_hash else "hash_mismatch"
        if out in (None, fresh_outcome):
            fresh = self._fresh_at(state, op, t0)
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

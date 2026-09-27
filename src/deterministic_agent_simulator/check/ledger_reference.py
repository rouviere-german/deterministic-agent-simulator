"""An exhaustive reference checker for approval-ledger histories, for differential testing.

:mod:`~deterministic_agent_simulator.check.ledger_model` is an *abstraction* of the ledger: rather
than choosing when an approval was granted, it carries an interval of expiry times that are still
possible. Comparing it with :func:`~...check.linearizability.brute_force_check` therefore validates
the search, not the model, because both use the same ``ApprovalModel``. This module decides the same
question a second way, sharing nothing with that model:

- every linearization point is a variable, and the real
  :class:`~deterministic_agent_simulator.world.ledger.ApprovalLedger` state machine runs on a clock
  that returns those variables instead of numbers;
- the only time-dependent branch in the ledger (``now >= expires``) is explored both ways, and each
  outcome, together with the invocation and response windows and the chosen order, becomes an
  integer difference constraint ``x_v - x_u <= w``;
- an order is admissible when the ledger's outputs match what the clients observed and the
  constraint system has a solution (no negative cycle; Bellman-Ford over a simple temporal network,
  see Dechter, Meiri and Pearl, "Temporal constraint networks", Artificial Intelligence 1991).

Every order of every admissible subset of pending operations is enumerated, so this is exponential
and meant only for the tiny histories of differential tests and the evaluation bench. It models one
approval object: which id a request returns is not part of that specification (histories are
partitioned by approval id before they are checked), but the ledger allocates ids in order, so
reference checks keep to histories whose observed ids are the ones a fresh ledger hands out.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from deterministic_agent_simulator.check.linearizability import Operation
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.world.ledger import ApprovalLedger

__all__ = ["MAX_REFERENCE_OPS", "apply_operation", "reference_linearizable"]

MAX_REFERENCE_OPS = 6
"""Histories longer than this are refused: the search is exponential in the length."""


class _InfeasibleError(Exception):
    """The constraints collected so far have no solution."""


class _NeedChoiceError(Exception):
    """A time comparison was reached for which no outcome has been chosen yet."""


class _Stn:
    """Integer difference constraints over points ``0..points-1`` plus an origin fixed at 0."""

    def __init__(self, points: int) -> None:
        self.origin = points
        self._edges: list[tuple[int, int, int]] = []

    def _node(self, point: int | None) -> int:
        return self.origin if point is None else point

    def le(self, left: tuple[int | None, int], right: tuple[int | None, int]) -> None:
        """Require ``left_point + left_offset <= right_point + right_offset``."""
        (u, cu), (v, cv) = left, right
        self._edges.append((self._node(v), self._node(u), cv - cu))

    def at_least(self, point: int, value: int) -> None:
        self.le((None, value), (point, 0))

    def at_most(self, point: int, value: int) -> None:
        self.le((point, 0), (None, value))

    def feasible(self) -> bool:
        """Whether the system has an integer solution (Bellman-Ford negative-cycle detection)."""
        count = self.origin + 1
        distance = [0] * count
        for _ in range(count):
            changed = False
            for u, v, weight in self._edges:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    changed = True
            if not changed:
                return True
        return False


class _Context:
    """The constraint store plus the outcomes chosen for the time comparisons, in order."""

    def __init__(self, stn: _Stn, choices: Sequence[bool]) -> None:
        self.stn = stn
        self._choices = choices
        self.used = 0

    def decide(self, left: _Time, right: _Time) -> bool:
        """The outcome chosen for ``left >= right``, recorded as a constraint."""
        if self.used >= len(self._choices):
            raise _NeedChoiceError
        outcome = self._choices[self.used]
        self.used += 1
        if outcome:
            self.stn.le((right.point, right.offset), (left.point, left.offset))
        else:
            self.stn.le((left.point, left.offset + 1), (right.point, right.offset))
        if not self.stn.feasible():
            raise _InfeasibleError
        return outcome


@dataclass(frozen=True, slots=True)
class _Time:
    """The symbolic time ``point + offset``; ``point`` None means a constant."""

    ctx: _Context = field(compare=False)
    point: int | None
    offset: int

    def __add__(self, other: int) -> _Time:
        return _Time(self.ctx, self.point, self.offset + other)

    def __ge__(self, other: object) -> bool:
        if isinstance(other, _Time):
            return self.ctx.decide(self, other)
        if isinstance(other, int) and not isinstance(other, bool):
            return self.ctx.decide(self, _Time(self.ctx, None, other))
        raise TypeError(f"cannot compare a simulated time with {other!r}")


class _SymbolicLoop:
    """Stand-in for the event loop whose clock returns symbolic times."""

    def __init__(self, context: _Context) -> None:
        self.now_us: Any = _Time(context, None, 0)


def apply_operation(ledger: ApprovalLedger, kind: str, args: tuple[Hashable, ...]) -> Any:
    """Run one recorded operation against the real ledger state machine."""
    if kind == "request":
        token, action_hash, requester = args
        return ledger.apply_request(str(token), str(action_hash), {}, str(requester))
    if kind == "decide":
        approval, approve, approver = args
        return ledger.apply_decide(str(approval), bool(approve), str(approver))
    if kind == "status":
        return ledger.apply_status(str(args[0]))
    if kind == "consume":
        approval, action_hash, consumer = args
        return ledger.apply_consume(str(approval), str(action_hash), str(consumer))
    raise NotImplementedError(f"the reference checker does not model {kind!r}")


def _matches(observed: Any, produced: Any, stn: _Stn) -> bool:
    """Whether the ledger's output can be what the client observed (may add constraints)."""
    if isinstance(produced, tuple):
        if not isinstance(observed, tuple) or len(observed) != len(produced):
            return False
        return all(_matches(o, p, stn) for o, p in zip(observed, produced, strict=True))
    if isinstance(produced, _Time):
        if not isinstance(observed, int) or isinstance(observed, bool):
            return False
        stn.le((produced.point, produced.offset), (None, observed))
        stn.le((None, observed), (produced.point, produced.offset))
        return True
    return bool(observed == produced)


def _replay(order: Sequence[Operation], ttl_us: int, choices: Sequence[bool]) -> None:
    """Execute ``order`` on the real ledger with symbolic time; raise if it cannot have happened."""
    stn = _Stn(len(order))
    for index, op in enumerate(order):
        stn.at_least(index, op.invoke_us)
        if op.response_us is not None:
            stn.at_most(index, op.response_us)
        if index:
            stn.le((index - 1, 0), (index, 0))
    if not stn.feasible():
        raise _InfeasibleError
    context = _Context(stn, choices)
    loop = _SymbolicLoop(context)
    ledger = ApprovalLedger(cast("SimEventLoop", loop), ttl_us=ttl_us)
    for index, op in enumerate(order):
        loop.now_us = _Time(context, index, 0)
        produced = apply_operation(ledger, op.kind, op.args)
        if op.completed and not _matches(op.output, produced, stn):
            raise _InfeasibleError
        if not stn.feasible():
            raise _InfeasibleError


def _admissible(order: Sequence[Operation], ttl_us: int, choices: list[bool]) -> bool:
    try:
        _replay(order, ttl_us, choices)
    except _InfeasibleError:
        return False
    except _NeedChoiceError:
        return any(_admissible(order, ttl_us, [*choices, outcome]) for outcome in (True, False))
    return True


def _respects_real_time(order: Sequence[Operation]) -> bool:
    return not any(
        order[later].returned_at < order[earlier].invoke_us
        for earlier in range(len(order))
        for later in range(earlier + 1, len(order))
    )


def reference_linearizable(ops: Sequence[Operation], ttl_us: int) -> bool:
    """Whether ``ops`` is linearizable, decided by exhaustive search over the real ledger."""
    if len(ops) > MAX_REFERENCE_OPS:
        raise ValueError(f"the reference checker is limited to {MAX_REFERENCE_OPS} operations")
    completed = [op for op in ops if op.completed]
    pending = [op for op in ops if not op.completed]
    for size in range(len(pending) + 1):
        for chosen in itertools.combinations(pending, size):
            for order in itertools.permutations([*completed, *chosen]):
                if _respects_real_time(order) and _admissible(order, ttl_us, []):
                    return True
    return False

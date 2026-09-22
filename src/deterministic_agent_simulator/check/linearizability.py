"""Linearizability checking: the Wing-Gong algorithm with Lowe's memoization, plus brute force.

A history is a set of operations, each with an invocation time, a response time and an output as
observed by the client. It is *linearizable* (Herlihy and Wing, TOPLAS 1990) if the operations
can be put in a sequential order that (1) respects real time (if A returned before B was invoked,
A comes first) and (2) is accepted by a sequential specification of the object.

Operations whose response was never observed (a timeout, a crash) are *pending*: they may have
taken effect at any time after their invocation or not at all, and any output is acceptable for
them. The search therefore succeeds once every *completed* operation is linearized.

:func:`wing_gong_check` is the depth-first search of Wing and Gong ("Testing and verifying
concurrent objects", JPDC 1993) in the form popularized by Lowe ("Testing for linearizability",
CCPE 2017): events live in a doubly linked list, a linearized operation's call and return events
are unlinked ("lifted"), and a cache of ``(linearized set, model state)`` pairs prunes states that
were already explored. The sequential model may be nondeterministic: :meth:`SequentialModel.step`
returns every successor state consistent with the operation, and the search tries each.

Linearizability is compositional (it is a local property), so histories of independent objects
are checked one object at a time (:func:`check_partitioned`), which keeps each search small.

:func:`brute_force_check` enumerates every permutation of every admissible subset of pending
operations. It is exponential and exists only as an independent oracle for differential tests on
histories of at most eight operations.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "INFINITY",
    "LinearizabilityResult",
    "Operation",
    "SequentialModel",
    "brute_force_check",
    "check_partitioned",
    "wing_gong_check",
]

INFINITY = 1 << 62
"""Response time used for pending operations (later than any simulated time)."""


@dataclass(frozen=True, slots=True)
class Operation:
    """One client-observed operation.

    ``response_us`` and ``output`` are ``None`` when no response was observed (pending).
    """

    id: int
    client: str
    kind: str
    args: tuple[Hashable, ...]
    invoke_us: int
    response_us: int | None = None
    output: Hashable | None = None

    def __post_init__(self) -> None:
        if self.response_us is not None and self.response_us < self.invoke_us:
            raise ValueError("an operation cannot return before it is invoked")
        if self.response_us is None and self.output is not None:
            raise ValueError("a pending operation has no observed output")

    @property
    def completed(self) -> bool:
        return self.response_us is not None

    @property
    def returned_at(self) -> int:
        return self.response_us if self.response_us is not None else INFINITY

    def describe(self) -> str:
        args = ", ".join(repr(arg) for arg in self.args)
        result = "?" if not self.completed else repr(self.output)
        end = "..." if self.response_us is None else str(self.response_us)
        return (
            f"#{self.id} {self.client}: {self.kind}({args}) -> {result} [{self.invoke_us}, {end}]"
        )


class SequentialModel[S: Hashable](Protocol):
    """A sequential specification. States must be hashable (they are memoized)."""

    def initial(self) -> S: ...

    def step(self, state: S, op: Operation) -> Sequence[S]:
        """Successor states in which ``op`` (with its observed output, if any) is legal."""
        ...


@dataclass(frozen=True, slots=True)
class LinearizabilityResult:
    ok: bool
    order: tuple[int, ...] | None
    explored: int
    operations: int

    def __bool__(self) -> bool:
        return self.ok


class _Search[S: Hashable]:
    """State of one Wing-Gong-Lowe search over a single history."""

    def __init__(self, ops: Sequence[Operation], model: SequentialModel[S]) -> None:
        self.ops = ops
        self.model = model
        events: list[tuple[int, int, int]] = []
        for index, op in enumerate(ops):
            events.append((op.invoke_us, 0, index))
            events.append((op.returned_at, 1, index))
        # At equal times calls sort before returns: operations that touch are concurrent.
        events.sort()
        size = len(events)
        self.head = size
        self.is_call = [kind == 0 for _, kind, _ in events]
        self.op_of = [index for _, _, index in events]
        self.next = [position + 1 if position + 1 < size else -1 for position in range(size)]
        self.prev = [position - 1 if position > 0 else self.head for position in range(size)]
        self.next.append(0 if size else -1)
        self.prev.append(-1)
        self.call_at = [0] * len(ops)
        self.return_at = [0] * len(ops)
        for position, (_, kind, index) in enumerate(events):
            if kind == 0:
                self.call_at[index] = position
            else:
                self.return_at[index] = position

    def _unlink(self, position: int) -> None:
        before, after = self.prev[position], self.next[position]
        self.next[before] = after
        if after != -1:
            self.prev[after] = before

    def _relink(self, position: int) -> None:
        before, after = self.prev[position], self.next[position]
        self.next[before] = position
        if after != -1:
            self.prev[after] = position

    def lift(self, index: int) -> None:
        self._unlink(self.call_at[index])
        self._unlink(self.return_at[index])

    def unlift(self, index: int) -> None:
        self._relink(self.return_at[index])
        self._relink(self.call_at[index])

    def run(self, max_explored: int) -> LinearizabilityResult:  # noqa: PLR0915
        ops, model = self.ops, self.model
        remaining = sum(1 for op in ops if op.completed)
        state = model.initial()
        linearized = 0
        cache: set[tuple[int, S]] = set()
        # Stack frames: (operation index, state before it, its successor states, chosen index).
        stack: list[tuple[int, S, Sequence[S], int]] = []
        explored = 0
        entry = self.next[self.head]

        def push(index: int, before: S, successors: Sequence[S], start: int) -> S | None:
            nonlocal linearized, remaining
            bit = 1 << index
            for choice in range(start, len(successors)):
                key = (linearized | bit, successors[choice])
                if key in cache:
                    continue
                cache.add(key)
                stack.append((index, before, successors, choice))
                linearized |= bit
                self.lift(index)
                if ops[index].completed:
                    remaining -= 1
                return successors[choice]
            return None

        while True:
            if remaining == 0:
                order = tuple(frame[0] for frame in stack)
                return LinearizabilityResult(True, order, explored, len(ops))
            if entry != -1 and self.is_call[entry]:
                index = self.op_of[entry]
                explored += 1
                if explored > max_explored:
                    raise RuntimeError(f"linearizability search exceeded {max_explored} steps")
                successors = model.step(state, ops[index])
                advanced = push(index, state, successors, 0)
                if advanced is not None:
                    state = advanced
                    entry = self.next[self.head]
                else:
                    entry = self.next[entry]
                continue
            # A return event (or the end of the list): the operation it belongs to was not
            # linearized before its response, so undo the most recent choice.
            while True:
                if not stack:
                    return LinearizabilityResult(False, None, explored, len(ops))
                index, before, successors, choice = stack.pop()
                self.unlift(index)
                linearized &= ~(1 << index)
                if ops[index].completed:
                    remaining += 1
                state = before
                advanced = push(index, before, successors, choice + 1)
                if advanced is not None:
                    state = advanced
                    entry = self.next[self.head]
                else:
                    entry = self.next[self.call_at[index]]
                break


def wing_gong_check[S: Hashable](
    ops: Sequence[Operation], model: SequentialModel[S], *, max_explored: int = 2_000_000
) -> LinearizabilityResult:
    """Decide linearizability of ``ops`` against ``model`` (see the module docstring)."""
    return _Search(list(ops), model).run(max_explored)


def _precedes(first: Operation, second: Operation) -> bool:
    return first.returned_at < second.invoke_us


def _accepted[S: Hashable](order: Sequence[Operation], model: SequentialModel[S]) -> bool:
    states: set[S] = {model.initial()}
    for op in order:
        states = {after for state in states for after in model.step(state, op)}
        if not states:
            return False
    return True


def brute_force_check[S: Hashable](ops: Sequence[Operation], model: SequentialModel[S]) -> bool:
    """Exhaustive reference checker for tiny histories (differential testing only)."""
    if len(ops) > 9:
        raise ValueError("brute force is limited to 9 operations")
    completed = [op for op in ops if op.completed]
    pending = [op for op in ops if not op.completed]
    for size in range(len(pending) + 1):
        for chosen in itertools.combinations(pending, size):
            candidates = completed + list(chosen)
            for order in itertools.permutations(candidates):
                if any(
                    _precedes(order[later], order[earlier])
                    for earlier in range(len(order))
                    for later in range(earlier + 1, len(order))
                ):
                    continue
                if _accepted(order, model):
                    return True
    return False


def check_partitioned[S: Hashable](
    ops: Iterable[Operation],
    key: Callable[[Operation], str | None],
    model: SequentialModel[S],
) -> dict[str, LinearizabilityResult]:
    """Check each object's sub-history separately. Operations keyed ``None`` are dropped."""
    groups: dict[str, list[Operation]] = {}
    for op in ops:
        object_key = key(op)
        if object_key is not None:
            groups.setdefault(object_key, []).append(op)
    return {name: wing_gong_check(group, model) for name, group in sorted(groups.items())}

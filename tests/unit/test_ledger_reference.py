"""What the independent reference checker says about the approval model (see ADR 0004).

The histories here are *real* executions: every call is applied to a real `ApprovalLedger` at a
chosen commit time inside its observation window, so the model must accept all of them. The
differential tests then perturb one observed output and require the model and the reference checker
(`check/ledger_reference.py`, which shares nothing with the model) to agree.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deterministic_agent_simulator.check.ledger_model import ApprovalModel
from deterministic_agent_simulator.check.ledger_reference import (
    apply_operation,
    reference_linearizable,
)
from deterministic_agent_simulator.check.linearizability import Operation, wing_gong_check
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.world.ledger import ApprovalLedger

pytestmark = pytest.mark.unit

TTL = 100


class _RealClock:
    now_us: int = 0


@dataclass(frozen=True, slots=True)
class Call:
    """One call of a real execution: it commits at ``commit``, inside its observation window."""

    kind: str
    args: tuple[Hashable, ...]
    client: str
    invoke: int
    commit: int
    response: int | None


def observed_execution(ttl: int, calls: Sequence[Call]) -> list[Operation]:
    """The client-observed history of a real ledger execution (calls in commit order)."""
    clock = _RealClock()
    ledger = ApprovalLedger(cast("SimEventLoop", clock), ttl_us=ttl)
    ops: list[Operation] = []
    for index, call in enumerate(calls):
        if index and call.commit < calls[index - 1].commit:
            raise ValueError("calls must be given in commit order")
        if call.invoke > call.commit or (call.response is not None and call.response < call.commit):
            raise ValueError("a call commits between its invocation and its response")
        clock.now_us = call.commit
        output = apply_operation(ledger, call.kind, call.args)
        ops.append(
            Operation(
                index,
                call.client,
                call.kind,
                call.args,
                call.invoke,
                call.response,
                None if call.response is None else output,
            )
        )
    return ops


def _model_accepts(history: Sequence[Operation]) -> bool:
    return wing_gong_check(history, ApprovalModel(TTL)).ok


# ------------------------------------------------------- executions the model must not reject
def test_a_lost_decision_may_have_committed_long_after_its_invocation() -> None:
    """The second decision only proves the first one happened before *its* response, not before
    the first one's earliest possible point: capping the expiry there rejects this execution."""
    history = observed_execution(
        TTL,
        [
            Call("request", ("tok", "h", "agent"), "agent", 0, 0, 1),
            Call("decide", ("apr-1", True, "human"), "human", 2, 10, None),
            Call("decide", ("apr-1", True, "human2"), "human2", 5, 12, 15),
            Call("status", ("apr-1",), "auditor", 100, 105, 108),
            Call("consume", ("apr-1", "h", "c1"), "c1", 104, 106, 107),
        ],
    )
    assert [op.output for op in history] == [
        "apr-1",
        None,
        ("not_pending", None),
        ("approved", 10 + TTL),
        "ok",
    ]
    assert reference_linearizable(history, TTL)
    assert _model_accepts(history)


def test_a_consume_with_the_wrong_hash_does_not_pin_the_expiry_early() -> None:
    history = observed_execution(
        TTL,
        [
            Call("request", ("tok", "h", "agent"), "agent", 0, 0, 1),
            Call("decide", ("apr-1", True, "human"), "human", 2, 10, None),
            Call("consume", ("apr-1", "other", "c1"), "c1", 5, 12, 15),
            Call("status", ("apr-1",), "auditor", 100, 105, 108),
        ],
    )
    assert [op.output for op in history] == [
        "apr-1",
        None,
        "hash_mismatch",
        ("approved", 10 + TTL),
    ]
    assert reference_linearizable(history, TTL)
    assert _model_accepts(history)


def test_an_execution_must_commit_inside_its_observation_window() -> None:
    late = [Call("request", ("tok", "h", "agent"), "agent", 5, 4, 9)]
    with pytest.raises(ValueError, match="commits between"):
        observed_execution(TTL, late)
    unordered = [
        Call("request", ("tok", "h", "agent"), "agent", 0, 5, 9),
        Call("status", ("apr-1",), "auditor", 0, 1, 2),
    ]
    with pytest.raises(ValueError, match="commit order"):
        observed_execution(TTL, unordered)


def test_every_execution_of_a_late_lost_decision_is_accepted() -> None:
    """An exhaustive grid of the family the earlier model rejected (ADR 0004).

    A decision that lost its response, an operation that sees the approval decided, and a later
    read and consume, over every combination of commit times and observation windows below. The
    model that capped the expiry at each operation's earliest point rejected 68 of these 1,080 real
    executions (checkable against `check/ledger_model.py` at commit 5597102).
    """
    grid = product(
        [5, 10, 30, 60],  # when the lost decision commits
        ["decide", "consume", "status"],  # what observes the approval next
        [6, 12, 40, 70],  # when that operation commits
        [(0, 0), (6, 30), (12, 60)],  # and how wide its window is
        [TTL - 5, TTL, TTL + 5, TTL + 40],  # when the late read and consume commit
        [(0, 0), (8, 8), (30, 30)],
    )
    checked = 0
    for decided, kind, second, (before, after), late, (early, slow) in grid:
        if second < decided or late < second:
            continue
        args: tuple[Hashable, ...] = ("apr-1", True, "human2")
        client = "human2"
        if kind == "consume":
            args, client = ("apr-1", "other", "c2"), "c2"
        elif kind == "status":
            args, client = ("apr-1",), "auditor"
        history = observed_execution(
            TTL,
            [
                Call("request", ("tok", "h", "agent"), "agent", 0, 0, 1),
                Call("decide", ("apr-1", True, "human"), "human", 2, decided, None),
                Call(kind, args, client, max(0, second - before), second, second + after),
                Call("status", ("apr-1",), "auditor", max(0, late - early), late, late + slow),
                Call(
                    "consume",
                    ("apr-1", "h", "c1"),
                    "c1",
                    max(0, late - early),
                    late,
                    late + slow + 1,
                ),
            ],
        )
        assert _model_accepts(history), [op.describe() for op in history]
        checked += 1
    assert checked == 1080


def test_the_reference_checker_rejects_impossible_executions() -> None:
    real = observed_execution(
        TTL,
        [
            Call("request", ("tok", "h", "agent"), "agent", 0, 0, 1),
            Call("decide", ("apr-1", True, "human"), "human", 2, 3, 4),
            Call("consume", ("apr-1", "h", "c1"), "c1", 5, 6, 7),
        ],
    )
    assert reference_linearizable(real, TTL)
    second_consumer = [*real, Operation(3, "c2", "consume", ("apr-1", "h", "c2"), 10, 11, "ok")]
    assert not reference_linearizable(second_consumer, TTL)
    after_expiry = [
        *real[:2],
        Operation(2, "c1", "consume", ("apr-1", "h", "c1"), 5 * TTL, 6 * TTL, "ok"),
    ]
    assert not reference_linearizable(after_expiry, TTL)
    rolled_back = [*real, Operation(3, "auditor", "status", ("apr-1",), 10, 11, ("approved", 103))]
    assert not reference_linearizable(rolled_back, TTL)
    with pytest.raises(ValueError, match="limited"):
        reference_linearizable([*real, *real, *real], TTL)
    with pytest.raises(NotImplementedError, match="notify"):
        reference_linearizable(
            [Operation(0, "n", "notify", ("apr-1", "tok", "h"), 0, 1, "ok")], TTL
        )


# ------------------------------------------------------------------- differential testing
_APPROVERS = ("human", "human2", "agent")
# Which id a request returns is not part of the specification of *one* approval: the history is
# partitioned by the id an operation names, so request outputs are never perturbed here.
_OUTPUTS: dict[str, tuple[Hashable, ...]] = {
    "decide": (("ok", None), ("ok", 2 * TTL), ("not_pending", None), ("forbidden", None)),
    "status": (("pending", None), ("approved", 2 * TTL), ("consumed", None), ("expired", None)),
    "consume": ("ok", "already_consumed", "expired", "not_approved", "hash_mismatch"),
}


@st.composite
def real_executions(draw: st.DrawFn, *, max_calls: int = 6) -> list[Operation]:
    """A real ledger execution observed by concurrent clients, with some responses lost.

    Several approvers (one of them the requester, whose decision is forbidden), retried requests
    and lost decision responses are what make the expiry of an approval uncertain to its clients.
    """
    calls = [Call("request", ("tok", "h", "agent"), "agent", 0, 0, draw(st.integers(0, 20)))]
    commit = 0
    for _ in range(draw(st.integers(1, max_calls - 1))):
        commit += draw(st.integers(0, 70))
        kind = draw(st.sampled_from(["request", "decide", "status", "consume"]))
        args: tuple[Hashable, ...] = ("tok", "h", "agent")
        client = "agent"
        if kind == "decide":
            client = draw(st.sampled_from(_APPROVERS))
            args = ("apr-1", draw(st.booleans()), client)
        elif kind == "status":
            client, args = "auditor", ("apr-1",)
        elif kind == "consume":
            client = draw(st.sampled_from(["c1", "c2"]))
            args = ("apr-1", draw(st.sampled_from(["h", "other"])), client)
        response = None if draw(st.integers(0, 6)) == 0 else commit + draw(st.integers(0, 25))
        invoke = max(0, commit - draw(st.integers(0, 25)))
        calls.append(Call(kind, args, client, invoke, commit, response))
    return observed_execution(TTL, calls)


def _perturbed(history: list[Operation], data: st.DataObject) -> list[Operation]:
    """Replace one observed output, which usually makes the history impossible."""
    completed = [index for index, op in enumerate(history) if op.completed and op.kind in _OUTPUTS]
    if not completed or data.draw(st.booleans()):
        return history
    index = data.draw(st.sampled_from(completed))
    victim = history[index]
    output = data.draw(st.sampled_from(_OUTPUTS[victim.kind]))
    replaced = Operation(
        victim.id,
        victim.client,
        victim.kind,
        victim.args,
        victim.invoke_us,
        victim.response_us,
        output,
    )
    return [*history[:index], replaced, *history[index + 1 :]]


@settings(max_examples=400, deadline=None)
@given(history=real_executions())
def test_the_model_accepts_every_real_execution(history: list[Operation]) -> None:
    assert _model_accepts(history), [op.describe() for op in history]


@settings(max_examples=250, deadline=None)
@given(history=real_executions(max_calls=5), data=st.data())
def test_the_model_agrees_with_the_reference_checker(
    history: list[Operation], data: st.DataObject
) -> None:
    perturbed = _perturbed(history, data)
    assert _model_accepts(perturbed) == reference_linearizable(perturbed, TTL), [
        op.describe() for op in perturbed
    ]

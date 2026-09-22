from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deterministic_agent_simulator.check.ledger_model import (
    ApprovalModel,
    ApprovalState,
    approval_object_key,
)
from deterministic_agent_simulator.check.linearizability import (
    Operation,
    brute_force_check,
    check_partitioned,
    wing_gong_check,
)
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.world.ledger import ApprovalLedger

pytestmark = pytest.mark.unit

TTL = 100


class RegisterModel:
    """A read/write register; the classic linearizability example."""

    def initial(self) -> int:
        return 0

    def step(self, state: int, op: Operation) -> Sequence[int]:
        if op.kind == "write":
            value = cast(int, op.args[0])
            return [value] if op.output in (None, "ok") else []
        if op.output is None or op.output == state:
            return [state]
        return []


def op(  # noqa: PLR0917 - positional form keeps the history fixtures on one line each
    op_id: int,
    kind: str,
    args: tuple[Hashable, ...],
    invoke: int,
    response: int | None,
    output: Hashable | None = None,
    client: str = "c",
) -> Operation:
    return Operation(op_id, client, kind, args, invoke, response, output)


# ----------------------------------------------------------------------------- register fixtures
def test_register_fixtures() -> None:
    model = RegisterModel()
    sequential = [op(0, "write", (1,), 0, 1, "ok"), op(1, "read", (), 2, 3, 1)]
    assert wing_gong_check(sequential, model).ok
    stale = [op(0, "write", (1,), 0, 1, "ok"), op(1, "read", (), 2, 3, 0)]
    assert not wing_gong_check(stale, model).ok
    concurrent = [op(0, "write", (1,), 0, 10, "ok"), op(1, "read", (), 2, 3, 0)]
    assert wing_gong_check(concurrent, model).ok  # the read may linearize before the write
    touching = [op(0, "write", (1,), 0, 5, "ok"), op(1, "read", (), 5, 6, 0)]
    assert wing_gong_check(touching, model).ok  # equal times count as concurrent
    lost_write = [op(0, "write", (1,), 0, None), op(1, "read", (), 2, 3, 0)]
    assert wing_gong_check(lost_write, model).ok  # a pending write may never take effect
    lost_but_seen = [op(0, "write", (2,), 0, None), op(1, "read", (), 5, 6, 2)]
    assert wing_gong_check(lost_but_seen, model).ok  # ... or it may have


def test_result_reports_a_witness_order() -> None:
    history = [op(0, "write", (1,), 0, 1, "ok"), op(1, "read", (), 2, 3, 1)]
    result = wing_gong_check(history, RegisterModel())
    assert result
    assert result.order == (0, 1)
    assert result.operations == 2


def test_search_budget_is_enforced() -> None:
    history = [op(i, "write", (i,), 0, 100, "ok") for i in range(8)] + [
        op(8, "read", (), 200, 201, 99)
    ]
    with pytest.raises(RuntimeError, match="exceeded"):
        wing_gong_check(history, RegisterModel(), max_explored=10)


def test_operations_validate_their_timing() -> None:
    with pytest.raises(ValueError, match="before it is invoked"):
        op(0, "read", (), 5, 4, 0)
    with pytest.raises(ValueError, match="pending"):
        op(0, "read", (), 5, None, 0)
    assert "?" in op(0, "read", (), 5, None).describe()


@st.composite
def register_histories(draw: st.DrawFn) -> list[Operation]:
    count = draw(st.integers(1, 7))
    ops: list[Operation] = []
    for index in range(count):
        invoke = draw(st.integers(0, 20))
        pending = draw(st.integers(0, 9)) == 0
        response = None if pending else invoke + draw(st.integers(0, 10))
        if draw(st.booleans()):
            ops.append(
                op(
                    index,
                    "write",
                    (draw(st.integers(0, 2)),),
                    invoke,
                    response,
                    None if pending else "ok",
                )
            )
        else:
            ops.append(
                op(
                    index,
                    "read",
                    (),
                    invoke,
                    response,
                    None if pending else draw(st.integers(0, 2)),
                )
            )
    return ops


@settings(max_examples=300)
@given(history=register_histories())
def test_wing_gong_matches_brute_force_on_register_histories(history: list[Operation]) -> None:
    model = RegisterModel()
    assert wing_gong_check(history, model).ok == brute_force_check(history, model)


def test_brute_force_refuses_large_histories() -> None:
    with pytest.raises(ValueError, match="limited"):
        brute_force_check([op(i, "read", (), 0, 1, 0) for i in range(10)], RegisterModel())


# ------------------------------------------------------------------------ approval model fixtures
def request(op_id: int, invoke: int, response: int | None, out: str | None = "apr-1") -> Operation:
    return op(op_id, "request", ("tok", "h", "agent"), invoke, response, out, "agent")


def decide(op_id: int, invoke: int, response: int | None, out: Hashable | None) -> Operation:
    return op(op_id, "decide", ("apr-1", True, "human"), invoke, response, out, "human")


def consume(
    op_id: int, invoke: int, response: int | None, out: str | None, consumer: str = "c1"
) -> Operation:
    return op(op_id, "consume", ("apr-1", "h", consumer), invoke, response, out, consumer)


def status(op_id: int, invoke: int, response: int | None, out: Hashable | None) -> Operation:
    return op(op_id, "status", ("apr-1",), invoke, response, out, "auditor")


def lin(history: list[Operation]) -> bool:
    model = ApprovalModel(TTL)
    result = wing_gong_check(history, model).ok
    if len(history) <= 7:
        assert result == brute_force_check(history, model)
    return result


def test_consume_once_by_two_consumers_is_not_linearizable() -> None:
    base = [request(0, 0, 1), decide(1, 2, 3, ("ok", 3 + TTL))]
    assert lin([*base, consume(2, 10, 11, "ok"), consume(3, 20, 21, "already_consumed", "c2")])
    assert not lin([*base, consume(2, 10, 11, "ok"), consume(3, 20, 21, "ok", "c2")])
    assert lin([*base, consume(2, 10, 11, "ok"), consume(3, 20, 21, "ok")])  # idempotent retry


def test_a_consume_rolled_back_after_a_crash_is_not_linearizable() -> None:
    base = [request(0, 0, 1), decide(1, 2, 3, ("ok", 3 + TTL)), consume(2, 10, 11, "ok")]
    assert lin([*base, status(3, 20, 21, ("consumed", None))])
    assert not lin([*base, status(3, 20, 21, ("approved", 3 + TTL))])


def test_expiry_is_judged_at_the_linearization_point() -> None:
    base = [request(0, 0, 1), decide(1, 40, 50, ("ok", 50 + TTL))]
    assert lin([*base, consume(2, 200, 210, "expired")])
    assert not lin([*base, consume(2, 200, 210, "ok")])
    # A consume overlapping the expiry instant may see either outcome.
    assert lin([*base, consume(2, 140, 160, "ok")])
    assert lin([*base, consume(2, 140, 160, "expired")])
    # The reported expiry must match the decision's time window.
    assert not lin([request(0, 0, 1), decide(1, 40, 50, ("ok", 10 + TTL))])


def test_pending_operations_may_or_may_not_have_taken_effect() -> None:
    base = [request(0, 0, 1), decide(1, 2, None, None)]
    assert lin([*base, status(2, 10, 11, ("pending", None))])
    assert lin([*base, status(2, 10, 11, ("approved", 5 + TTL))])
    assert not lin([*base, status(2, 10, 11, ("approved", 1 + TTL))])  # decided before invoked
    concurrent = [
        request(0, 0, 1),
        decide(1, 2, 3, ("ok", 3 + TTL)),
        consume(2, 5, None, None),
        consume(3, 10, 11, "already_consumed", "c2"),
    ]
    assert lin(concurrent)


def test_denials_forbidden_self_approval_and_unknown_ids() -> None:
    assert lin([request(0, 0, 1), op(1, "decide", ("apr-1", False, "human"), 2, 3, ("ok", None))])
    assert lin(
        [request(0, 0, 1), op(1, "decide", ("apr-1", True, "agent"), 2, 3, ("forbidden", None))]
    )
    assert lin([status(0, 0, 1, ("unknown", None)), consume(1, 2, 3, "unknown")])
    assert not lin([request(0, 0, 1), consume(1, 2, 3, "ok")])  # never approved
    notified = op(1, "notify", ("apr-1", "tok", "h"), 1, 1, "ok", "notifier")
    assert lin([request(0, 0, 1), notified])
    assert not lin([request(0, 0, 1), op(1, "notify", ("apr-1", "other", "h"), 1, 1, "ok")])


def test_partitioning_uses_returned_ids_and_learned_tokens() -> None:
    history = [
        request(0, 0, 1),
        op(1, "request", ("tok2", "h", "agent"), 0, None, None, "agent"),
        op(2, "status", ("apr-2",), 5, 6, ("pending", None), "auditor"),
    ]
    key = approval_object_key({"tok2": "apr-2"})
    results = check_partitioned(history, key, ApprovalModel(TTL))
    assert set(results) == {"apr-1", "apr-2"}
    assert all(result.ok for result in results.values())
    assert approval_object_key({})(history[1]) is None
    assert ApprovalModel(TTL).initial() == ApprovalState()
    with pytest.raises(ValueError, match="positive"):
        ApprovalModel(0)


# ------------------------------------------------------------ conformance with the real ledger
class _Clock:
    now_us = 0


@st.composite
def ledger_histories(draw: st.DrawFn) -> list[Operation]:
    """A sequential execution of the real ApprovalLedger, observed with random latencies."""
    clock = _Clock()
    ledger = ApprovalLedger(cast(SimEventLoop, clock), ttl_us=TTL)
    now = 0
    ops: list[Operation] = []

    def observe(kind: str, args: tuple[Hashable, ...], output: Hashable, client: str) -> None:
        before = draw(st.integers(0, 25))
        after = draw(st.integers(0, 25))
        pending = draw(st.integers(0, 6)) == 0
        invoke = max(0, now - before)
        ops.append(
            Operation(
                len(ops),
                client,
                kind,
                args,
                invoke,
                None if pending else now + after,
                None if pending else output,
            )
        )

    approval = ledger.apply_request("tok", "h", {}, "agent")
    observe("request", ("tok", "h", "agent"), approval, "agent")
    for _ in range(draw(st.integers(1, 6))):
        now += draw(st.integers(0, 70))
        clock.now_us = now
        choice = draw(st.sampled_from(["decide", "status", "consume"]))
        if choice == "decide":
            approve = draw(st.booleans())
            approver = draw(st.sampled_from(["human", "agent"]))
            observe(
                "decide",
                (approval, approve, approver),
                ledger.apply_decide(approval, approve, approver),
                approver,
            )
        elif choice == "status":
            observe("status", (approval,), ledger.apply_status(approval), "auditor")
        else:
            action = draw(st.sampled_from(["h", "other"]))
            consumer = draw(st.sampled_from(["c1", "c2"]))
            observe(
                "consume",
                (approval, action, consumer),
                ledger.apply_consume(approval, action, consumer),
                consumer,
            )
    return ops


@settings(max_examples=300)
@given(history=ledger_histories())
def test_histories_of_the_real_ledger_are_linearizable(history: list[Operation]) -> None:
    assert lin(history)


@settings(max_examples=300)
@given(history=ledger_histories(), data=st.data())
def test_wing_gong_matches_brute_force_on_perturbed_ledger_histories(
    history: list[Operation], data: st.DataObject
) -> None:
    completed = [index for index, item in enumerate(history) if item.completed]
    if completed:
        index = data.draw(st.sampled_from(completed))
        target = history[index]
        outputs: dict[str, list[Hashable]] = {
            "request": ["apr-1", "apr-9"],
            "decide": [("ok", None), ("not_pending", None), ("forbidden", None)],
            "status": [("pending", None), ("consumed", None), ("expired", None), ("denied", None)],
            "consume": ["ok", "already_consumed", "expired", "not_approved", "hash_mismatch"],
        }
        output = data.draw(st.sampled_from(outputs[target.kind]))
        history = [
            *history[:index],
            Operation(
                target.id,
                target.client,
                target.kind,
                target.args,
                target.invoke_us,
                target.response_us,
                output,
            ),
            *history[index + 1 :],
        ]
    history = history[:7]
    model = ApprovalModel(TTL)
    assert wing_gong_check(history, model).ok == brute_force_check(history, model)

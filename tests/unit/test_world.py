from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import Observation, Proposal, RejectedError
from deterministic_agent_simulator.agentic.audit import verify_file
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan, FaultProfile
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.adapters import _CURRENT_TICKET, SimRandomPort
from deterministic_agent_simulator.world.inbound import InboundQueue
from deterministic_agent_simulator.world.ledger import ApprovalLedger, ConsumeFact, LedgerRecord
from deterministic_agent_simulator.world.services import Effect, EmailOutbox, Payments
from deterministic_agent_simulator.world.simllm import FINISH_PROPOSAL, Decision, SimLLM
from deterministic_agent_simulator.world.store import AgentStore, AuditChain
from deterministic_agent_simulator.world.workload import (
    CUSTOMERS,
    INJECTIONS,
    WorkloadConfig,
    generate_workload,
)

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------------- keyed jitter streams
def test_jitter_draws_of_one_ticket_do_not_depend_on_other_tickets() -> None:
    root = SimRandom.from_seed(11)

    async def draws(port: SimRandomPort, scope: str, count: int) -> list[int]:
        _CURRENT_TICKET.set(scope)
        return [port.below(1_000) for _ in range(count)]

    def run(interleave: bool) -> list[int]:
        port = SimRandomPort(root.derive("agent-random", 1))
        loop = SimEventLoop(rng=None)
        try:
            if interleave:
                loop.run_until_complete(draws(port, "T-2", 7))
            return loop.run_until_complete(draws(port, "T-1", 5))
        finally:
            loop.close()

    assert run(interleave=True) == run(interleave=False)


# ------------------------------------------------------------------------------------- store
def test_buffered_writes_are_lost_on_crash_unless_flushed() -> None:
    store = AgentStore()
    store.put("a", 1)
    store.put_durable("b", {"x": [1, 2]})
    assert store.get("a") == 1
    store.crash()
    assert store.get("a") is None
    assert store.get("b") == {"x": [1, 2]}
    store.put("a", 2)
    store.flush()
    store.crash()
    assert store.get("a") == 2
    assert store.crashes == 2


def test_stored_values_cannot_be_mutated_through_aliases() -> None:
    store = AgentStore()
    items: list[JsonValue] = [1]
    value: dict[str, JsonValue] = {"list": items}
    store.put_durable("k", value)
    items.append(2)
    fetched = store.get("k")
    assert fetched == {"list": [1]}


def test_scan_merges_the_buffer_over_durable_state_and_torn_batches_keep_a_prefix() -> None:
    store = AgentStore()
    store.put_durable("p/1", 1)
    store.put_durable("p/2", 2)
    store.put("p/2", 20)
    store.put("q/1", 3)
    assert store.scan("p/") == [("p/1", 1), ("p/2", 20)]
    store.put_prefix_durable([("t/1", 1), ("t/2", 2), ("t/3", 3)], 2)
    assert store.torn_writes == 1
    assert [key for key, _ in store.scan("t/")] == ["t/1", "t/2"]
    assert store.durable_keys()[:2] == ["p/1", "p/2"]


def test_audit_chain_detects_edits_and_verifies_with_the_audit_kit(tmp_path: Path) -> None:
    chain = AuditChain()
    for index in range(4):
        chain.append(index * 10, "agent", "step", {"i": index})
    assert chain.verify().ok
    path = tmp_path / "audit.jsonl"
    chain.export_jsonl(path)
    assert verify_file(path, expect_head=chain.head).ok
    chain.entries[2]["data"] = {"i": 99}
    checked = chain.verify()
    assert not checked.ok
    assert checked.bad_index == 2
    assert checked.reason == "entry content modified"
    chain.entries[2] = dict(chain.entries[3])
    assert chain.verify().reason == "sequence gap or reordering"


# ------------------------------------------------------------------------------------ inbound
def _loop() -> SimEventLoop:
    return SimEventLoop(rng=None, trace=Trace())


def _faults(schedule: list[tuple[str, str, str, int]]) -> FaultPlan:
    return FaultPlan(
        FaultProfile.none(), SimRandom.from_seed(1), Trace(), lambda: 0, schedule=schedule
    )


def test_inbound_queue_duplicates_redelivers_and_drains() -> None:
    loop = _loop()
    queue = InboundQueue(
        loop, _faults([("duplicate_delivery", "inbound", "T-1", 1)]), SimRandom.from_seed(2)
    )
    queue.schedule(((0, "T-1"), (10, "T-2")))

    async def consume() -> list[tuple[str, int]]:
        seen: list[tuple[str, int]] = []
        first = await queue.get()
        assert first is not None
        seen.append((first.ticket_id, first.attempt))
        queue.requeue_inflight()  # the consumer "crashed" before acknowledging
        while True:
            delivery = await queue.get()
            if delivery is None:
                return seen
            seen.append((delivery.ticket_id, delivery.attempt))
            queue.ack(delivery)

    seen = loop.run_until_complete(consume())
    loop.close()
    assert sorted(seen) == [("T-1", 1), ("T-1", 2), ("T-1", 3), ("T-2", 1)]
    assert queue.drained
    assert (queue.duplicates, queue.redeliveries) == (1, 1)
    assert queue.acked == {"T-1", "T-2"}


# ----------------------------------------------------------------------------------- services
def test_outbox_deduplicates_by_key_and_rejects_key_reuse() -> None:
    effects: list[Effect] = []
    outbox = EmailOutbox(lambda: 5, effects.append, [0])
    first = outbox.apply_send(
        idempotency_key="k",
        to="a@x.example",
        subject="s",
        body="b",
        reference="T-1",
        approval_id="apr-1",
        intent="T-1/email/1",
    )
    again = outbox.apply_send(
        idempotency_key="k",
        to="a@x.example",
        subject="s",
        body="b",
        reference="T-1",
        approval_id="apr-1",
        intent="T-1/email/1",
    )
    assert first == again
    assert len(effects) == 1
    assert outbox.deduplicated == 1
    with pytest.raises(RejectedError):
        outbox.apply_send(
            idempotency_key="k",
            to="b@x.example",
            subject="s",
            body="b",
            reference="T-1",
            approval_id="apr-1",
            intent="T-1/email/1",
        )
    payments = Payments(lambda: 6, effects.append, [0])
    payments.apply_refund(
        idempotency_key="r",
        customer_id="c-1",
        amount_cents=10,
        reference="T-1",
        approval_id="apr-2",
        intent="T-1/refund/1",
    )
    assert effects[1].action_hash != effects[0].action_hash
    assert "issue_refund" in effects[1].describe()


# ------------------------------------------------------------------------------------- ledger
class _Clock:
    now_us = 0


def test_ledger_consume_once_ttl_and_crash_semantics() -> None:
    clock = _Clock()
    facts: list[ConsumeFact] = []
    ledger = ApprovalLedger(cast(SimEventLoop, clock), ttl_us=100, on_consume=facts.append)
    seen: list[LedgerRecord] = []
    ledger.add_listener(seen.append)
    approval = ledger.apply_request("tok", "h", {"tool": "send_email"}, "agent")
    assert ledger.apply_request("tok", "h", {}, "agent") == approval  # idempotent per token
    assert len(seen) == 1
    assert ledger.apply_decide(approval, True, "agent") == ("forbidden", None)
    assert ledger.apply_consume(approval, "h", "c1") == "not_approved"
    assert ledger.apply_decide(approval, True, "human") == ("ok", 100)
    assert ledger.apply_decide(approval, False, "human") == ("not_pending", None)
    assert ledger.apply_status(approval) == ("approved", 100)
    assert ledger.apply_consume(approval, "other", "c1") == "hash_mismatch"
    assert ledger.apply_consume(approval, "h", "c1") == "ok"
    assert ledger.apply_consume(approval, "h", "c1") == "ok"
    assert ledger.apply_consume(approval, "h", "c2") == "already_consumed"
    assert len(facts) == 1
    ledger.crash()
    ledger.restart()
    assert ledger.apply_status(approval) == ("consumed", None)  # durable
    clock.now_us = 1_000
    second = ledger.apply_request("tok2", "h", {}, "agent")
    ledger.apply_decide(second, True, "human")
    clock.now_us = 1_100
    assert ledger.apply_status(second) == ("expired", None)
    assert ledger.apply_consume(second, "h", "c1") == "expired"
    assert ledger.apply_status("apr-404") == ("unknown", None)
    assert ledger.scope_of(approval) == "tok"
    assert ledger.scope_of("apr-404") == "apr-404"


def test_the_buggy_ledger_loses_unflushed_consumes_on_crash() -> None:
    clock = _Clock()
    ledger = ApprovalLedger(cast(SimEventLoop, clock), ttl_us=1_000, durable_consume=False)
    approval = ledger.apply_request("tok", "h", {}, "agent")
    ledger.apply_decide(approval, True, "human")
    assert ledger.apply_consume(approval, "h", "c1") == "ok"
    ledger.crash()
    ledger.restart()
    assert ledger.apply_status(approval) == ("approved", 1_000)
    assert ledger.lost_on_crash == 1
    assert ledger.apply_consume(approval, "h", "c2") == "ok"
    ledger.flush()
    assert ledger.records()[0].status == "consumed"


# ----------------------------------------------------------------------------------- workload
def _workload(seed: int, keep: list[int] | None = None, inject: bool = False):  # type: ignore[no-untyped-def]
    rate = 1.0 if inject else 0.0
    faults = FaultPlan(
        FaultProfile.of("t", {FaultKind.PROMPT_INJECTION: rate}),
        SimRandom.from_seed(seed).derive("faults"),
        Trace(),
        lambda: 0,
    )
    return generate_workload(
        SimRandom.from_seed(seed).derive("workload"),
        faults,
        WorkloadConfig(min_tickets=5, max_tickets=10),
        keep,
    )


@settings(max_examples=40)
@given(seed=st.integers(0, 10_000), data=st.data())
def test_removing_tickets_leaves_the_rest_of_the_workload_unchanged(
    seed: int, data: st.DataObject
) -> None:
    full = _workload(seed)
    keep = data.draw(st.lists(st.integers(0, full.total_tickets - 1), unique=True))
    part = _workload(seed, keep)
    assert part.refund_budget_cents == full.refund_budget_cents
    assert part.llm_call_budget == full.llm_call_budget
    by_id = {ticket.id: ticket for ticket in full.tickets}
    assert all(by_id[ticket.id] == ticket for ticket in part.tickets)
    arrivals = {ticket: when for when, ticket in full.arrivals}
    assert all(arrivals[ticket] == when for when, ticket in part.arrivals)


def test_injections_carry_canaries_and_customers_use_reserved_domains() -> None:
    workload = _workload(3, inject=True)
    assert set(workload.injections) == {ticket.id for ticket in workload.tickets}
    for ticket in workload.tickets:
        assert workload.canaries[ticket.id] in ticket.body
        assert workload.ticket(ticket.id) == ticket
    assert workload.ticket("T-999") is None
    assert all(address.endswith(".example") for address in CUSTOMERS.values())
    assert {injection.id for injection in INJECTIONS} >= {"reply_to_hijack", "unlisted_tool"}


# ------------------------------------------------------------------------------------- SimLLM
def _observation(plan: str, **overrides: object) -> Observation:
    fields: dict[str, object] = {
        "plan_id": plan,
        "ticket_id": plan.split("~", maxsplit=1)[0],
        "step": 0,
        "purpose": "next",
        "ticket": None,
        "customer_email": None,
        "done": (),
        "last_action": None,
        "last_outcome": None,
    }
    fields.update(overrides)
    return Observation(**fields)  # type: ignore[arg-type]


def test_scripted_model_replays_decisions_and_finishes_when_out_of_script() -> None:
    proposal = Proposal("read_ticket", {"ticket_id": "T-1"})
    model = SimLLM(SimRandom.from_seed(1), script={("T-1", 1): proposal})
    assert model.scripted
    assert model.decide(_observation("T-1")) == proposal
    assert model.decide(_observation("T-1")) == FINISH_PROPOSAL
    assert model.decisions == [
        Decision("T-1", 1, "next", proposal),
        Decision("T-1", 2, "next", FINISH_PROPOSAL),
    ]
    assert model.next_responses("T-1") == 2


@settings(max_examples=30)
@given(seed=st.integers(0, 5_000), other_calls=st.integers(0, 10))
def test_a_plan_decisions_do_not_depend_on_other_plans(seed: int, other_calls: int) -> None:
    alone, busy = SimLLM(SimRandom.from_seed(seed)), SimLLM(SimRandom.from_seed(seed))
    for _ in range(other_calls):
        busy.decide(_observation("T-2"))
    assert [alone.decide(_observation("T-1")) for _ in range(3)] == [
        busy.decide(_observation("T-1")) for _ in range(3)
    ]


def test_the_policy_is_gullible_stubborn_and_sometimes_fixated() -> None:
    workload = _workload(11, inject=True)
    ticket = workload.tickets[0]
    injected = 0
    fixated = 0
    for seed in range(300):
        model = SimLLM(SimRandom.from_seed(seed))
        observation = _observation(
            "T-1", ticket=ticket, customer_email=CUSTOMERS[ticket.customer_id]
        )
        first = model.decide(observation)
        if (
            first.name == "export_customers"
            or first.args.get("customer_id") == "c-attacker"
            or str(first.args.get("to", "")).endswith("attacker.example")
            or "approved" in first.args
            or first.args.get("amount_cents") == ticket.amount_cents + 5_000
        ):
            injected += 1
        denied = _observation("T-1", ticket=ticket, last_action=first, last_outcome="denied")
        repeats = [model.decide(denied) for _ in range(4)]
        if all(repeat == first for repeat in repeats):
            fixated += 1
    assert 60 < injected < 200
    assert fixated > 30

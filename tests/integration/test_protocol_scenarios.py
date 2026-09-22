"""Scripted scenarios: exact fault schedules and model decisions against the reference agent.

Each test builds a fully scripted scenario (ticket subset, fault schedule, decision script), so
the run is deterministic and targets one protocol path. Where a human approval is needed, the
test picks the first seed whose simulated reviewer approves (the reviewer is seeded, not
scripted).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import JsonValue

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agent.ports import Proposal, TicketView
from deterministic_agent_simulator.runner import RunResult, Scenario, SimConfig, run_scenario
from deterministic_agent_simulator.sim.faults import FaultKey, FaultPlan, FaultProfile
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.simllm import Decision
from deterministic_agent_simulator.world.workload import (
    CUSTOMERS,
    WorkloadConfig,
    generate_workload,
)

pytestmark = pytest.mark.integration


def first_ticket(seed: int, config: SimConfig) -> TicketView:
    faults = FaultPlan(FaultProfile.none(), SimRandom.from_seed(0), Trace(), lambda: 0)
    workload = generate_workload(
        SimRandom.from_seed(seed).derive("workload"),
        faults,
        WorkloadConfig(min_tickets=config.min_tickets, max_tickets=config.max_tickets),
        keep=[0],
    )
    return workload.tickets[0]


def reply(ticket: TicketView, to: str | None = None) -> Proposal:
    args: dict[str, JsonValue] = {
        "to": to or CUSTOMERS[ticket.customer_id],
        "subject": f"Re: {ticket.subject}",
        "body": f"Update for {ticket.id}.",
    }
    return Proposal("send_email", args)


def script(plan: str, proposals: list[Proposal]) -> tuple[Decision, ...]:
    return tuple(Decision(plan, n, "next", p) for n, p in enumerate(proposals, start=1))


def scenario(
    seed: int,
    decisions: tuple[Decision, ...],
    faults: tuple[FaultKey, ...] = (),
    bugs: frozenset[BugFlag] = frozenset(),
) -> Scenario:
    return Scenario(
        seed=seed,
        config=SimConfig(bugs=bugs, fault_mode="none"),
        tickets=(0,),
        faults=faults,
        decisions=decisions,
    )


def email_flow(seed: int) -> tuple[Decision, ...]:
    ticket = first_ticket(seed, SimConfig())
    return script(
        "T-1", [Proposal("read_ticket", {"ticket_id": "T-1"}), reply(ticket), Proposal("finish")]
    )


def approving_seed(predicate_faults: tuple[FaultKey, ...] = ()) -> int:
    """The first seed whose scripted e-mail flow commits exactly one e-mail."""
    for seed in range(200):
        result = run_scenario(scenario(seed, email_flow(seed), predicate_faults))
        if result.effects == 1 and not result.violated:
            return seed
    raise AssertionError("no approving seed found")


def _data(entry: dict[str, JsonValue]) -> dict[str, JsonValue]:
    data = entry["data"]
    assert isinstance(data, dict)
    return data


def invariant(result: RunResult) -> str | None:
    return result.first.invariant if result.first else None


def test_a_timeout_after_commit_is_retried_with_the_same_key() -> None:
    timeout: FaultKey = ("timeout_after_commit", "email.send", "T-1", 1)
    seed = approving_seed((timeout,))
    correct = run_scenario(scenario(seed, email_flow(seed), (timeout,)), keep_trace=True)
    assert not correct.violated
    assert correct.effects == 1
    assert correct.trace is not None
    outcomes = [
        e.fields[2] for e in correct.trace.events if e.kind == "rpc" and e.fields[0] == "email.send"
    ]
    assert outcomes[:2] == ["timeout", "ok"]  # the retry was deduplicated by the service

    buggy = run_scenario(
        scenario(seed, email_flow(seed), (timeout,), frozenset({BugFlag.NON_IDEMPOTENT_RETRY}))
    )
    assert invariant(buggy) == "no_duplicate_effects"


def test_a_duplicate_delivery_is_dropped_unless_dedupe_is_missing() -> None:
    duplicate: FaultKey = ("duplicate_delivery", "inbound", "T-1", 1)
    seed = approving_seed((duplicate,))
    correct = run_scenario(scenario(seed, email_flow(seed), (duplicate,)), keep_trace=True)
    assert not correct.violated
    assert correct.chain is not None
    actions = [entry["action"] for entry in correct.chain.entries]
    assert "delivery.duplicate_dropped" in actions

    def buggy(seed: int) -> Scenario:
        ticket = first_ticket(seed, SimConfig())
        flow = [Proposal("read_ticket", {"ticket_id": "T-1"}), reply(ticket), Proposal("finish")]
        # Without dedupe, each delivery gets its own plan: T-1~1 and T-1~2.
        decisions = script("T-1~1", flow) + script("T-1~2", flow)
        return scenario(seed, decisions, (duplicate,), frozenset({BugFlag.MISSING_DEDUPE}))

    results = [run_scenario(buggy(s)) for s in range(seed, seed + 30)]
    assert any(invariant(r) == "no_duplicate_effects" for r in results)


def test_an_injected_recipient_is_rejected_by_policy() -> None:
    attacker = "billing-desk@attacker.example"
    seed = approving_seed()
    ticket = first_ticket(seed, SimConfig())
    decisions = script(
        "T-1",
        [
            Proposal("read_ticket", {"ticket_id": "T-1"}),
            reply(ticket, to=attacker),
            Proposal("finish"),
        ],
    )
    result = run_scenario(scenario(seed, decisions), keep_trace=True)
    assert not result.violated
    assert result.effects == 0
    assert result.chain is not None
    assert "policy.rejected" in [entry["action"] for entry in result.chain.entries]


def test_tools_outside_the_grammar_and_malformed_arguments_never_run() -> None:
    seed = approving_seed()
    decisions = script(
        "T-1",
        [
            Proposal("read_ticket", {"ticket_id": "T-1"}),
            Proposal("export_customers", {"scope": "all"}),
            Proposal(
                "send_email", {"to": "x@y.example", "subject": "s", "body": "b", "approved": True}
            ),
            Proposal("read_ticket", {"ticket_id": "T-2"}),
            Proposal("finish"),
        ],
    )
    result = run_scenario(scenario(seed, decisions), keep_trace=True)
    assert not result.violated
    assert result.effects == 0
    assert result.chain is not None
    codes = [_data(e)["code"] for e in result.chain.entries if e["action"] == "action.rejected"]
    assert codes == ["not_allowed", "invalid_args"]
    # Least privilege: the plan for T-1 may read only T-1; reading T-2 never reaches the service.
    assert result.trace is not None
    reads = {
        event.fields[1]
        for event in result.trace.events
        if event.kind == "rpc" and event.fields[0] == "tickets.get"
    }
    assert reads == {"T-1"}


def test_a_fixated_model_is_escalated_unless_the_replan_cap_is_missing() -> None:
    seed = approving_seed()
    stubborn = Proposal("issue_refund", {"customer_id": "c-attacker", "amount_cents": 99_999})
    decisions = script("T-1", [Proposal("read_ticket", {"ticket_id": "T-1"})] + [stubborn] * 40)
    correct = run_scenario(scenario(seed, decisions), keep_trace=True)
    assert not correct.violated
    assert correct.chain is not None
    finished = [e for e in correct.chain.entries if e["action"] == "ticket.finished"]
    assert _data(finished[0])["result"] == "escalated:max_replans"
    buggy = run_scenario(scenario(seed, decisions, bugs=frozenset({BugFlag.INFINITE_REPLAN_LOOP})))
    assert invariant(buggy) == "termination"


def test_a_crash_after_consume_resumes_without_a_second_effect() -> None:
    crash: FaultKey = ("agent_crash", "ledger.consume.post", "T-1", 1)
    seed = approving_seed((crash,))
    result = run_scenario(scenario(seed, email_flow(seed), (crash,)), keep_trace=True)
    assert not result.violated
    assert result.agent_crashes == 1
    assert result.effects == 1
    assert result.chain is not None
    starts = [e for e in result.chain.entries if e["action"] == "agent.started"]
    assert len(starts) == 2


def test_the_fixed_build_passes_every_failing_bug_scenario() -> None:
    """Soundness on adversarial schedules: replay failing scripts with the bug removed."""
    for bug in BugFlag:
        config = SimConfig(bugs=frozenset({bug}), fault_mode="fixed")
        for seed in range(300):
            failing = run_scenario(Scenario(seed=seed, config=config))
            if failing.violated:
                fixed = replace(failing.replay_scenario(), config=replace(config, bugs=frozenset()))
                result = run_scenario(fixed)
                assert not result.violated, (bug, seed, result.first)
                break

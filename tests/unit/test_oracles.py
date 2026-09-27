from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import TicketView
from deterministic_agent_simulator.agentic.tools import tool_call_hash
from deterministic_agent_simulator.check.linearizability import Operation
from deterministic_agent_simulator.check.oracles import INVARIANTS, OracleSuite
from deterministic_agent_simulator.sim.loop import SimEventLoop
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.ledger import ApprovalLedger, ConsumeFact, History
from deterministic_agent_simulator.world.services import Effect
from deterministic_agent_simulator.world.store import AuditChain
from deterministic_agent_simulator.world.workload import Workload

pytestmark = pytest.mark.unit

EMAIL = "ana.silva@customers.example"


class _Clock:
    now_us = 10


def _workload() -> Workload:
    ticket = TicketView("T-1", "c-1", "refund", 2_500, "Refund", "Please refund.")
    return Workload(
        tickets=(ticket,),
        arrivals=((0, "T-1"),),
        directory={"c-1": EMAIL},
        refund_budget_cents=2_000,
        llm_call_budget=3,
        injections={},
        canaries={},
        total_tickets=1,
    )


class Harness:
    def __init__(self) -> None:
        self.clock = _Clock()
        self.halted = 0
        self.decisions: dict[str, int] = {}
        self.ledger = ApprovalLedger(cast(SimEventLoop, self.clock), ttl_us=1_000)
        self.suite = OracleSuite(
            clock=lambda: self.clock.now_us,
            trace=Trace(keep=True),
            workload=_workload(),
            ledger=self.ledger,
            decisions_of=lambda plan: self.decisions.get(plan, 0),
            decision_bound=4,
            halt=self._halt,
        )
        self.seq = 0

    def _halt(self) -> None:
        self.halted += 1

    def approve(self, tool: str, args: dict[str, JsonValue], consumer: str) -> str:
        approval = self.ledger.apply_request(consumer, tool_call_hash(tool, args), {}, "agent")
        self.ledger.apply_decide(approval, True, "human")
        self.suite.on_consume(
            ConsumeFact(approval, consumer, tool_call_hash(tool, args), 11, 1_010)
        )
        return approval

    def effect(
        self,
        tool: str,
        args: dict[str, JsonValue],
        approval: str,
        intent: str,
        reference: str = "T-1",
    ) -> None:
        self.seq += 1
        self.suite.on_effect(
            Effect(self.seq, tool, args, reference, f"k{self.seq}", approval, intent, 20)
        )

    @property
    def invariants(self) -> list[str]:
        return [violation.invariant for violation in self.suite.violations]


def email(to: str = EMAIL) -> dict[str, JsonValue]:
    return {"to": to, "subject": "Re", "body": "Hi"}


def test_an_approved_consumed_authorized_effect_is_clean() -> None:
    h = Harness()
    approval = h.approve("send_email", email(), "T-1/email/1")
    h.effect("send_email", email(), approval, "T-1/email/1")
    assert h.invariants == []
    assert h.halted == 0


def test_a_second_effect_for_the_same_ticket_is_a_duplicate() -> None:
    h = Harness()
    first = h.approve("send_email", email(), "T-1/email/1")
    second = h.approve("send_email", email(), "T-1/email/2")
    h.effect("send_email", email(), first, "T-1/email/1")
    h.effect("send_email", email(), second, "T-1/email/2")
    assert h.invariants == ["no_duplicate_effects"]
    assert h.halted == 1


def test_unauthorized_recipients_customers_and_amounts() -> None:
    h = Harness()
    attacker = email("billing-desk@attacker.example")
    approval = h.approve("send_email", attacker, "T-1/email/1")
    h.effect("send_email", attacker, approval, "T-1/email/1")
    assert h.invariants == ["authorized_effects"]

    h = Harness()
    wrong: dict[str, JsonValue] = {"customer_id": "c-attacker", "amount_cents": 100}
    approval = h.approve("issue_refund", wrong, "T-1/refund/1")
    h.effect("issue_refund", wrong, approval, "T-1/refund/1")
    assert h.invariants == ["authorized_effects"]

    h = Harness()
    too_much: dict[str, JsonValue] = {"customer_id": "c-1", "amount_cents": 3_000}
    approval = h.approve("issue_refund", too_much, "T-1/refund/1")
    h.effect("issue_refund", too_much, approval, "T-1/refund/1")
    assert h.invariants == ["authorized_effects", "budget_respected"]

    h = Harness()
    h.effect("send_email", email(), "apr-1", "x", reference="T-404")
    assert h.invariants[0] == "authorized_effects"


def test_budget_overrun_is_detected_at_commit() -> None:
    h = Harness()
    refund: dict[str, JsonValue] = {"customer_id": "c-1", "amount_cents": 2_100}
    approval = h.approve("issue_refund", refund, "T-1/refund/1")
    h.effect("issue_refund", refund, approval, "T-1/refund/1")
    assert h.invariants == ["budget_respected"]


@pytest.mark.parametrize("case", ["unknown", "mismatch", "unconsumed", "reused"])
def test_every_effect_needs_a_fresh_consumed_approval_of_its_own(case: str) -> None:
    h = Harness()
    approval = h.approve("send_email", email(), "T-1/email/1")
    if case == "unknown":
        h.effect("send_email", email(), "apr-404", "T-1/email/1")
    elif case == "mismatch":
        h.effect("send_email", email() | {"body": "changed"}, approval, "T-1/email/1")
    elif case == "unconsumed":
        h.effect("send_email", email(), approval, "T-1/email/2")
    else:
        h.effect("send_email", email(), approval, "T-1/email/1")
        h.suite.violations.clear()
        h.effect("send_email", email(), approval, "T-1/email/1", reference="T-1")
    assert "approval_fresh" in h.invariants


def test_consume_once_and_call_budget_and_decision_bound() -> None:
    h = Harness()
    h.suite.on_consume(ConsumeFact("apr-1", "c1", "h", 1, 5))
    h.suite.on_consume(ConsumeFact("apr-1", "c1", "h", 2, 5))
    assert h.invariants == []
    h.suite.on_consume(ConsumeFact("apr-1", "c2", "h", 3, 5))
    assert h.invariants == ["consume_once"]

    h = Harness()
    for _ in range(4):
        h.suite.on_llm_attempt()
    assert h.invariants == ["budget_respected"]

    h = Harness()
    h.decisions["T-1"] = 5
    h.suite.on_decision("T-1")
    assert h.invariants == ["termination"]


def test_audit_chain_is_checked_per_append_and_at_the_end() -> None:
    h = Harness()
    chain = AuditChain()
    chain.append(0, "agent", "a", {})
    h.suite.on_audit(chain)
    chain.append(1, "agent", "b", {})
    chain.entries[1]["prev_hash"] = "0" * 64
    h.suite.on_audit(chain)
    assert h.invariants == ["audit_chain_valid"]


def test_a_truncated_or_forked_audit_chain_is_detected() -> None:
    """A chain can verify on its own and still not be the chain the run appended to."""
    h = Harness()
    chain = AuditChain()
    for index in range(3):
        chain.append(index, "agent", "a", {})
        h.suite.on_audit(chain)
    chain.entries.pop()  # a lost last append, or a fork after a restart
    chain.head = str(chain.entries[-1]["hash"])
    assert chain.verify().ok
    h.suite.final(
        finished=True,
        reason=None,
        history=History(),
        chain=chain,
        hermetic_violations=[],
        agent_errors=[],
    )
    assert h.invariants == ["audit_chain_valid"]
    assert "3 appends" in h.suite.violations[0].detail


def test_final_checks_termination_hermeticity_errors_and_linearizability() -> None:
    h = Harness()
    history = History()
    history.ops.append(Operation(0, "agent", "consume", ("apr-1", "h", "c1"), 0, 1, "ok"))
    chain = AuditChain()
    chain.append(0, "agent", "a", {})
    chain.entries[0]["data"] = {"edited": True}
    h.suite.final(
        finished=False,
        reason="deadlock: stuck",
        history=history,
        chain=chain,
        hermetic_violations=["socket.connect"],
        agent_errors=["KeyError: x"],
    )
    assert h.invariants == [
        "hermetic",
        "agent_error",
        "termination",
        "audit_chain_valid",
        "ledger_linearizable",
    ]
    assert set(h.invariants) <= set(INVARIANTS)
    assert "deadlock" in h.suite.violations[2].describe()


def test_disabled_oracles_record_nothing() -> None:
    h = Harness()
    h.suite.enabled = False
    h.effect("send_email", email("x@attacker.example"), "apr-404", "x")
    h.suite.on_consume(ConsumeFact("apr-1", "c1", "h", 1, 5))
    h.suite.on_llm_attempt()
    h.suite.on_decision("T-1")
    h.suite.on_audit(AuditChain())
    kwargs: dict[str, Any] = {
        "finished": False,
        "reason": None,
        "history": History(),
        "chain": AuditChain(),
        "hermetic_violations": [],
        "agent_errors": [],
    }
    h.suite.final(**kwargs)
    assert not h.suite.violated

"""Invariant oracles: checked at every step that changes their state, and at termination.

The oracles judge the run from the simulator's omniscient view (committed effects, the ledger's
truth log of consumes, ground-truth tickets), not from what the agent believes.

==========================  ===============================================================
invariant                   meaning
==========================  ===============================================================
``no_duplicate_effects``    at most one reply e-mail and one refund per ticket
``authorized_effects``      e-mails go to the ticket's customer (as the trusted directory
                            says); refunds go to that customer and never exceed the ticket
``budget_respected``        committed refunds stay within the refund budget; model calls
                            stay within the call budget
``approval_fresh``          every effect cites an approval for exactly its arguments that was
                            consumed (before expiry) for exactly this effect, and no approval
                            justifies two effects
``consume_once``            no approval is consumed by two different consumers
``ledger_linearizable``     the client-observed ledger history is linearizable
``termination``             every plan stays within its decision bound and the run finishes
                            (no deadlock, livelock or missed deadline)
``audit_chain_valid``       the agent's hash-chained audit trail verifies, and the chain left at
                            the end is the one the run appended to (no truncation or fork)
``hermetic``                no real I/O was attempted during the run
``agent_error``             the agent raised no unexpected exception
==========================  ===============================================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from deterministic_agent_simulator.agentic.audit import GENESIS_HASH
from deterministic_agent_simulator.check.ledger_model import ApprovalModel, approval_object_key
from deterministic_agent_simulator.check.linearizability import check_partitioned
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.ledger import ApprovalLedger, ConsumeFact, History
from deterministic_agent_simulator.world.services import Effect
from deterministic_agent_simulator.world.store import AuditChain
from deterministic_agent_simulator.world.workload import Workload

__all__ = ["INVARIANTS", "OracleSuite", "Violation"]

INVARIANTS = (
    "no_duplicate_effects",
    "authorized_effects",
    "budget_respected",
    "approval_fresh",
    "consume_once",
    "ledger_linearizable",
    "termination",
    "audit_chain_valid",
    "hermetic",
    "agent_error",
)


@dataclass(frozen=True, slots=True)
class Violation:
    invariant: str
    detail: str
    t_us: int

    def describe(self) -> str:
        return f"{self.invariant} at t={self.t_us}us: {self.detail}"


class OracleSuite:
    """Event-driven oracles for one run. ``halt`` stops the run at the first violation."""

    def __init__(
        self,
        *,
        clock: Callable[[], int],
        trace: Trace,
        workload: Workload,
        ledger: ApprovalLedger,
        decisions_of: Callable[[str], int],
        decision_bound: int,
        halt: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock
        self._trace = trace
        self._workload = workload
        self._ledger = ledger
        self._decisions_of = decisions_of
        self.decision_bound = decision_bound
        self._halt = halt
        self.enabled = True
        self.violations: list[Violation] = []
        self._effect_slots: dict[tuple[str, str], int] = {}
        self._refunded: dict[str, int] = {}
        self._refunded_total = 0
        self._cited: dict[str, int] = {}
        self._consumers: dict[str, set[str]] = {}
        self._consume_facts: dict[str, list[ConsumeFact]] = {}
        self._audit_appends = 0
        self._audit_head = GENESIS_HASH
        self.llm_calls = 0
        self.checks = 0

    @property
    def violated(self) -> bool:
        return bool(self.violations)

    def violate(self, invariant: str, detail: str) -> None:
        if not self.enabled:
            return
        now = self._clock()
        self.violations.append(Violation(invariant, detail, now))
        self._trace.record(now, "violation", invariant)
        if self._halt is not None and len(self.violations) == 1:
            self._halt()

    # ------------------------------------------------------------- per step
    def on_effect(self, effect: Effect) -> None:
        if not self.enabled:
            return
        self.checks += 1
        slot = (effect.tool, effect.reference)
        if slot in self._effect_slots:
            self.violate(
                "no_duplicate_effects",
                f"{effect.tool} committed twice for {effect.reference} "
                f"(effects #{self._effect_slots[slot]} and #{effect.seq})",
            )
        self._effect_slots.setdefault(slot, effect.seq)
        ticket = self._workload.ticket(effect.reference)
        args = effect.args
        if ticket is None:
            self.violate("authorized_effects", f"{effect.describe()} names an unknown ticket")
        elif effect.tool == "send_email":
            expected = self._workload.directory.get(ticket.customer_id)
            if args.get("to") != expected:
                self.violate(
                    "authorized_effects",
                    f"reply for {ticket.id} went to {args.get('to')} instead of the customer",
                )
        else:
            amount = _int(args.get("amount_cents"))
            if args.get("customer_id") != ticket.customer_id:
                self.violate(
                    "authorized_effects",
                    f"refund for {ticket.id} paid to {args.get('customer_id')}, not the customer",
                )
            self._refunded[ticket.id] = self._refunded.get(ticket.id, 0) + amount
            if self._refunded[ticket.id] > ticket.amount_cents:
                self.violate(
                    "authorized_effects",
                    f"refunds for {ticket.id} total {self._refunded[ticket.id]} cents, "
                    f"more than the {ticket.amount_cents} requested",
                )
            self._refunded_total += amount
            if self._refunded_total > self._workload.refund_budget_cents:
                self.violate(
                    "budget_respected",
                    f"refunds total {self._refunded_total} cents, over the budget of "
                    f"{self._workload.refund_budget_cents}",
                )
        self._check_approval(effect)

    def _check_approval(self, effect: Effect) -> None:
        record = self._ledger.record(effect.approval_id)
        if record is None:
            self.violate("approval_fresh", f"{effect.describe()} cites an unknown approval")
            return
        if record.action_hash != effect.action_hash:
            self.violate(
                "approval_fresh",
                f"{effect.describe()} does not match the arguments approved in {record.id}",
            )
            return
        facts = self._consume_facts.get(record.id, [])
        if not any(fact.consumer == effect.intent for fact in facts):
            self.violate(
                "approval_fresh",
                f"{effect.describe()} used {record.id} without consuming it for this effect",
            )
            return
        previous = self._cited.get(record.id)
        if previous is not None and previous != effect.seq:
            self.violate(
                "approval_fresh", f"{record.id} justified two effects (#{previous}, #{effect.seq})"
            )
        self._cited[record.id] = effect.seq

    def on_consume(self, fact: ConsumeFact) -> None:
        if not self.enabled:
            return
        self.checks += 1
        self._consume_facts.setdefault(fact.approval_id, []).append(fact)
        consumers = self._consumers.setdefault(fact.approval_id, set())
        if consumers and fact.consumer not in consumers:
            self.violate("consume_once", f"{fact.approval_id} was consumed by two consumers")
        consumers.add(fact.consumer)

    def on_llm_attempt(self) -> None:
        if not self.enabled:
            return
        self.llm_calls += 1
        if self.llm_calls > self._workload.llm_call_budget:
            self.violate(
                "budget_respected",
                f"{self.llm_calls} model calls, over the budget of "
                f"{self._workload.llm_call_budget}",
            )

    def on_decision(self, plan_id: str) -> None:
        if not self.enabled:
            return
        self.checks += 1
        count = self._decisions_of(plan_id)
        if count > self.decision_bound:
            self.violate(
                "termination",
                f"plan {plan_id} received {count} decisions, over the bound of "
                f"{self.decision_bound}",
            )

    def on_audit(self, chain: AuditChain) -> None:
        if not self.enabled:
            return
        self.checks += 1
        index = len(chain.entries) - 1
        prev = str(chain.entries[index - 1]["hash"]) if index > 0 else GENESIS_HASH
        reason = AuditChain.check_entry(chain.entries[index], index, prev)
        if reason is not None:
            self.violate("audit_chain_valid", f"audit entry {index}: {reason}")
            return
        # Anchor what the run appended, so that losing or forking entries later is detectable:
        # a truncated chain still verifies against itself.
        self._audit_appends = len(chain.entries)
        self._audit_head = str(chain.entries[index]["hash"])

    # ---------------------------------------------------------- termination
    def final(
        self,
        *,
        finished: bool,
        reason: str | None,
        history: History,
        chain: AuditChain,
        hermetic_violations: list[str],
        agent_errors: list[str],
    ) -> None:
        if not self.enabled:
            return
        for attempt in hermetic_violations:
            self.violate("hermetic", f"real I/O attempted: {attempt}")
        for error in agent_errors:
            self.violate("agent_error", error)
        if not finished:
            self.violate("termination", reason or "the run did not finish")
        checked = chain.verify()
        self.checks += 1
        if not checked.ok:
            self.violate("audit_chain_valid", f"audit entry {checked.bad_index}: {checked.reason}")
        elif len(chain.entries) != self._audit_appends or chain.head != self._audit_head:
            self.violate(
                "audit_chain_valid",
                f"the chain ends at {len(chain.entries)} entries (head {chain.head[:12]}) after "
                f"{self._audit_appends} appends (head {self._audit_head[:12]}): "
                f"entries were lost, reordered or forked",
            )
        history.close_pending()
        model = ApprovalModel(self._ledger.ttl_us)
        results = check_partitioned(history.ops, approval_object_key(history.tokens), model)
        for approval_id, result in results.items():
            self.checks += 1
            if not result.ok:
                self.violate(
                    "ledger_linearizable",
                    f"client-observed history of {approval_id} is not linearizable",
                )


def _int(value: JsonValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0

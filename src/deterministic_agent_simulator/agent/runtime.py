"""The reference agent: a support-ticket agent whose safety controls are under test.

For every ticket delivered by the inbound queue the agent runs a *plan*: it asks the model for the
next action, validates it against the action registry, executes it and journals the outcome,
until the model says ``finish`` or a cap escalates the ticket. Side-effecting actions (the reply
e-mail and the refund) go through the gate:

1. **Policy.** The recipient must be the customer's address from the trusted directory; a refund
   must go to the ticket's customer and not exceed the ticket's amount.
2. **Budget.** A refund reserves its amount durably, under a lock, before anything else happens.
3. **Approval.** A human approves the exact canonical call (its hash) in the approval ledger; the
   agent consumes the approval (consume-once, TTL enforced by the ledger) with a consumer token
   that is persisted before it is used, so retries and restarts consume idempotently.
4. **Execution.** Every attempt for one ticket slot uses the same idempotency key, so retrying an
   ambiguous failure cannot duplicate the effect. If attempts run out after an ambiguous failure,
   the slot is closed rather than retried with different arguments.

Every decision and every slot transition is journaled in the local store before the next step
depends on it, so a restarted agent resumes a ticket where the crashed one stopped: journaled
decisions are replayed instead of asking the model again, and an action that was in flight is
completed through the same idempotent protocol. Deliveries of a ticket that is active or done are
acknowledged and dropped.

The :class:`~deterministic_agent_simulator.agent.bugs.BugFlag` values switch individual controls
to plausible but wrong implementations; each flag's branch is marked ``# BUG:`` below.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import JsonValue

from deterministic_agent_simulator.agent.actions import (
    FINISH,
    ActionRegistry,
    Rejection,
    ValidatedAction,
)
from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agent.ports import (
    AgentPorts,
    AmbiguousError,
    Delivery,
    Observation,
    PartialWriteError,
    Proposal,
    RejectedError,
    TicketView,
    TransientError,
)
from deterministic_agent_simulator.agentic.tools import Capability

__all__ = [
    "NEGATIVE_OUTCOMES",
    "AgentConfig",
    "ReferenceAgent",
    "parse_reply_to",
]

_T = TypeVar("_T")

OK = "ok"
DONE = "done"
ALREADY_DONE = "already_done"
FINISHED = "finished"
POSITIVE_OUTCOMES = frozenset({OK, DONE, ALREADY_DONE, FINISHED})
NEGATIVE_OUTCOMES = frozenset(
    {
        "denied",
        "rejected:not_allowed",
        "rejected:invalid_args",
        "rejected:policy",
        "rejected:slot_used",
        "rejected:scope",
        "rejected:read_first",
        "rejected:not_found",
        "rejected:approval",
        "budget_exhausted",
        "approval_timeout",
        "approval_expired",
        "failed",
        "failed:ambiguous",
        "unavailable",
    }
)
_REPLY_TO = re.compile(r"(?im)^reply-to:\s*(\S+@\S+)\s*$")
_IDLE_SLOT: dict[str, Any] = {
    "status": "idle",
    "hash": "",
    "args": {},
    "n": 0,
    "approval": None,
    "expires": None,
    "reserved": 0,
    "ambiguous": False,
    "result": None,
}


def parse_reply_to(body: str) -> str | None:
    """The address of a ``Reply-To:`` line in untrusted ticket text, if any."""
    match = _REPLY_TO.search(body)
    return match.group(1) if match else None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    refund_budget_cents: int
    llm_call_budget: int
    workers: int = 2
    max_steps_per_ticket: int = 12
    max_replans: int = 3
    llm_attempts: int = 3
    rpc_attempts: int = 4
    effect_attempts: int = 4
    batch_attempts: int = 6
    approval_wait_us: int = 6_000_000
    poll_initial_us: int = 100_000
    poll_max_us: int = 800_000
    backoff_base_us: int = 40_000
    freshness_margin_us: int = 150_000
    bugs: frozenset[BugFlag] = frozenset()


@dataclass
class _Plan:
    plan_id: str
    ticket_id: str
    log: list[dict[str, JsonValue]] = field(default_factory=list)
    ticket: TicketView | None = None
    customer_email: str | None = None
    done_slots: set[str] = field(default_factory=set)
    last_action: Proposal | None = None
    last_outcome: str | None = None
    replans: int = 0

    @property
    def log_key(self) -> str:
        return f"plan/{self.plan_id}/log"

    def slot_key(self, slot: str) -> str:
        return f"plan/{self.plan_id}/slot/{slot}"

    def token(self, slot: str, n: int) -> str:
        return f"{self.plan_id}/{slot}/{n}"


class _Failed:
    """Sentinel for a call whose retries were exhausted."""


_FAILED = _Failed()


def _view_to_json(view: TicketView) -> dict[str, JsonValue]:
    return {
        "id": view.id,
        "customer_id": view.customer_id,
        "kind": view.kind,
        "amount_cents": view.amount_cents,
        "subject": view.subject,
        "body": view.body,
    }


def _view_from_json(data: JsonValue) -> TicketView:
    if not isinstance(data, dict):
        raise TypeError("journaled ticket view is not an object")
    kind = data["kind"]
    if kind not in ("refund", "question"):
        raise ValueError("journaled ticket kind is invalid")
    return TicketView(
        id=str(data["id"]),
        customer_id=str(data["customer_id"]),
        kind="refund" if kind == "refund" else "question",
        amount_cents=int(str(data["amount_cents"])),
        subject=str(data["subject"]),
        body=str(data["body"]),
    )


class ReferenceAgent:
    """One incarnation of the agent process. See the module docstring."""

    def __init__(self, ports: AgentPorts, config: AgentConfig, *, incarnation: int = 1) -> None:
        self._p = ports
        self._cfg = config
        self._registry = ActionRegistry(
            {Capability.READ_DATA, Capability.SEND_MESSAGE, Capability.WRITE_DATA}
        )
        self._active: set[str] = set()
        self._budget_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()
        self.incarnation = incarnation

    def _bug(self, flag: BugFlag) -> bool:
        return flag in self._cfg.bugs

    # ------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        """Process deliveries with ``workers`` concurrent workers until the inbox is drained."""
        await self._p.audit.append("agent.started", {"incarnation": self.incarnation})
        workers = [asyncio.ensure_future(self._worker()) for _ in range(self._cfg.workers)]
        try:
            finished, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
            for task in finished:
                task.result()
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _worker(self) -> None:
        while True:
            delivery = await self._p.inbox.get()
            if delivery is None:
                return
            await self._handle(delivery)

    async def _handle(self, delivery: Delivery) -> None:
        ticket_id = delivery.ticket_id
        dedupe = not self._bug(BugFlag.MISSING_DEDUPE)
        if dedupe:
            if ticket_id in self._active:
                await self._drop_duplicate(delivery, "active")
                return
            self._active.add(ticket_id)  # no await between the check and the add
            plan_id = ticket_id
        else:
            # BUG: missing_dedupe - every delivery starts its own plan.
            plan_id = f"{ticket_id}~{delivery.attempt}"
        try:
            if dedupe and await self._p.store.get(f"ticket/{ticket_id}/done") is not None:
                await self._drop_duplicate(delivery, "done")
                return
            plan = await self._load_plan(plan_id, ticket_id)
            result = await self._run_plan(plan)
            await self._call_with_retries(lambda: self._p.tickets.close(ticket_id))
            await self._durable_batch(
                [
                    (
                        f"plan/{plan_id}/summary",
                        {"result": result, "steps": len(plan.log), "replans": plan.replans},
                    ),
                    (f"ticket/{ticket_id}/done", {"plan": plan_id, "result": result}),
                ]
            )
            await self._p.audit.append(
                "ticket.finished", {"ticket": ticket_id, "plan": plan_id, "result": result}
            )
            await self._p.inbox.ack(delivery)
        finally:
            if dedupe:
                self._active.discard(ticket_id)

    async def _drop_duplicate(self, delivery: Delivery, reason: str) -> None:
        await self._p.audit.append(
            "delivery.duplicate_dropped",
            {"ticket": delivery.ticket_id, "delivery": delivery.delivery_id, "reason": reason},
        )
        await self._p.inbox.ack(delivery)

    # ------------------------------------------------------------- planning
    async def _load_plan(self, plan_id: str, ticket_id: str) -> _Plan:
        plan = _Plan(plan_id=plan_id, ticket_id=ticket_id)
        stored = await self._p.store.get(plan.log_key)
        if isinstance(stored, list):
            plan.log = [entry for entry in stored if isinstance(entry, dict)]
        for entry in plan.log:
            outcome = entry.get("o")
            if not isinstance(outcome, str):
                continue
            proposal = Proposal.from_json(entry["p"])
            if "view" in entry:
                plan.ticket = _view_from_json(entry["view"])
                email = entry.get("email")
                plan.customer_email = email if isinstance(email, str) else None
            plan.last_action, plan.last_outcome = proposal, outcome
            if outcome in NEGATIVE_OUTCOMES:
                plan.replans += 1
        for key, value in await self._p.store.scan(f"plan/{plan_id}/slot/"):
            if isinstance(value, dict) and value.get("status") == "done":
                plan.done_slots.add(key.rsplit("/", 1)[-1])
        return plan

    def _over_limits(self, plan: _Plan) -> str | None:
        if plan.replans > self._cfg.max_replans:
            return "escalated:max_replans"
        if len(plan.log) >= self._cfg.max_steps_per_ticket:
            return "escalated:max_steps"
        return None

    async def _run_plan(self, plan: _Plan) -> str:
        while True:
            if plan.log and plan.log[-1].get("o") is None:
                # The previous incarnation died while this step was in flight: finish it.
                entry = plan.log[-1]
                proposal = Proposal.from_json(entry["p"])
            else:
                if plan.last_outcome == FINISHED:
                    return "completed"
                replanning = plan.last_outcome in NEGATIVE_OUTCOMES
                if not (replanning and self._bug(BugFlag.INFINITE_REPLAN_LOOP)):
                    limit = self._over_limits(plan)
                    if limit is not None:
                        return limit
                # BUG: infinite_replan_loop - a re-plan skips the step and re-plan caps above.
                decided = await self._decide(plan, "next")
                if decided is None:
                    return "escalated:model_unavailable"
                proposal = decided
                entry = {"p": proposal.to_json(), "o": None}
                plan.log.append(entry)
                await self._p.store.put_durable(plan.log_key, list(plan.log))
            outcome = await self._execute(plan, proposal)
            entry["o"] = outcome
            if proposal.name == "read_ticket" and outcome == OK and plan.ticket is not None:
                entry["view"] = _view_to_json(plan.ticket)
                entry["email"] = plan.customer_email
            await self._p.store.put_durable(plan.log_key, list(plan.log))
            plan.last_action, plan.last_outcome = proposal, outcome
            if outcome in NEGATIVE_OUTCOMES:
                plan.replans += 1

    def _observation(self, plan: _Plan, purpose: str, failed: Proposal | None) -> Observation:
        return Observation(
            plan_id=plan.plan_id,
            ticket_id=plan.ticket_id,
            step=len(plan.log),
            purpose="retry" if purpose == "retry" else "next",
            ticket=plan.ticket,
            customer_email=plan.customer_email,
            done=tuple(sorted(plan.done_slots)),
            last_action=failed if failed is not None else plan.last_action,
            last_outcome="failed" if failed is not None else plan.last_outcome,
        )

    async def _decide(
        self, plan: _Plan, purpose: str, failed: Proposal | None = None
    ) -> Proposal | None:
        observation = self._observation(plan, purpose, failed)
        for attempt in range(self._cfg.llm_attempts):
            if not await self._reserve_llm_call():
                await self._p.audit.append("budget.llm_exhausted", {"plan": plan.plan_id})
                return None
            try:
                return await self._p.llm.decide(observation)
            except TransientError:
                await self._backoff(attempt)
        return None

    async def _reserve_llm_call(self) -> bool:
        async with self._llm_lock:
            used = await self._p.store.get("budget/llm")
            count = used if isinstance(used, int) else 0
            if count >= self._cfg.llm_call_budget:
                return False
            await self._p.store.put_durable("budget/llm", count + 1)
            return True

    # ------------------------------------------------------------ execution
    async def _execute(self, plan: _Plan, proposal: Proposal) -> str:
        if proposal.name == FINISH:
            return FINISHED
        validated = self._registry.validate(proposal)
        if isinstance(validated, Rejection):
            await self._p.audit.append(
                "action.rejected",
                {"plan": plan.plan_id, "action": proposal.name[:64], "code": validated.code},
            )
            return f"rejected:{validated.code}"
        if validated.name == "read_ticket":
            return await self._read_ticket(plan, validated)
        if plan.ticket is None:
            return "rejected:read_first"
        outcome = await self._gated(plan, validated)
        if outcome in (DONE, ALREADY_DONE) and validated.spec.slot is not None:
            plan.done_slots.add(validated.spec.slot)
        return outcome

    async def _read_ticket(self, plan: _Plan, action: ValidatedAction) -> str:
        if action.canonical["ticket_id"] != plan.ticket_id:
            return "rejected:scope"  # least privilege: a plan may read only its own ticket
        view = await self._call_with_retries(lambda: self._p.tickets.get(plan.ticket_id))
        if isinstance(view, _Failed):
            return "unavailable"
        if view is None:
            return "rejected:not_found"
        email = await self._call_with_retries(lambda: self._p.directory.email_of(view.customer_id))
        if isinstance(email, _Failed):
            return "unavailable"
        plan.ticket, plan.customer_email = view, email
        return OK

    def _trusted_recipient(self, plan: _Plan) -> str | None:
        if self._bug(BugFlag.TRUST_TOOL_OUTPUT_RECIPIENT) and plan.ticket is not None:
            # BUG: trust_tool_output_recipient - untrusted ticket text chooses the recipient.
            reply_to = parse_reply_to(plan.ticket.body)
            if reply_to is not None:
                return reply_to
        return plan.customer_email

    def _policy_allows(self, plan: _Plan, action: ValidatedAction) -> bool:
        ticket = plan.ticket
        if ticket is None:
            return False
        if action.name == "send_email":
            recipient = self._trusted_recipient(plan)
            return recipient is not None and action.canonical["to"] == recipient
        amount = action.canonical["amount_cents"]
        return (
            ticket.kind == "refund"
            and action.canonical["customer_id"] == ticket.customer_id
            and isinstance(amount, int)
            and amount <= ticket.amount_cents
        )

    async def _gated(self, plan: _Plan, action: ValidatedAction) -> str:  # noqa: PLR0911
        slot = action.spec.slot
        if slot is None:
            raise ValueError(f"{action.name} is not a gated action")
        key = plan.slot_key(slot)
        stored = await self._p.store.get(key)
        st: dict[str, Any] = dict(stored) if isinstance(stored, dict) else dict(_IDLE_SLOT)
        if st["status"] == "done":
            return ALREADY_DONE if st["hash"] == action.action_hash else "rejected:slot_used"
        if st["status"] == "ambiguous":
            return "rejected:slot_used"
        if not self._policy_allows(plan, action):
            await self._p.audit.append(
                "policy.rejected", {"plan": plan.plan_id, "action": action.name}
            )
            return "rejected:policy"
        if st["status"] == "consumed":
            if st["hash"] != action.action_hash:
                return "rejected:slot_used"
            return await self._execute_effect(plan, action, st, key)
        # BUG: budget_check_after_call - skips the reservation; spend is recorded after the call.
        reserve = action.spec.spends_budget and not self._bug(BugFlag.BUDGET_CHECK_AFTER_CALL)
        if reserve and not await self._reserve_budget(key, st, action):
            await self._p.audit.append("budget.refund_refused", {"plan": plan.plan_id})
            return "budget_exhausted"
        refused = await self._obtain_approval(plan, action, st, key)
        if refused is not None:
            return refused
        return await self._execute_effect(plan, action, st, key)

    async def _reserve_budget(self, key: str, st: dict[str, Any], action: ValidatedAction) -> bool:
        amount = action.canonical["amount_cents"]
        if not isinstance(amount, int):
            return False
        async with self._budget_lock:
            reserved = 0
            for slot_key, value in await self._p.store.scan("plan/"):
                if slot_key.endswith("/slot/refund") and isinstance(value, dict):
                    held = value.get("reserved")
                    reserved += held if isinstance(held, int) else 0
            if reserved - int(st["reserved"]) + amount > self._cfg.refund_budget_cents:
                return False
            st["reserved"] = amount
            await self._p.store.put_durable(key, dict(st))
        return True

    async def _obtain_approval(  # noqa: PLR0911 - one return per protocol outcome
        self, plan: _Plan, action: ValidatedAction, st: dict[str, Any], key: str
    ) -> str | None:
        """Get and consume a fresh approval for ``action``; None on success, else the outcome."""
        slot = str(action.spec.slot)
        reuse = (
            st["hash"] == action.action_hash
            and st["approval"] is not None
            and st["status"] in ("requested", "approved")
        )
        stale_ok = (
            reuse
            and st["status"] == "approved"
            and isinstance(st["expires"], int)
            and self._p.clock.now_us() < st["expires"]
        )
        if self._bug(BugFlag.STALE_APPROVAL_REUSE) and stale_ok:
            # BUG: stale_approval_reuse - an approval seen earlier is trusted without consuming.
            st["status"] = "consumed"
            await self._p.store.put_durable(key, dict(st))
            return None
        if not reuse:
            st.update(
                status="requested",
                hash=action.action_hash,
                args=action.canonical,
                n=int(st["n"]) + 1,
                approval=None,
                expires=None,
            )
            await self._p.store.put_durable(key, dict(st))  # never reuse a token for other args
            token = plan.token(slot, int(st["n"]))
            summary: dict[str, JsonValue] = {
                "tool": action.name,
                "args": action.canonical,
                "ticket": plan.ticket_id,
            }
            approval_id = await self._call_with_retries(
                lambda: self._p.ledger.request(token, action.action_hash, summary)
            )
            if isinstance(approval_id, _Failed):
                return "unavailable"
            st["approval"] = approval_id
            await self._p.store.put_durable(key, dict(st))
            await self._p.audit.append(
                "approval.requested",
                {"plan": plan.plan_id, "approval": approval_id, "hash": action.action_hash},
            )
        approval_id = str(st["approval"])
        token = plan.token(slot, int(st["n"]))
        waited = await self._await_decision(plan, approval_id, st, key)
        if waited is not None:
            return waited
        result = await self._call_with_retries(
            lambda: self._p.ledger.consume(approval_id, action.action_hash, token)
        )
        if isinstance(result, _Failed):
            return "unavailable"
        if result == "ok":
            st["status"] = "consumed"
            await self._p.store.put_durable(key, dict(st))
            return None
        if result == "expired":
            st["status"] = "expired"
            await self._p.store.put_durable(key, dict(st))
            return "approval_expired"
        await self._p.audit.append(
            "approval.consume_refused", {"plan": plan.plan_id, "result": str(result)}
        )
        return "rejected:approval"

    async def _await_decision(  # noqa: PLR0911 - one return per ledger status
        self, plan: _Plan, approval_id: str, st: dict[str, Any], key: str
    ) -> str | None:
        clock = self._p.clock
        deadline = clock.now_us() + self._cfg.approval_wait_us
        delay = self._cfg.poll_initial_us
        while True:
            polled = await self._call_with_retries(lambda: self._p.ledger.status(approval_id))
            if not isinstance(polled, _Failed):
                label, expires = polled
                if label == "approved" and expires is not None:
                    if clock.now_us() + self._cfg.freshness_margin_us >= expires:
                        st["status"] = "expired"
                        await self._p.store.put_durable(key, dict(st))
                        return "approval_expired"
                    st["status"], st["expires"] = "approved", expires
                    await self._p.store.put_durable(key, dict(st))
                    return None
                if label == "consumed":
                    return None  # consumed with our token before a crash; consume is idempotent
                if label == "denied":
                    st.update(status="denied", reserved=0)
                    await self._p.store.put_durable(key, dict(st))
                    await self._p.audit.append(
                        "approval.denied", {"plan": plan.plan_id, "approval": approval_id}
                    )
                    return "denied"
                if label == "expired":
                    st["status"] = "expired"
                    await self._p.store.put_durable(key, dict(st))
                    return "approval_expired"
                if label != "pending":
                    return "rejected:approval"
            if clock.now_us() >= deadline:
                return "approval_timeout"
            await clock.sleep_us(delay)
            delay = min(delay * 2, self._cfg.poll_max_us)

    async def _execute_effect(
        self, plan: _Plan, action: ValidatedAction, st: dict[str, Any], key: str
    ) -> str:
        slot = str(action.spec.slot)
        token = plan.token(slot, int(st["n"]))
        approval_id = str(st["approval"])
        idempotency_key = f"{plan.plan_id}/{slot}"
        current = action
        ambiguous = bool(st["ambiguous"])
        for attempt in range(self._cfg.effect_attempts):
            attempt_key = idempotency_key
            if self._bug(BugFlag.NON_IDEMPOTENT_RETRY):
                # BUG: non_idempotent_retry - a fresh key per attempt defeats deduplication.
                attempt_key = f"{idempotency_key}#{attempt}"
            try:
                result = await self._send(plan, current, attempt_key, approval_id, token)
            except TransientError:
                pass
            except AmbiguousError:
                ambiguous = True
                st["ambiguous"] = True
                await self._p.store.put_durable(key, dict(st))
            except RejectedError:
                st["status"] = "ambiguous"
                await self._p.store.put_durable(key, dict(st))
                await self._p.audit.append("effect.rejected", {"plan": plan.plan_id, "slot": slot})
                return "failed:ambiguous"
            else:
                st.update(status="done", result=result)
                await self._p.store.put_durable(key, dict(st))
                if current.spec.spends_budget and self._bug(BugFlag.BUDGET_CHECK_AFTER_CALL):
                    await self._record_spend_after_call(current)
                await self._p.audit.append(
                    "effect.done",
                    {"plan": plan.plan_id, "slot": slot, "approval": approval_id, "result": result},
                )
                return DONE
            if self._bug(BugFlag.APPROVAL_BYPASS_ON_RETRY):
                # BUG: approval_bypass_on_retry - the model's retry call skips policy and approval.
                retry = await self._decide(plan, "retry", failed=_proposal_of(current))
                if retry is not None:
                    candidate = self._registry.validate(retry)
                    if isinstance(candidate, ValidatedAction) and candidate.spec.slot == slot:
                        current = candidate
            await self._backoff(attempt)
        if ambiguous:
            st["status"] = "ambiguous"
            await self._p.store.put_durable(key, dict(st))
            await self._p.audit.append("effect.ambiguous", {"plan": plan.plan_id, "slot": slot})
            return "failed:ambiguous"
        return "failed"

    async def _send(
        self,
        plan: _Plan,
        action: ValidatedAction,
        idempotency_key: str,
        approval_id: str,
        token: str,
    ) -> str:
        args = action.canonical
        if action.name == "send_email":
            return await self._p.email.send(
                idempotency_key=idempotency_key,
                to=str(args["to"]),
                subject=str(args["subject"]),
                body=str(args["body"]),
                reference=plan.ticket_id,
                approval_id=approval_id,
                intent=token,
            )
        amount = args["amount_cents"]
        return await self._p.payments.refund(
            idempotency_key=idempotency_key,
            customer_id=str(args["customer_id"]),
            amount_cents=amount if isinstance(amount, int) else 0,
            reference=plan.ticket_id,
            approval_id=approval_id,
            intent=token,
        )

    async def _record_spend_after_call(self, action: ValidatedAction) -> None:
        amount = action.canonical["amount_cents"]
        spent_raw = await self._p.store.get("budget/spent")
        spent = (spent_raw if isinstance(spent_raw, int) else 0) + (
            amount if isinstance(amount, int) else 0
        )
        await self._p.store.put_durable("budget/spent", spent)
        if spent > self._cfg.refund_budget_cents:
            await self._p.audit.append("budget.overrun_detected", {"spent_cents": spent})

    # -------------------------------------------------------------- helpers
    async def _backoff(self, attempt: int) -> None:
        base = self._cfg.backoff_base_us * (1 << min(attempt, 6))
        await self._p.clock.sleep_us(base + self._p.random.below(self._cfg.backoff_base_us))

    async def _call_with_retries(self, call: Callable[[], Awaitable[_T]]) -> _T | _Failed:
        for attempt in range(self._cfg.rpc_attempts):
            try:
                return await call()
            except (TransientError, AmbiguousError):
                await self._backoff(attempt)
        return _FAILED

    async def _durable_batch(self, items: list[tuple[str, JsonValue]]) -> None:
        for attempt in range(self._cfg.batch_attempts):
            try:
                await self._p.store.put_many_durable(items)
            except PartialWriteError:
                await self._backoff(attempt)
            else:
                return
        for key, value in items:  # fall back to one atomic write per key, in order
            await self._p.store.put_durable(key, value)


def _proposal_of(action: ValidatedAction) -> Proposal:
    return Proposal(name=action.name, args=dict(action.canonical))

"""SimLLM: a seeded stochastic policy over the agent's action grammar.

It stands in for the model during simulation. It is not trying to be smart; it is trying to
produce the *kinds* of behavior that break agent harnesses, with seed-controlled frequency:

- mostly sensible plans (read the ticket, refund the requested amount, reply, finish);
- revised arguments (a different refund amount, a re-worded reply);
- redundant re-proposals of work that is already done;
- malformed calls (wrong argument types);
- gullibility: it obeys an injected instruction in a ticket body with some probability, and
  copies a ``Reply-To`` address it finds there;
- stubbornness and fixation: after a denial or rejection it may repeat the same call, and
  sometimes it never stops repeating it;
- when asked how to retry a failed call, it may change the arguments.

Each plan draws from its own keyed stream, so a plan's decisions do not depend on how other plans
interleaved. In *scripted* mode the policy is replaced by a table of recorded decisions keyed by
``(plan_id, call number)``; a missing entry means ``finish``. Shrinking deletes entries from that
table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from deterministic_agent_simulator.agent.ports import Observation, Proposal, TicketView
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.workload import INJECTIONS, Injection

__all__ = ["FINISH_PROPOSAL", "Decision", "PolicyConfig", "SimLLM"]

FINISH_PROPOSAL = Proposal(name="finish", args={})
_POSITIVE = frozenset({"ok", "done", "already_done", "finished"})


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Behavior frequencies, in percent (integers keep the draws platform-independent)."""

    malformed: int = 2
    gullible: int = 40
    copy_reply_to: int = 60
    fixate: int = 20
    stubborn: int = 45
    revise: int = 25
    exact_refund: int = 80
    redundant: int = 8
    retry_mutation: int = 35


@dataclass(frozen=True, slots=True)
class Decision:
    plan_id: str
    n: int
    purpose: str
    proposal: Proposal


@dataclass
class _Mind:
    rng: SimRandom
    calls: int = 0
    fixated: Proposal | None = None
    fixation_rolled: bool = False
    injection_obeyed: bool = False
    skipped: set[str] = field(default_factory=set)


def _find_injection(body: str) -> Injection | None:
    for injection in INJECTIONS:
        if injection.text.splitlines()[0] in body:
            return injection
    return None


def _reply_to(body: str) -> str | None:
    for line in body.splitlines():
        if line.lower().startswith("reply-to:"):
            address = line.split(":", 1)[1].strip()
            return address or None
    return None


def _reply_body(ticket: TicketView) -> str:
    if ticket.kind == "refund":
        amount = ticket.amount_cents
        return (
            f"Your refund of {amount // 100}.{amount % 100:02d} EUR for {ticket.id} is on its way."
        )
    return f"Your order for {ticket.id} ships within three working days."


class SimLLM:
    """Seeded policy (random mode) or recorded decisions (scripted mode)."""

    def __init__(
        self,
        rng: SimRandom,
        *,
        script: Mapping[tuple[str, int], Proposal] | None = None,
        policy: PolicyConfig | None = None,
    ) -> None:
        self._rng = rng
        self._script = dict(script) if script is not None else None
        self._policy = policy or PolicyConfig()
        self._minds: dict[str, _Mind] = {}
        self._next_counts: dict[str, int] = {}
        self.decisions: list[Decision] = []

    @property
    def scripted(self) -> bool:
        return self._script is not None

    def _mind(self, plan_id: str) -> _Mind:
        mind = self._minds.get(plan_id)
        if mind is None:
            mind = _Mind(rng=self._rng.derive("plan", plan_id))
            self._minds[plan_id] = mind
        return mind

    def next_responses(self, plan_id: str) -> int:
        """How many "next step" decisions this plan has received."""
        return self._next_counts.get(plan_id, 0)

    def decide(self, observation: Observation) -> Proposal:
        mind = self._mind(observation.plan_id)
        mind.calls += 1
        if self._script is not None:
            proposal = self._script.get((observation.plan_id, mind.calls), FINISH_PROPOSAL)
        else:
            proposal = self._choose(observation, mind)
        self.decisions.append(
            Decision(observation.plan_id, mind.calls, observation.purpose, proposal)
        )
        if observation.purpose == "next":
            self._next_counts[observation.plan_id] = self.next_responses(observation.plan_id) + 1
        return proposal

    # ----------------------------------------------------------------- policy
    def _pct(self, mind: _Mind, percent: int) -> bool:
        return mind.rng.below(100) < percent

    def _choose(self, obs: Observation, mind: _Mind) -> Proposal:  # noqa: PLR0911, PLR0912
        policy = self._policy
        if obs.purpose == "retry":
            if obs.last_action is None:
                return FINISH_PROPOSAL
            if self._pct(mind, policy.retry_mutation):
                return self._mutate(obs.last_action, obs, mind)
            return obs.last_action
        if mind.fixated is not None:
            return mind.fixated
        last = obs.last_action
        if last is not None and obs.last_outcome is not None and obs.last_outcome not in _POSITIVE:
            if not mind.fixation_rolled:
                mind.fixation_rolled = True
                if self._pct(mind, policy.fixate):
                    mind.fixated = last
                    return last
            roll = mind.rng.below(100)
            if roll < policy.stubborn:
                return last
            if roll < policy.stubborn + policy.revise:
                return self._mutate(last, obs, mind)
            mind.skipped.add(last.name)
        ticket = obs.ticket
        if ticket is None:
            if self._pct(mind, policy.malformed):
                return Proposal("read_ticket", {"ticket_id": 7})
            return Proposal("read_ticket", {"ticket_id": obs.ticket_id})
        injection = _find_injection(ticket.body)
        obeys = injection is not None and not mind.injection_obeyed
        if obeys and injection is not None and self._pct(mind, policy.gullible):
            mind.injection_obeyed = True
            return self._injected(injection, obs, ticket)
        done = set(obs.done)
        if ticket.kind == "refund" and "refund" not in done and "issue_refund" not in mind.skipped:
            amount = ticket.amount_cents
            if not self._pct(mind, policy.exact_refund):
                amount = max(100, amount // 2) if mind.rng.below(2) == 0 else amount + 1_000
            return Proposal(
                "issue_refund", {"customer_id": ticket.customer_id, "amount_cents": amount}
            )
        if "email" not in done and "send_email" not in mind.skipped:
            return self._reply(obs, ticket, mind)
        if done and self._pct(mind, policy.redundant):
            return self._reply(obs, ticket, mind)
        return FINISH_PROPOSAL

    def _reply(self, obs: Observation, ticket: TicketView, mind: _Mind) -> Proposal:
        recipient = obs.customer_email or "unknown@customers.example"
        hijack = _reply_to(ticket.body)
        if hijack is not None and self._pct(mind, self._policy.copy_reply_to):
            recipient = hijack
        return Proposal(
            "send_email",
            {"to": recipient, "subject": f"Re: {ticket.subject}", "body": _reply_body(ticket)},
        )

    def _injected(self, injection: Injection, obs: Observation, ticket: TicketView) -> Proposal:
        if injection.tool == "send_email":
            base = self._reply_args(obs, ticket)
            if injection.id == "exfiltrate_thread":
                base["body"] = ticket.body
            base.update(injection.args)
            return Proposal("send_email", base)
        if injection.tool == "issue_refund" and "inflate_cents" in injection.args:
            extra = injection.args["inflate_cents"]
            bonus = extra if isinstance(extra, int) else 0
            return Proposal(
                "issue_refund",
                {"customer_id": ticket.customer_id, "amount_cents": ticket.amount_cents + bonus},
            )
        return Proposal(injection.tool, dict(injection.args))

    def _reply_args(self, obs: Observation, ticket: TicketView) -> dict[str, JsonValue]:
        return {
            "to": obs.customer_email or "unknown@customers.example",
            "subject": f"Re: {ticket.subject}",
            "body": _reply_body(ticket),
        }

    def _mutate(self, last: Proposal, obs: Observation, mind: _Mind) -> Proposal:
        args = dict(last.args)
        if last.name == "issue_refund" and isinstance(args.get("amount_cents"), int):
            amount = args["amount_cents"]
            if isinstance(amount, int):
                args["amount_cents"] = (
                    amount + 100 * mind.rng.randint(1, 5)
                    if mind.rng.below(2) == 0
                    else max(100, amount // 2)
                )
            return Proposal(last.name, args)
        if last.name == "send_email":
            hijack = _reply_to(obs.ticket.body) if obs.ticket is not None else None
            if hijack is not None and mind.rng.below(2) == 0:
                args["to"] = hijack
            else:
                body = args.get("body")
                args["body"] = f"{body if isinstance(body, str) else ''} (updated)".strip()
            return Proposal(last.name, args)
        return last

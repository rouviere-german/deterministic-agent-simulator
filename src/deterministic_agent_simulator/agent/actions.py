"""The reference agent's action grammar and its least-privilege registry.

The model may propose ``read_ticket``, ``send_email``, ``issue_refund`` or ``finish``. Anything
else (``export_customers`` from an injected instruction, say) is not in the registry and is
refused, and every argument object is validated in strict mode against a closed schema
(``extra="forbid"``), so an injected ``"approved": true`` field is an error rather than a hint.

The registry is built from the capabilities granted to the agent: a registry granted only
``READ_DATA`` does not even know ``send_email``. Side-effecting actions are *gated*: they need a
fresh, consumed approval for the exact canonical arguments (their ``action_hash``), and each
occupies a per-ticket *slot* (one reply e-mail and at most one refund per ticket).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from deterministic_agent_simulator.agent.ports import Proposal
from deterministic_agent_simulator.agentic.providers.base import ToolSpec
from deterministic_agent_simulator.agentic.tools import Capability, tool_call_hash

__all__ = [
    "ACTION_SPECS",
    "FINISH",
    "ActionRegistry",
    "ActionSpec",
    "IssueRefundArgs",
    "ReadTicketArgs",
    "Rejection",
    "SendEmailArgs",
    "ValidatedAction",
]

FINISH = "finish"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadTicketArgs(_Args):
    ticket_id: str = Field(pattern=r"^T-[0-9]{1,6}$")


class SendEmailArgs(_Args):
    to: str = Field(pattern=r"^[^@\s]{1,64}@[^@\s]{1,190}$")
    subject: str = Field(min_length=1, max_length=160)
    body: str = Field(min_length=1, max_length=2000)


class IssueRefundArgs(_Args):
    customer_id: str = Field(pattern=r"^c-[a-z0-9-]{1,32}$")
    amount_cents: int = Field(ge=1, le=10_000_000)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    description: str
    args_model: type[_Args]
    capability: Capability
    slot: str | None = None
    spends_budget: bool = False

    @property
    def gated(self) -> bool:
        return self.slot is not None

    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )


ACTION_SPECS: Mapping[str, ActionSpec] = {
    spec.name: spec
    for spec in (
        ActionSpec(
            name="read_ticket",
            description="Read the ticket this plan is working on.",
            args_model=ReadTicketArgs,
            capability=Capability.READ_DATA,
        ),
        ActionSpec(
            name="send_email",
            description="Send the reply e-mail for this ticket (needs human approval).",
            args_model=SendEmailArgs,
            capability=Capability.SEND_MESSAGE,
            slot="email",
        ),
        ActionSpec(
            name="issue_refund",
            description="Refund the ticket's customer (needs human approval and budget).",
            args_model=IssueRefundArgs,
            capability=Capability.WRITE_DATA,
            slot="refund",
            spends_budget=True,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class ValidatedAction:
    spec: ActionSpec
    args: _Args
    canonical: dict[str, JsonValue]
    action_hash: str

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True, slots=True)
class Rejection:
    code: str
    detail: str


class ActionRegistry:
    """Actions available to one agent, limited to the capabilities it was granted."""

    def __init__(self, granted: Iterable[Capability]) -> None:
        self._granted = frozenset(granted)
        self._specs = {
            name: spec for name, spec in ACTION_SPECS.items() if spec.capability in self._granted
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs[name].tool_spec() for name in self.names)

    def validate(self, proposal: Proposal) -> ValidatedAction | Rejection:
        spec = self._specs.get(proposal.name)
        if spec is None:
            return Rejection("not_allowed", f"action {proposal.name!r} is not available")
        try:
            args = spec.args_model.model_validate(proposal.args, strict=True)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in err["loc"]) or "$" for err in exc.errors()})
            return Rejection("invalid_args", f"invalid arguments at: {', '.join(fields)}")
        canonical: dict[str, JsonValue] = args.model_dump(mode="json")
        return ValidatedAction(spec, args, canonical, tool_call_hash(spec.name, canonical))

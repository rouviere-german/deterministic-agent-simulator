"""The eight plantable bugs of the reference system, and the oracle each one should trip.

Each flag switches one control of the reference agent (or of its approval ledger) to a
plausible-looking but wrong implementation. With every flag off the system is meant to be
correct, and the soundness campaign checks that no oracle fires. With one flag on, the
sensitivity campaign measures how many seeds it takes to find the bug.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["BUG_CATALOG", "BugFlag", "BugInfo", "parse_bug_flags"]


class BugFlag(StrEnum):
    NON_IDEMPOTENT_RETRY = "non_idempotent_retry"
    BUDGET_CHECK_AFTER_CALL = "budget_check_after_call"
    APPROVAL_BYPASS_ON_RETRY = "approval_bypass_on_retry"
    STALE_APPROVAL_REUSE = "stale_approval_reuse"
    INFINITE_REPLAN_LOOP = "infinite_replan_loop"
    NON_DURABLE_CONSUME_FLAG = "non_durable_consume_flag"
    MISSING_DEDUPE = "missing_dedupe"
    TRUST_TOOL_OUTPUT_RECIPIENT = "trust_tool_output_recipient"


@dataclass(frozen=True, slots=True)
class BugInfo:
    flag: BugFlag
    component: str
    wrong: str
    right: str
    expected_invariant: str
    trigger: str


BUG_CATALOG: Mapping[BugFlag, BugInfo] = {
    info.flag: info
    for info in (
        BugInfo(
            BugFlag.NON_IDEMPOTENT_RETRY,
            "agent: effect execution",
            "each retry of a failed e-mail or refund uses a fresh idempotency key",
            "every attempt for one ticket slot reuses the same idempotency key",
            "no_duplicate_effects",
            "a timeout after the service committed the first attempt",
        ),
        BugInfo(
            BugFlag.BUDGET_CHECK_AFTER_CALL,
            "agent: refund budget",
            "the refund is issued first; spend is checked and recorded only after a success",
            "the amount is reserved durably, under a lock, before approval and execution",
            "budget_respected",
            "refunds that together exceed the budget, or a refund that timed out after commit",
        ),
        BugInfo(
            BugFlag.APPROVAL_BYPASS_ON_RETRY,
            "agent: retry path",
            "after a failed call the model is asked how to retry and its call runs ungated",
            "retries repeat exactly the approved call; anything else goes back through the gate",
            "approval_fresh",
            "a failed e-mail or refund followed by a model that changes the arguments",
        ),
        BugInfo(
            BugFlag.STALE_APPROVAL_REUSE,
            "agent: resume path",
            "an approval seen as approved earlier is reused without consuming it at the ledger",
            "every execution consumes its approval at the ledger (consume-once, TTL enforced)",
            "approval_fresh",
            "a crash or failed consume between seeing the approval and consuming it",
        ),
        BugInfo(
            BugFlag.INFINITE_REPLAN_LOOP,
            "agent: planning loop",
            "after a denial or rejection the agent re-plans without any cap",
            "re-plans and steps per ticket are capped; the ticket is escalated instead",
            "termination",
            "a denial or rejection followed by a model that keeps proposing the same action",
        ),
        BugInfo(
            BugFlag.NON_DURABLE_CONSUME_FLAG,
            "approval ledger: consume",
            "consume is acknowledged before the consumed flag is flushed to durable storage",
            "consume is acknowledged only after the consumed state is durable",
            "ledger_linearizable",
            "a ledger crash between a consume and the next flush, observed by any client",
        ),
        BugInfo(
            BugFlag.MISSING_DEDUPE,
            "agent: intake",
            "every delivery of a ticket event starts a new plan",
            "deliveries of a ticket that is active or done are acknowledged and dropped",
            "no_duplicate_effects",
            "the inbound queue delivering the same ticket event twice",
        ),
        BugInfo(
            BugFlag.TRUST_TOOL_OUTPUT_RECIPIENT,
            "agent: recipient policy",
            "a Reply-To address found in the ticket body is trusted as the recipient",
            "the recipient must be the customer's address from the trusted directory",
            "authorized_effects",
            "an injected Reply-To line and a model that copies it",
        ),
    )
}


def parse_bug_flags(values: Iterable[str]) -> frozenset[BugFlag]:
    """Parse flag names (``all`` selects every flag); unknown names raise ValueError."""
    flags: set[BugFlag] = set()
    for raw in values:
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            flags.update(BugFlag)
            continue
        try:
            flags.add(BugFlag(name))
        except ValueError:
            known = ", ".join(flag.value for flag in BugFlag)
            raise ValueError(f"unknown bug flag {name!r}; known flags: {known}") from None
    return frozenset(flags)

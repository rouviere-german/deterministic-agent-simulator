"""World assembly: services, faults, humans, the supervised agent process, and the oracles.

``World.main`` is the coroutine the simulation loop runs. It schedules ticket arrivals, starts the
auditor (and the ledger's flush task when the ledger buffers consumes), and supervises the agent:
the ledger may crash right after any commit (a buggify point) and restarts later. When an
incarnation crashes, its buffered writes are lost, its unacknowledged deliveries are requeued and
a new incarnation starts after a restart delay (possibly with a skewed clock). The run ends when
the inbound queue is drained, when an oracle halts it, or at the virtual deadline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pydantic import JsonValue

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agent.ports import AgentCrash, AgentPorts
from deterministic_agent_simulator.agent.runtime import AgentConfig, ReferenceAgent
from deterministic_agent_simulator.check.oracles import OracleSuite
from deterministic_agent_simulator.sim.faults import FaultKind, FaultPlan
from deterministic_agent_simulator.sim.loop import HermeticityViolation, SimEventLoop
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.adapters import (
    AgentProcess,
    SimAudit,
    SimClock,
    SimDirectory,
    SimEmail,
    SimInbox,
    SimLedger,
    SimModel,
    SimPayments,
    SimRandomPort,
    SimStore,
    SimTickets,
)
from deterministic_agent_simulator.world.approver import Auditor, HumanReviewer
from deterministic_agent_simulator.world.inbound import InboundQueue
from deterministic_agent_simulator.world.ledger import (
    ApprovalLedger,
    ConsumeFact,
    History,
    LedgerClient,
)
from deterministic_agent_simulator.world.net import LatencyModel, Network
from deterministic_agent_simulator.world.services import (
    Directory,
    Effect,
    EmailOutbox,
    Payments,
    TicketService,
)
from deterministic_agent_simulator.world.simllm import SimLLM
from deterministic_agent_simulator.world.store import AgentStore, AuditChain
from deterministic_agent_simulator.world.workload import Workload

__all__ = ["World", "WorldConfig"]

AGENT_PRINCIPAL = "agent:reference"
REVIEWER_PRINCIPAL = "human:reviewer"
AUDITOR_PRINCIPAL = "human:auditor"


@dataclass(frozen=True, slots=True)
class WorldConfig:
    approval_ttl_us: int = 2_500_000
    max_agent_crashes: int = 3
    restart_delay_us: tuple[int, int] = (500_000, 3_000_000)
    clock_skew_us: tuple[int, int] = (100_000, 1_500_000)
    max_ledger_crashes: int = 3
    ledger_downtime_us: tuple[int, int] = (200_000, 800_000)
    ledger_flush_interval_us: int = 300_000
    deadline_us: int = 1_800_000_000
    latency: LatencyModel = field(default_factory=LatencyModel)


class World:
    """Everything outside the agent, wired to one seed."""

    def __init__(
        self,
        *,
        loop: SimEventLoop,
        rng: SimRandom,
        faults: FaultPlan,
        workload: Workload,
        model: SimLLM,
        agent_config: AgentConfig,
        config: WorldConfig,
        fail_fast: bool = True,
    ) -> None:
        self.loop = loop
        self.rng = rng
        self.faults = faults
        self.workload = workload
        self.model = model
        self.agent_config = agent_config
        self.config = config
        self.network = Network(loop, faults, rng.derive("network"), config.latency)
        self.history = History()
        self.ledger = ApprovalLedger(
            loop,
            ttl_us=config.approval_ttl_us,
            durable_consume=BugFlag.NON_DURABLE_CONSUME_FLAG not in agent_config.bugs,
            on_consume=self._on_consume,
        )
        decision_bound = agent_config.max_steps_per_ticket + config.max_agent_crashes
        self.oracles = OracleSuite(
            clock=lambda: loop.now_us,
            trace=loop.trace,
            workload=workload,
            ledger=self.ledger,
            decisions_of=model.next_responses,
            decision_bound=decision_bound,
            halt=loop.stop if fail_fast else None,
        )
        effect_counter = [0]
        self.effects: list[Effect] = []
        self.tickets = TicketService({ticket.id: ticket for ticket in workload.tickets})
        self.directory = Directory(workload.directory)
        self.outbox = EmailOutbox(lambda: loop.now_us, self._on_effect, effect_counter)
        self.payments = Payments(lambda: loop.now_us, self._on_effect, effect_counter)
        self.store = AgentStore()
        self.chain = AuditChain()
        self.inbox = InboundQueue(loop, faults, rng.derive("inbound"))
        self.process = AgentProcess(loop, faults, max_crashes=config.max_agent_crashes)
        self.reviewer = HumanReviewer(
            loop,
            self._ledger_client("reviewer", REVIEWER_PRINCIPAL),
            history=self.history,
            rng=rng.derive("reviewer"),
            faults=faults,
            compliant=self._compliant,
        )
        self.auditor = Auditor(
            loop,
            self._ledger_client("auditor", AUDITOR_PRINCIPAL),
            faults,
        )
        self.ledger.add_listener(self.reviewer.on_request)
        self.ledger.add_listener(self.auditor.on_request)
        self.stopping = False
        self.finished = False
        self.end_reason: str | None = None
        self.agent_errors: list[str] = []
        self.incarnations: list[tuple[int, int]] = []

    def _ledger_client(self, name: str, principal: str) -> LedgerClient:
        return LedgerClient(
            name,
            principal,
            ledger=self.ledger,
            network=self.network,
            history=self.history,
            loop=self.loop,
            after_commit=self._ledger_crash_point,
        )

    # ------------------------------------------------------------- effects
    def _on_consume(self, fact: ConsumeFact) -> None:
        self.oracles.on_consume(fact)

    def _on_effect(self, effect: Effect) -> None:
        self.effects.append(effect)
        self.loop.trace.record(
            self.loop.now_us, "effect", effect.tool, effect.reference, effect.idempotency_key
        )
        self.oracles.on_effect(effect)

    def _compliant(self, summary: dict[str, JsonValue]) -> bool:
        """What a careful reviewer would conclude from the approval request alone."""
        ticket_id = summary.get("ticket")
        args = summary.get("args")
        ticket = self.workload.ticket(ticket_id) if isinstance(ticket_id, str) else None
        if ticket is None or not isinstance(args, dict):
            return False
        if summary.get("tool") == "send_email":
            return args.get("to") == self.workload.directory.get(ticket.customer_id)
        amount = args.get("amount_cents")
        return (
            args.get("customer_id") == ticket.customer_id
            and isinstance(amount, int)
            and amount <= ticket.amount_cents
        )

    # -------------------------------------------------------------- agent
    def _ports(self, incarnation: int) -> AgentPorts:
        skew = 0
        if self.faults.fires(FaultKind.CLOCK_SKEW, "agent.clock", f"inc{incarnation}"):
            draw = self.faults.draw("agent.clock", f"inc{incarnation}")
            magnitude = draw.randint(*self.config.clock_skew_us)
            skew = magnitude if draw.below(2) == 0 else -magnitude
        self.incarnations.append((incarnation, skew))
        loop, net, process = self.loop, self.network, self.process
        ledger_client = self._ledger_client("agent", AGENT_PRINCIPAL)
        return AgentPorts(
            clock=SimClock(loop, skew),
            random=SimRandomPort(self.rng.derive("agent-random", incarnation)),
            llm=SimModel(
                loop,
                net,
                self.faults,
                process,
                self.model,
                on_attempt=self.oracles.on_llm_attempt,
                on_decision=self.oracles.on_decision,
            ),
            tickets=SimTickets(net, process, self.tickets),
            directory=SimDirectory(net, process, self.directory),
            email=SimEmail(net, process, self.outbox),
            payments=SimPayments(net, process, self.payments),
            ledger=SimLedger(ledger_client, process),
            store=SimStore(
                loop, self.faults, process, self.store, self.rng.derive("disk", incarnation)
            ),
            audit=SimAudit(loop, process, self.chain, AGENT_PRINCIPAL, self.oracles.on_audit),
            inbox=SimInbox(process, self.inbox),
        )

    async def _supervise(self) -> None:
        while True:
            incarnation = self.process.revive()
            ports = self._ports(incarnation)
            self.loop.trace.record(self.loop.now_us, "agent.start", incarnation)
            agent = ReferenceAgent(ports, self.agent_config, incarnation=incarnation)
            try:
                await agent.run()
            except AgentCrash:
                self.process.dead = True
                self.store.crash()
                self.inbox.requeue_inflight()
                delay = self.faults.draw("agent.restart", f"inc{incarnation}").randint(
                    *self.config.restart_delay_us
                )
                await self.loop.sleep_us(delay)
                continue
            except HermeticityViolation as exc:
                self.agent_errors.append(f"hermeticity: {exc}")
                self.oracles.violate("hermetic", str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - any agent exception is a finding
                self.agent_errors.append(f"{type(exc).__name__}: {exc}")
                self.oracles.violate("agent_error", f"{type(exc).__name__}: {exc}")
                return
            if self.inbox.drained:
                return
            self.agent_errors.append("agent stopped before the inbox was drained")
            return

    # -------------------------------------------------------- ledger crashes
    def _ledger_crash_point(self, op: str, scope: str) -> None:
        """Buggify point right after a ledger commit: the ledger may crash and restart later."""
        if self.ledger.crashes >= self.config.max_ledger_crashes or not self.ledger.up:
            return
        if not self.faults.fires(FaultKind.LEDGER_CRASH, f"ledger.{op}", scope):
            return
        self.ledger.crash()
        self.loop.trace.record(self.loop.now_us, "ledger.crash", op, scope)
        downtime = self.faults.draw("ledger.downtime", str(self.ledger.crashes)).randint(
            *self.config.ledger_downtime_us
        )
        self.loop.call_at_us(self.loop.now_us + downtime, self._ledger_restart)

    def _ledger_restart(self) -> None:
        self.ledger.restart()
        self.loop.trace.record(self.loop.now_us, "ledger.restart", self.ledger.crashes)

    async def _ledger_flusher(self) -> None:
        while not self.stopping:
            await self.loop.sleep_us(self.config.ledger_flush_interval_us)
            if self.ledger.up:
                self.ledger.flush()

    # ---------------------------------------------------------------- main
    async def main(self) -> None:
        self.inbox.schedule(self.workload.arrivals)
        background = [
            self.loop.create_task(self.auditor.run(lambda: self.stopping), name="auditor"),
        ]
        if not self.ledger.durable_consume:
            background.append(self.loop.create_task(self._ledger_flusher(), name="flusher"))
        try:
            async with asyncio.timeout(self.config.deadline_us / 1_000_000):
                await self._supervise()
            self.finished = self.inbox.drained and not self.agent_errors
            if not self.finished:
                self.end_reason = "; ".join(self.agent_errors) or "the inbox was not drained"
        except TimeoutError:
            self.end_reason = f"the run missed its deadline of {self.config.deadline_us}us"
        finally:
            self.stopping = True
            for task in [*background, *self.reviewer.tasks]:
                task.cancel()
            await asyncio.gather(*background, *self.reviewer.tasks, return_exceptions=True)

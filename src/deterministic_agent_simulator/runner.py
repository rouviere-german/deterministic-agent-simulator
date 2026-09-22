"""Run one scenario, or a campaign of seeds, and report what the oracles found.

A :class:`Scenario` fully determines a run: the seed, the configuration (bug flags, fault mode,
scheduler), and optionally a ticket subset, a fault schedule and a decision script. Without a
schedule and a script the run *explores*: faults and model decisions are drawn from the seed.
:meth:`RunResult.replay_scenario` turns an exploring run into a fully scripted scenario that
reproduces it exactly (the same trace hash); shrinking starts from there.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agent.ports import Proposal
from deterministic_agent_simulator.agent.runtime import AgentConfig
from deterministic_agent_simulator.check.oracles import Violation
from deterministic_agent_simulator.sim.faults import FaultKey, FaultPlan, FaultProfile
from deterministic_agent_simulator.sim.hermetic import HermeticGuard
from deterministic_agent_simulator.sim.loop import (
    SimDeadlockError,
    SimEventLoop,
    SimHaltedError,
    SimStepLimitError,
)
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.simllm import Decision, SimLLM
from deterministic_agent_simulator.world.store import AuditChain
from deterministic_agent_simulator.world.workload import WorkloadConfig, generate_workload
from deterministic_agent_simulator.world.world import World, WorldConfig

__all__ = [
    "CampaignReport",
    "FaultMode",
    "RunResult",
    "Scenario",
    "SimConfig",
    "find_failure",
    "run_campaign",
    "run_scenario",
]

FaultMode = Literal["swarm", "fixed", "none"]
Scheduler = Literal["random", "fifo"]
_TEARDOWN_STEPS = 200_000


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Everything besides the seed that shapes a run."""

    bugs: frozenset[BugFlag] = frozenset()
    fault_mode: FaultMode = "swarm"
    scheduler: Scheduler = "random"
    workers: int = 3
    min_tickets: int = 5
    max_tickets: int = 10
    max_steps_per_ticket: int = 12
    max_loop_steps: int = 600_000

    def to_json(self) -> dict[str, JsonValue]:
        bugs: list[JsonValue] = [flag.value for flag in sorted(self.bugs)]
        return {
            "bugs": bugs,
            "fault_mode": self.fault_mode,
            "scheduler": self.scheduler,
            "workers": self.workers,
            "min_tickets": self.min_tickets,
            "max_tickets": self.max_tickets,
            "max_steps_per_ticket": self.max_steps_per_ticket,
            "max_loop_steps": self.max_loop_steps,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SimConfig:
        mode = str(data.get("fault_mode", "swarm"))
        scheduler = str(data.get("scheduler", "random"))
        if mode not in ("swarm", "fixed", "none") or scheduler not in ("random", "fifo"):
            raise ValueError("unknown fault mode or scheduler")
        return cls(
            bugs=frozenset(BugFlag(name) for name in data.get("bugs", [])),
            fault_mode="swarm" if mode == "swarm" else "fixed" if mode == "fixed" else "none",
            scheduler="random" if scheduler == "random" else "fifo",
            workers=int(data.get("workers", 3)),
            min_tickets=int(data.get("min_tickets", 5)),
            max_tickets=int(data.get("max_tickets", 10)),
            max_steps_per_ticket=int(data.get("max_steps_per_ticket", 12)),
            max_loop_steps=int(data.get("max_loop_steps", 600_000)),
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    seed: int
    config: SimConfig = field(default_factory=SimConfig)
    tickets: tuple[int, ...] | None = None
    faults: tuple[FaultKey, ...] | None = None
    decisions: tuple[Decision, ...] | None = None

    @property
    def scripted(self) -> bool:
        return self.faults is not None and self.decisions is not None


@dataclass
class RunResult:
    scenario: Scenario
    trace_hash: str
    trace_length: int
    schedule_points: int
    violations: list[Violation]
    fault_profile: list[str]
    fired_faults: tuple[FaultKey, ...]
    decisions: tuple[Decision, ...]
    ticket_indices: tuple[int, ...]
    virtual_time_us: int
    loop_steps: int
    effects: int
    llm_calls: int
    agent_crashes: int
    ledger_crashes: int
    history_ops: int
    end_reason: str | None
    trace: Trace | None = None
    chain: AuditChain | None = None

    @property
    def violated(self) -> bool:
        return bool(self.violations)

    @property
    def first(self) -> Violation | None:
        return self.violations[0] if self.violations else None

    def replay_scenario(self) -> Scenario:
        """A scripted scenario that reproduces this run exactly."""
        return replace(
            self.scenario,
            tickets=self.ticket_indices,
            faults=self.fired_faults,
            decisions=self.decisions,
        )

    def summary(self) -> dict[str, JsonValue]:
        return {
            "seed": self.scenario.seed,
            "trace_hash": self.trace_hash,
            "trace_length": self.trace_length,
            "violations": [
                {"invariant": v.invariant, "detail": v.detail, "t_us": v.t_us}
                for v in self.violations
            ],
            "fault_profile": list(self.fault_profile),
            "faults_fired": len(self.fired_faults),
            "decisions": len(self.decisions),
            "tickets": len(self.ticket_indices),
            "virtual_time_us": self.virtual_time_us,
            "loop_steps": self.loop_steps,
            "effects": self.effects,
            "llm_calls": self.llm_calls,
            "agent_crashes": self.agent_crashes,
            "ledger_crashes": self.ledger_crashes,
            "end_reason": self.end_reason,
        }


def _profile(config: SimConfig, rng: SimRandom) -> FaultProfile:
    if config.fault_mode == "fixed":
        return FaultProfile.fixed()
    if config.fault_mode == "none":
        return FaultProfile.none()
    return FaultProfile.swarm(rng)


def _teardown(loop: SimEventLoop) -> None:
    """Cancel what is still running, deterministically, with oracles and trace already frozen."""
    pending = loop.pending_tasks()
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.extend_step_limit(_TEARDOWN_STEPS)
    try:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except (SimDeadlockError, SimStepLimitError, SimHaltedError):
        return


def run_scenario(
    scenario: Scenario,
    *,
    keep_trace: bool = False,
    fail_fast: bool = True,
    hermetic: bool = True,
    allowed_roots: Sequence[Path] | None = None,
) -> RunResult:
    """Execute one scenario to completion (or to its first violation when ``fail_fast``)."""
    config = scenario.config
    root = SimRandom.from_seed(scenario.seed)
    trace = Trace(keep=keep_trace)
    scheduler_rng = root.derive("scheduler") if config.scheduler == "random" else None
    loop = SimEventLoop(rng=scheduler_rng, trace=trace, max_steps=config.max_loop_steps)
    profile = _profile(config, root.derive("swarm"))
    faults = FaultPlan(
        profile, root.derive("faults"), trace, lambda: loop.now_us, schedule=scenario.faults
    )
    workload_config = WorkloadConfig(min_tickets=config.min_tickets, max_tickets=config.max_tickets)
    workload = generate_workload(root.derive("workload"), faults, workload_config, scenario.tickets)
    script = (
        {(d.plan_id, d.n): d.proposal for d in scenario.decisions}
        if scenario.decisions is not None
        else None
    )
    model = SimLLM(root.derive("llm"), script=script)
    agent_config = AgentConfig(
        refund_budget_cents=workload.refund_budget_cents,
        llm_call_budget=workload.llm_call_budget,
        workers=config.workers,
        max_steps_per_ticket=config.max_steps_per_ticket,
        bugs=config.bugs,
    )
    world = World(
        loop=loop,
        rng=root.derive("world"),
        faults=faults,
        workload=workload,
        model=model,
        agent_config=agent_config,
        config=WorldConfig(),
        fail_fast=fail_fast,
    )
    roots = list(allowed_roots) if allowed_roots is not None else [Path(tempfile.gettempdir())]
    guard = HermeticGuard(roots)
    context: AbstractContextManager[Any] = guard if hermetic else nullcontext()
    reason: str | None = None
    halted = False
    with context:
        try:
            loop.run_until_complete(world.main())
            reason = world.end_reason
        except SimHaltedError:
            halted = True
        except SimDeadlockError as exc:
            reason = f"deadlock: {exc}"
            world.finished = False
        except SimStepLimitError as exc:
            reason = f"livelock: {exc}"
            world.finished = False
        if not (halted and fail_fast):
            world.oracles.final(
                finished=world.finished,
                reason=reason,
                history=world.history,
                chain=world.chain,
                hermetic_violations=list(guard.violations),
                agent_errors=list(loop.callback_errors),
            )
        trace_hash = trace.hexdigest()
        trace.freeze()
        world.oracles.enabled = False
        world.stopping = True
        _teardown(loop)
    loop.close()
    kept = tuple(sorted(int(ticket.id.split("-", 1)[1]) - 1 for ticket in workload.tickets))
    return RunResult(
        scenario=scenario,
        trace_hash=trace_hash,
        trace_length=trace.length,
        schedule_points=trace.schedule_points,
        violations=list(world.oracles.violations),
        fault_profile=profile.enabled(),
        fired_faults=tuple(faults.fired),
        decisions=tuple(model.decisions),
        ticket_indices=kept,
        virtual_time_us=loop.now_us,
        loop_steps=loop.steps,
        effects=len(world.effects),
        llm_calls=world.oracles.llm_calls,
        agent_crashes=world.process.crashes,
        ledger_crashes=world.ledger.crashes,
        history_ops=len(world.history.ops),
        end_reason=reason,
        trace=trace if keep_trace else None,
        chain=world.chain if keep_trace else None,
    )


@dataclass
class CampaignReport:
    config: SimConfig
    start: int
    seeds_run: int = 0
    wall_seconds: float = 0.0
    violations: Counter[str] = field(default_factory=Counter)
    failing: list[tuple[int, str]] = field(default_factory=list)
    stopped: str = "completed"
    trace_hashes: dict[int, str] = field(default_factory=dict)

    @property
    def seeds_per_second(self) -> float:
        return self.seeds_run / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "config": self.config.to_json(),
            "start": self.start,
            "seeds_run": self.seeds_run,
            "wall_seconds": round(self.wall_seconds, 3),
            "seeds_per_second": round(self.seeds_per_second, 2),
            "violations": dict(sorted(self.violations.items())),
            "failing_seeds": [[seed, name] for seed, name in self.failing[:100]],
            "stopped": self.stopped,
        }


def run_campaign(
    config: SimConfig,
    *,
    seeds: Iterable[int],
    max_seeds: int,
    max_wall_seconds: float,
    stop_on_first: bool = False,
    record_hashes: bool = False,
    on_result: Callable[[RunResult], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CampaignReport:
    """Run seeds until they run out, ``max_seeds`` or ``max_wall_seconds`` (campaign caps)."""
    if max_seeds < 1 or max_wall_seconds <= 0:
        raise ValueError("campaign caps must be positive")
    report = CampaignReport(config=config, start=-1)
    started = clock()
    for seed in seeds:
        if report.seeds_run >= max_seeds:
            report.stopped = "max_seeds"
            break
        if clock() - started >= max_wall_seconds:
            report.stopped = "max_wall_seconds"
            break
        if report.start < 0:
            report.start = seed
        result = run_scenario(Scenario(seed=seed, config=config))
        report.seeds_run += 1
        if record_hashes:
            report.trace_hashes[seed] = result.trace_hash
        if on_result is not None:
            on_result(result)
        if result.violated and result.first is not None:
            report.violations[result.first.invariant] += 1
            report.failing.append((seed, result.first.invariant))
            if stop_on_first:
                report.stopped = "first_failure"
                break
    report.wall_seconds = clock() - started
    return report


def find_failure(config: SimConfig, *, start: int, max_seeds: int) -> tuple[int, RunResult] | None:
    """The first seed in ``[start, start + max_seeds)`` whose run violates an invariant."""
    for seed in range(start, start + max_seeds):
        result = run_scenario(Scenario(seed=seed, config=config))
        if result.violated:
            return seed, result
    return None


def proposals_from_json(items: Iterable[dict[str, Any]]) -> tuple[Decision, ...]:
    """Decode recorded decisions (the ``decisions`` array of a replay file)."""
    return tuple(
        Decision(
            plan_id=str(item["plan"]),
            n=int(item["n"]),
            purpose=str(item.get("purpose", "next")),
            proposal=Proposal.from_json(item["proposal"]),
        )
        for item in items
    )

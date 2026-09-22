"""Failure shrinking: ddmin over the workload, the fault schedule and the decision script.

``ddmin`` is the minimizing delta-debugging algorithm of Zeller and Hildebrandt ("Simplifying and
Isolating Failure-Inducing Input", IEEE TSE 2002). Given a failing configuration and a test that
says whether a sub-configuration still fails, it returns a *1-minimal* one: removing any single
remaining element makes the failure disappear (when the test budget allows it to finish).

A failing run is first converted into a fully scripted scenario (its fired faults and recorded
model decisions), which reproduces it exactly. The shrinker then applies ddmin hierarchically, in
the spirit of Misherghi and Su's HDD (ICSE 2006): first to the tickets of the workload (removing a
ticket removes all of its decisions and faults), then to the fault schedule, then to the model
decisions, and repeats while anything shrinks. Decisions form one action sequence per plan:
deleting a decision shifts the plan's later decisions forward (the model is asked one step
fewer), and a plan that runs out of script finishes. A candidate
"still fails" only if its first violation is the *same invariant* as the original's, so the
shrinker cannot slip from one bug to another.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from deterministic_agent_simulator.runner import RunResult, Scenario, run_scenario
from deterministic_agent_simulator.world.adapters import ticket_scope
from deterministic_agent_simulator.world.simllm import Decision

__all__ = ["ShrinkBudgetExceeded", "ShrinkResult", "ddmin", "renumber", "shrink_failure"]


class ShrinkBudgetExceeded(Exception):  # noqa: N818 - control flow, not an error condition
    """Raised by a test function when the shrinker's test budget is used up."""


def _split[T](items: Sequence[T], parts: int) -> list[list[T]]:
    size, extra = divmod(len(items), parts)
    chunks: list[list[T]] = []
    start = 0
    for index in range(parts):
        end = start + size + (1 if index < extra else 0)
        chunks.append(list(items[start:end]))
        start = end
    return [chunk for chunk in chunks if chunk]


def ddmin[T](items: Sequence[T], fails: Callable[[Sequence[T]], bool]) -> list[T]:
    """Return a 1-minimal failing subsequence of ``items`` (``fails(items)`` must be True).

    The search runs over element positions, so elements need not be hashable; results of
    ``fails`` are cached per candidate. If ``fails`` raises :class:`ShrinkBudgetExceeded`, the
    smallest failing candidate found so far is returned.
    """
    cache: dict[tuple[int, ...], bool] = {}

    def test(candidate: Sequence[int]) -> bool:
        key = tuple(candidate)
        if key not in cache:
            cache[key] = fails([items[index] for index in key])
        return cache[key]

    current = list(range(len(items)))
    try:
        if current and test([]):
            return []
        granularity = 2
        while len(current) >= 2:
            chunks = _split(current, granularity)
            reduced = False
            for chunk in chunks:
                if test(chunk):
                    current, granularity, reduced = chunk, 2, True
                    break
            if not reduced:
                for skip in range(len(chunks)):
                    complement = [i for j, part in enumerate(chunks) if j != skip for i in part]
                    if test(complement):
                        current = complement
                        granularity = max(granularity - 1, 2)
                        reduced = True
                        break
            if not reduced:
                if granularity >= len(current):
                    break
                granularity = min(len(current), granularity * 2)
    except ShrinkBudgetExceeded:
        pass
    return [items[index] for index in current]


@dataclass(frozen=True, slots=True)
class ShrinkResult:
    invariant: str
    original: Scenario
    shrunk: Scenario
    original_length: int
    shrunk_length: int
    tests: int
    final: RunResult

    @property
    def ratio(self) -> float:
        return self.original_length / max(1, self.shrunk_length)


def _prune(scenario: Scenario) -> Scenario:
    """Drop script entries and faults that belong to tickets the scenario no longer has."""
    if scenario.tickets is None:
        return scenario
    kept = {f"T-{index + 1}" for index in scenario.tickets}
    decisions = scenario.decisions
    if decisions is not None:
        decisions = tuple(d for d in decisions if ticket_scope(d.plan_id) in kept)
    faults = scenario.faults
    if faults is not None:
        faults = tuple(
            key for key in faults if not key[2].startswith("T-") or ticket_scope(key[2]) in kept
        )
    return replace(scenario, decisions=decisions, faults=faults)


def _shrink_tickets(base: Scenario, fails: Callable[[Scenario], bool]) -> Scenario:
    kept = ddmin(base.tickets or (), lambda sub: fails(_prune(replace(base, tickets=tuple(sub)))))
    return _prune(replace(base, tickets=tuple(kept)))


def _shrink_faults(base: Scenario, fails: Callable[[Scenario], bool]) -> Scenario:
    kept = ddmin(base.faults or (), lambda sub: fails(replace(base, faults=tuple(sub))))
    return replace(base, faults=tuple(kept))


def renumber(decisions: Sequence[Decision]) -> tuple[Decision, ...]:
    """Number each plan's remaining decisions 1, 2, 3... in their original order.

    A scripted model answers call ``n`` of a plan with the decision numbered ``n``; renumbering
    after a deletion therefore means "the plan's action sequence without that step".
    """
    counts: dict[str, int] = {}
    renumbered: list[Decision] = []
    for decision in decisions:
        counts[decision.plan_id] = counts.get(decision.plan_id, 0) + 1
        renumbered.append(replace(decision, n=counts[decision.plan_id]))
    return tuple(renumbered)


def _shrink_decisions(base: Scenario, fails: Callable[[Scenario], bool]) -> Scenario:
    kept = ddmin(base.decisions or (), lambda sub: fails(replace(base, decisions=renumber(sub))))
    return replace(base, decisions=renumber(kept))


def shrink_failure(failing: RunResult, *, max_tests: int = 600, rounds: int = 4) -> ShrinkResult:
    """Minimize a failing run to a small scripted scenario with the same first violation."""
    first = failing.first
    if first is None:
        raise ValueError("the run did not fail; there is nothing to shrink")
    target = first.invariant
    tests = 0

    def still_fails(candidate: Scenario) -> bool:
        nonlocal tests
        if tests >= max_tests:
            raise ShrinkBudgetExceeded
        tests += 1
        result = run_scenario(candidate)
        return result.first is not None and result.first.invariant == target

    scenario = failing.replay_scenario()
    if not still_fails(scenario):
        raise RuntimeError("the scripted replay of the failing run did not fail; replay diverged")

    def size(candidate: Scenario) -> int:
        return (
            len(candidate.tickets or ())
            + len(candidate.faults or ())
            + len(candidate.decisions or ())
        )

    for _ in range(rounds):
        before = size(scenario)
        scenario = _shrink_tickets(scenario, still_fails)
        scenario = _shrink_faults(scenario, still_fails)
        scenario = _shrink_decisions(scenario, still_fails)
        if size(scenario) == before or tests >= max_tests:
            break
    final = run_scenario(scenario, keep_trace=True)
    if final.first is None or final.first.invariant != target:
        raise RuntimeError("the shrunk scenario no longer fails; the replay is not deterministic")
    return ShrinkResult(
        invariant=target,
        original=failing.replay_scenario(),
        shrunk=scenario,
        original_length=failing.trace_length,
        shrunk_length=final.trace_length,
        tests=tests,
        final=final,
    )

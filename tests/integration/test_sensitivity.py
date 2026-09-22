"""Bug-flag sensitivity and soundness smoke tests (the per-push CI job runs larger campaigns)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from deterministic_agent_simulator.agent.bugs import BUG_CATALOG, BugFlag
from deterministic_agent_simulator.runner import (
    Scenario,
    SimConfig,
    find_failure,
    run_campaign,
    run_scenario,
)

pytestmark = pytest.mark.integration

# Seeds searched per bug; generous next to the medians measured by bench/bench_sim.py.
SEARCH_LIMIT = dict.fromkeys(BugFlag, 400) | {BugFlag.STALE_APPROVAL_REUSE: 1_500}


@pytest.mark.parametrize("bug", list(BugFlag))
def test_each_planted_bug_is_found_and_trips_its_oracle(bug: BugFlag) -> None:
    config = SimConfig(bugs=frozenset({bug}), fault_mode="fixed")
    found = find_failure(config, start=0, max_seeds=SEARCH_LIMIT[bug])
    assert found is not None, f"{bug.value} not found in {SEARCH_LIMIT[bug]} seeds"
    seed, result = found
    assert result.first is not None
    expected = BUG_CATALOG[bug].expected_invariant
    invariants = {violation.invariant for violation in result.violations}
    assert expected in invariants, f"{bug.value} at seed {seed} tripped {invariants}"


@pytest.mark.parametrize("mode", ["swarm", "fixed"])
def test_the_clean_build_is_sound_on_a_sample_of_seeds(mode: str) -> None:
    config = SimConfig(fault_mode="swarm" if mode == "swarm" else "fixed")
    report = run_campaign(config, seeds=range(50_000, 50_060), max_seeds=60, max_wall_seconds=300)
    assert report.seeds_run == 60
    assert report.failing == [], report.failing


def test_campaign_caps_are_enforced() -> None:
    config = SimConfig()
    report = run_campaign(config, seeds=range(100), max_seeds=3, max_wall_seconds=300)
    assert (report.seeds_run, report.stopped) == (3, "max_seeds")
    ticks = iter([0.0, 0.0, 10.0, 10.0, 10.0])
    timed = run_campaign(
        config, seeds=range(100), max_seeds=100, max_wall_seconds=5, clock=lambda: next(ticks)
    )
    assert timed.stopped == "max_wall_seconds"
    with pytest.raises(ValueError, match="positive"):
        run_campaign(config, seeds=range(1), max_seeds=0, max_wall_seconds=1)
    first = run_campaign(
        SimConfig(bugs=frozenset({BugFlag.MISSING_DEDUPE}), fault_mode="fixed"),
        seeds=range(100),
        max_seeds=100,
        max_wall_seconds=300,
        stop_on_first=True,
        record_hashes=True,
    )
    assert first.stopped == "first_failure"
    assert len(first.trace_hashes) == first.seeds_run
    assert first.to_json()["failing_seeds"]


def _hostile(args: Mapping[str, object], name: str) -> bool:
    to = args.get("to")
    return (
        name == "export_customers"
        or args.get("customer_id") == "c-attacker"
        or (isinstance(to, str) and to.endswith("@attacker.example"))
        or "approved" in args
    )


def test_obeyed_injections_never_become_unauthorized_effects_on_the_clean_build() -> None:
    """Security property as an oracle: the model obeys injected text; the controls hold."""
    config = SimConfig(fault_mode="fixed")
    obeyed = 0
    for seed in range(70_000, 70_060):
        result = run_scenario(Scenario(seed=seed, config=config))
        obeyed += sum(1 for d in result.decisions if _hostile(d.proposal.args, d.proposal.name))
        assert not result.violated, (seed, result.first)
    assert obeyed >= 50, f"only {obeyed} hostile proposals; the injection fault is not exercised"

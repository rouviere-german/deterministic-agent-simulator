from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deterministic_agent_simulator.sim.faults import (
    FIXED_RATES,
    FaultKind,
    FaultPlan,
    FaultProfile,
)
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace

pytestmark = pytest.mark.unit


def test_trace_hash_is_deterministic_and_order_sensitive() -> None:
    first, second, swapped = Trace(), Trace(), Trace()
    for trace in (first, second):
        trace.record(1, "rpc", "email.send", "T-1", "ok", 0)
        trace.record(2, "effect", "send_email", "T-1", None)
    swapped.record(2, "effect", "send_email", "T-1", None)
    swapped.record(1, "rpc", "email.send", "T-1", "ok", 0)
    assert first.hexdigest() == second.hexdigest()
    assert first.hexdigest() != swapped.hexdigest()


def test_field_boundaries_cannot_collide() -> None:
    a, b = Trace(), Trace()
    a.record(0, "x", "ab", "c")
    b.record(0, "x", "a", "bc")
    assert a.hexdigest() != b.hexdigest()


def test_scheduling_choices_are_hashed_but_not_counted() -> None:
    plain, scheduled = Trace(keep=True), Trace(keep=True)
    for trace in (plain, scheduled):
        trace.record(0, "inbound", "T-1", 1)
    scheduled.schedule(3, 1)
    assert plain.length == scheduled.length == 1
    assert scheduled.schedule_points == 1
    assert plain.hexdigest() != scheduled.hexdigest()
    assert scheduled.kinds() == {"inbound": 1}


def test_a_frozen_trace_ignores_further_events() -> None:
    trace = Trace(keep=True)
    trace.record(0, "a")
    digest = trace.hexdigest()
    trace.freeze()
    trace.record(1, "b")
    trace.schedule(2, 0)
    assert trace.frozen
    assert trace.hexdigest() == digest
    assert trace.length == 1
    assert trace.events[0].render() == '[0,"a"]'


def test_profiles() -> None:
    assert FaultProfile.none().enabled() == []
    fixed = FaultProfile.fixed()
    assert set(fixed.enabled()) == {kind.value for kind in FIXED_RATES}
    assert fixed.rate(FaultKind.ERROR_5XX) == FIXED_RATES[FaultKind.ERROR_5XX]
    with pytest.raises(ValueError, match="within"):
        FaultProfile.of("bad", {FaultKind.ERROR_5XX: 1.5})


def test_swarm_profiles_vary_by_seed_and_are_reproducible() -> None:
    profiles = [FaultProfile.swarm(SimRandom.from_seed(seed)) for seed in range(40)]
    again = [FaultProfile.swarm(SimRandom.from_seed(seed)) for seed in range(40)]
    assert profiles == again
    enabled = {tuple(profile.enabled()) for profile in profiles}
    assert len(enabled) > 10  # different seeds switch different fault kinds off
    for profile in profiles:
        for kind, rate in profile.rates:
            assert 0 < rate <= min(1.0, FIXED_RATES[kind] * 4)


def _plan(seed: int, schedule: list[tuple[str, str, str, int]] | None = None) -> FaultPlan:
    rng = SimRandom.from_seed(seed)
    return FaultPlan(
        FaultProfile.of("half", dict.fromkeys(FaultKind, 0.5)),
        rng,
        Trace(),
        lambda: 0,
        schedule=schedule,
    )


@settings(max_examples=40)
@given(
    seed=st.integers(0, 10_000),
    calls=st.lists(
        st.tuples(
            st.sampled_from([FaultKind.ERROR_5XX, FaultKind.AGENT_CRASH]),
            st.sampled_from(["email.send", "ledger.consume"]),
            st.sampled_from(["T-1", "T-2"]),
        ),
        max_size=40,
    ),
)
def test_replaying_the_fired_keys_reproduces_every_decision(
    seed: int, calls: list[tuple[FaultKind, str, str]]
) -> None:
    explore = _plan(seed)
    decisions = [explore.fires(kind, point, scope) for kind, point, scope in calls]
    replay = _plan(seed + 1, schedule=list(explore.fired))
    assert [replay.fires(kind, point, scope) for kind, point, scope in calls] == decisions
    assert replay.scheduled
    assert replay.fired == explore.fired


@settings(max_examples=30)
@given(seed=st.integers(0, 10_000), noise=st.integers(0, 30))
def test_decisions_at_one_point_ignore_activity_at_others(seed: int, noise: int) -> None:
    quiet, noisy = _plan(seed), _plan(seed)
    for _ in range(noise):
        noisy.fires(FaultKind.ERROR_5XX, "tickets.get", "T-9")
    first = [quiet.fires(FaultKind.AGENT_CRASH, "store.get.pre", "T-1") for _ in range(10)]
    second = [noisy.fires(FaultKind.AGENT_CRASH, "store.get.pre", "T-1") for _ in range(10)]
    assert first == second


def test_disabled_kinds_never_fire_and_draw_streams_are_keyed() -> None:
    plan = FaultPlan(FaultProfile.none(), SimRandom.from_seed(1), Trace(), lambda: 0)
    assert not any(plan.fires(FaultKind.ERROR_5XX, "x", "y") for _ in range(100))
    assert plan.fired == []
    a = plan.draw("ledger.downtime", "1").next_u64()
    assert a == plan.draw("ledger.downtime", "1").next_u64()
    assert a != plan.draw("ledger.downtime", "2").next_u64()

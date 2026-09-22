from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from deterministic_agent_simulator.sim.rand import SimRandom

pytestmark = pytest.mark.unit


def _zero_state() -> SimRandom:
    return SimRandom(bytes(32))


def test_splitmix64_matches_the_published_reference_sequence() -> None:
    # SplitMix64 seeded with 0 (Steele, Lea and Flood 2014; the sequence used by xoshiro seeding).
    rng = _zero_state()
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_same_seed_same_stream_and_different_seeds_differ() -> None:
    a, b, c = SimRandom.from_seed(7), SimRandom.from_seed(7), SimRandom.from_seed(8)
    first = [a.next_u64() for _ in range(5)]
    assert first == [b.next_u64() for _ in range(5)]
    assert first != [c.next_u64() for _ in range(5)]


@given(draws=st.integers(min_value=0, max_value=50), label=st.text(max_size=12))
def test_derived_streams_do_not_depend_on_parent_draws(draws: int, label: str) -> None:
    parent = SimRandom.from_seed(99)
    before = parent.derive(label, 3)
    for _ in range(draws):
        parent.next_u64()
    after = parent.derive(label, 3)
    assert [before.next_u64() for _ in range(4)] == [after.next_u64() for _ in range(4)]


def test_label_encoding_is_injective() -> None:
    root = SimRandom.from_seed(1)
    assert root.derive("ab", "c").key != root.derive("a", "bc").key
    assert root.derive("1").key != root.derive(1, "").key
    assert root.derive("x").key != root.key


@given(seed=st.integers(0, 2**32), n=st.integers(1, 10**9))
def test_below_stays_in_range(seed: int, n: int) -> None:
    rng = SimRandom.from_seed(seed)
    assert all(0 <= rng.below(n) < n for _ in range(20))


def test_below_is_close_to_uniform() -> None:
    rng = SimRandom.from_seed(5)
    counts = [0] * 6
    draws = 60_000
    for _ in range(draws):
        counts[rng.below(6)] += 1
    expected = draws / 6
    chi_square = sum((count - expected) ** 2 / expected for count in counts)
    assert chi_square < 20.5  # p ~ 0.001 for 5 degrees of freedom


def test_random_and_chance_edges() -> None:
    rng = SimRandom.from_seed(3)
    values = [rng.random() for _ in range(1000)]
    assert all(0.0 <= value < 1.0 for value in values)
    state = rng.derive("probe")
    assert not state.chance(0.0)
    assert state.chance(1.0)
    hits = sum(rng.chance(0.25) for _ in range(8000))
    assert 1700 < hits < 2300


def test_choice_shuffle_and_weights() -> None:
    rng = SimRandom.from_seed(11)
    items = list(range(20))
    rng.shuffle(items)
    assert sorted(items) == list(range(20))
    assert rng.choice("abc") in "abc"
    assert all(rng.weighted_index([0, 3, 0, 1]) in (1, 3) for _ in range(200))
    assert 0 <= rng.randint(-3, 3) + 3 <= 6


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda r: r.below(0), ValueError),
        (lambda r: r.randint(5, 1), ValueError),
        (lambda r: r.choice([]), ValueError),
        (lambda r: r.weighted_index([0, 0]), ValueError),
        (lambda r: r.weighted_index([2, -1]), ValueError),
    ],
)
def test_invalid_arguments_are_rejected(call: object, error: type[Exception]) -> None:
    rng = SimRandom.from_seed(0)
    with pytest.raises(error):
        call(rng)  # type: ignore[operator]


def test_keys_must_be_32_bytes() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        SimRandom(b"short")
    assert SimRandom.from_seed(0).below(1) == 0

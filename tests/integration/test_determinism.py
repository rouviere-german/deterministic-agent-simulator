from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.runner import Scenario, SimConfig, run_scenario

pytestmark = pytest.mark.integration

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_trace_hashes.json"


@pytest.mark.parametrize("seed", [0, 7, 1234])
def test_reruns_of_a_seed_produce_identical_traces(seed: int) -> None:
    hashes = {run_scenario(Scenario(seed=seed)).trace_hash for _ in range(10)}
    assert len(hashes) == 1


def test_different_seeds_produce_different_traces() -> None:
    hashes = {run_scenario(Scenario(seed=seed)).trace_hash for seed in range(25)}
    assert len(hashes) == 25


@settings(max_examples=25)
@given(
    seed=st.integers(0, 2**31),
    bugs=st.sets(st.sampled_from(list(BugFlag)), max_size=2),
    mode=st.sampled_from(["swarm", "fixed"]),
)
def test_the_scripted_replay_of_any_run_reproduces_its_trace_exactly(
    seed: int, bugs: set[BugFlag], mode: str
) -> None:
    config = SimConfig(bugs=frozenset(bugs), fault_mode="swarm" if mode == "swarm" else "fixed")
    explored = run_scenario(Scenario(seed=seed, config=config))
    replayed = run_scenario(explored.replay_scenario())
    assert replayed.trace_hash == explored.trace_hash
    assert [v.invariant for v in replayed.violations] == [v.invariant for v in explored.violations]


def test_golden_hashes_match_on_this_platform() -> None:
    """The committed hashes were computed on Windows; CI checks Linux against the same file."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    config = SimConfig.from_json(golden["config"])
    for seed in sorted(golden["hashes"], key=int)[:25]:
        assert (
            run_scenario(Scenario(seed=int(seed), config=config)).trace_hash
            == (golden["hashes"][seed])
        ), f"seed {seed} diverged from the golden trace hash"


def test_trace_hashes_do_not_depend_on_python_hash_randomization() -> None:
    code = (
        "from deterministic_agent_simulator.runner import Scenario, run_scenario\n"
        "print(','.join(run_scenario(Scenario(seed=s)).trace_hash for s in range(4)))\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
            env=env,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1

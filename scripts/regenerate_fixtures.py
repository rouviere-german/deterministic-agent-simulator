#!/usr/bin/env python3
"""Regenerate the committed simulation fixtures after an intentional simulator change.

Writes:
- ``tests/fixtures/golden_trace_hashes.json``: trace hashes of seeds 0-199 (swarm faults, default
  configuration). CI recomputes them on Linux and Windows and fails on any difference.
- ``tests/regressions/``: an example produced by the tool itself: the first seed on which the
  planted ``non_idempotent_retry`` bug is found, shrunk, signed with the test-only key (derived
  from a public label, exactly as ``tests/conftest.py`` does) and turned into a regression test.

Any change to the simulator's behavior changes trace hashes; review the diff before committing.

Usage: uv run python scripts/regenerate_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.emit import emit_regression_test
from deterministic_agent_simulator.replay import Keyring, build_repro, write_repro
from deterministic_agent_simulator.runner import Scenario, SimConfig, find_failure, run_scenario
from deterministic_agent_simulator.shrink import shrink_failure

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_trace_hashes.json"
REGRESSIONS = ROOT / "tests" / "regressions"
TEST_KEY = hashlib.sha256(b"deterministic-agent-simulator test-only replay key").hexdigest()
GOLDEN_SEEDS = 200


def golden() -> None:
    config = SimConfig()
    hashes = {
        str(seed): run_scenario(Scenario(seed=seed, config=config)).trace_hash
        for seed in range(GOLDEN_SEEDS)
    }
    document = {"config": config.to_json(), "hashes": hashes}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN.relative_to(ROOT)} ({GOLDEN_SEEDS} seeds)")


def regression_example() -> None:
    config = SimConfig(bugs=frozenset({BugFlag.NON_IDEMPOTENT_RETRY}), fault_mode="swarm")
    found = find_failure(config, start=0, max_seeds=2_000)
    if found is None:
        raise SystemExit("the planted bug was not found")
    seed, failing = found
    first = failing.first
    if first is None:
        raise SystemExit("the failing run has no violation")
    shrunk = shrink_failure(failing)
    keyring = Keyring({"tests": TEST_KEY.encode("utf-8")}, "tests")
    document = build_repro(
        shrunk.shrunk,
        violation=first,
        trace_hash=shrunk.final.trace_hash,
        trace_length=shrunk.shrunk_length,
        original_trace_length=shrunk.original_length,
        keyring=keyring,
    )
    for stale in REGRESSIONS.glob("*_seed_*"):
        stale.unlink()
    repro = REGRESSIONS / f"repro_seed_{seed}_{first.invariant}.json"
    write_repro(repro, document)
    test = emit_regression_test(repro, document)
    print(f"wrote {repro.relative_to(ROOT)} and {test.relative_to(ROOT)}")
    print(f"shrunk {shrunk.original_length} -> {shrunk.shrunk_length} trace events")


def main() -> int:
    golden()
    regression_example()
    return 0


if __name__ == "__main__":
    sys.exit(main())

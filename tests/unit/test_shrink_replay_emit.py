from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agent.ports import Proposal
from deterministic_agent_simulator.check.oracles import INVARIANTS
from deterministic_agent_simulator.core.config import Settings
from deterministic_agent_simulator.core.tracing import build_tracer_provider
from deterministic_agent_simulator.emit import emit_regression_test, regression_test_source
from deterministic_agent_simulator.replay import (
    MAX_REPRO_BYTES,
    Keyring,
    ReproError,
    ReproSignatureError,
    build_repro,
    load_repro,
    parse_repro,
    replay_file,
    scenario_from_document,
    scenario_to_document,
    write_repro,
)
from deterministic_agent_simulator.report import (
    export_trace_spans,
    failure_report,
    redact_report_text,
)
from deterministic_agent_simulator.runner import (
    RunResult,
    Scenario,
    SimConfig,
    find_failure,
    run_scenario,
)
from deterministic_agent_simulator.shrink import (
    ShrinkBudgetExceeded,
    ShrinkResult,
    ddmin,
    renumber,
    shrink_failure,
)
from deterministic_agent_simulator.sim.metrics import create_sim_metrics
from deterministic_agent_simulator.sim.trace import Trace
from deterministic_agent_simulator.world.simllm import Decision

pytestmark = pytest.mark.unit

KEY = b"k" * 32
KEYRING = Keyring({"test": KEY}, "test")


# --------------------------------------------------------------------- action-sequence deletion
@given(
    plans=st.lists(st.sampled_from(["T-1", "T-2", "T-3~2"]), min_size=0, max_size=20),
    data=st.data(),
)
def test_deleting_decisions_shifts_each_plans_later_steps_forward(
    plans: list[str], data: st.DataObject
) -> None:
    counts: dict[str, int] = {}
    decisions = []
    for index, plan in enumerate(plans):
        counts[plan] = counts.get(plan, 0) + 1
        decisions.append(Decision(plan, counts[plan], "next", Proposal(f"step{index}")))
    kept = [d for d in decisions if data.draw(st.booleans())]
    renumbered = renumber(kept)
    # The same proposals, in the same order, numbered 1..k within each plan.
    assert [d.proposal for d in renumbered] == [d.proposal for d in kept]
    for plan in set(plans):
        numbers = [d.n for d in renumbered if d.plan_id == plan]
        assert numbers == list(range(1, len(numbers) + 1))
    assert renumber(decisions) == tuple(decisions)  # a full recorded script is already numbered


# ------------------------------------------------------------------------------------ ddmin
def test_ddmin_finds_the_failure_inducing_pair() -> None:
    items = list(range(40))
    assert ddmin(items, lambda sub: {7, 31} <= set(sub)) == [7, 31]


@settings(max_examples=80)
@given(
    size=st.integers(1, 30),
    culprits=st.sets(st.integers(0, 29), min_size=1, max_size=4),
)
def test_ddmin_results_fail_and_are_one_minimal(size: int, culprits: set[int]) -> None:
    culprits = {c for c in culprits if c < size} or {0}
    tests: list[int] = []

    def fails(sub: Sequence[int]) -> bool:
        tests.append(len(sub))
        return culprits <= set(sub)

    result = ddmin(list(range(size)), fails)
    assert fails(result)
    for index in range(len(result)):
        assert not fails(result[:index] + result[index + 1 :])
    assert sorted(result) == sorted(culprits)


def test_ddmin_returns_the_empty_list_when_nothing_is_needed_and_honors_budgets() -> None:
    assert ddmin([1, 2, 3], lambda sub: True) == []
    calls = 0

    def limited(sub: Sequence[int]) -> bool:
        nonlocal calls
        calls += 1
        if calls > 3:
            raise ShrinkBudgetExceeded
        return 5 in sub

    result = ddmin(list(range(10)), limited)
    assert 5 in result  # the best failing candidate found before the budget ran out


# ------------------------------------------------------------------------------- shrink real
@pytest.fixture(scope="module")
def shrunk() -> tuple[RunResult, ShrinkResult]:
    config = SimConfig(bugs=frozenset({BugFlag.NON_IDEMPOTENT_RETRY}), fault_mode="fixed")
    found = find_failure(config, start=0, max_seeds=300)
    assert found is not None
    _, failing = found
    return failing, shrink_failure(failing)


def test_shrinking_keeps_the_invariant_and_reduces_the_trace(
    shrunk: tuple[RunResult, ShrinkResult],
) -> None:
    failing, result = shrunk
    assert failing.first is not None
    assert result.invariant == failing.first.invariant
    assert result.final.first is not None
    assert result.final.first.invariant == result.invariant
    assert result.shrunk_length < result.original_length
    assert result.ratio > 2
    original = failing.replay_scenario()
    assert set(result.shrunk.tickets or ()) <= set(original.tickets or ())
    assert set(result.shrunk.faults or ()) <= set(original.faults or ())
    # The shrunk scenario is itself deterministic.
    assert run_scenario(result.shrunk).trace_hash == result.final.trace_hash


def test_shrinking_a_passing_run_is_refused() -> None:
    passing = run_scenario(Scenario(seed=1))
    assert not passing.violated
    with pytest.raises(ValueError, match="nothing to shrink"):
        shrink_failure(passing)


# ------------------------------------------------------------------------------ replay files
def _document(shrunk: tuple[RunResult, ShrinkResult], keyring: Keyring = KEYRING) -> dict[str, Any]:
    failing, result = shrunk
    assert failing.first is not None
    return build_repro(
        result.shrunk,
        violation=failing.first,
        trace_hash=result.final.trace_hash,
        trace_length=result.shrunk_length,
        original_trace_length=result.original_length,
        keyring=keyring,
    )


def test_signed_repros_round_trip(shrunk: tuple[RunResult, ShrinkResult], tmp_path: Path) -> None:
    path = tmp_path / "repro.json"
    write_repro(path, _document(shrunk))
    scenario, expected = load_repro(path, KEYRING)
    assert scenario == shrunk[1].shrunk
    assert expected["trace_hash"] == shrunk[1].final.trace_hash
    assert scenario_from_document(scenario_to_document(scenario)) == scenario


@settings(max_examples=40)
@given(
    field=st.sampled_from(
        ["seed", "tickets", "faults", "decisions", "config", "expected", "key_id", "created_by"]
    )
)
def test_any_tampered_field_is_rejected(shrunk: tuple[RunResult, ShrinkResult], field: str) -> None:
    document = json.loads(json.dumps(_document(shrunk)))
    value = document[field]
    if isinstance(value, int):
        document[field] = value + 1
    elif isinstance(value, str):
        document[field] = value + "x"
    elif isinstance(value, list):
        document[field] = [*value, value[0]] if value else [0]
    else:
        document[field] = {**value, "extra": 1}
    with pytest.raises(ReproSignatureError):
        parse_repro(json.dumps(document).encode(), KEYRING)


def test_unsigned_foreign_oversized_and_malformed_files_are_rejected(
    shrunk: tuple[RunResult, ShrinkResult],
) -> None:
    document = _document(shrunk)
    unsigned = {k: v for k, v in document.items() if k != "signature"}
    with pytest.raises(ReproSignatureError, match="not signed"):
        parse_repro(json.dumps(unsigned).encode(), KEYRING)
    foreign = _document(shrunk, Keyring({"other": b"o" * 32}, "other"))
    with pytest.raises(ReproSignatureError, match="unknown key"):
        parse_repro(json.dumps(foreign).encode(), KEYRING)
    with pytest.raises(ReproError, match="too large"):
        parse_repro(b" " * (MAX_REPRO_BYTES + 1), KEYRING)
    with pytest.raises(ReproError, match="JSON"):
        parse_repro(b"{not json", KEYRING)
    with pytest.raises(ReproError, match="object"):
        parse_repro(b"[]", KEYRING)
    signed_garbage = KEYRING.sign({"format": "das-repro/1", "seed": "x"})
    with pytest.raises(ReproError, match="schema"):
        parse_repro(json.dumps(signed_garbage).encode(), KEYRING)


def test_hostile_json_is_rejected_instead_of_escaping_as_another_error() -> None:
    """Inputs that make json.loads raise something other than JSONDecodeError."""
    with pytest.raises(ReproError, match="JSON"):
        parse_repro(b'{"seed": ' + b"9" * 5_000 + b"}", KEYRING)  # int_max_str_digits
    with pytest.raises(ReproError, match="JSON"):
        parse_repro(b"[" * 100_000, KEYRING)  # RecursionError in the parser
    deep = b'{"a":' * 200 + b"1" + b"}" * 200
    with pytest.raises(ReproError, match="nested too deeply"):
        parse_repro(deep, KEYRING)


def test_keyring_requirements() -> None:
    with pytest.raises(ReproError, match="32 bytes"):
        Keyring({"k": b"short"}, "k")
    with pytest.raises(ReproError, match="signing key"):
        Keyring({"k": KEY}, "missing")
    with pytest.raises(ReproError, match="APP_REPLAY_HMAC_KEY"):
        Keyring.from_settings(Settings(_env_file=None))
    ring = Keyring.from_settings(
        Settings(_env_file=None, replay_hmac_key="z" * 32, replay_key_id="ci")
    )
    assert ring.signing_key_id == "ci"


def test_replaying_with_recorded_and_current_builds(
    shrunk: tuple[RunResult, ShrinkResult], tmp_path: Path
) -> None:
    path = tmp_path / "repro.json"
    write_repro(path, _document(shrunk))
    settings = Settings(_env_file=None)
    recorded = replay_file(path, build="recorded", keyring=KEYRING, settings=settings)
    assert recorded.invariant == shrunk[1].invariant
    assert recorded.trace_hash == recorded.expected_trace_hash
    assert recorded.build_flags == ["non_idempotent_retry"]
    current = replay_file(path, build="current", keyring=KEYRING, settings=settings)
    assert current.violations == []
    assert current.report == "no invariant was violated"
    buggy = Settings(_env_file=None, sim_bug_flags=["non_idempotent_retry"])
    assert replay_file(path, build="current", keyring=KEYRING, settings=buggy).violations


# --------------------------------------------------------------------------- emitted tests
def test_emitted_regression_tests_are_valid_python(
    shrunk: tuple[RunResult, ShrinkResult], tmp_path: Path
) -> None:
    document = _document(shrunk)
    repro = tmp_path / "repro_seed_1.json"
    write_repro(repro, document)
    test_path = emit_regression_test(repro, document)
    source = test_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(names) == 2
    assert all(name.startswith("test_seed_") for name in names)
    assert "repro_seed_1.json" in source


@settings(max_examples=60)
@given(seed=st.integers(0, 2**63 - 1), invariant=st.sampled_from(INVARIANTS))
def test_every_seed_and_invariant_emits_compilable_python(seed: int, invariant: str) -> None:
    document = {
        "seed": seed,
        "config": {"bugs": ["non_idempotent_retry"]},
        "expected": {"invariant": invariant, "trace_length": 4, "original_trace_length": 40},
    }
    source = regression_test_source(document, "repro.json")
    compile(source, "<emitted>", "exec")
    assert f"def test_seed_{seed}_{invariant}_reproduces_with_recorded_flags(" in source


def test_negative_seeds_are_refused_by_emission_and_by_the_replay_schema(
    shrunk: tuple[RunResult, ShrinkResult],
) -> None:
    """`seed_-3_...` is not an identifier, so a negative seed may not reach a generated test."""
    document = {
        "seed": -3,
        "config": {"bugs": []},
        "expected": {
            "invariant": "no_duplicate_effects",
            "trace_length": 1,
            "original_trace_length": 2,
        },
    }
    with pytest.raises(ValueError, match="negative"):
        regression_test_source(document, "repro.json")
    body = {k: v for k, v in _document(shrunk).items() if k != "signature"}
    signed = KEYRING.sign({**body, "seed": -3})
    with pytest.raises(ReproError, match="schema"):
        parse_repro(json.dumps(signed).encode(), KEYRING)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["expected"].update(invariant="rm -rf"), "unknown invariant"),
        (lambda d: d["config"].update(bugs=["x'); import os; ('"]), "bug flag"),
    ],
)
def test_emission_refuses_unexpected_content(
    shrunk: tuple[RunResult, ShrinkResult], change: Any, message: str
) -> None:
    document = json.loads(json.dumps(_document(shrunk)))
    change(document)
    with pytest.raises(ValueError, match=message):
        regression_test_source(document, "repro.json")
    with pytest.raises(ValueError, match="names"):
        regression_test_source(_document(shrunk), "../evil.json")


# ---------------------------------------------------------------------------------- reports
def test_reports_redact_canaries_and_pii(shrunk: tuple[RunResult, ShrinkResult]) -> None:
    text = "sent CANARY-00ff00ff00ff00ff to ana.silva@customers.example"
    redacted = redact_report_text(text)
    assert "CANARY-" not in redacted
    assert "ana.silva" not in redacted
    report = failure_report(*shrunk)
    blob = json.dumps(report)
    assert "CANARY-" not in blob
    assert "@customers.example" not in blob
    assert report["invariant"] == "no_duplicate_effects"


def test_trace_spans_export_rpcs_as_child_spans(shrunk: tuple[RunResult, ShrinkResult]) -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    trace = shrunk[1].final.trace
    assert trace is not None
    count = export_trace_spans(trace, provider, name="run")
    spans = exporter.get_finished_spans()
    assert count == len(spans)
    rpcs = sum(1 for event in trace.events if event.kind == "rpc")
    assert count == rpcs + 1
    assert all(span.start_time is not None and span.end_time is not None for span in spans)
    with pytest.raises(ValueError, match="keep=False"):
        export_trace_spans(Trace(), provider, name="x")


def test_sim_metrics_render_prometheus_families() -> None:
    metrics = create_sim_metrics()
    metrics.observe_run("swarm", 10, None)
    metrics.observe_run("swarm", 12, "termination")
    metrics.set_throughput(12.5)
    text = metrics.render().decode()
    assert 'das_seeds_total{mode="swarm"} 2.0' in text
    assert 'das_invariant_violations_total{invariant="termination"} 1.0' in text
    assert "das_seeds_per_second 12.5" in text
    assert "das_trace_events_total 22.0" in text

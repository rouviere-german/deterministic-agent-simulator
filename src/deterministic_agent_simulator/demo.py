"""Offline end-to-end demo: find a planted bug, shrink it, sign it, test it, and check soundness.

1. Plant ``non_idempotent_retry`` and run seeds until an oracle fires.
2. Shrink the failing run with ddmin and write it as an HMAC-signed replay file.
3. Show that a tampered copy of the file is rejected.
4. Generate a pytest regression test and run it against the buggy build (it fails) and against
   the clean build (it passes).
5. Export the shrunk trace as OpenTelemetry spans and its audit chain as verifiable JSONL.
6. Run the clean build over 500 seeds with swarm fault profiles: no invariant may fail.

Everything runs in-process: seeded SimLLM, no network, no API key. If APP_REPLAY_HMAC_KEY is not
set, the demo signs with a random key that lives only for this process.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import JsonValue

from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agentic.audit import verify_file
from deterministic_agent_simulator.core.config import get_settings
from deterministic_agent_simulator.core.tracing import build_tracer_provider
from deterministic_agent_simulator.emit import emit_regression_test
from deterministic_agent_simulator.replay import (
    Keyring,
    ReproSignatureError,
    build_repro,
    load_repro,
    write_repro,
)
from deterministic_agent_simulator.report import export_trace_spans, failure_report
from deterministic_agent_simulator.runner import (
    CampaignReport,
    RunResult,
    SimConfig,
    find_failure,
    run_campaign,
)
from deterministic_agent_simulator.shrink import ShrinkResult, shrink_failure
from deterministic_agent_simulator.sim.metrics import create_sim_metrics

__all__ = ["run_demo"]

DEMO_BUG = BugFlag.NON_IDEMPOTENT_RETRY


@contextlib.contextmanager
def _environment(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set APP_* variables (the generated test reads them through Settings)."""
    saved = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def _run_test(function: Callable[[], None]) -> str:
    try:
        function()
    except AssertionError:
        return "failed"
    return "passed"


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _regression_test(
    test_path: Path, seed: int, invariant: str, key: str, key_id: str
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Run the generated test against the buggy build, then against the clean build."""
    module = _load_module(test_path)
    reproduce = getattr(module, f"test_seed_{seed}_{invariant}_reproduces_with_recorded_flags")
    holds = getattr(module, f"test_seed_{seed}_{invariant}_holds_on_current_build")
    env = {"APP_REPLAY_HMAC_KEY": key, "APP_REPLAY_KEY_ID": key_id}
    with _environment({**env, "APP_SIM_BUG_FLAGS": DEMO_BUG.value}):
        buggy: dict[str, JsonValue] = {
            "reproduces": _run_test(reproduce),
            "holds": _run_test(holds),
        }
    with _environment({**env, "APP_SIM_BUG_FLAGS": ""}):
        clean: dict[str, JsonValue] = {
            "reproduces": _run_test(reproduce),
            "holds": _run_test(holds),
        }
    return buggy, clean


def _tamper_rejected(document: dict[str, Any], workdir: Path, keyring: Keyring) -> bool:
    tampered = dict(document)
    tickets = list(document["tickets"])
    tampered["tickets"] = [*tickets, 0] if 0 not in tickets else []
    tampered_path = workdir / "tampered.json"
    write_repro(tampered_path, tampered)
    try:
        load_repro(tampered_path, keyring)
    except ReproSignatureError:
        return True
    return False


def _observability(shrunk: ShrinkResult, workdir: Path, seed: int) -> tuple[int, bool]:
    """Export the shrunk trace as OpenTelemetry spans and its audit chain as JSONL."""
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("deterministic-agent-simulator", extra_exporter=exporter)
    trace = shrunk.final.trace
    spans = export_trace_spans(trace, provider, name=f"seed-{seed}") if trace is not None else 0
    provider.shutdown()
    finished = [
        {"name": s.name, "start_ns": s.start_time, "end_ns": s.end_time}
        for s in exporter.get_finished_spans()
    ]
    (workdir / "shrunk_spans.json").write_text(
        json.dumps(finished, indent=2) + "\n", encoding="utf-8"
    )
    chain = shrunk.final.chain
    if chain is None:
        return spans, False
    audit_path = workdir / "shrunk_audit.jsonl"
    chain.export_jsonl(audit_path)
    return spans, verify_file(audit_path, expect_head=chain.head).ok


def _clean_campaign(workdir: Path, seeds: int) -> CampaignReport:
    metrics = create_sim_metrics()

    def observe(result: RunResult) -> None:
        first = result.first
        metrics.observe_run("swarm", result.trace_length, first.invariant if first else None)

    clean = run_campaign(
        SimConfig(fault_mode="swarm"),
        seeds=range(seeds),
        max_seeds=seeds,
        max_wall_seconds=900.0,
        on_result=observe,
    )
    metrics.set_throughput(clean.seeds_per_second)
    (workdir / "metrics.prom").write_bytes(metrics.render())
    return clean


def _keyring() -> tuple[Keyring, str, str]:
    settings = get_settings()
    configured = settings.replay_hmac_key
    key = configured.get_secret_value() if configured is not None else secrets.token_hex(32)
    key_id = settings.replay_key_id if configured is not None else "demo-ephemeral"
    return Keyring({key_id: key.encode("utf-8")}, key_id), key, key_id


def run_demo(
    workdir: Path, *, clean_seeds: int = 500, max_search: int = 2_000
) -> dict[str, JsonValue]:
    workdir.mkdir(parents=True, exist_ok=True)
    keyring, key, key_id = _keyring()

    # 1. find the planted bug
    found = find_failure(
        SimConfig(bugs=frozenset({DEMO_BUG}), fault_mode="swarm"), start=0, max_seeds=max_search
    )
    if found is None or found[1].first is None:
        raise RuntimeError(f"{DEMO_BUG.value} was not found in {max_search} seeds")
    seed, failing = found
    first = found[1].first

    # 2. shrink and sign
    shrunk = shrink_failure(failing)
    document = build_repro(
        shrunk.shrunk,
        violation=first,
        trace_hash=shrunk.final.trace_hash,
        trace_length=shrunk.shrunk_length,
        original_trace_length=shrunk.original_length,
        keyring=keyring,
    )
    repro_path = workdir / "regressions" / f"repro_seed_{seed}_{first.invariant}.json"
    for stale in repro_path.parent.glob("*"):
        stale.unlink()
    write_repro(repro_path, document)

    # 3. tampering is detected; 4. the regression test fails on the buggy build only
    tamper_rejected = _tamper_rejected(document, workdir, keyring)
    test_path = emit_regression_test(repro_path, document)
    buggy_build, clean_build = _regression_test(test_path, seed, first.invariant, key, key_id)

    # 5. observability; 6. soundness of the clean build
    spans, audit_ok = _observability(shrunk, workdir, seed)
    report = failure_report(failing, shrunk)
    (workdir / "failure_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    clean = _clean_campaign(workdir, clean_seeds)
    return {
        "bug": DEMO_BUG.value,
        "failing_seed": seed,
        "seeds_searched": seed + 1,
        "invariant": first.invariant,
        "detail": str(report["detail"]),
        "original_trace_events": shrunk.original_length,
        "shrunk_trace_events": shrunk.shrunk_length,
        "reduction_ratio": round(shrunk.ratio, 2),
        "shrink_tests": shrunk.tests,
        "shrunk_scenario": {
            "tickets": len(shrunk.shrunk.tickets or ()),
            "faults": [list(k) for k in shrunk.shrunk.faults or ()],
            "decisions": len(shrunk.shrunk.decisions or ()),
        },
        "replay_file": repro_path.name,
        "replay_key_id": key_id,
        "tampered_copy_rejected": tamper_rejected,
        "regression_test": test_path.name,
        "regression_test_on_buggy_build": buggy_build,
        "regression_test_on_clean_build": clean_build,
        "otel_spans_exported": spans,
        "audit_chain_verified": audit_ok,
        "clean_build": {
            "seeds": clean.seeds_run,
            "violations": dict(clean.violations),
            "seeds_per_second": round(clean.seeds_per_second, 1),
        },
    }

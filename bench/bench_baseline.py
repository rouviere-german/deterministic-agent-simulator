"""Micro-benchmarks for the baseline components and the simulator kernel, with CI thresholds.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts. The
evaluation metrics of the simulator itself (bug sensitivity, soundness, shrinking, determinism,
checker agreement, throughput) are measured by bench/bench_sim.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deterministic_agent_simulator.agentic.audit import AuditLog, verify_file
from deterministic_agent_simulator.agentic.providers.base import ChatMessage, LLMRequest
from deterministic_agent_simulator.check.ledger_model import ApprovalModel
from deterministic_agent_simulator.check.linearizability import Operation, wing_gong_check
from deterministic_agent_simulator.core.ratelimit import RateLimiter
from deterministic_agent_simulator.core.redaction import redact_text
from deterministic_agent_simulator.runner import Scenario, SimConfig, run_scenario
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.ledger import ApprovalLedger

if TYPE_CHECKING:
    from deterministic_agent_simulator.sim.loop import SimEventLoop

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = [sample / 1_000.0 for sample in samples]
    percentiles = statistics.quantiles(micros, n=100, method="inclusive")
    return {
        "runs": runs,
        "p50_us": round(statistics.median(micros), 1),
        "p95_us": round(percentiles[94], 1),
    }


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def bench_request_hash() -> dict[str, float]:
    messages = tuple(ChatMessage.user(f"message {i} " + "x" * 400) for i in range(10))
    request = LLMRequest(system="bench", messages=messages)
    return _timed(request.sha256, runs=2_000)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return _timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=200
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(1_000):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return _timed(lambda: verify_file(path), runs=10, warmup=1)


def bench_sim_random() -> dict[str, float]:
    rng = SimRandom.from_seed(0)
    return _timed(lambda: [rng.below(1000) for _ in range(1_000)], runs=200)


def bench_sim_run() -> dict[str, float]:
    """One complete simulated run (5-10 tickets, swarm faults, every oracle)."""
    seeds = iter(range(10**9))
    return _timed(lambda: run_scenario(Scenario(seed=next(seeds), config=SimConfig())), runs=60)


class _Clock:
    now_us = 0


def _ledger_history(ops: int, ttl: int) -> list[Operation]:
    """A linearizable history of the real ledger: sequential commits, overlapping observations."""
    clock = _Clock()
    ledger = ApprovalLedger(cast("SimEventLoop", clock), ttl_us=ttl)
    rng = SimRandom.from_seed(7)
    history: list[Operation] = []

    def observe(kind: str, args: tuple[Hashable, ...], output: Hashable, client: str) -> None:
        now = clock.now_us
        invoke, response = max(0, now - rng.below(40)), now + rng.below(40)
        history.append(Operation(len(history), client, kind, args, invoke, response, output))

    approval = ledger.apply_request("tok", "h", {}, "agent")
    observe("request", ("tok", "h", "agent"), approval, "agent")
    clock.now_us = 20
    observe(
        "decide", (approval, True, "human"), ledger.apply_decide(approval, True, "human"), "human"
    )
    while len(history) < ops:
        clock.now_us += 10 + rng.below(20)
        if rng.below(4) == 0:
            consumer = rng.choice(("c1", "c2"))
            result = ledger.apply_consume(approval, "h", consumer)
            observe("consume", (approval, "h", consumer), result, consumer)
        else:
            observe("status", (approval,), ledger.apply_status(approval), "auditor")
    return history


def bench_wing_gong() -> dict[str, float]:
    history = _ledger_history(40, ttl=1_000)
    model = ApprovalModel(1_000)
    if not wing_gong_check(history, model).ok:
        raise AssertionError("the benchmark history must be linearizable")
    return _timed(lambda: wing_gong_check(history, model), runs=100)


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "llm_request_sha256_10_messages": bench_request_hash(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_1000_entries": bench_audit_verify(workdir),
            "sim_random_1000_draws": bench_sim_random(),
            "sim_run_one_seed": bench_sim_run(),
            "wing_gong_ledger_history_40_ops": bench_wing_gong(),
        }


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            value = results[name][metric]
            if value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="baseline micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

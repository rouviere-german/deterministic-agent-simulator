"""Evaluation of the simulator itself, on CPU, with regression thresholds for CI.

Usage:
  uv run python bench/bench_sim.py                    # full evaluation (a few minutes)
  uv run python bench/bench_sim.py --profile ci       # smaller sample (per-push CI job)
  uv run python bench/bench_sim.py --check            # also enforce bench/sim_thresholds.json
  uv run python bench/bench_sim.py --only soundness --soundness-seeds 50000   # nightly size

Metrics (the evaluation section of the project spec):

1. sensitivity: for each of the 8 planted bugs and each fault mode (swarm, fixed), ``searches``
   independent searches over disjoint seed ranges run up to ``search_limit`` seeds each and stop at
   the first violated invariant. Reported: searches that found the bug, the median number of
   seeds to the first failure, and which invariants fired. A bug counts as found only if *every*
   search found it within the limit.
2. soundness: ``soundness_seeds`` seeds with every bug off and swarm faults; any violation fails.
3. shrinking: the first failing run of ``shrinks_per_bug`` fixed-mode searches per bug is shrunk
   with ddmin; reported: trace-length reduction ratio (original / shrunk), per bug and median.
4. determinism: ``determinism_seeds`` seeds each re-run ``reruns`` times must give one trace hash
   per seed, and the first ``golden_seeds`` committed golden hashes (computed on Windows; CI
   recomputes them on Linux and Windows) must match.
5. checker: Wing-Gong vs the brute-force permutation checker on random histories of at most 8
   operations (a register model, and the approval-ledger model on real ledger executions with
   perturbed outputs), plus known linearizable and non-linearizable fixtures. Brute force uses the
   same sequential model, so it validates the search; the approval *model* is validated against
   ``check/ledger_reference.py``, an independent checker that executes the real ledger.
6. throughput: seeds per second (and per minute) of the soundness campaign.
7. scheduler: how often the seeded tie-breaker has a choice (callbacks ready at the same virtual
   instant) and how often the seeded order changes what happens compared with FIFO order.

Results go to bench/results/sim_latest.json and are printed as Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deterministic_agent_simulator.agent.bugs import BUG_CATALOG, BugFlag
from deterministic_agent_simulator.check.ledger_model import ApprovalModel
from deterministic_agent_simulator.check.ledger_reference import (
    MAX_REFERENCE_OPS,
    reference_linearizable,
)
from deterministic_agent_simulator.check.linearizability import (
    Operation,
    brute_force_check,
    wing_gong_check,
)
from deterministic_agent_simulator.runner import (
    FaultMode,
    RunResult,
    Scenario,
    SimConfig,
    run_campaign,
    run_scenario,
)
from deterministic_agent_simulator.shrink import shrink_failure
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.world.ledger import ApprovalLedger

if TYPE_CHECKING:
    from deterministic_agent_simulator.sim.loop import SimEventLoop

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "sim_latest.json"
THRESHOLDS = ROOT / "bench" / "sim_thresholds.json"
GOLDEN = ROOT / "tests" / "fixtures" / "golden_trace_hashes.json"
SECTIONS = ("sensitivity", "soundness", "shrinking", "determinism", "checker", "scheduler")
SEARCH_BASE = 1_000_000  # sensitivity seeds; disjoint from soundness (0...) and the golden seeds
MODES: tuple[FaultMode, FaultMode] = ("swarm", "fixed")


@dataclass(frozen=True, slots=True)
class Profile:
    searches: int
    search_limit: int
    soundness_seeds: int
    shrinks_per_bug: int
    determinism_seeds: int
    reruns: int
    golden_seeds: int
    checker_histories: int
    scheduler_seeds: int


PROFILES = {
    "full": Profile(11, 10_000, 5_000, 3, 20, 100, 200, 1_000, 300),
    "ci": Profile(3, 10_000, 2_000, 1, 10, 20, 200, 300, 50),
}


# --------------------------------------------------------------------------------- sensitivity
def _first_failure(config: SimConfig, start: int, limit: int) -> tuple[int, RunResult] | None:
    for offset in range(limit):
        result = run_scenario(Scenario(seed=start + offset, config=config))
        if result.violated:
            return offset + 1, result
    return None


def sensitivity(profile: Profile) -> tuple[dict[str, Any], dict[BugFlag, list[RunResult]]]:
    rows: dict[str, Any] = {}
    failures: dict[BugFlag, list[RunResult]] = {bug: [] for bug in BugFlag}
    for bug in BugFlag:
        row: dict[str, Any] = {"expected_invariant": BUG_CATALOG[bug].expected_invariant}
        for mode in MODES:
            config = SimConfig(bugs=frozenset({bug}), fault_mode=mode)
            seeds_needed: list[int] = []
            invariants: Counter[str] = Counter()
            started = time.perf_counter()
            for search in range(profile.searches):
                found = _first_failure(
                    config, SEARCH_BASE + search * profile.search_limit, profile.search_limit
                )
                if found is None:
                    continue
                needed, result = found
                seeds_needed.append(needed)
                if result.first is not None:
                    invariants[result.first.invariant] += 1
                if mode == "fixed":
                    failures[bug].append(result)
            row[mode] = {
                "found": len(seeds_needed),
                "searches": profile.searches,
                "median_seeds_to_failure": statistics.median(seeds_needed)
                if seeds_needed
                else None,
                "max_seeds_to_failure": max(seeds_needed) if seeds_needed else None,
                "invariants": dict(sorted(invariants.items())),
                "seconds": round(time.perf_counter() - started, 1),
            }
        rows[bug.value] = row
    found_everywhere = sum(
        1 for row in rows.values() if all(row[m]["found"] == row[m]["searches"] for m in MODES)
    )
    return {"bugs": rows, "bugs_found": found_everywhere, "limit": profile.search_limit}, failures


# ---------------------------------------------------------------------------------- soundness
def soundness(profile: Profile) -> dict[str, Any]:
    report = run_campaign(
        SimConfig(fault_mode="swarm"),
        seeds=range(profile.soundness_seeds),
        max_seeds=profile.soundness_seeds,
        max_wall_seconds=24 * 3600.0,
    )
    return {
        "seeds": report.seeds_run,
        "violations": sum(report.violations.values()),
        "by_invariant": dict(report.violations),
        "failing_seeds": [seed for seed, _ in report.failing[:20]],
        "wall_seconds": round(report.wall_seconds, 1),
        "seeds_per_second": round(report.seeds_per_second, 1),
        "seeds_per_minute": round(report.seeds_per_second * 60),
    }


# ---------------------------------------------------------------------------------- shrinking
def shrinking(profile: Profile, failures: dict[BugFlag, list[RunResult]]) -> dict[str, Any]:
    per_bug: dict[str, Any] = {}
    ratios: list[float] = []
    preserved = 0
    total = 0
    for bug, runs in failures.items():
        rows = []
        for failing in runs[: profile.shrinks_per_bug]:
            started = time.perf_counter()
            shrunk = shrink_failure(failing)
            total += 1
            preserved += int(
                shrunk.final.first is not None
                and failing.first is not None
                and shrunk.final.first.invariant == failing.first.invariant
            )
            ratios.append(shrunk.ratio)
            rows.append(
                {
                    "seed": failing.scenario.seed,
                    "original": shrunk.original_length,
                    "shrunk": shrunk.shrunk_length,
                    "ratio": round(shrunk.ratio, 1),
                    "tests": shrunk.tests,
                    "tickets": len(shrunk.shrunk.tickets or ()),
                    "faults": len(shrunk.shrunk.faults or ()),
                    "decisions": len(shrunk.shrunk.decisions or ()),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
        per_bug[bug.value] = rows
    return {
        "per_bug": per_bug,
        "shrinks": total,
        "invariant_preserved": preserved,
        "median_ratio": round(statistics.median(ratios), 2) if ratios else None,
        "min_ratio": round(min(ratios), 2) if ratios else None,
        "max_ratio": round(max(ratios), 2) if ratios else None,
    }


# -------------------------------------------------------------------------------- determinism
def determinism(profile: Profile) -> dict[str, Any]:
    mismatched = []
    config = SimConfig()
    for seed in range(profile.determinism_seeds):
        hashes = {
            run_scenario(Scenario(seed=seed, config=config)).trace_hash
            for _ in range(profile.reruns)
        }
        if len(hashes) != 1:
            mismatched.append(seed)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    golden_config = SimConfig.from_json(golden["config"])
    seeds = sorted(golden["hashes"], key=int)[: profile.golden_seeds]
    golden_mismatches = [
        seed
        for seed in seeds
        if run_scenario(Scenario(seed=int(seed), config=golden_config)).trace_hash
        != golden["hashes"][seed]
    ]
    return {
        "seeds": profile.determinism_seeds,
        "reruns_per_seed": profile.reruns,
        "mismatched_seeds": len(mismatched),
        "golden_seeds": len(seeds),
        "golden_mismatches": len(golden_mismatches),
        "platform": platform.system(),
    }


# ------------------------------------------------------------------------------------ checker
class RegisterModel:
    """A read/write register, the textbook linearizability example."""

    def initial(self) -> int:
        return 0

    def step(self, state: int, op: Operation) -> Sequence[int]:
        if op.kind == "write":
            value = cast("int", op.args[0])
            return [value] if op.output in (None, "ok") else []
        return [state] if op.output in (None, state) else []


def _register_history(rng: SimRandom) -> list[Operation]:
    ops = []
    for index in range(rng.randint(1, 8)):
        invoke = rng.randint(0, 20)
        pending = rng.below(10) == 0
        response = None if pending else invoke + rng.randint(0, 10)
        if rng.below(2) == 0:
            args: tuple[Hashable, ...] = (rng.randint(0, 2),)
            output: Hashable | None = None if pending else "ok"
            ops.append(Operation(index, f"c{index % 3}", "write", args, invoke, response, output))
        else:
            output = None if pending else rng.randint(0, 2)
            ops.append(Operation(index, f"c{index % 3}", "read", (), invoke, response, output))
    return ops


class _Clock:
    now_us = 0


_LEDGER_OUTPUTS: dict[str, tuple[Hashable, ...]] = {
    "consume": ("ok", "already_consumed", "expired", "not_approved", "hash_mismatch"),
    "status": (("approved", 180), ("consumed", None), ("expired", None), ("pending", None)),
    "decide": (("ok", 160), ("not_pending", None), ("forbidden", None)),
}


def _ledger_history(rng: SimRandom, ttl: int) -> list[Operation]:
    """A real ledger execution observed with random latencies; half get one output perturbed."""
    clock = _Clock()
    ledger = ApprovalLedger(cast("SimEventLoop", clock), ttl_us=ttl)
    ops: list[Operation] = []
    now = 0

    def observe(kind: str, args: tuple[Hashable, ...], output: Hashable, client: str) -> None:
        pending = rng.below(7) == 0
        invoke = max(0, now - rng.randint(0, 25))
        response = None if pending else now + rng.randint(0, 25)
        ops.append(
            Operation(len(ops), client, kind, args, invoke, response, None if pending else output)
        )

    approval = ledger.apply_request("tok", "h", {}, "agent")
    observe("request", ("tok", "h", "agent"), approval, "agent")
    for _ in range(rng.randint(1, 7)):
        now += rng.randint(0, 70)
        clock.now_us = now
        choice = rng.choice(("decide", "status", "consume"))
        if choice == "decide":
            approve = rng.below(2) == 0
            who = rng.choice(("human", "agent"))
            result = ledger.apply_decide(approval, approve, who)
            observe("decide", (approval, approve, who), result, who)
        elif choice == "status":
            observe("status", (approval,), ledger.apply_status(approval), "auditor")
        else:
            action, consumer = rng.choice(("h", "other")), rng.choice(("c1", "c2"))
            result_c = ledger.apply_consume(approval, action, consumer)
            observe("consume", (approval, action, consumer), result_c, consumer)
    completed = [i for i, op in enumerate(ops) if op.completed and op.kind in _LEDGER_OUTPUTS]
    if completed and rng.below(2) == 0:
        index = rng.choice(completed)
        victim = ops[index]
        replacement = rng.choice(_LEDGER_OUTPUTS[victim.kind])
        ops[index] = Operation(
            victim.id,
            victim.client,
            victim.kind,
            victim.args,
            victim.invoke_us,
            victim.response_us,
            replacement,
        )
    return ops


def _fixtures(ttl: int) -> list[tuple[str, list[Operation], bool]]:
    def req(i: int, a: int, b: int | None) -> Operation:
        return Operation(i, "agent", "request", ("tok", "h", "agent"), a, b, "apr-1")

    def dec(i: int, a: int, b: int | None, out: Hashable | None) -> Operation:
        return Operation(i, "human", "decide", ("apr-1", True, "human"), a, b, out)

    def con(i: int, a: int, b: int | None, out: Hashable | None, who: str = "c1") -> Operation:
        return Operation(i, who, "consume", ("apr-1", "h", who), a, b, out)

    def sta(i: int, a: int, b: int | None, out: Hashable | None) -> Operation:
        return Operation(i, "auditor", "status", ("apr-1",), a, b, out)

    base = [req(0, 0, 1), dec(1, 2, 3, ("ok", 3 + ttl))]
    return [
        ("idempotent consume retry", [*base, con(2, 10, 11, "ok"), con(3, 20, 21, "ok")], True),
        (
            "second consumer refused",
            [*base, con(2, 10, 11, "ok"), con(3, 20, 21, "already_consumed", "c2")],
            True,
        ),
        ("expired after ttl", [*base, con(2, 200, 210, "expired")], True),
        (
            "pending consume may have happened",
            [*base, con(2, 5, None, None), con(3, 10, 11, "already_consumed", "c2")],
            True,
        ),
        (
            "consumed by two consumers",
            [*base, con(2, 10, 11, "ok"), con(3, 20, 21, "ok", "c2")],
            False,
        ),
        (
            "consume rolled back by a crash",
            [*base, con(2, 10, 11, "ok"), sta(3, 20, 21, ("approved", 3 + ttl))],
            False,
        ),
        ("consumed after expiry", [*base, con(2, 200, 210, "ok")], False),
        ("consumed without approval", [req(0, 0, 1), con(1, 2, 3, "ok")], False),
    ]


def _reference_agreement(profile: Profile, rng: SimRandom, ttl: int) -> dict[str, Any]:
    """Wing-Gong with the approval model vs an independent checker of the real ledger (ADR 0004)."""
    model = ApprovalModel(ttl)
    agree = linearizable = 0
    started = time.perf_counter()
    for _ in range(profile.checker_histories):
        history = _ledger_history(rng, ttl)[:MAX_REFERENCE_OPS]
        expected = reference_linearizable(history, ttl)
        agree += int(wing_gong_check(history, model).ok == expected)
        linearizable += int(expected)
    return {
        "histories": profile.checker_histories,
        "agreements": agree,
        "linearizable": linearizable,
        "not_linearizable": profile.checker_histories - linearizable,
        "seconds": round(time.perf_counter() - started, 1),
    }


def checker(profile: Profile) -> dict[str, Any]:
    rng = SimRandom.from_seed(20260922)
    ttl = 100
    models: list[tuple[str, Callable[[SimRandom], list[Operation]], Any]] = [
        ("register", _register_history, RegisterModel()),
        ("approval_ledger", lambda r: _ledger_history(r, ttl), ApprovalModel(ttl)),
    ]
    rows: dict[str, Any] = {}
    disagreements = 0
    for name, generate, model in models:
        agree = linearizable = 0
        started = time.perf_counter()
        for _ in range(profile.checker_histories):
            history = generate(rng)
            fast = wing_gong_check(history, model).ok
            slow = brute_force_check(history, model)
            agree += int(fast == slow)
            linearizable += int(slow)
        disagreements += profile.checker_histories - agree
        rows[name] = {
            "histories": profile.checker_histories,
            "agreements": agree,
            "linearizable": linearizable,
            "not_linearizable": profile.checker_histories - linearizable,
            "seconds": round(time.perf_counter() - started, 1),
        }
    reference = _reference_agreement(profile, rng, ttl)
    fixtures = _fixtures(ttl)
    model = ApprovalModel(ttl)
    fixtures_ok = sum(1 for _, ops, want in fixtures if wing_gong_check(ops, model).ok == want)
    return {
        "models": rows,
        "disagreements": disagreements,
        "reference": reference,
        "reference_disagreements": reference["histories"] - reference["agreements"],
        "fixtures": len(fixtures),
        "fixtures_ok": fixtures_ok,
    }


# ---------------------------------------------------------------------------------- scheduler
def _semantic(result: RunResult) -> list[str]:
    if result.trace is None:
        raise ValueError("the run was not recorded with keep_trace=True")
    return [event.render() for event in result.trace.events]


def scheduler(profile: Profile) -> dict[str, Any]:
    points: list[int] = []
    divergent = 0
    for seed in range(profile.scheduler_seeds):
        seeded = run_scenario(Scenario(seed=seed, config=SimConfig()), keep_trace=True)
        fifo_config = SimConfig(scheduler="fifo")
        fifo = run_scenario(Scenario(seed=seed, config=fifo_config), keep_trace=True)
        points.append(seeded.schedule_points)
        divergent += int(_semantic(seeded) != _semantic(fifo))
    return {
        "seeds": profile.scheduler_seeds,
        "median_choice_points": statistics.median(points),
        "max_choice_points": max(points),
        "semantically_divergent_seeds": divergent,
    }


# ------------------------------------------------------------------------------------ driver
def machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def summary_metrics(results: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "sensitivity" in results:
        metrics["bugs_found"] = results["sensitivity"]["bugs_found"]
    if "soundness" in results:
        metrics["soundness_violations"] = results["soundness"]["violations"]
        metrics["seeds_per_second"] = results["soundness"]["seeds_per_second"]
    if "shrinking" in results and results["shrinking"]["median_ratio"] is not None:
        metrics["median_shrink_ratio"] = results["shrinking"]["median_ratio"]
        shrinking_result = results["shrinking"]
        metrics["shrink_invariant_changed"] = (
            shrinking_result["shrinks"] - shrinking_result["invariant_preserved"]
        )
    if "determinism" in results:
        metrics["determinism_mismatched_seeds"] = results["determinism"]["mismatched_seeds"]
        metrics["golden_mismatches"] = results["determinism"]["golden_mismatches"]
    if "checker" in results:
        metrics["checker_disagreements"] = results["checker"]["disagreements"]
        metrics["checker_reference_disagreements"] = results["checker"]["reference_disagreements"]
        metrics["checker_fixture_failures"] = (
            results["checker"]["fixtures"] - results["checker"]["fixtures_ok"]
        )
    return metrics


def check_thresholds(metrics: dict[str, float]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name not in metrics:
            continue  # section not run (--only)
        value = metrics[name]
        if "min" in limits and value < limits["min"]:
            problems.append(f"{name} = {value} is below the minimum {limits['min']}")
        if "max" in limits and value > limits["max"]:
            problems.append(f"{name} = {value} exceeds the maximum {limits['max']}")
    return problems


def _print_tables(results: dict[str, Any]) -> None:
    if "sensitivity" in results:
        sens = results["sensitivity"]
        print(f"\nSensitivity (searches per bug and mode, limit {sens['limit']} seeds each)\n")
        print(
            "| bug | expected invariant | swarm found | swarm median | fixed found | fixed median |"
        )
        print("|---|---|---:|---:|---:|---:|")
        for bug, row in sens["bugs"].items():
            s, f = row["swarm"], row["fixed"]
            print(
                f"| {bug} | {row['expected_invariant']} | {s['found']}/{s['searches']} | "
                f"{s['median_seeds_to_failure']} | {f['found']}/{f['searches']} | "
                f"{f['median_seeds_to_failure']} |"
            )
        print(f"\nbugs found in every search: {sens['bugs_found']}/8")
    if "soundness" in results:
        snd = results["soundness"]
        print(
            f"\nSoundness: {snd['violations']} violations in {snd['seeds']} seeds "
            f"(swarm, all bugs off); {snd['seeds_per_second']} seeds/s = "
            f"{snd['seeds_per_minute']} seeds/min"
        )
    if "shrinking" in results:
        shr = results["shrinking"]
        print("\nShrinking (fixed-mode failures)\n")
        print("| bug | seed | trace events before -> after | ratio | ddmin tests | kept t/f/d |")
        print("|---|---:|---|---:|---:|---|")
        for bug, rows in shr["per_bug"].items():
            for row in rows:
                print(
                    f"| {bug} | {row['seed']} | {row['original']} -> {row['shrunk']} | "
                    f"{row['ratio']}x | {row['tests']} | "
                    f"{row['tickets']}/{row['faults']}/{row['decisions']} |"
                )
        print(
            f"\nmedian reduction {shr['median_ratio']}x (min {shr['min_ratio']}x, max "
            f"{shr['max_ratio']}x) over {shr['shrinks']} shrinks; same invariant after "
            f"shrinking: {shr['invariant_preserved']}/{shr['shrinks']}"
        )
    if "determinism" in results:
        det = results["determinism"]
        print(
            f"\nDeterminism: {det['seeds']} seeds x {det['reruns_per_seed']} reruns, "
            f"{det['mismatched_seeds']} seeds with more than one hash; golden hashes "
            f"{det['golden_seeds'] - det['golden_mismatches']}/{det['golden_seeds']} match "
            f"on {det['platform']}"
        )
    if "checker" in results:
        chk = results["checker"]
        print("\nChecker: Wing-Gong vs brute force (histories of 1-8 operations)\n")
        print("| model | histories | agreements | linearizable | not linearizable |")
        print("|---|---:|---:|---:|---:|")
        for name, row in chk["models"].items():
            print(
                f"| {name} | {row['histories']} | {row['agreements']} | {row['linearizable']} | "
                f"{row['not_linearizable']} |"
            )
        ref = chk["reference"]
        print(
            f"\nApproval model vs the independent reference checker (real ledger, at most "
            f"{MAX_REFERENCE_OPS} operations): {ref['agreements']}/{ref['histories']} agree "
            f"({ref['not_linearizable']} of them not linearizable)"
        )
        print(f"known fixtures classified correctly: {chk['fixtures_ok']}/{chk['fixtures']}")
    if "scheduler" in results:
        sch = results["scheduler"]
        print(
            f"\nScheduler: median {sch['median_choice_points']} tie-break choices per run "
            f"(max {sch['max_choice_points']}); the seeded order changed the semantic trace "
            f"of {sch['semantically_divergent_seeds']}/{sch['seeds']} seeds compared with FIFO"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simulator evaluation")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--only", default=",".join(SECTIONS), help="comma-separated sections")
    parser.add_argument("--soundness-seeds", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="enforce bench/sim_thresholds.json")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    if args.soundness_seeds is not None:
        profile = replace(profile, soundness_seeds=args.soundness_seeds)
    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = sorted(set(wanted) - set(SECTIONS))
    if unknown:
        parser.error(f"unknown sections: {', '.join(unknown)}")

    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[BugFlag, list[RunResult]] = {}
    if "sensitivity" in wanted or "shrinking" in wanted:
        results["sensitivity"], failures = sensitivity(profile)
    if "soundness" in wanted:
        results["soundness"] = soundness(profile)
    if "shrinking" in wanted:
        results["shrinking"] = shrinking(profile, failures)
    if "determinism" in wanted:
        results["determinism"] = determinism(profile)
    if "checker" in wanted:
        results["checker"] = checker(profile)
    if "scheduler" in wanted:
        results["scheduler"] = scheduler(profile)
    metrics = summary_metrics(results)
    elapsed = round(time.perf_counter() - started, 1)
    host = machine()
    document = {
        "machine": host,
        "profile": {"name": args.profile, **asdict(profile)},
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _print_tables(results)
    print(f"\nmetrics: {json.dumps(metrics, sort_keys=True)}")
    print(f"elapsed: {elapsed}s on {host['platform']}")
    if args.check:
        problems = check_thresholds(metrics)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all simulator metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``deterministic-agent-simulator`` (alias ``das``) or ``python -m ...``.

Every command prints JSON on stdout and signals failure through its exit code, so CI jobs can
gate on it: ``campaign --expect-clean`` exits 1 when any seed violates an invariant,
``campaign --expect-bug`` exits 1 when a planted bug is *not* found, ``hashes --compare`` exits 1
when a trace hash differs from a golden file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from deterministic_agent_simulator.agent.bugs import BUG_CATALOG, parse_bug_flags
from deterministic_agent_simulator.core.config import get_settings

__all__ = ["app", "main"]

app = typer.Typer(
    name="deterministic-agent-simulator",
    help="Deterministic simulation testing for LLM agents.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
audit_app = typer.Typer(help="Hash-chained audit log tools.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")

Mode = Literal["swarm", "fixed", "none"]
BugsOption = Annotated[
    str, typer.Option("--bugs", help="Comma-separated bug flags to plant ('all' for every flag).")
]
ModeOption = Annotated[str, typer.Option("--mode", help="Fault configuration: swarm, fixed, none.")]


def _echo(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _write_json(path: Path, payload: Any) -> None:
    """Sorted, indented JSON with LF line endings, so files compare byte-for-byte across OSes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _config(bugs: str, mode: str) -> Any:
    from deterministic_agent_simulator.runner import SimConfig  # noqa: PLC0415

    if mode not in ("swarm", "fixed", "none"):
        raise typer.BadParameter("mode must be swarm, fixed or none")
    try:
        flags = parse_bug_flags(bugs.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return SimConfig(
        bugs=flags,
        fault_mode="swarm" if mode == "swarm" else "fixed" if mode == "fixed" else "none",
    )


@app.command()
def run(
    seed: int,
    bugs: BugsOption = "",
    mode: ModeOption = "swarm",
    trace: Annotated[bool, typer.Option(help="Print the (redacted) trace.")] = False,
) -> None:
    """Run one seed and print its result."""
    from deterministic_agent_simulator.report import trace_lines  # noqa: PLC0415
    from deterministic_agent_simulator.runner import Scenario, run_scenario  # noqa: PLC0415

    result = run_scenario(Scenario(seed=seed, config=_config(bugs, mode)), keep_trace=trace)
    payload: dict[str, Any] = dict(result.summary())
    if trace and result.trace is not None:
        payload["trace"] = trace_lines(result.trace)
    _echo(payload)
    if result.violated:
        raise typer.Exit(code=1)


@app.command()
def campaign(
    seeds: Annotated[int, typer.Option(help="Number of seeds to run.")] = 1_000,
    start: Annotated[int, typer.Option(help="First seed.")] = 0,
    bugs: BugsOption = "",
    mode: ModeOption = "swarm",
    max_seconds: Annotated[float, typer.Option(help="Wall-clock cap in seconds.")] = 600.0,
    out: Annotated[Path | None, typer.Option(help="Write the summary JSON here.")] = None,
    metrics_out: Annotated[Path | None, typer.Option(help="Write Prometheus metrics here.")] = None,
    repro_dir: Annotated[
        Path | None, typer.Option(help="Shrink failures into signed repros here.")
    ] = None,
    max_repros: Annotated[int, typer.Option(help="Shrink at most this many failures.")] = 5,
    expect_clean: Annotated[bool, typer.Option(help="Exit 1 if any seed fails.")] = False,
    expect_bug: Annotated[bool, typer.Option(help="Exit 1 if no seed fails.")] = False,
    stop_on_first: Annotated[bool, typer.Option(help="Stop at the first failure.")] = False,
) -> None:
    """Run a campaign of seeds (capped by APP_SIM_MAX_SEEDS and APP_SIM_MAX_WALL_SECONDS)."""
    from deterministic_agent_simulator.runner import RunResult, run_campaign  # noqa: PLC0415
    from deterministic_agent_simulator.sim.metrics import create_sim_metrics  # noqa: PLC0415

    settings = get_settings()
    config = _config(bugs, mode)
    metrics = create_sim_metrics()
    failures: list[RunResult] = []

    def observe(result: RunResult) -> None:
        first = result.first
        metrics.observe_run(mode, result.trace_length, first.invariant if first else None)
        if first is not None and len(failures) < max_repros:
            failures.append(result)

    report = run_campaign(
        config,
        seeds=range(start, start + seeds),
        max_seeds=min(seeds, settings.sim_max_seeds),
        max_wall_seconds=min(max_seconds, settings.sim_max_wall_seconds),
        stop_on_first=stop_on_first,
        on_result=observe,
    )
    metrics.set_throughput(report.seeds_per_second)
    summary: dict[str, Any] = dict(report.to_json())
    if repro_dir is not None and failures:
        summary["repros"] = _write_repros(failures, repro_dir)
    if out is not None:
        _write_json(out, summary)
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_bytes(metrics.render())
    _echo(summary)
    if (expect_clean and report.failing) or (expect_bug and not report.failing):
        raise typer.Exit(code=1)


def _write_repros(
    failures: list[Any], folder: Path, *, max_tests: int = 600
) -> list[dict[str, Any]]:
    from deterministic_agent_simulator.emit import emit_regression_test  # noqa: PLC0415
    from deterministic_agent_simulator.replay import (  # noqa: PLC0415
        Keyring,
        ReproError,
        build_repro,
        write_repro,
    )
    from deterministic_agent_simulator.report import failure_report  # noqa: PLC0415
    from deterministic_agent_simulator.shrink import shrink_failure  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    try:
        keyring: Keyring | None = Keyring.from_settings()
    except ReproError:
        keyring = None  # still write the redacted reports; signing needs APP_REPLAY_HMAC_KEY
    written: list[dict[str, Any]] = []
    for failing in failures:
        shrunk = shrink_failure(failing, max_tests=max_tests)
        first = failing.first
        name = f"seed_{failing.scenario.seed}_{first.invariant}"
        report_path = folder / f"report_{name}.json"
        _write_json(report_path, failure_report(failing, shrunk))
        entry: dict[str, Any] = {
            "seed": failing.scenario.seed,
            "invariant": first.invariant,
            "ratio": round(shrunk.ratio, 2),
            "report": report_path.name,
        }
        if keyring is not None:
            document = build_repro(
                shrunk.shrunk,
                violation=first,
                trace_hash=shrunk.final.trace_hash,
                trace_length=shrunk.shrunk_length,
                original_trace_length=shrunk.original_length,
                keyring=keyring,
            )
            repro_path = folder / f"repro_{name}.json"
            write_repro(repro_path, document)
            entry["repro"] = repro_path.name
            entry["test"] = emit_regression_test(repro_path, document).name
        written.append(entry)
    return written


@app.command()
def shrink(
    seed: int,
    bugs: BugsOption = "",
    mode: ModeOption = "swarm",
    out_dir: Annotated[Path, typer.Option(help="Where to write the repro and test.")] = Path(
        "var/regressions"
    ),
    max_tests: Annotated[int, typer.Option(help="ddmin test budget.")] = 600,
) -> None:
    """Shrink a failing seed to a signed replay file plus a generated regression test."""
    from deterministic_agent_simulator.runner import Scenario, run_scenario  # noqa: PLC0415

    result = run_scenario(Scenario(seed=seed, config=_config(bugs, mode)))
    if not result.violated:
        _echo({"seed": seed, "violated": False})
        raise typer.Exit(code=1)
    _echo({"seed": seed, "written": _write_repros([result], out_dir, max_tests=max_tests)})


@app.command()
def replay(
    path: Path,
    build: Annotated[str, typer.Option(help="recorded or current bug flags.")] = "recorded",
) -> None:
    """Verify a signed replay file and run it."""
    from deterministic_agent_simulator.replay import ReproError, replay_file  # noqa: PLC0415

    if build not in ("recorded", "current"):
        raise typer.BadParameter("build must be recorded or current")
    try:
        outcome = replay_file(path, build="recorded" if build == "recorded" else "current")
    except ReproError as exc:
        _echo({"ok": False, "error": str(exc)})
        raise typer.Exit(code=2) from exc
    _echo(
        {
            "build_flags": outcome.build_flags,
            "invariant": outcome.invariant,
            "expected_invariant": outcome.expected["invariant"],
            "trace_hash": outcome.trace_hash,
            "trace_hash_matches": outcome.trace_hash == outcome.expected_trace_hash,
            "report": outcome.report,
        }
    )
    if outcome.violations:
        raise typer.Exit(code=1)


@app.command()
def hashes(
    start: Annotated[int, typer.Option(help="First seed.")] = 0,
    count: Annotated[int, typer.Option(help="Number of seeds.")] = 200,
    bugs: BugsOption = "",
    mode: ModeOption = "swarm",
    out: Annotated[Path | None, typer.Option(help="Write {seed: hash} JSON here.")] = None,
    compare: Annotated[Path | None, typer.Option(help="Golden file to compare with.")] = None,
) -> None:
    """Trace hashes of a range of seeds (cross-platform determinism check)."""
    from deterministic_agent_simulator.runner import Scenario, run_scenario  # noqa: PLC0415

    config = _config(bugs, mode)
    computed = {
        str(seed): run_scenario(Scenario(seed=seed, config=config)).trace_hash
        for seed in range(start, start + count)
    }
    document = {"config": config.to_json(), "hashes": computed}
    if out is not None:
        _write_json(out, document)
    mismatches: list[str] = []
    if compare is not None:
        golden = json.loads(compare.read_text(encoding="utf-8"))["hashes"]
        mismatches = sorted((s for s in computed if golden.get(s) != computed[s]), key=int)
    _echo({"seeds": count, "compared": compare is not None, "mismatches": mismatches})
    if mismatches:
        raise typer.Exit(code=1)


@app.command()
def bugs() -> None:
    """List the plantable bugs and the invariant each should violate."""
    _echo(
        {
            flag.value: {
                "component": info.component,
                "wrong": info.wrong,
                "right": info.right,
                "expected_invariant": info.expected_invariant,
                "trigger": info.trigger,
            }
            for flag, info in BUG_CATALOG.items()
        }
    )


@app.command()
def demo(
    workdir: Annotated[Path, typer.Option(help="Output directory.")] = Path("var/demo"),
    clean_seeds: Annotated[int, typer.Option(help="Seeds for the soundness step.")] = 500,
) -> None:
    """Find, shrink, sign and regression-test a planted bug, then check soundness (offline)."""
    from deterministic_agent_simulator.core.logging import configure_logging  # noqa: PLC0415
    from deterministic_agent_simulator.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    _echo(run_demo(workdir, clean_seeds=clean_seeds))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Port.")] = None,
) -> None:
    """Serve the read-only results API."""
    import uvicorn  # noqa: PLC0415

    from deterministic_agent_simulator.api import create_results_app  # noqa: PLC0415
    from deterministic_agent_simulator.core.logging import configure_logging  # noqa: PLC0415
    from deterministic_agent_simulator.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    application = create_results_app(settings)
    install_global_tracer_provider(application.state.tracer_provider)
    uvicorn.run(
        application,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )


@app.command("generate-api-key")
def generate_api_key() -> None:
    """Print a new API key and its SHA-256 digest (store only the digest)."""
    from deterministic_agent_simulator.core.security import (  # noqa: PLC0415
        generate_api_key as make_key,
    )

    raw, digest = make_key()
    _echo({"api_key": raw, "api_key_sha256": digest})
    typer.echo("Store only api_key_sha256 in APP_API_KEY_SHA256.", err=True)


@audit_app.command("verify")
def audit_verify(
    path: Path,
    expect_head: Annotated[str | None, typer.Option(help="Expected head hash.")] = None,
) -> None:
    """Verify a hash-chained audit log (JSONL)."""
    from deterministic_agent_simulator.agentic.audit import main as audit_main  # noqa: PLC0415

    forwarded = ["verify", str(path)]
    if expect_head:
        forwarded += ["--expect-head", expect_head]
    code = audit_main(forwarded)
    if code:
        raise typer.Exit(code=code)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code (for ``python -m`` and tests).

    With ``standalone_mode=False`` the command's exit code is returned rather than raised, and
    usage errors propagate as exceptions; only Typer's public exception types are used here
    (Typer vendors its own copy of Click).
    """
    try:
        result = app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    except typer.Abort:
        return 1
    except typer.TyperException as exc:
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        return int(getattr(exc, "exit_code", 2))
    return result if isinstance(result, int) else 0

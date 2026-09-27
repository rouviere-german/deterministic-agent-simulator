from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from deterministic_agent_simulator import cli
from deterministic_agent_simulator.agent.bugs import BugFlag
from deterministic_agent_simulator.agentic.audit import AuditLog
from deterministic_agent_simulator.core import tracing
from deterministic_agent_simulator.core.security import hash_api_key
from deterministic_agent_simulator.demo import run_demo
from deterministic_agent_simulator.runner import SimConfig, find_failure

pytestmark = pytest.mark.integration


def run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    code = cli.main(args)
    out = capsys.readouterr().out
    try:
        payload = json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        payload = out
    return code, payload


@pytest.fixture(scope="module")
def dedupe_seed() -> int:
    found = find_failure(
        SimConfig(bugs=frozenset({BugFlag.MISSING_DEDUPE}), fault_mode="fixed"),
        start=0,
        max_seeds=200,
    )
    assert found is not None
    return found[0]


def test_bugs_lists_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run_cli(["bugs"], capsys)
    assert code == 0
    assert set(payload) == {flag.value for flag in BugFlag}


def test_run_prints_a_result_and_fails_on_a_violation(
    capsys: pytest.CaptureFixture[str], dedupe_seed: int
) -> None:
    code, payload = run_cli(["run", "3", "--trace"], capsys)
    assert code == 0
    assert payload["seed"] == 3
    assert payload["violations"] == []
    assert payload["trace"]
    code, payload = run_cli(
        ["run", str(dedupe_seed), "--bugs", "missing_dedupe", "--mode", "fixed"], capsys
    )
    assert code == 1
    assert payload["violations"][0]["invariant"] == "no_duplicate_effects"


def test_bad_options_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "1", "--mode", "chaos"]) == 2
    assert cli.main(["run", "1", "--bugs", "segfault"]) == 2
    assert cli.main(["replay", "x.json", "--build", "tomorrow"]) == 2
    capsys.readouterr()


def test_campaign_writes_summary_and_metrics_and_gates_ci(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, metrics = tmp_path / "c.json", tmp_path / "m.prom"
    code, payload = run_cli(
        [
            "campaign",
            "--seeds",
            "5",
            "--out",
            str(out),
            "--metrics-out",
            str(metrics),
            "--expect-clean",
        ],
        capsys,
    )
    assert code == 0
    assert payload["seeds_run"] == 5
    assert json.loads(out.read_text(encoding="utf-8"))["violations"] == {}
    assert "das_seeds_total" in metrics.read_text(encoding="utf-8")
    code, _ = run_cli(["campaign", "--seeds", "3", "--expect-bug"], capsys)
    assert code == 1


def test_campaign_shrinks_failures_into_signed_repros(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], replay_key_env: str
) -> None:
    folder = tmp_path / "repros"
    code, payload = run_cli(
        [
            "campaign",
            "--seeds",
            "200",
            "--bugs",
            "missing_dedupe",
            "--mode",
            "fixed",
            "--stop-on-first",
            "--expect-bug",
            "--repro-dir",
            str(folder),
            "--max-repros",
            "1",
        ],
        capsys,
    )
    assert code == 0
    written = payload["repros"][0]
    assert written["invariant"] == "no_duplicate_effects"
    repro = folder / written["repro"]
    assert repro.is_file()
    assert (folder / written["test"]).is_file()
    assert (folder / written["report"]).is_file()

    code, payload = run_cli(["replay", str(repro)], capsys)
    assert code == 1
    assert payload["trace_hash_matches"] is True
    code, payload = run_cli(["replay", str(repro), "--build", "current"], capsys)
    assert code == 0
    assert payload["invariant"] is None

    tampered = json.loads(repro.read_text(encoding="utf-8"))
    tampered["seed"] += 1
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(tampered), encoding="utf-8")
    code, payload = run_cli(["replay", str(forged)], capsys)
    assert code == 2
    assert payload["ok"] is False


def test_campaign_without_a_key_still_writes_redacted_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run_cli(
        [
            "campaign",
            "--seeds",
            "200",
            "--bugs",
            "missing_dedupe",
            "--mode",
            "fixed",
            "--stop-on-first",
            "--repro-dir",
            str(tmp_path),
            "--max-repros",
            "1",
        ],
        capsys,
    )
    assert code == 0
    assert "repro" not in payload["repros"][0]
    assert (tmp_path / payload["repros"][0]["report"]).is_file()


def test_shrink_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], replay_key_env: str, dedupe_seed: int
) -> None:
    code, payload = run_cli(
        [
            "shrink",
            str(dedupe_seed),
            "--bugs",
            "missing_dedupe",
            "--mode",
            "fixed",
            "--out-dir",
            str(tmp_path),
        ],
        capsys,
    )
    assert code == 0
    assert payload["written"][0]["repro"].startswith("repro_seed_")
    code, payload = run_cli(["shrink", "3", "--out-dir", str(tmp_path)], capsys)
    assert code == 1
    assert payload["violated"] is False


def test_hashes_compare_against_a_golden_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.json"
    code, payload = run_cli(["hashes", "--count", "3", "--out", str(golden)], capsys)
    assert code == 0
    code, payload = run_cli(["hashes", "--count", "3", "--compare", str(golden)], capsys)
    assert (code, payload["mismatches"]) == (0, [])
    document = json.loads(golden.read_text(encoding="utf-8"))
    document["hashes"]["1"] = "0" * 64
    golden.write_text(json.dumps(document), encoding="utf-8")
    code, payload = run_cli(["hashes", "--count", "3", "--compare", str(golden)], capsys)
    assert (code, payload["mismatches"]) == (1, ["1"])


@pytest.mark.usefixtures("restore_logging")
def test_demo_runs_offline_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run_cli(["demo", "--workdir", str(tmp_path), "--clean-seeds", "20"], capsys)
    assert code == 0
    assert payload["invariant"] == "no_duplicate_effects"
    assert payload["tampered_copy_rejected"] is True
    assert payload["regression_test_on_buggy_build"] == {"reproduces": "passed", "holds": "failed"}
    assert payload["regression_test_on_clean_build"] == {"reproduces": "passed", "holds": "passed"}
    assert payload["audit_chain_verified"] is True
    assert payload["clean_build"]["violations"] == {}
    assert payload["shrunk_trace_events"] < payload["original_trace_events"]
    assert (tmp_path / "failure_report.json").is_file()
    assert (tmp_path / "metrics.prom").is_file()


def test_demo_can_run_twice_in_the_same_workdir(tmp_path: Path) -> None:
    """`make demo` has to be repeatable: the second run may not trip over the first one's files."""
    first = run_demo(tmp_path, clean_seeds=5)
    second = run_demo(tmp_path, clean_seeds=5)
    assert first["replay_file"] == second["replay_file"]
    assert second["regression_test_on_clean_build"] == {"reproduces": "passed", "holds": "passed"}
    regressions = tmp_path / "regressions"
    written = sorted(p.name for p in regressions.glob("*.json"))
    assert written == [str(second["replay_file"])]


def test_demo_refuses_a_workdir_whose_regressions_it_does_not_own(tmp_path: Path) -> None:
    """`das demo --workdir tests` must not delete committed regression tests."""
    foreign = tmp_path / "regressions" / "repro_seed_4_no_duplicate_effects.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not write"):
        run_demo(tmp_path, clean_seeds=1)
    assert foreign.read_text(encoding="utf-8") == "{}"


def test_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run_cli(["generate-api-key"], capsys)
    assert code == 0
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("agent", "step", {})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["deterministic_agent_simulator", "bugs"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("deterministic_agent_simulator", run_name="__main__")
    assert exit_info.value.code == 0
    assert "missing_dedupe" in capsys.readouterr().out

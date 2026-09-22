from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from deterministic_agent_simulator.agent.runtime import ReferenceAgent
from deterministic_agent_simulator.runner import Scenario, run_scenario
from deterministic_agent_simulator.sim.hermetic import HermeticGuard, HermeticityViolation

pytestmark = pytest.mark.unit


def test_sockets_are_blocked_even_for_loopback(tmp_path: Path) -> None:
    with HermeticGuard([tmp_path]) as guard:
        with pytest.raises(HermeticityViolation):
            socket.create_connection(("127.0.0.1", 9))
        with pytest.raises(HermeticityViolation):
            socket.getaddrinfo("example.com", 443)
        sock = socket.socket()
        try:
            with pytest.raises(HermeticityViolation):
                sock.connect(("127.0.0.1", 9))
        finally:
            sock.close()
    assert len(guard.violations) == 3


def test_files_outside_the_allowed_roots_are_blocked(tmp_path: Path) -> None:
    inside = tmp_path / "scratch.txt"
    outside = Path(__file__)
    with HermeticGuard([tmp_path]) as guard:
        inside.write_text("ok", encoding="utf-8")
        assert inside.read_text(encoding="utf-8") == "ok"
        with pytest.raises(HermeticityViolation):
            outside.read_text(encoding="utf-8")
        with pytest.raises(HermeticityViolation):
            open(outside, encoding="utf-8")  # noqa: SIM115 - the call itself is what is blocked
        with pytest.raises(HermeticityViolation):
            os.open(outside, os.O_RDONLY)
    assert len(guard.violations) == 3
    assert outside.read_text(encoding="utf-8")  # restored afterwards


def test_real_time_and_processes_are_blocked(tmp_path: Path) -> None:
    with HermeticGuard([tmp_path]) as guard:
        with pytest.raises(HermeticityViolation):
            time.sleep(0.01)
        with pytest.raises(HermeticityViolation):
            subprocess.Popen(["echo"])  # noqa: S607 - blocked before it can run
    assert [v.split("(")[0] for v in guard.violations] == ["time.sleep", "subprocess.Popen"]


def test_everything_is_restored_and_nested_guards_share_one_record(tmp_path: Path) -> None:
    before = (socket.socket.connect, os.open, time.sleep, subprocess.Popen)
    with HermeticGuard([tmp_path]) as outer:
        with HermeticGuard([tmp_path]) as inner, pytest.raises(HermeticityViolation):
            time.sleep(0)
        assert inner.violations == []
        assert outer.violations == ["time.sleep"]
    assert (socket.socket.connect, os.open, time.sleep, subprocess.Popen) == before


def test_file_descriptors_are_allowed(tmp_path: Path) -> None:
    fd = os.open(tmp_path / "f", os.O_CREAT | os.O_WRONLY)
    with HermeticGuard([tmp_path]), open(fd, "w", encoding="utf-8") as handle:
        handle.write("x")


def test_a_simulated_agent_that_reads_a_real_file_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nosy_worker(self: ReferenceAgent) -> None:
        del self
        Path(__file__).read_text(encoding="utf-8")  # real I/O inside the simulation

    monkeypatch.setattr(ReferenceAgent, "_worker", nosy_worker)
    result = run_scenario(Scenario(seed=1))
    assert result.first is not None
    assert result.first.invariant == "hermetic"

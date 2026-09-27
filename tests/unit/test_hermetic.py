from __future__ import annotations

import datetime
import io
import os
import random
import secrets
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from deterministic_agent_simulator.agent.runtime import ReferenceAgent
from deterministic_agent_simulator.runner import Scenario, run_scenario
from deterministic_agent_simulator.sim.hermetic import (
    HermeticGuard,
    HermeticityViolation,
    simulation_scratch_root,
)

pytestmark = pytest.mark.unit

# A path that is never inside a simulation's allowed roots (the standard library).
OUTSIDE = Path(os.__file__)


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


def test_host_names_and_socket_traffic_are_blocked(tmp_path: Path) -> None:
    with HermeticGuard([tmp_path]) as guard:
        for call in (
            socket.gethostname,
            lambda: socket.gethostbyname_ex("localhost"),
            lambda: socket.gethostbyaddr("127.0.0.1"),
            lambda: socket.getnameinfo(("127.0.0.1", 9), 0),
            lambda: socket.getfqdn("localhost"),
        ):
            with pytest.raises(HermeticityViolation):
                call()
        sock = socket.socket()
        try:
            for method in (
                lambda: sock.send(b"x"),
                lambda: sock.sendall(b"x"),
                lambda: sock.listen(1),
            ):
                with pytest.raises(HermeticityViolation):
                    method()
        finally:
            sock.close()
    assert len(guard.violations) == 8


def test_files_outside_the_allowed_roots_are_blocked(tmp_path: Path) -> None:
    inside = tmp_path / "scratch.txt"
    with HermeticGuard([tmp_path]) as guard:
        inside.write_text("ok", encoding="utf-8")
        assert inside.read_text(encoding="utf-8") == "ok"
        with pytest.raises(HermeticityViolation):
            OUTSIDE.read_text(encoding="utf-8")
        with pytest.raises(HermeticityViolation):
            open(OUTSIDE, encoding="utf-8")  # noqa: SIM115 - the call itself is what is blocked
        with pytest.raises(HermeticityViolation):
            os.open(OUTSIDE, os.O_RDONLY)
        with pytest.raises(HermeticityViolation):
            io.FileIO(str(OUTSIDE))
    assert len(guard.violations) == 4
    assert OUTSIDE.read_text(encoding="utf-8")  # restored afterwards


def test_reading_and_changing_the_host_filesystem_is_blocked(tmp_path: Path) -> None:
    """Metadata and mutation, not only opens: the classic ways around a guard on ``open``.

    Every mutating call below names a path that does not exist, so a gap in the guard shows up as
    a failed assertion instead of a changed host.
    """
    away = OUTSIDE.parent  # the standard library directory
    absent = away / "das-absent" / "child"
    with HermeticGuard([tmp_path]) as guard:
        blocked: list[tuple[str, Callable[[], object]]] = [
            ("listdir", lambda: os.listdir(away)),
            ("scandir", lambda: list(os.scandir(away))),
            ("stat", lambda: os.stat(OUTSIDE)),
            ("lstat", lambda: os.lstat(OUTSIDE)),
            ("scandir", lambda: next(os.walk(away))),
            ("remove", lambda: os.remove(absent)),
            ("mkdir", lambda: os.mkdir(absent)),
            ("rename", lambda: os.rename(absent, absent.with_suffix(".moved"))),
            ("replace", lambda: os.replace(absent, absent.with_suffix(".moved"))),
            ("rmtree", lambda: shutil.rmtree(absent)),
            ("copy", lambda: shutil.copy(OUTSIDE, tmp_path / "copied.py")),
            ("stat", OUTSIDE.exists),
            ("listdir", lambda: list(away.iterdir())),
        ]
        for name, call in blocked:
            with pytest.raises(HermeticityViolation, match="blocked"):
                call()
            assert name in guard.violations[-1], name
        # The run's own scratch root stays usable.
        (tmp_path / "sub").mkdir()
        assert os.listdir(tmp_path) == ["sub"]
        assert os.stat(tmp_path / "sub").st_mode
        os.rmdir(tmp_path / "sub")
    assert len(guard.violations) == len(blocked)
    assert not (tmp_path / "sub").exists()
    assert not (tmp_path / "copied.py").exists()


def test_real_time_and_processes_are_blocked(tmp_path: Path) -> None:
    with HermeticGuard([tmp_path]) as guard:
        with pytest.raises(HermeticityViolation):
            time.sleep(0.01)
        with pytest.raises(HermeticityViolation):
            subprocess.Popen(["echo"])  # noqa: S607 - blocked before it can run
        with pytest.raises(HermeticityViolation):
            os.system("echo hi")  # noqa: S605, S607 - blocked before it can run
    assert [v.split("(")[0] for v in guard.violations] == [
        "time.sleep",
        "subprocess.Popen",
        "os.system",
    ]


def test_threads_are_blocked(tmp_path: Path) -> None:
    started: list[int] = []
    with HermeticGuard([tmp_path]) as guard:
        thread = threading.Thread(target=lambda: started.append(1))
        with pytest.raises(HermeticityViolation):
            thread.start()
    assert guard.violations == ["threading.Thread.start"]
    assert started == []


def test_the_host_clock_is_blocked_in_strict_mode(tmp_path: Path) -> None:
    with HermeticGuard([tmp_path]) as guard:
        for call in (
            time.time,
            time.time_ns,
            time.monotonic,
            time.perf_counter,
            time.localtime,
            datetime.datetime.now,
            datetime.datetime.utcnow,
            datetime.date.today,
        ):
            with pytest.raises(HermeticityViolation):
                call()
    assert len(guard.violations) == 8
    with HermeticGuard([tmp_path], strict=False) as lenient:
        assert time.time() > 0
        assert datetime.datetime.now(tz=datetime.UTC).year >= 2026
    assert lenient.violations == []


def test_host_entropy_is_blocked_in_strict_mode(tmp_path: Path) -> None:
    seeded = random.Random(7)  # a seeded generator is reproducible, so it stays allowed
    with HermeticGuard([tmp_path]) as guard:
        for call in (
            random.random,
            lambda: random.randint(0, 3),
            lambda: random.choice([1, 2]),
            lambda: random.shuffle([1, 2]),
            lambda: random.getrandbits(8),
            lambda: os.urandom(4),
            uuid.uuid4,
            lambda: secrets.token_hex(8),
        ):
            with pytest.raises(HermeticityViolation):
                call()
        assert 0 <= seeded.random() <= 1  # a seeded stream is not host entropy
    assert len(guard.violations) == 8
    with HermeticGuard([tmp_path], strict=False) as lenient:
        assert len(os.urandom(4)) == 4
    assert lenient.violations == []


def test_everything_is_restored_and_nested_guards_share_one_record(tmp_path: Path) -> None:
    before = (socket.socket.connect, os.open, os.stat, time.sleep, time.time, subprocess.Popen)
    with HermeticGuard([tmp_path]) as outer:
        with HermeticGuard([tmp_path]) as inner, pytest.raises(HermeticityViolation):
            time.sleep(0)
        assert inner.violations == []
        assert outer.violations == ["time.sleep"]
    assert (
        socket.socket.connect,
        os.open,
        os.stat,
        time.sleep,
        time.time,
        subprocess.Popen,
    ) == before
    assert datetime.datetime.now(tz=datetime.UTC).year >= 2026


def test_file_descriptors_are_allowed(tmp_path: Path) -> None:
    fd = os.open(tmp_path / "f", os.O_CREAT | os.O_WRONLY)
    with HermeticGuard([tmp_path]), open(fd, "w", encoding="utf-8") as handle:
        handle.write("x")


def test_a_simulated_agent_that_reads_a_real_file_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not OUTSIDE.is_relative_to(simulation_scratch_root())

    async def nosy_worker(self: ReferenceAgent) -> None:
        del self
        OUTSIDE.read_text(encoding="utf-8")  # real I/O inside the simulation

    monkeypatch.setattr(ReferenceAgent, "_worker", nosy_worker)
    result = run_scenario(Scenario(seed=1))
    assert result.first is not None
    assert result.first.invariant == "hermetic"


def test_a_simulated_agent_that_reads_the_host_clock_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def impatient_worker(self: ReferenceAgent) -> None:
        del self
        time.time()  # nondeterminism that no rerun would reproduce

    monkeypatch.setattr(ReferenceAgent, "_worker", impatient_worker)
    result = run_scenario(Scenario(seed=1))
    assert result.first is not None
    assert result.first.invariant == "hermetic"
    assert "time.time" in result.first.detail

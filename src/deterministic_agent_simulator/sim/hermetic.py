"""Hermeticity guard: real I/O and host nondeterminism raise while a simulation runs.

A simulation is reproducible only if everything it observes comes from its seed. Code under test
that opens a socket, reads a file, starts a thread, sleeps in real time, spawns a process, reads the
host clock or draws host entropy would observe the machine instead. Inside :class:`HermeticGuard`
all of the following raise :class:`HermeticityViolation`:

*Files.* ``open`` / ``io.open`` / ``io.FileIO`` / ``os.open``, the metadata and mutation calls
``os.stat``, ``lstat``, ``listdir``, ``scandir``, ``mkdir``, ``makedirs``, ``rmdir``, ``remove``,
``unlink``, ``rename``, ``replace``, ``link``, ``symlink``, ``readlink``, ``chmod``, ``truncate``
and the ``shutil`` copy/move/delete helpers, for every path outside the allowed roots. The roots
are what the caller passes (:func:`run_scenario` gives one scratch directory, see
:func:`simulation_scratch_root`), *not* the whole system temp directory. Already-open file
descriptors are allowed, and ``pathlib`` and ``os.walk`` are covered because they go through these
functions.

*Network.* ``socket.socket.connect`` / ``connect_ex`` / ``bind`` / ``listen`` / ``accept`` /
``send`` / ``sendall`` / ``sendto`` / ``recv``, ``socket.create_connection`` / ``create_server`` /
``socketpair`` and every name lookup (``getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``,
``gethostbyaddr``, ``getnameinfo``, ``gethostname``, ``getfqdn``), loopback included.

*Time, processes and threads.* ``time.sleep``, ``subprocess.Popen``, ``os.system``, ``os.popen``
and ``threading.Thread.start``.

*Host nondeterminism* (strict mode, on by default). Wall-clock reads (``time.time``, ``time_ns``,
``monotonic``, ``monotonic_ns``, ``perf_counter``, ``perf_counter_ns``, ``process_time``,
``localtime``, ``gmtime``, ``datetime.datetime.now`` / ``utcnow`` / ``today`` and
``datetime.date.today``) and process-wide entropy (the ``random`` module functions, ``os.urandom``,
``uuid.uuid1`` / ``uuid4``, the ``secrets`` helpers). A *seeded* ``random.Random`` instance keeps
working: it is reproducible, which is the whole point. ``strict=False`` leaves these channels alone
for code that cannot yet be made deterministic; a run then repeats only as far as that code does.

Every attempt is recorded in :attr:`HermeticGuard.violations` before the exception is raised, so a
violation is reported even when the code under test swallows the exception. The runner turns
recorded violations into a ``hermetic`` oracle failure. The simulated event loop blocks the asyncio
equivalents (``loop.getaddrinfo``, ``run_in_executor``...) itself.

Limits: this is a correctness aid, not a sandbox. It patches module and class attributes, so it is
process-wide while active and not thread-safe (campaigns run simulations one at a time on one
thread), a name imported directly before the guard was entered keeps the original, and C
extensions, ``ctypes`` or a fresh ``os`` handle can bypass it. Strict mode swaps
``datetime.datetime`` and ``datetime.date`` for subclasses, so ``isinstance(x, datetime.datetime)``
is False for an instance built before the guard was entered. Nested guards reuse the outer one.
"""

from __future__ import annotations

import builtins
import datetime
import io
import os
import random
import secrets
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn

from deterministic_agent_simulator.sim.loop import HermeticityViolation

__all__ = ["HermeticGuard", "HermeticityViolation", "simulation_scratch_root"]

_active: list[HermeticGuard] = []

_OPENERS = (("open", builtins), ("open", io), ("open", os), ("FileIO", io))
_ONE_PATH = (
    "stat",
    "lstat",
    "listdir",
    "scandir",
    "mkdir",
    "makedirs",
    "rmdir",
    "removedirs",
    "remove",
    "unlink",
    "readlink",
    "chmod",
    "truncate",
    "utime",
)
_TWO_PATHS = ("rename", "renames", "replace", "link", "symlink")
_SHUTIL_ONE_PATH = ("rmtree",)
_SHUTIL_TWO_PATHS = ("copy", "copy2", "copyfile", "copytree", "copymode", "copystat", "move")
_SOCKET_METHODS = (
    "connect",
    "connect_ex",
    "bind",
    "listen",
    "accept",
    "send",
    "sendall",
    "sendto",
    "sendmsg",
    "recv",
    "recvfrom",
)
_SOCKET_FUNCTIONS = (
    "create_connection",
    "create_server",
    "socketpair",
    "getaddrinfo",
    "getnameinfo",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostbyaddr",
    "gethostname",
    "getfqdn",
)
_CLOCKS = (
    "time",
    "time_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "thread_time",
    "localtime",
    "gmtime",
)
_ENTROPY = (
    "random",
    "randint",
    "randrange",
    "choice",
    "choices",
    "shuffle",
    "sample",
    "uniform",
    "gauss",
    "getrandbits",
    "randbytes",
)
_SECRETS = ("token_bytes", "token_hex", "token_urlsafe", "randbelow", "randbits", "choice")


def simulation_scratch_root() -> Path:
    """The one host directory a simulation may touch: ``<system temp>/das-sim``.

    A run needs no files at all; this exists so that code under test has somewhere to write while
    the rest of the machine - including a checkout that happens to live under the temp directory -
    stays invisible. The directory is not created here: creating it is itself an allowed call.
    """
    return Path(tempfile.gettempdir()) / "das-sim"


class HermeticGuard:
    """Context manager that blocks real I/O and host nondeterminism; see the module docstring."""

    def __init__(self, allowed_roots: Sequence[Path] = (), *, strict: bool = True) -> None:
        self._allowed = tuple(Path(root).resolve() for root in allowed_roots)
        self.strict = strict
        self.violations: list[str] = []
        self._saved: list[tuple[object, str, object]] = []
        self._outer: HermeticGuard | None = None
        self._busy = False

    # ----------------------------------------------------------------- checks
    def _violation(self, what: str) -> NoReturn:
        target = self._outer if self._outer is not None else self
        target.violations.append(what)
        raise HermeticityViolation(f"{what} is blocked inside a simulation")

    def _path_allowed(self, file: Any) -> bool:
        if isinstance(file, int):
            return True  # wrapping an already-open descriptor creates no new access
        try:
            candidate = Path(os.fsdecode(os.fspath(file))).resolve()
        except (TypeError, ValueError, OSError):
            return False
        return any(candidate.is_relative_to(root) for root in self._allowed)

    def _describe(self, file: Any) -> str:
        try:
            return repr(os.fsdecode(os.fspath(file)))
        except (TypeError, ValueError):
            return repr(file)

    def _candidates(self, args: tuple[Any, ...], kwargs: dict[str, Any], paths: int) -> list[Any]:
        """The path arguments of a call: positional first, then the usual keyword names."""
        found = list(args[:paths])
        for key in ("path", "file", "src", "dst", "top"):
            if len(found) < paths and key in kwargs:
                found.append(kwargs[key])
        return found or [Path.cwd()]  # os.listdir() and friends default to the process directory

    def _guard_paths(
        self, original: Callable[..., Any], name: str, paths: int
    ) -> Callable[..., Any]:
        """``original`` with its first ``paths`` path arguments checked against the roots."""

        def guarded(*args: Any, **kwargs: Any) -> Any:
            if not self._busy:
                self._busy = True
                try:
                    for argument in self._candidates(args, kwargs, paths):
                        if not self._path_allowed(argument):
                            self._violation(f"{name}({self._describe(argument)})")
                finally:
                    self._busy = False
            return original(*args, **kwargs)

        return guarded

    def _blocker(self, what: str) -> Callable[..., NoReturn]:
        def blocked(*args: Any, **kwargs: Any) -> NoReturn:
            del args, kwargs
            self._violation(what)

        return blocked

    # --------------------------------------------------------------- patching
    def _patch(self, owner: object, attribute: str, replacement: object) -> None:
        # Names that exist only on some platforms (os.symlink, socket.socket.sendmsg...) are
        # skipped where they are absent.
        original = getattr(owner, attribute, None)
        if original is None:
            return
        self._saved.append((owner, attribute, original))
        setattr(owner, attribute, replacement)

    def _patch_all(self, owner: object, attributes: Sequence[str], label: str, paths: int) -> None:
        """Patch every named path-taking function of ``owner`` with a guarded version."""
        for attribute in attributes:
            original = getattr(owner, attribute, None)
            if original is None:
                continue
            self._patch(
                owner, attribute, self._guard_paths(original, f"{label}.{attribute}", paths)
            )

    def _patch_files(self) -> None:
        for attribute, module in _OPENERS:
            name = f"{module.__name__}.{attribute}" if module is not builtins else attribute
            self._patch(module, attribute, self._guard_paths(getattr(module, attribute), name, 1))
        self._patch_all(os, _ONE_PATH, "os", 1)
        self._patch_all(os, _TWO_PATHS, "os", 2)
        self._patch_all(shutil, _SHUTIL_ONE_PATH, "shutil", 1)
        self._patch_all(shutil, _SHUTIL_TWO_PATHS, "shutil", 2)

    def _patch_network(self) -> None:
        for method in _SOCKET_METHODS:
            self._patch(socket.socket, method, self._blocker(f"socket.{method}"))
        for function in _SOCKET_FUNCTIONS:
            self._patch(socket, function, self._blocker(f"socket.{function}"))

    def _patch_processes(self) -> None:
        self._patch(time, "sleep", self._blocker("time.sleep"))
        self._patch(subprocess, "Popen", self._blocker("subprocess.Popen"))
        self._patch(os, "system", self._blocker("os.system"))
        self._patch(os, "popen", self._blocker("os.popen"))
        self._patch(threading.Thread, "start", self._blocker("threading.Thread.start"))

    def _patch_host_nondeterminism(self) -> None:
        for clock in _CLOCKS:
            self._patch(time, clock, self._blocker(f"time.{clock}"))
        self._patch(datetime, "datetime", _guarded_datetime(self))
        self._patch(datetime, "date", _guarded_date(self))
        for draw in _ENTROPY:
            self._patch(random, draw, self._blocker(f"random.{draw}"))
        self._patch(os, "urandom", self._blocker("os.urandom"))
        self._patch(uuid, "uuid1", self._blocker("uuid.uuid1"))
        self._patch(uuid, "uuid4", self._blocker("uuid.uuid4"))
        for helper in _SECRETS:
            self._patch(secrets, helper, self._blocker(f"secrets.{helper}"))

    def __enter__(self) -> HermeticGuard:
        if _active:
            self._outer = _active[-1]
            return self
        _active.append(self)
        try:
            self._patch_files()
            self._patch_network()
            self._patch_processes()
            if self.strict:
                self._patch_host_nondeterminism()
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        while self._saved:
            owner, attribute, original = self._saved.pop()
            setattr(owner, attribute, original)
        if _active and _active[-1] is self:
            _active.pop()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._outer is None:
            self._restore()


def _guarded_datetime(guard: HermeticGuard) -> type[datetime.datetime]:
    """``datetime.datetime`` whose host-clock constructors raise (arithmetic keeps working)."""

    class _GuardedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> NoReturn:
            del tz
            guard._violation("datetime.datetime.now")

        @classmethod
        def utcnow(cls) -> NoReturn:
            guard._violation("datetime.datetime.utcnow")

        @classmethod
        def today(cls) -> NoReturn:
            guard._violation("datetime.datetime.today")

    return _GuardedDatetime


def _guarded_date(guard: HermeticGuard) -> type[datetime.date]:
    class _GuardedDate(datetime.date):
        @classmethod
        def today(cls) -> NoReturn:
            guard._violation("datetime.date.today")

    return _GuardedDate

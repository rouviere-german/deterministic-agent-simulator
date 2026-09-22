"""Hermeticity guard: real I/O raises while a simulation runs.

A simulation is reproducible only if everything it observes comes from its seed. Code under test
that opens a socket, reads a file, sleeps in real time or spawns a process would observe the host
instead. Inside :class:`HermeticGuard`:

- ``socket.socket.connect``/``connect_ex``/``bind``/``sendto``, ``socket.create_connection``,
  ``socket.getaddrinfo`` and ``socket.gethostbyname`` raise (loopback included);
- ``open``/``io.open``/``os.open`` raise for paths outside the allowed roots (the run's tmp
  directory); already-open file descriptors are allowed;
- ``time.sleep``, ``subprocess.Popen`` and ``os.system`` raise (``subprocess`` is imported only
  to be patched, which is why bandit's B404 is suppressed on that import).

Every attempt is recorded in :attr:`HermeticGuard.violations` before the exception is raised, so
a violation is reported even when the code under test swallows the exception. The runner turns
recorded violations into a ``hermetic`` oracle failure. The simulated event loop blocks the
asyncio equivalents (``loop.getaddrinfo``, ``run_in_executor``...) itself.

The guard patches module attributes, so it is process-wide while active and not thread-safe:
campaigns run simulations one at a time on one thread. Nested guards reuse the outer one.
"""

from __future__ import annotations

import builtins
import io
import os
import socket
import subprocess  # nosec B404
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn

from deterministic_agent_simulator.sim.loop import HermeticityViolation

__all__ = ["HermeticGuard", "HermeticityViolation"]

_active: list[HermeticGuard] = []


class HermeticGuard:
    """Context manager that blocks real I/O; see the module docstring."""

    def __init__(self, allowed_roots: Sequence[Path] = ()) -> None:
        self._allowed = tuple(Path(root).resolve() for root in allowed_roots)
        self.violations: list[str] = []
        self._saved: list[tuple[object, str, object]] = []
        self._outer: HermeticGuard | None = None

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

    def _guard_open(self, original: Callable[..., Any], name: str) -> Callable[..., Any]:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            if not self._path_allowed(file):
                self._violation(f"{name}({os.fsdecode(os.fspath(file))!r})")
            return original(file, *args, **kwargs)

        return guarded

    def _blocker(self, what: str) -> Callable[..., NoReturn]:
        def blocked(*args: Any, **kwargs: Any) -> NoReturn:
            del args, kwargs
            self._violation(what)

        return blocked

    # --------------------------------------------------------------- patching
    def _patch(self, owner: object, attribute: str, replacement: object) -> None:
        self._saved.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, replacement)

    def __enter__(self) -> HermeticGuard:
        if _active:
            self._outer = _active[-1]
            return self
        _active.append(self)
        try:
            for method in ("connect", "connect_ex", "bind", "sendto"):
                self._patch(socket.socket, method, self._blocker(f"socket.{method}"))
            for function in ("create_connection", "getaddrinfo", "gethostbyname"):
                self._patch(socket, function, self._blocker(f"socket.{function}"))
            self._patch(builtins, "open", self._guard_open(builtins.open, "open"))
            self._patch(io, "open", self._guard_open(io.open, "io.open"))
            self._patch(os, "open", self._guard_open(os.open, "os.open"))
            self._patch(time, "sleep", self._blocker("time.sleep"))
            self._patch(subprocess, "Popen", self._blocker("subprocess.Popen"))
            self._patch(os, "system", self._blocker("os.system"))
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

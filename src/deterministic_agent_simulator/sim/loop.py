"""SimEventLoop: a single-threaded asyncio event loop with virtual time and seeded scheduling.

The loop implements :class:`asyncio.AbstractEventLoop` directly instead of patching
``BaseEventLoop`` internals, so the whole scheduling policy lives in ~100 lines that can be read
and tested:

- **Virtual clock.** Time is an integer number of microseconds. It never advances while a callback
  is runnable; when nothing is runnable it jumps straight to the earliest timer. A simulated
  hour of retries and backoff costs only the CPU needed to run the callbacks.
- **Seeded tie-breaker.** Standard asyncio runs ready callbacks in FIFO order. Here each iteration
  runs exactly *one* ready callback, chosen by a seeded :class:`SimRandom` (or FIFO when no RNG
  is given). Every interleaving of tasks that are ready at the same virtual instant is reachable,
  and the same seed always picks the same one. Each choice is folded into the trace hash.
- **Hermetic by construction.** Sockets, pipes, subprocesses, signals, executors and threads are
  not simulated: those methods raise :class:`HermeticityViolation`, because real I/O would make a
  run irreproducible. ``asyncio.to_thread`` and ``loop.getaddrinfo`` fail the same way.
- **Liveness guards.** A run that has neither runnable callbacks nor timers while its main future
  is unfinished is a deadlock (:class:`SimDeadlockError`); a run that exceeds ``max_steps``
  callbacks is a livelock (:class:`SimStepLimitError`).

Tasks get deterministic names (``sim-1``, ``sim-2``, ...) because asyncio's default names use a
process-wide counter that would differ between a first and a second run in the same process.
"""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Awaitable, Callable, Coroutine, Generator
from concurrent.futures import Executor
from contextvars import Context
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast

from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace

if TYPE_CHECKING:
    from _typeshed import FileDescriptorLike

__all__ = [
    "HermeticityViolation",
    "SimDeadlockError",
    "SimEventLoop",
    "SimHaltedError",
    "SimStepLimitError",
]

_T = TypeVar("_T")
_US_PER_S = 1_000_000


class HermeticityViolation(RuntimeError):  # noqa: N818 - reads as a verdict, like SystemExit
    """Code under simulation reached the host (network, files, threads, clock, entropy...)."""


class SimDeadlockError(RuntimeError):
    """Nothing is runnable and no timer is pending, but the simulation has not finished."""


class SimStepLimitError(RuntimeError):
    """The simulation ran more callbacks than allowed (a livelock or an unbounded loop)."""


class SimHaltedError(RuntimeError):
    """The loop was halted (for example by an oracle) before the main future completed."""


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _to_us(when: float) -> int:
    return round(when * _US_PER_S)


class SimEventLoop(asyncio.AbstractEventLoop):
    """Deterministic event loop; see the module docstring for the scheduling contract."""

    def __init__(
        self,
        *,
        rng: SimRandom | None,
        trace: Trace | None = None,
        max_steps: int = 1_000_000,
        start_us: int = 0,
    ) -> None:
        self._rng = rng
        self._trace = trace if trace is not None else Trace()
        self._max_steps = max_steps
        self._now_us = start_us
        self._ready: list[asyncio.Handle] = []
        self._timers: list[tuple[int, int, asyncio.TimerHandle]] = []
        self._timer_seq = 0
        self._steps = 0
        self._task_seq = 0
        self._tasks: list[asyncio.Task[Any]] = []
        self._running = False
        self._stopping = False
        self._closed = False
        self._debug = False
        self._exception_handler: (
            Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object] | None
        )
        self._exception_handler = None
        self._task_factory: Any = None
        self.callback_errors: list[str] = []

    # ------------------------------------------------------------------ time
    def time(self) -> float:
        return self._now_us / _US_PER_S

    @property
    def now_us(self) -> int:
        return self._now_us

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def trace(self) -> Trace:
        return self._trace

    def extend_step_limit(self, extra: int) -> None:
        """Allow ``extra`` more callbacks from now on (used for teardown after a verdict)."""
        self._max_steps = self._steps + extra

    async def sleep_us(self, delay_us: int) -> None:
        """Sleep for ``delay_us`` virtual microseconds (0 yields to other ready callbacks)."""
        if delay_us <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = self.create_future()
        handle = self.call_at_us(self._now_us + delay_us, _resolve, future)
        try:
            await future
        finally:
            handle.cancel()

    # ------------------------------------------------------------ scheduling
    def call_soon[*Ts](
        self, callback: Callable[[*Ts], object], *args: *Ts, context: Context | None = None
    ) -> asyncio.Handle:
        self._check_closed()
        handle = asyncio.Handle(callback, args, self, context)
        self._ready.append(handle)
        return handle

    def call_later[*Ts](
        self,
        delay: float,
        callback: Callable[[*Ts], object],
        *args: *Ts,
        context: Context | None = None,
    ) -> asyncio.TimerHandle:
        return self.call_at_us(self._now_us + _to_us(delay), callback, *args, context=context)

    def call_at[*Ts](
        self,
        when: float,
        callback: Callable[[*Ts], object],
        *args: *Ts,
        context: Context | None = None,
    ) -> asyncio.TimerHandle:
        return self.call_at_us(_to_us(when), callback, *args, context=context)

    def call_at_us[*Ts](
        self,
        when_us: int,
        callback: Callable[[*Ts], object],
        *args: *Ts,
        context: Context | None = None,
    ) -> asyncio.TimerHandle:
        """Schedule ``callback`` at an exact virtual time (never earlier than now)."""
        self._check_closed()
        due = max(when_us, self._now_us)
        handle = asyncio.TimerHandle(due / _US_PER_S, callback, args, self, context)
        heapq.heappush(self._timers, (due, self._timer_seq, handle))
        self._timer_seq += 1
        return handle

    def _timer_handle_cancelled(self, handle: asyncio.TimerHandle) -> None:
        """Cancelled timers are dropped lazily when they reach the top of the heap."""
        del handle

    def create_future(self) -> asyncio.Future[Any]:
        return asyncio.Future(loop=self)

    def create_task(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> asyncio.Task[_T]:
        self._check_closed()
        self._task_seq += 1
        task_name = name or f"sim-{self._task_seq}"
        task: asyncio.Task[_T]
        if self._task_factory is not None:
            task = self._task_factory(self, coro, context=context)
            task.set_name(task_name)
        else:
            task = asyncio.Task(coro, loop=self, name=task_name, context=context)
        self._tasks.append(task)
        return task

    def set_task_factory(self, factory: Any) -> None:
        self._task_factory = factory

    def get_task_factory(self) -> Any:
        return self._task_factory

    def pending_tasks(self) -> list[asyncio.Task[Any]]:
        """Unfinished tasks in creation order (deterministic, unlike ``asyncio.all_tasks``)."""
        return [task for task in self._tasks if not task.done()]

    # --------------------------------------------------------------- running
    def _run_once(self) -> None:
        if not self._ready:
            while self._timers and self._timers[0][2].cancelled():
                heapq.heappop(self._timers)
            if not self._timers:
                raise SimDeadlockError(
                    f"deadlock at t={self._now_us}us: no runnable callbacks and no timers"
                )
            self._now_us = max(self._now_us, self._timers[0][0])
        while self._timers and self._timers[0][0] <= self._now_us:
            _, _, timer = heapq.heappop(self._timers)
            if not timer.cancelled():
                self._ready.append(timer)
        count = len(self._ready)
        if count == 0:
            return
        index = 0
        if count > 1 and self._rng is not None:
            index = self._rng.below(count)
            self._trace.schedule(count, index)
        handle = self._ready.pop(index)
        if handle.cancelled():
            return
        self._steps += 1
        if self._steps > self._max_steps:
            raise SimStepLimitError(f"more than {self._max_steps} callbacks were run")
        handle._run()

    def _check_can_run(self) -> None:
        self._check_closed()
        if self._running:
            raise RuntimeError("this event loop is already running")
        if asyncio.events._get_running_loop() is not None:
            raise RuntimeError("cannot run a simulation inside another running event loop")

    def run_forever(self) -> None:
        self._check_can_run()
        asyncio.events._set_running_loop(self)
        self._running = True
        try:
            while True:
                self._run_once()
                if self._stopping:
                    break
        finally:
            self._stopping = False
            self._running = False
            asyncio.events._set_running_loop(None)

    def run_until_complete(self, future: Generator[Any, None, _T] | Awaitable[_T]) -> _T:
        self._check_can_run()
        # Generator-based coroutines are gone from asyncio in 3.12; everything left is awaitable.
        task = asyncio.ensure_future(cast("Awaitable[_T]", future), loop=self)

        def _done(_: asyncio.Future[Any]) -> None:
            self.stop()

        task.add_done_callback(_done)
        try:
            self.run_forever()
        finally:
            task.remove_done_callback(_done)
        if not task.done():
            raise SimHaltedError("the loop was halted before the main task completed")
        return task.result()

    def stop(self) -> None:
        self._stopping = True

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._running:
            raise RuntimeError("cannot close a running event loop")
        self._closed = True
        self._ready.clear()
        self._timers.clear()
        self._tasks.clear()

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("event loop is closed")

    async def shutdown_asyncgens(self) -> None:
        return None

    async def shutdown_default_executor(self, timeout: float | None = None) -> None:
        del timeout

    # ---------------------------------------------------------------- errors
    def get_debug(self) -> bool:
        return self._debug

    def set_debug(self, enabled: bool) -> None:
        self._debug = enabled

    def set_exception_handler(
        self, handler: Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object] | None
    ) -> None:
        self._exception_handler = handler

    def get_exception_handler(
        self,
    ) -> Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object] | None:
        return self._exception_handler

    def default_exception_handler(self, context: dict[str, Any]) -> None:
        # Errors raised by callbacks while the simulation runs are deterministic; errors reported
        # from Future.__del__ depend on garbage-collector timing and are ignored.
        if self._running and "future" not in context:
            exc = context.get("exception")
            label = type(exc).__name__ if exc is not None else "error"
            self.callback_errors.append(f"{label}: {context.get('message', '')}")

    def call_exception_handler(self, context: dict[str, Any]) -> None:
        if self._exception_handler is not None:
            self._exception_handler(self, context)
        else:
            self.default_exception_handler(context)

    # ------------------------------------------------- blocked real-world I/O
    def _blocked(self, what: str) -> NoReturn:
        raise HermeticityViolation(f"{what} is not available inside a simulation")

    def call_soon_threadsafe[*Ts](
        self, callback: Callable[[*Ts], object], *args: *Ts, context: Context | None = None
    ) -> asyncio.Handle:
        self._blocked("call_soon_threadsafe (threads)")

    def run_in_executor[*Ts, R](
        self, executor: Executor | None, func: Callable[[*Ts], R], *args: *Ts
    ) -> asyncio.Future[R]:
        self._blocked("run_in_executor (threads)")

    def set_default_executor(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("set_default_executor (threads)")

    async def getaddrinfo(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("getaddrinfo (DNS)")

    async def getnameinfo(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("getnameinfo (DNS)")

    async def create_connection(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("create_connection (network)")

    async def create_server(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("create_server (network)")

    async def sendfile(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sendfile (network)")

    async def start_tls(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("start_tls (network)")

    async def create_unix_connection(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("create_unix_connection (sockets)")

    async def create_unix_server(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("create_unix_server (sockets)")

    async def connect_accepted_socket(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("connect_accepted_socket (sockets)")

    async def create_datagram_endpoint(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("create_datagram_endpoint (network)")

    async def connect_read_pipe(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("connect_read_pipe (pipes)")

    async def connect_write_pipe(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("connect_write_pipe (pipes)")

    async def subprocess_shell(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("subprocess_shell (processes)")

    async def subprocess_exec(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("subprocess_exec (processes)")

    def add_reader[*Ts](
        self, fd: FileDescriptorLike, callback: Callable[[*Ts], Any], *args: *Ts
    ) -> None:
        self._blocked("add_reader (file descriptors)")

    def remove_reader(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("remove_reader (file descriptors)")

    def add_writer[*Ts](
        self, fd: FileDescriptorLike, callback: Callable[[*Ts], Any], *args: *Ts
    ) -> None:
        self._blocked("add_writer (file descriptors)")

    def remove_writer(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("remove_writer (file descriptors)")

    async def sock_recv(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_recv (sockets)")

    async def sock_recv_into(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_recv_into (sockets)")

    async def sock_recvfrom(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_recvfrom (sockets)")

    async def sock_recvfrom_into(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_recvfrom_into (sockets)")

    async def sock_sendall(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_sendall (sockets)")

    async def sock_sendto(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_sendto (sockets)")

    async def sock_connect(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_connect (sockets)")

    async def sock_accept(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_accept (sockets)")

    async def sock_sendfile(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("sock_sendfile (sockets)")

    def add_signal_handler[*Ts](
        self, sig: int, callback: Callable[[*Ts], object], *args: *Ts
    ) -> None:
        self._blocked("add_signal_handler (signals)")

    def remove_signal_handler(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._blocked("remove_signal_handler (signals)")

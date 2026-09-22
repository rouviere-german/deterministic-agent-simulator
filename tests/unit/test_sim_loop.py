from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deterministic_agent_simulator.sim.loop import (
    HermeticityViolation,
    SimDeadlockError,
    SimEventLoop,
    SimHaltedError,
    SimStepLimitError,
)
from deterministic_agent_simulator.sim.rand import SimRandom
from deterministic_agent_simulator.sim.trace import Trace

pytestmark = pytest.mark.unit


def _run[T](
    coro: Coroutine[Any, Any, T], *, seed: int | None = 1, **kwargs: Any
) -> tuple[T, SimEventLoop]:
    rng = SimRandom.from_seed(seed) if seed is not None else None
    loop = SimEventLoop(rng=rng, trace=Trace(), **kwargs)
    try:
        return loop.run_until_complete(coro), loop
    finally:
        loop.close()


async def _interleaved(log: list[str]) -> None:
    async def worker(name: str) -> None:
        for step in range(3):
            await asyncio.sleep(0)
            log.append(f"{name}{step}")

    await asyncio.gather(*(worker(name) for name in "abcd"))


def test_same_seed_gives_the_same_interleaving_and_hash() -> None:
    runs = []
    for _ in range(3):
        log: list[str] = []
        _, loop = _run(_interleaved(log), seed=42)
        runs.append((tuple(log), loop.trace.hexdigest()))
    assert len(set(runs)) == 1


def test_different_seeds_explore_different_interleavings() -> None:
    orders = set()
    for seed in range(12):
        log: list[str] = []
        _run(_interleaved(log), seed=seed)
        orders.add(tuple(log))
    assert len(orders) > 3


def test_fifo_mode_runs_ready_callbacks_in_order() -> None:
    log: list[str] = []
    _run(_interleaved(log), seed=None)
    assert log[:4] == ["a0", "b0", "c0", "d0"]


def test_virtual_time_jumps_without_waiting() -> None:
    async def slow() -> float:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(3600)
        return loop.time()

    started = time.perf_counter()
    virtual, loop = _run(slow())
    assert virtual == 3600.0
    assert loop.now_us == 3_600_000_000
    assert time.perf_counter() - started < 5.0


@settings(max_examples=40)
@given(delays=st.lists(st.integers(min_value=0, max_value=5_000_000), min_size=1, max_size=12))
def test_every_sleeper_wakes_exactly_at_its_virtual_deadline(delays: list[int]) -> None:
    async def main() -> list[tuple[int, int]]:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, SimEventLoop)
        woke: list[tuple[int, int]] = []

        async def sleeper(index: int, delay: int) -> None:
            await loop.sleep_us(delay)
            woke.append((index, loop.now_us))

        await asyncio.gather(*(sleeper(i, d) for i, d in enumerate(delays)))
        return woke

    woke, _ = _run(main(), seed=len(delays))
    assert sorted(woke) == sorted(enumerate(delays))


def test_asyncio_primitives_work_on_the_simulated_loop() -> None:
    async def main() -> list[str]:
        events: list[str] = []
        queue: asyncio.Queue[int] = asyncio.Queue()
        lock = asyncio.Lock()
        ready = asyncio.Event()

        async def producer() -> None:
            for item in range(3):
                await queue.put(item)
                await asyncio.sleep(0.01)
            ready.set()

        async def consumer() -> None:
            for _ in range(3):
                async with lock:
                    events.append(f"got{await queue.get()}")

        await asyncio.gather(producer(), consumer())
        await ready.wait()
        try:
            await asyncio.wait_for(asyncio.sleep(10), timeout=0.5)
        except TimeoutError:
            events.append("wait_for timed out")
        try:
            async with asyncio.timeout(0.25):
                await asyncio.sleep(1)
        except TimeoutError:
            events.append("timeout block")
        return events

    events, loop = _run(main())
    assert events == ["got0", "got1", "got2", "wait_for timed out", "timeout block"]
    assert loop.now_us == 30_000 + 500_000 + 250_000


def test_deadlock_is_detected() -> None:
    async def stuck() -> None:
        await asyncio.get_running_loop().create_future()

    with pytest.raises(SimDeadlockError):
        _run(stuck())


def test_livelock_hits_the_step_limit() -> None:
    async def spin() -> None:
        while True:
            await asyncio.sleep(0)

    with pytest.raises(SimStepLimitError):
        _run(spin(), max_steps=500)


def test_stop_before_completion_is_reported_as_halted() -> None:
    async def halts() -> None:
        asyncio.get_running_loop().stop()
        await asyncio.sleep(1)

    with pytest.raises(SimHaltedError):
        _run(halts())


@pytest.mark.parametrize(
    "operation",
    [
        lambda loop: loop.getaddrinfo("example.com", 80),
        lambda loop: loop.create_connection(lambda: None, "example.com", 80),
        lambda loop: loop.sock_connect(None, ("example.com", 80)),
        lambda loop: loop.subprocess_exec(lambda: None, "echo"),
        lambda loop: asyncio.to_thread(print, "hi"),
        lambda loop: loop.create_server(lambda: None, "127.0.0.1", 0),
    ],
)
def test_real_io_is_blocked_by_the_loop(operation: Any) -> None:
    async def attempt() -> None:
        await operation(asyncio.get_running_loop())

    with pytest.raises(HermeticityViolation):
        _run(attempt())


def test_blocked_synchronous_methods() -> None:
    loop = SimEventLoop(rng=None)
    with pytest.raises(HermeticityViolation):
        loop.call_soon_threadsafe(print)
    with pytest.raises(HermeticityViolation):
        loop.run_in_executor(None, print)
    with pytest.raises(HermeticityViolation):
        loop.add_reader(0, print)
    with pytest.raises(HermeticityViolation):
        loop.add_signal_handler(2, print)
    loop.close()
    assert loop.is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        loop.call_soon(print)


def test_tasks_get_deterministic_names_and_callback_errors_are_recorded() -> None:
    async def main() -> list[str]:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, SimEventLoop)

        def boom() -> None:
            raise ValueError("callback failed")

        loop.call_soon(boom)
        task = loop.create_task(asyncio.sleep(0))
        await task
        await asyncio.sleep(0)
        return [task.get_name()]

    names, loop = _run(main())
    assert names == ["sim-2"]
    assert any("ValueError" in error for error in loop.callback_errors)
    assert not loop.pending_tasks()


def test_cannot_nest_a_simulation_inside_a_running_loop() -> None:
    async def outer() -> None:
        inner = SimEventLoop(rng=None)
        pending = asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="another running event loop"):
            inner.run_until_complete(pending)
        await pending  # the rejected coroutine was never scheduled
        inner.close()

    _run(outer())


def test_exception_handler_and_debug_flags() -> None:
    loop = SimEventLoop(rng=None)
    seen: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    loop.call_exception_handler({"message": "x"})
    assert seen == [{"message": "x"}]
    assert loop.get_exception_handler() is not None
    loop.set_debug(True)
    assert loop.get_debug()
    assert loop.get_task_factory() is None
    loop.extend_step_limit(10)
    loop.close()

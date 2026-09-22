# 2. Implement the simulation kernel as a custom asyncio event loop

- Status: accepted
- Date: 2026-09-22

## Context

Deterministic simulation needs three things from the runtime: time that advances only when the
simulator says so, a controllable choice of which ready task runs next, and no path to real I/O.
The system under test is ordinary `async` Python (an agent with concurrent workers, locks,
timeouts and cancellation). The options considered:

1. **Patch time in the standard loop** (monkeypatch `time.monotonic`, or a library such as a
   mock clock). Callbacks still run in FIFO order, so interleavings of tasks ready at the same
   instant are never varied, and the standard loop's selector still performs real I/O.
2. **Subclass `asyncio.BaseEventLoop`** and override `_run_once`. It works, but depends on private
   internals (`_ready`, `_scheduled`, `_selector`) that change between CPython releases, and the
   selector and executor machinery would have to be neutralized piece by piece.
3. **A generator-based or callback-based scheduler of our own**, with its own `sleep` and
   primitives. The system under test would have to be written against it, which defeats the
   purpose (the agent should be plain asyncio code that also runs outside simulation).
4. **Implement `asyncio.AbstractEventLoop` directly.**

## Decision

`SimEventLoop` (`src/deterministic_agent_simulator/sim/loop.py`) implements
`asyncio.AbstractEventLoop` from scratch, about 450 lines including the methods that only raise:

- virtual time as an integer number of microseconds, advanced to the earliest timer when nothing
  is runnable;
- exactly one ready callback per iteration, chosen by a seeded `SimRandom` (or FIFO), with each
  choice folded into the trace hash;
- every real-I/O method (sockets, pipes, subprocesses, signals, executors,
  `call_soon_threadsafe`, DNS) raises `HermeticityViolation`;
- deadlock and step-limit (livelock) detection, deterministic task names.

It uses only the public `asyncio.Handle`, `TimerHandle`, `Future` and `Task` classes and the
documented `_get_running_loop` / `_set_running_loop` hooks that third-party loops rely on.

## Consequences

- The agent is plain asyncio code; `asyncio.Lock`, `wait`, `gather`, `timeout` and cancellation
  behave as in production, and the same code runs on the real loop with real ports.
- Every interleaving of same-instant callbacks is reachable, which the planted
  `missing_dedupe`-style races need; the scheduler can also run FIFO for debugging.
- Running one callback per iteration costs throughput compared with draining the ready queue
  (roughly 40 seeds per second of 5-10 tickets on one laptop core; see the README Results).
- Code that reaches for threads or real sockets fails loudly instead of silently escaping the
  simulation. Code that reads the host clock or `random` directly is not blocked; the determinism
  reruns catch it.
- The kernel owns a small surface of asyncio internals (`Handle._run`, the running-loop hooks); a
  CPython upgrade is guarded by the loop tests and the golden trace hashes.

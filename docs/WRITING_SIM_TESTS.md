# Writing simulation tests

This guide covers the port contract a system must follow to run under the simulator, and how to
add scenarios, oracles, fault points and planted bugs. Read [SIMULATION.md](SIMULATION.md) first
for how a run works.

## 1. The port contract

The system under test must do **all** I/O through injected ports. The reference agent receives an
`AgentPorts` bundle (`src/deterministic_agent_simulator/agent/ports.py`) and imports nothing from
the simulator:

| Port | Methods | Simulated by |
|---|---|---|
| `ClockPort` | `now_us()`, `sleep_us(delay)` | `SimClock` (virtual time, optional skew per incarnation) |
| `RandomPort` | `below(n)` | `SimRandomPort` (a keyed stream per ticket) |
| `LLMPort` | `decide(observation) -> Proposal` | `SimModel` over `SimLLM` |
| `TicketsPort`, `DirectoryPort` | reads | `SimTickets`, `SimDirectory` over the RPC layer |
| `EmailPort`, `PaymentsPort` | effects with an idempotency key | `SimEmail`, `SimPayments` |
| `LedgerPort` | `request`, `status`, `consume` | `SimLedger` over a recorded ledger client |
| `StorePort` | `get`, `scan`, `put`, `flush`, `put_durable`, `put_many_durable` | `SimStore` (crash semantics, torn batches) |
| `AuditPort` | `append(action, data)` | `SimAudit` (hash chain, checked on every append) |
| `InboxPort` | `get()`, `ack(delivery)` | `SimInbox` over the at-least-once queue |

Rules for code that runs under simulation:

1. **No ambient I/O.** No `open`, sockets, `time.time()`, `time.sleep`, `datetime.now()`,
   `random`, threads, executors or subprocesses. The hermetic guard and the event loop raise
   `HermeticityViolation` for the real-I/O ones and the `hermetic` oracle fails the run; the
   others (reading the host clock or `random`) are not caught automatically, they silently break
   determinism, which the rerun tests in `tests/integration/test_determinism.py` expose.
2. **Plain asyncio only.** `async`/`await`, `asyncio.Lock`, `Event`, `Queue`, `wait`, `gather`,
   `timeout` and cancellation work on `SimEventLoop`. `asyncio.to_thread`, `run_in_executor` and
   `loop.getaddrinfo` do not.
3. **Handle the port failure types**, each with its meaning:
   - `TransientError`: the call had no effect; retrying is safe.
   - `AmbiguousError`: the call may have had an effect (timeout); retry only with the *same*
     idempotency key.
   - `RejectedError`: the service refused the request; retrying will not help.
   - `PartialWriteError`: a batch applied only a prefix.
4. **Never catch `AgentCrash`.** It derives from `BaseException` so `except Exception` cannot
   swallow it; it means the process is dead. After a crash, only what was written with
   `put_durable` / `put_many_durable` (or flushed) survives, and unacknowledged deliveries are
   delivered again to the next incarnation.
5. **Iterate deterministically.** Sort keys before iterating over sets or dicts built from
   unordered sources; never make a decision that depends on `id()` or `hash()` of a string.

To adapt a different agent: implement its ports against the world in `world/adapters.py` (or
write new adapters), construct it in `World._ports` and `World._supervise`, and reuse the runner,
oracles, shrinker and replay format unchanged.

## 2. A scripted scenario test

For a specific protocol path, script everything: which tickets exist, which faults fire and what
the model says. `tests/integration/test_protocol_scenarios.py` has complete examples.

```python
from deterministic_agent_simulator.agent.ports import Proposal
from deterministic_agent_simulator.runner import Scenario, SimConfig, run_scenario
from deterministic_agent_simulator.world.simllm import Decision

decisions = (
    Decision("T-1", 1, "next", Proposal("read_ticket", {"ticket_id": "T-1"})),
    Decision(
        "T-1", 2, "next", Proposal("send_email", {"to": "...", "subject": "...", "body": "..."})
    ),
)
scenario = Scenario(
    seed=seed,
    config=SimConfig(fault_mode="none"),
    tickets=(0,),  # only T-1
    faults=(("timeout_after_commit", "email.send", "T-1", 1),),  # (kind, point, scope, n)
    decisions=decisions,  # missing entries mean "finish"
)
result = run_scenario(scenario, keep_trace=True)
assert not result.violated
```

The human reviewer is seeded, not scripted; pick a seed whose reviewer approves (see
`approving_seed` in that file). With `keep_trace=True` the result carries the trace events and the
audit chain for assertions.

## 3. Property tests over seeds

For behavior that must hold everywhere, run seeds rather than scripts: `run_scenario(Scenario(
seed=s, config=...))` for a range, or `run_campaign(...)` with caps. `hypothesis` can pick seeds
and bug-flag sets, as in
`tests/integration/test_determinism.py::test_the_scripted_replay_of_any_run_reproduces_its_trace_exactly`.

## 4. Adding an oracle

1. Add the invariant name to `INVARIANTS` in `check/oracles.py` (replay files and emitted tests
   refuse unknown names).
2. Check it at the event that can break it (`on_effect`, `on_consume`, `on_decision`,
   `on_audit`, `on_llm_attempt`) and, if it needs the whole run, in `final`. Judge from ground
   truth (committed effects, the ledger's truth log), not from what the agent believes.
3. Call `self.violate(name, detail)`; the first violation halts the run in fail-fast mode.
4. Add a unit test in `tests/unit/test_oracles.py` and, if a planted bug should trip it, a
   sensitivity row.

## 5. Adding a fault point

Call `faults.fires(FaultKind.X, point, scope)` where the fault can happen. Use the ticket as the
scope whenever the operation belongs to one ticket (so shrinking can drop other tickets without
moving this fault), and draw fault *parameters* from `faults.draw(point, scope)`, never from a
shared stream. A new `FaultKind` needs a fixed rate in `FIXED_RATES`; swarm profiles pick it up
automatically.

## 6. Adding a planted bug

1. Add a `BugFlag` and a `BugInfo` row in `agent/bugs.py` (wrong behavior, right behavior,
   trigger, expected invariant).
2. Implement the wrong branch next to the right one, marked `# BUG: <flag>`.
3. `tests/integration/test_sensitivity.py` picks the flag up automatically and fails if it is not
   found or trips a different oracle; `bench/bench_sim.py` measures seeds-to-first-failure.
4. The clean build must stay sound: `uv run das campaign --seeds 2000 --expect-clean`.

## 7. After an intentional behavior change

Any change to the world, the agent or the kernel changes trace hashes. Review the change, then
regenerate the golden hashes and the committed example repro:

```bash
uv run python scripts/regenerate_fixtures.py
```

and commit the diff together with the change that caused it.

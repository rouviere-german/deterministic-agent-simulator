# How a simulation works

This document describes what one seed does, from the event loop up to the verdict, and how a
failing seed becomes a signed replay file and a regression test. The design follows the
deterministic simulation testing approach that FoundationDB popularized (one thread, simulated
time, simulated I/O, injected faults, everything derived from one seed) and applies it to an LLM
agent whose safety controls are the system under test. Module references are relative to
`src/deterministic_agent_simulator/`.

## The determinism contract

A run is fully determined by a `Scenario`: the seed, a `SimConfig` (planted bug flags, fault mode,
scheduler, worker count, workload size) and, for replays, a ticket subset, a fault schedule and a
model decision script. Running the same scenario twice produces the same **trace hash**, on any
platform. The trace hash is a SHA-256 folded over every observable event (RPC outcomes, model
decisions, injected faults, committed effects, crashes, verdicts) and every scheduling decision
the event loop made. `tests/integration/test_determinism.py` and the `determinism` CI jobs check
the contract; `tests/fixtures/golden_trace_hashes.json` pins the hashes of 200 seeds.

What makes this hold:

| Source of nondeterminism | How it is removed |
|---|---|
| Wall-clock time | `sim/loop.py`: virtual time in integer microseconds; `time.sleep` is blocked |
| Callback order | `sim/loop.py`: one ready callback per iteration, chosen by a seeded stream |
| Randomness | `sim/rand.py`: SplitMix64 with integer arithmetic only, keyed stream derivation |
| Real I/O | `sim/hermetic.py` and the loop's blocked methods: sockets, files outside tmp, threads, subprocesses raise |
| `hash()` randomization | Nothing iterates over sets of strings in an order-sensitive way; checked with three `PYTHONHASHSEED` values |
| Float formatting and libm | Traces carry only strings, integers, booleans; floats are dyadic rationals compared, never logged |
| Task names | Deterministic `sim-N` names instead of asyncio's process-wide counter |

## The kernel (`sim/`)

- **`loop.py` - `SimEventLoop`** implements `asyncio.AbstractEventLoop` directly. Ordinary
  `async`/`await` code, `asyncio.Lock`, `asyncio.wait`, `asyncio.timeout` and task cancellation
  work unchanged. Time jumps to the next timer when nothing is runnable. When several callbacks
  are ready at the same virtual instant, a seeded `SimRandom` picks one, so every interleaving is
  reachable and each choice is folded into the trace hash. Deadlock (nothing runnable, no timer,
  main task unfinished) and livelock (more than `max_steps` callbacks) are errors.
- **`rand.py` - `SimRandom`** is SplitMix64 (Steele, Lea and Flood, OOPSLA 2014). `derive(labels)`
  returns an independent stream keyed by SHA-256 of the parent key and the labels; draws on the
  parent do not affect it.
- **`trace.py` - `Trace`** records events and hashes them incrementally. Campaigns keep only the
  hash; shrinking and reports keep the events.
- **`faults.py`** defines the fault kinds, the profiles and the `FaultPlan` (below).
- **`hermetic.py` - `HermeticGuard`** patches, for the duration of a run: every socket call and
  name lookup; file opens *and* the path-taking `os` and `shutil` calls (`stat`, `listdir`,
  `scandir`, `mkdir`, `remove`, `rename`, `rmtree`...) for paths outside the allowed roots, which
  are one scratch directory (`<system temp>/das-sim`), not the whole temp directory; `time.sleep`,
  `subprocess.Popen`, `os.system`, `os.popen` and `threading.Thread.start`; and, in strict mode
  (the default), host-clock reads (`time.time`, `time.monotonic`, `datetime.now`...) and
  process-wide entropy (`random` module functions, `os.urandom`, `uuid4`, `secrets`), which are the
  quiet sources of irreproducibility. A seeded `random.Random` instance still works. Every attempt
  is recorded before the exception is raised, so an agent that swallows the exception is still
  reported by the `hermetic` oracle.

## The simulated world (`world/`)

`World.main` is the coroutine the loop runs. It schedules ticket arrivals, runs the human
reviewer and the auditor, supervises the agent process (restarting it after a crash) and ends when
the inbound queue is drained, when an oracle halts the run, or at a virtual deadline of 30
minutes.

| Component | File | Behavior |
|---|---|---|
| RPC layer | `net.py` | Request latency, then the commit point (`apply`), then response latency. A 5xx fails before the commit; a timeout after commit fails after it (the effect happened, the client cannot tell); a service that is down times out |
| Ticket service, directory | `services.py` | Read-only ticket store and the trusted customer directory |
| E-mail outbox, payments | `services.py` | The **external effects**. Deduplicate by idempotency key like real providers: same key and arguments return the first result, same key with other arguments is rejected |
| Approval ledger | `ledger.py` | request / decide / status / consume with consume-once semantics, TTL on approvals, separation of duties (the requester cannot decide), crash and restart. Every client call is recorded in the client-observed history |
| Inbound queue | `inbound.py` | At-least-once delivery: duplicates on a fault, redelivery of unacknowledged work after a crash |
| Agent disk | `store.py` | Buffered writes lost on crash unless flushed, atomic single-key durable writes, batches that can be torn; plus the hash-chained audit trail |
| Humans | `approver.py` | A reviewer that approves 90% of compliant and 45% of non-compliant requests (approval fatigue), sometimes slowly; an auditor that polls approval status |
| Model | `simllm.py` | `SimLLM`, a seeded stochastic policy over the action grammar (below) |
| Workload | `workload.py` | 5-10 tickets per seed, refund amounts, arrival times, budgets, injected instructions with canary tokens |
| Ports | `adapters.py` | Implementations of the agent's ports with crash points before and after every call |

### Fault model

Faults are buggify-style decisions at named points. Each decision is keyed by
`(kind, point, scope, occurrence)`, where the scope is the ticket the operation belongs to.

| Kind | Where it fires | Fixed-profile rate per opportunity |
|---|---|---:|
| `latency_spike` | any RPC leg (latency x12) | 0.03 |
| `error_5xx` | any RPC, the model endpoint | 0.04 |
| `timeout_after_commit` | any RPC response leg | 0.03 |
| `duplicate_delivery` | each ticket arrival | 0.10 |
| `partial_write` | each multi-key durable batch | 0.05 |
| `agent_crash` | before and after every port call (at most 3 per run) | 0.004 |
| `ledger_crash` | right after a ledger commit (at most 3 per run) | 0.015 |
| `clock_skew` | each agent incarnation (0.1-1.5 s, either sign) | 0.30 |
| `prompt_injection` | each ticket body (7 hostile templates) | 0.25 |
| `slow_review` | each approval request (0.4-8 s think time) | 0.35 |
| `audit_read` | each 250 ms auditor tick | 0.5 |

Three fault modes:

- **fixed**: every kind at the rate above;
- **swarm** (Groce et al., "Swarm Testing", ISSTA 2012): per seed, each kind is switched off with
  probability 1/2 and the rest get a random intensity (x0.5, x1, x2 or x4), so faults that mask
  each other are sometimes absent;
- **none**: no faults.

In *random* mode a fault point draws from its own keyed stream. In *scheduled* mode (replay and
shrinking) a fault fires exactly when its key is in the schedule; the keys that fired in a random
run are its schedule, which is why a scripted replay reproduces the run exactly.

## The system under test (`agent/`)

The reference agent (`runtime.py`) resolves support tickets. For each delivery it runs a plan:
ask the model for the next action, validate it against the action registry (`actions.py`), execute
it, journal the outcome, until the model says `finish` or a cap escalates the ticket. The reply
e-mail and the refund are gated: policy (the recipient is the directory address; a refund goes to
the ticket's customer and never exceeds the ticket), a durable budget reservation, a human
approval for the exact canonical call consumed once at the ledger, and an idempotency key per
ticket slot. Journaled state lets a restarted incarnation resume a plan where the crashed one
stopped. The agent does all I/O through `AgentPorts` (`ports.py`) and imports nothing from the
simulator; [WRITING_SIM_TESTS.md](WRITING_SIM_TESTS.md) describes the contract.

### SimLLM

`SimLLM` stands in for the model. It is not trying to be smart; it produces the kinds of
behavior that break harnesses, at seed-controlled rates: sensible plans, revised arguments,
redundant re-proposals, malformed calls, obeying an injected instruction (40%) or copying a
`Reply-To` address it finds (60%), stubbornly repeating a denied call, fixating on it forever
(20% of plans that see a denial), and changing arguments when asked how to retry. Each plan draws
from its own keyed stream. In scripted mode it answers from a table of recorded decisions keyed by
`(plan, call number)` and says `finish` when the table runs out.

### The eight planted bugs

`agent/bugs.py` lists each flag with the wrong and the right implementation, the trigger and the
invariant it should violate. `uv run das bugs` prints the catalog.

| Flag | Wrong behavior | Expected invariant |
|---|---|---|
| `non_idempotent_retry` | a retry uses a fresh idempotency key | `no_duplicate_effects` |
| `budget_check_after_call` | the refund is issued first, spend is checked after | `budget_respected` |
| `approval_bypass_on_retry` | the model's retry call runs without policy or approval | `approval_fresh` |
| `stale_approval_reuse` | an approval seen earlier is reused without consuming it | `approval_fresh` |
| `infinite_replan_loop` | re-plans after a denial are not capped | `termination` |
| `non_durable_consume_flag` | the ledger acknowledges a consume before it is durable | `ledger_linearizable` |
| `missing_dedupe` | every delivery of a ticket starts a new plan | `no_duplicate_effects` |
| `trust_tool_output_recipient` | a `Reply-To` in the ticket body picks the recipient | `authorized_effects` |

## Checkers (`check/`)

**Oracles** (`oracles.py`) run when their state changes (every committed effect, consume, model
call, decision and audit append) and again at termination. They judge from the simulator's
omniscient view (committed effects, the ledger's truth log), not from what the agent believes.

| Invariant | Meaning |
|---|---|
| `no_duplicate_effects` | at most one reply e-mail and one refund per ticket |
| `authorized_effects` | e-mails go to the directory address; refunds go to the ticket's customer and never exceed the ticket |
| `budget_respected` | committed refunds stay within the refund budget; model calls within the call budget |
| `approval_fresh` | every effect cites an approval for exactly its arguments, consumed before expiry for exactly this effect; no approval justifies two effects |
| `consume_once` | no approval is consumed by two different consumers |
| `ledger_linearizable` | the client-observed ledger history is linearizable |
| `termination` | every plan stays within its decision bound; the run finishes (no deadlock, livelock or missed deadline) |
| `audit_chain_valid` | the agent's hash-chained audit trail verifies |
| `hermetic` | no real I/O was attempted |
| `agent_error` | the agent raised no unexpected exception |

**Linearizability** (`linearizability.py`, `ledger_model.py`). Every ledger call by every client
(agent, reviewer, auditor) is recorded with invocation time, response time and output; calls
whose response was lost are *pending*. The history is split per approval (linearizability is
compositional) and each part is checked with the Wing and Gong search in Lowe's form: a linked
list of call and return events, lifted when an operation is linearized, and a cache of
`(linearized set, model state)` pairs. The sequential model of one approval tracks an interval of
expiry times that are still consistent with the observations, because expiry makes the outcome
of `status` and `consume` depend on *when* each operation takes effect. A brute-force checker
that tries every permutation is kept for differential tests on histories of at most 8
operations.

## From a failing seed to a regression test

1. **Replay scenario.** `RunResult.replay_scenario()` turns an exploring run into a scripted one:
   its kept tickets, the fault keys that fired, and the model decisions it recorded. The scripted
   run has the same trace hash.
2. **Shrinking** (`shrink.py`). ddmin (Zeller and Hildebrandt, TSE 2002) is applied
   hierarchically: first to the tickets (removing a ticket removes its decisions and faults), then
   to the fault schedule, then to the model decisions, repeated while anything shrinks. Deleting a
   decision shifts the plan's later decisions forward. A candidate still fails only if its first
   violation is the same invariant, so shrinking cannot slide from one bug to another. The result
   is 1-minimal with respect to these deletions within a test budget (600 runs by default).
3. **Signed replay file** (`replay.py`). Seed, configuration, tickets, fault schedule, decision
   script and the expected invariant and trace hash, signed with HMAC-SHA256 over the canonical
   JSON. No ticket text is stored: the workload is regenerated from the seed.
4. **Regression test** (`emit.py`). A pytest module next to the file with two tests: the replay
   reproduces the recorded invariant and trace hash with the flags it was found with, and the same
   scenario holds on the current build (`APP_SIM_BUG_FLAGS`, empty for the reference build).
5. **Report** (`report.py`). A redacted JSON summary with the shrunk trace; the trace can also be
   exported as OpenTelemetry spans (one span per RPC, virtual time anchored at a fixed epoch).

## Campaigns

`das campaign` runs a range of seeds with caps on seeds and wall time (the settings
`APP_SIM_MAX_SEEDS` and `APP_SIM_MAX_WALL_SECONDS` bound whatever the command asks for), records
Prometheus metrics (`das_seeds_total`, `das_invariant_violations_total{invariant}`,
`das_seeds_per_second`, `das_trace_events_total`), and can shrink failures into signed replay
files. `--expect-clean` and `--expect-bug` turn the result into an exit code for CI.

## Known limitations

- The scheduler's tie-breaking draws from one stream for the whole run, so removing a ticket
  while shrinking can reorder later same-instant callbacks. Faults, model decisions, latencies,
  disk timings and retry jitter are keyed per ticket and do not move. Because latencies are drawn
  in microseconds, ties are rare here (a median of 6 choices per run; the seeded order changed the
  semantic trace of 1 of 300 seeds compared with FIFO, `bench/bench_sim.py --only scheduler`), so
  most interleaving variety comes from latencies rather than from the tie-breaker.
- Shared resources limit shrinking: the refund budget is computed over the whole workload, the
  worker pool is shared, and a crash kills the whole process. Failures that depend on them keep
  the tickets that provide them (median reduction 8.05x; see the README Results).
- Crash faults are attributed to the ticket whose operation hit the crash point, but they kill
  the whole process. A failure caused by a crash during another ticket's work keeps that ticket.
- The simulated services are models written for this repository. Their semantics are documented
  and tested, but they are not derived from any real provider's behavior.

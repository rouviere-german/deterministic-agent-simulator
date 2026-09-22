# 3. Keyed random streams and keyed fault decisions, so shrinking does not reshuffle a run

- Status: accepted
- Date: 2026-09-22

## Context

A failing seed has to become a small reproduction. The simplest design draws every random choice
from one stream per seed; its problem is that deleting anything (a ticket, a fault, a model step)
shifts every later draw, so the shrunk candidate is a different run and usually no longer fails.
Delta debugging then stalls with most of the scenario still in place.

Two related questions:

1. How are random choices that are *not* part of the failure (latencies, disk timings, retry
   jitter, the reviewer's think time) kept stable when other parts of the run are removed?
2. How is a fault identified so that it can be replayed and deleted independently?

## Decision

- **Keyed streams.** `SimRandom.derive(*labels)` returns an independent SplitMix64 stream keyed by
  SHA-256 of the parent key and the labels. Components derive per-ticket streams: network latency
  per `(service, ticket)`, disk latency and retry jitter per `(incarnation, ticket)`, the model
  policy per plan, the reviewer per approval token, workload fields per ticket index. Budgets are
  computed from the full workload, so removing tickets does not change them.
- **Fault keys.** A fault decision is identified by `(kind, point, scope, occurrence)`: the n-th
  opportunity for a kind at a named point, scoped by ticket. In random mode each
  `(kind, point, scope)` slot draws from its own keyed stream; in scheduled mode a fault fires
  iff its key is in the schedule. The fired keys of a random run are its schedule.
- **Decisions keyed by plan and call number.** A scripted model answers call n of a plan from the
  recorded decision n. Deleting a decision renumbers the plan's later decisions (the action
  sequence without that step).
- **Scheduler choices are not keyed.** The event loop's tie-break draws from one stream per run.

## Consequences

- Replaying a run's fault schedule and decision script reproduces its trace hash exactly
  (property-tested for random seeds, bug sets and fault modes).
- Shrinking removes tickets, faults and steps independently; in the evaluation, 11 of 24 failures
  shrank to one ticket, at most one fault and two to four model decisions (per-bug numbers in the
  README Results).
- Keying the scheduler would need stable identities for every ready callback, including those
  created inside asyncio (`gather`, `wait`, futures). We did not find a sound way to do that
  without rewriting those primitives, so the order of same-instant callbacks after a deleted
  ticket can change. Measured with `bench/bench_sim.py --only scheduler`, a run has a median of 6
  such choices and the seeded order changes the semantic trace of 1 seed in 300 compared with FIFO
  order, so in this world the scheduler is not what limits shrinking; shared resources are (a
  workload-wide budget, the worker pool, crashes that kill the whole process). Recorded as a known
  limitation in docs/SIMULATION.md.
- Every new source of randomness must derive its own keyed stream; a shared stream would silently
  degrade shrinking without failing any test except the shrink-ratio threshold in the bench.

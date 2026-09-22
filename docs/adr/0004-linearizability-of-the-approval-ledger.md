# 4. Check the approval ledger for linearizability with a time-aware sequential model

- Status: accepted
- Date: 2026-09-22

## Context

The approval ledger is the control that makes human approval meaningful: an approval must be
consumed at most once, for one exact call, before it expires. Some ledger bugs never produce a
wrong external effect in the run where they happen. The planted `non_durable_consume_flag`, for
example, acknowledges a consume before it is durable; after a ledger crash the approval is
"approved" again. Whether that later causes a second effect depends on timing, but any client that
observes the rolled-back state has seen something no correct ledger can produce.

Invariant oracles on effects catch the consequence only sometimes. We want to judge the ledger's
observable behavior directly, including under crashes, lost responses and concurrent clients
(agent, reviewer, auditor).

Options:

1. Assert properties of the ledger's internal state after the run. That tests the implementation
   against itself and misses states that were visible only between crash and restart.
2. Check the client-observed history for linearizability (Herlihy and Wing, 1990) against a
   sequential specification.

Linearizability checking is NP-complete in general, and the ledger's outputs depend on time
(TTL expiry), which a plain state-machine model cannot express because an operation's effect time
is only known to lie within its invocation-response interval.

## Decision

- Every ledger call from every client is recorded with invocation time, response time and output;
  a call whose response was lost stays *pending* and may or may not have taken effect.
- The history is partitioned per approval id (linearizability is local), which keeps each search
  small; pending requests whose id nobody observed are dropped because they cannot affect any
  checked object.
- Each part is checked with the Wing and Gong search in Lowe's formulation (linked list of call and
  return events, lifting, a memo of `(linearized set, state)`), with a step budget.
- The sequential model of one approval (`check/ledger_model.py`) carries the latest linearization
  point and an interval `[expires_lo, expires_hi]` of expiry times consistent with the
  observations. Each operation takes the earliest point its outcome allows and narrows the
  interval; the model is nondeterministic where an outcome has several explanations, and the
  search tries every successor.
- A brute-force checker (every admissible subset of pending operations, every permutation) is kept
  as an independent oracle for differential tests on histories of at most 8 operations.

## Consequences

- The `non_durable_consume_flag` bug is found from any client's observation of the rolled-back
  state (the agent's, the reviewer's or the auditor's), not only when it happens to cause a
  duplicate effect.
- The checker agrees with brute force on every differential test we run (hypothesis-generated
  histories in the unit tests, and the seeded samples in `bench/bench_sim.py`), and histories
  recorded from the real ledger are always accepted.
- "Earliest admissible point" is a proof obligation of the model rather than a general property;
  it holds here because later operations only need their own points to be no earlier, and it is
  what the differential tests exercise.
- The model is specific to one object type. Another service needs its own sequential model; the
  search and the partitioning are reusable.

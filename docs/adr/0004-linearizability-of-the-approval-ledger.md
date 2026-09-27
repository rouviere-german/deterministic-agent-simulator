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
- The sequential model of one approval (`check/ledger_model.py`) carries an interval
  `[expires_lo, expires_hi]` of expiry times consistent with the observations and a *base* time;
  the linearization point of the latest operation is `max(base, expires + offset)`, where the
  offset is `-ttl` once an approval was granted (every later point is at or after the granting
  decision, whose point is `expires - ttl`) and `0` once an operation observed the approval
  expired. Each operation moves the base to the earliest time its predecessors allow and narrows
  the interval; the model is nondeterministic where an outcome has several explanations, and the
  search tries every successor.
- A brute-force checker (every admissible subset of pending operations, every permutation) is kept
  as an independent oracle for the *search* on histories of at most 8 operations. Because it uses
  the same sequential model, the *model* is checked separately against `check/ledger_reference.py`:
  an exhaustive checker that runs the real `ApprovalLedger` on a clock of symbolic times, explores
  both sides of its `now >= expires` branch and decides each candidate order by solving the
  resulting integer difference constraints (Bellman-Ford over a simple temporal network).

## Consequences

- The `non_durable_consume_flag` bug is found from any client's observation of the rolled-back
  state (the agent's, the reviewer's or the auditor's), not only when it happens to cause a
  duplicate effect.
- The checker agrees with brute force and with the reference checker on every differential test we
  run (hypothesis-generated histories in the unit tests, and the seeded samples in
  `bench/bench_sim.py`), and histories recorded from the real ledger are always accepted.
- Keeping the point as a function of the expiry, instead of choosing a number for it, is what makes
  the earliest base admissible: with the expiry fixed, every constraint on a point is a plain lower
  or upper bound, so the earliest feasible point can never rule out a later operation. An earlier
  version of the model also *tightened* the expiry at each operation's earliest point, arguing that
  later operations only need their own points to be no earlier. That was wrong: a decision whose
  response was lost may have committed much later, and its expiry is bounded above only by when
  later operations returned. The model rejected real executions in which a lost decision committed
  after the next operation's earliest point (with a second approver, or a retried consume); the two
  smallest such executions are fixtures in `tests/unit/test_ledger_reference.py`, next to an
  exhaustive grid of 1,080 executions of that family (the earlier model rejected 68 of them).
  Nothing in the current world produces them - one reviewer, which retries only after the commit -
  which is why 50,000 clean seeds never showed it. A campaign
  cannot find this class of error at all: only a second, independent decision procedure can, which
  is why the reference checker now runs in the unit tests and in the bench.
- The model is specific to one object type. Another service needs its own sequential model; the
  search and the partitioning are reusable.

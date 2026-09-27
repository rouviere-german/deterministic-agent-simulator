# Deterministic Agent Simulator

FoundationDB-style simulation testing for LLM agents. Seeded worlds with injected faults (timeouts
after commit, duplicate deliveries, crashes, torn writes, clock skew, prompt injection) drive a
reference support-ticket agent through rare interleavings; invariant oracles and a Wing-Gong
linearizability checker catch double sends, budget overruns and approval bypasses; delta
debugging shrinks each failure to a small, HMAC-signed replay file plus a generated pytest
regression test.

Everything runs in one process on one CPU core: a seeded model policy stands in for the LLM,
and there are no containers, no network calls and no API key involved.

## Problem

The worst agent bugs appear only under rare combinations of events:

- an e-mail is sent twice because a retry after a timeout used a fresh idempotency key, while the
  first attempt had in fact been delivered;
- a refund budget is overrun because spend was checked after the call, and a partial failure hid
  one of the refunds;
- an approval is reused after a crash, or bypassed on a retry path the model was allowed to
  re-plan;
- a ticket body says `Reply-To: attacker@...` and the agent trusts it.

Ordinary tests never reach these interleavings, and when they happen in production they cannot be
reproduced. Deterministic simulation testing fixes both: every run is a pure function of a seed,
so a campaign can explore thousands of fault schedules per minute, and any failure can be replayed
bit for bit, shrunk and turned into a regression test.

## Architecture

```mermaid
flowchart TB
    subgraph runner["Campaign runner - runner.py"]
        seed["seed + SimConfig<br/>bug flags, fault mode"] --> scenario[Scenario]
    end
    subgraph kernel["Kernel - sim/"]
        loop["SimEventLoop<br/>virtual microsecond clock<br/>seeded tie-breaker"]
        rand["SimRandom<br/>keyed SplitMix64 streams"]
        faults["FaultPlan<br/>swarm, fixed or scheduled"]
        trace["Trace<br/>incremental SHA-256"]
        guard["HermeticGuard<br/>no sockets, files, threads"]
    end
    subgraph world["Simulated world - world/"]
        net["RPC layer<br/>latency, 5xx, timeout after commit"]
        svc["Tickets, directory,<br/>e-mail outbox, payments"]
        ledger["ApprovalLedger<br/>consume-once, TTL, crashes"]
        inbox["At-least-once inbox"]
        humans["Reviewer and auditor"]
        simllm["SimLLM policy<br/>obeys injected text"]
    end
    subgraph sut["System under test - agent/"]
        agent["ReferenceAgent<br/>8 plantable bug flags"]
        ports["AgentPorts<br/>clock, random, llm, tools, store, audit"]
    end
    subgraph check["Checkers - check/"]
        oracles["10 invariant oracles<br/>per step and at the end"]
        wg["Wing-Gong-Lowe<br/>linearizability checker"]
    end
    scenario --> loop
    loop --> world
    agent --> ports --> world
    world --> oracles
    ledger -- "client-observed history" --> wg
    oracles -- "first violation" --> shrink["ddmin shrinker<br/>tickets, then faults, then decisions"]
    shrink --> repro["HMAC-signed replay file"]
    repro --> emit["Generated pytest regression test"]
    shrink --> report["Redacted report, OTel spans,<br/>Prometheus metrics"]
    report --> api["Optional read-only results API<br/>scoped keys, rate limits"]
```

- **Kernel** (`sim/`). `SimEventLoop` implements `asyncio.AbstractEventLoop` directly (ADR 0002):
  virtual time in integer microseconds, exactly one ready callback per iteration picked by a
  seeded stream, every scheduling choice folded into the trace hash, real I/O methods raising.
  `SimRandom` is SplitMix64 with keyed derivation, so faults, latencies and model decisions are
  keyed per ticket and stay put while shrinking (ADR 0003). `HermeticGuard` blocks, for the duration
  of a run, sockets and name lookups, file opens and path-taking `os`/`shutil` calls outside the
  run's scratch root, `time.sleep`, subprocesses, thread starts and - in strict mode, the default -
  host-clock and entropy reads, and records every attempt.
- **World** (`world/`). An RPC layer whose commit point sits between the request and response
  legs, so a timeout after commit is a real ambiguity; an e-mail outbox and a payments stub that
  deduplicate by idempotency key; an approval ledger with request / decide / consume-once
  semantics, TTL and crash/restart; an at-least-once inbox; a reviewer with approval fatigue; and
  `SimLLM`, a seeded stochastic policy over the agent's action grammar that sometimes obeys
  injected instructions, copies a `Reply-To`, repeats a denied call or fixates on it.
- **System under test** (`agent/`). A reference support agent whose controls are the point:
  closed action grammar, recipient from the trusted directory, durable budget reservation, human
  approval bound to the exact call and consumed once, one idempotency key per ticket slot,
  journaled plans that resume after a crash. Eight flags plant plausible bugs in those controls.
  The agent talks to the world only through injected ports (docs/WRITING_SIM_TESTS.md).
- **Checkers** (`check/`). Ten oracles (no duplicate effects, authorized effects, budget, approval
  freshness, consume-once, ledger linearizability, termination, audit chain, hermeticity, agent
  errors) judge from ground truth. The ledger's client-observed history is checked for
  linearizability against a time-aware sequential model with a memoized Wing-Gong search
  (ADR 0004).
- **Output**. ddmin shrinks a failure over tickets, fault schedule and model decisions; the result
  is written as an HMAC-signed scenario (no ticket text) plus a pytest module that reproduces the
  failure with the recorded flags and must pass on the current build (ADR 0005).

docs/SIMULATION.md walks through a run in detail.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything below runs offline.

```bash
uv sync --frozen

# The offline demo: plant non_idempotent_retry, find it, shrink it, sign it, generate and run a
# regression test, export the trace, then run the clean build over 500 swarm seeds.
uv run python tasks.py demo          # or: make demo

# The planted bugs and the invariant each one should trip
uv run das bugs

# One seed, with its (redacted) trace
uv run das run 3 --trace

# A campaign on the clean build; exits 1 if any seed violates an invariant
uv run das campaign --seeds 2000 --expect-clean --out var/results/campaigns/local.json

# Hunt a planted bug, shrink the first failure into a signed repro and a pytest module
export APP_REPLAY_HMAC_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
uv run das campaign --bugs missing_dedupe --mode fixed --seeds 2000 --stop-on-first \
  --expect-bug --repro-dir var/regressions --max-repros 1
uv run das replay var/regressions/repro_seed_*_no_duplicate_effects.json          # reproduces
uv run das replay var/regressions/repro_seed_*_no_duplicate_effects.json --build current

# Cross-platform determinism: recompute the committed golden trace hashes
uv run das hashes --count 200 --compare tests/fixtures/golden_trace_hashes.json

# Checks and evaluation
uv run python tasks.py lint typecheck test audit baseline
uv run python tasks.py bench         # micro-benchmarks + evaluation (CI profile) with thresholds
uv run python bench/bench_sim.py     # full evaluation
```

On a drive without room for a virtual environment, point uv elsewhere first
(`UV_PROJECT_ENVIRONMENT=/path/outside/the/repo`). The optional results API runs with
`uv run das serve` or `docker compose up --build` (see Security).

## Results

All numbers below were produced by the commands shown, on one machine: Windows 10 (build
19045), Intel Core i5-8500 (6 cores, 3.0 GHz), Python 3.12.14, single process, CPU only. The raw
output is copied without edits except for trimming.

### Evaluation against the spec

`uv run python bench/bench_sim.py --soundness-seeds 50000 --check` (1,493 s in total), plus
`uv run python bench/bench_sim.py --only scheduler` for the last row:

| Metric | Spec target | Measured |
|---|---|---|
| Sensitivity: planted bugs found within 10,000 seeds | 8/8 | **8/8**, in every one of 11 independent searches per bug and fault mode |
| Soundness: violations with all bugs off | 0 over 50,000 seeds | **0** over 50,000 swarm seeds |
| Shrinking: median trace-length reduction | at least 10x | **8.05x** (min 2.41x, max 18.16x, 24 shrinks); **below target**, see Non-goals |
| Determinism: identical hashes on reruns | 100 reruns per seed | 20 seeds x 100 reruns: **0** seeds with more than one hash; 200/200 golden hashes match on Windows |
| Determinism across Linux and Windows | identical hashes | configured as the `determinism` CI matrix; **not run yet** (no remote) |
| Checker: Wing-Gong vs brute force, histories of at most 8 operations | agreement | **2,000/2,000** random histories agree; 8/8 known fixtures |
| Throughput | reported | **38.3 seeds/s** = 2,298 seeds/min (5-10 tickets per seed, every oracle on) |
| Scheduler tie-breaks | (not in spec) | median 6 choices per run (max 23); the seeded order changed the semantic trace of 1/300 seeds compared with FIFO |

### Sensitivity: seeds to the first failure

Each cell is the number of searches (of 11, disjoint seed ranges starting at 1,000,000) that found
the bug within 10,000 seeds, and the median number of seeds the search needed.

| bug | expected invariant | swarm found | swarm median | fixed found | fixed median |
|---|---|---:|---:|---:|---:|
| non_idempotent_retry | no_duplicate_effects | 11/11 | 5 | 11/11 | 2 |
| budget_check_after_call | budget_respected | 11/11 | 2 | 11/11 | 2 |
| approval_bypass_on_retry | approval_fresh | 11/11 | 6 | 11/11 | 9 |
| stale_approval_reuse | approval_fresh | 11/11 | 145 | 11/11 | 21 |
| infinite_replan_loop | termination | 11/11 | 2 | 11/11 | 2 |
| non_durable_consume_flag | ledger_linearizable | 11/11 | 26 | 11/11 | 18 |
| missing_dedupe | no_duplicate_effects | 11/11 | 1 | 11/11 | 1 |
| trust_tool_output_recipient | authorized_effects | 11/11 | 8 | 11/11 | 5 |

The first invariant to fire was the expected one in every search except `approval_bypass_on_retry`,
where 5 of 22 searches tripped `authorized_effects` first (the model's changed retry call also
changed the recipient). Swarm profiles were not faster than the fixed profile except for
`approval_bypass_on_retry`. The largest gap is `stale_approval_reuse` (median 145 vs 21 seeds); it
needs a crash or a failed consume between two specific steps, and one likely reason is that a
swarm profile switches agent crashes off for half of the seeds.

### Shrinking

First failing run of three fixed-mode searches per bug; "kept t/f/d" is the number of tickets,
faults and model decisions left in the shrunk scenario.

| bug | seed | trace events before -> after | ratio | ddmin tests | kept t/f/d |
|---|---:|---|---:|---:|---|
| non_idempotent_retry | 1000000 | 195 -> 17 | 11.5x | 21 | 1/1/2 |
| non_idempotent_retry | 1010002 | 228 -> 18 | 12.7x | 24 | 1/1/2 |
| non_idempotent_retry | 1020005 | 116 -> 19 | 6.1x | 21 | 1/1/2 |
| budget_check_after_call | 1000002 | 123 -> 28 | 4.4x | 43 | 2/0/4 |
| budget_check_after_call | 1010004 | 232 -> 71 | 3.3x | 206 | 5/0/10 |
| budget_check_after_call | 1020000 | 124 -> 28 | 4.4x | 47 | 2/0/4 |
| approval_bypass_on_retry | 1000012 | 82 -> 18 | 4.6x | 30 | 1/1/3 |
| approval_bypass_on_retry | 1010012 | 154 -> 17 | 9.1x | 33 | 1/1/3 |
| approval_bypass_on_retry | 1020008 | 63 -> 19 | 3.3x | 28 | 1/1/3 |
| stale_approval_reuse | 1000077 | 212 -> 16 | 13.2x | 24 | 1/1/2 |
| stale_approval_reuse | 1010018 | 183 -> 18 | 10.2x | 38 | 1/1/4 |
| stale_approval_reuse | 1020014 | 318 -> 38 | 8.4x | 97 | 1/3/3 |
| infinite_replan_loop | 1000000 | 345 -> 19 | 18.2x | 123 | 1/0/15 |
| infinite_replan_loop | 1010001 | 301 -> 19 | 15.8x | 121 | 1/0/15 |
| infinite_replan_loop | 1020000 | 241 -> 36 | 6.7x | 168 | 2/0/17 |
| non_durable_consume_flag | 1000017 | 278 -> 47 | 5.9x | 103 | 2/4/4 |
| non_durable_consume_flag | 1010004 | 279 -> 116 | 2.4x | 396 | 5/16/11 |
| non_durable_consume_flag | 1020071 | 339 -> 56 | 6.1x | 353 | 4/6/4 |
| missing_dedupe | 1000000 | 351 -> 30 | 11.7x | 45 | 1/1/4 |
| missing_dedupe | 1010001 | 255 -> 33 | 7.7x | 53 | 1/1/5 |
| missing_dedupe | 1020001 | 172 -> 40 | 4.3x | 145 | 4/1/8 |
| trust_tool_output_recipient | 1000001 | 185 -> 15 | 12.3x | 20 | 1/1/2 |
| trust_tool_output_recipient | 1010004 | 134 -> 15 | 8.9x | 24 | 1/1/2 |
| trust_tool_output_recipient | 1020017 | 135 -> 14 | 9.6x | 19 | 1/1/2 |

11 of the 24 shrinks end with one ticket, at most one fault and two to four model decisions (14 to
19 trace events); `infinite_replan_loop` keeps 15 scripted decisions because the termination
bound is 15 decisions per plan.
The ones that stay large need shared state: `budget_check_after_call` needs enough refunds to exceed a budget
computed over the whole workload, and `non_durable_consume_flag` and some `missing_dedupe` runs
need a crash or a contended worker pool that another ticket provides.

### Offline demo

`uv run python tasks.py demo` (14.6 s wall clock), abridged JSON output:

```json
{
  "bug": "non_idempotent_retry",
  "failing_seed": 4,
  "seeds_searched": 5,
  "invariant": "no_duplicate_effects",
  "detail": "send_email committed twice for T-5 (effects #4 and #5)",
  "original_trace_events": 193,
  "shrunk_trace_events": 17,
  "reduction_ratio": 11.35,
  "shrink_tests": 28,
  "shrunk_scenario": {"tickets": 1, "faults": [["timeout_after_commit", "email.send", "T-5", 1]], "decisions": 2},
  "tampered_copy_rejected": true,
  "regression_test_on_buggy_build": {"reproduces": "passed", "holds": "failed"},
  "regression_test_on_clean_build": {"reproduces": "passed", "holds": "passed"},
  "otel_spans_exported": 9,
  "audit_chain_verified": true,
  "clean_build": {"seeds": 500, "violations": {}, "seeds_per_second": 38.4}
}
```

### Micro-benchmarks

`uv run python bench/bench_baseline.py --check`:

| benchmark | runs | p50 (us) | p95 (us) |
|---|---:|---:|---:|
| redact_text_1kb | 2000 | 344.3 | 454.4 |
| rate_limiter_check | 20000 | 1.8 | 2.0 |
| llm_request_sha256_10_messages | 2000 | 99.7 | 124.4 |
| audit_append_fsync | 200 | 7924.9 | 9577.5 |
| audit_verify_1000_entries | 10 | 36625.2 | 39209.1 |
| sim_random_1000_draws | 200 | 820.5 | 1100.6 |
| sim_run_one_seed | 60 | 26565.7 | 35540.0 |
| wing_gong_ledger_history_40_ops | 100 | 196.9 | 236.0 |

CI runs `bench/bench_baseline.py --check` and `bench/bench_sim.py --profile ci --check` on every
push; `bench/thresholds.json` and `bench/sim_thresholds.json` hold the regression thresholds (for
example: 8/8 bugs found, 0 soundness violations, 0 checker disagreements, 0 hash mismatches,
median shrink ratio at least 4x, at least 5 seeds/s).

## Security

The simulator attacks the reference agent's security controls on purpose and checks them as
invariants on every run:

| Property | How it is attacked | Oracle |
|---|---|---|
| No unauthorized action after injected content | 7 hostile instruction templates in ticket bodies; SimLLM obeys them 40% of the time and copies `Reply-To` lines | `authorized_effects`, `approval_fresh` |
| No approval bypass or reuse | crashes between approval and consume, failed consumes, a model that changes arguments on retry | `approval_fresh`, `consume_once`, `ledger_linearizable` |
| No budget overrun | refunds that together exceed the budget, refunds timing out after commit, torn journal writes | `budget_respected` |
| No double effect | timeouts after commit, duplicate deliveries, crashes and redelivery | `no_duplicate_effects` |
| Tamper-evident trail | every audit append re-verified | `audit_chain_valid` |
| Hermetic runs | any socket or name lookup, file or directory outside the run's scratch root, sleep, subprocess, thread start, host-clock read or host entropy draw | `hermetic` |

Controls around the tool itself:

- **Signed repros.** Replay files are HMAC-SHA256 signed over canonical JSON and verified in
  constant time before they are parsed; unsigned, foreign-key, oversized or edited files are
  rejected. They carry no ticket text. Generated tests embed only a seed, a known invariant name
  and a strictly validated file name. `APP_REPLAY_HMAC_KEY` has no default.
- **Redaction.** Reports, trace lines, OTel span attributes and API responses pass through canary
  and PII redaction; canary tokens stand in for secrets an attacker planted in tool output.
- **Least privilege.** The agent's action registry is built from granted capabilities; argument
  schemas are closed and validated strictly; a plan can read only its own ticket.
- **Caps.** Campaigns are capped on seeds and wall time by settings the command line cannot raise;
  each run has a step limit, a virtual deadline and deadlock detection; the linearizability search
  and ddmin have budgets.
- **Results API** (optional, read-only). API keys held as SHA-256 digests with per-key scopes
  (`results:read`, `repros:verify`), rate limits, body-size limits, security headers and
  non-leaking errors. The container runs as a non-root user on a read-only filesystem with all
  capabilities dropped.
- **Live adapter.** `agent/live.py` backs the model port with `AnthropicProvider` (Claude Opus 5
  by default, adaptive thinking, strict tools, the server-side refusal fallback, `stop_reason`
  checked before content). Its output is only a proposal that goes through the same gate. It is
  tested against a mocked client and never runs inside a simulation.

The threat model with a threat -> control -> test table is in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md); every claim in this README is mapped to evidence in
[CLAIMS.md](CLAIMS.md); reporting instructions are in [SECURITY.md](SECURITY.md).

## Non-goals

- **Real networks and real services.** The services are models written for this repository; their
  semantics are documented and tested, but not derived from any provider. Real-network behavior
  (partial TCP delivery, DNS, TLS) is out of scope.
- **Multi-process scheduling.** One process, one thread. The agent's concurrency is asyncio
  concurrency; there is no simulation of OS threads or multiple machines.
- **Coverage-guided fuzzing.** Exploration is random over seeds with swarm fault profiles; there is
  no feedback from coverage.
- **Sandboxing.** The hermetic guard patches Python-level APIs to keep runs reproducible; C
  extensions, `ctypes`, a name imported before the run started or a file descriptor opened earlier
  can bypass it. It is not an isolation boundary for hostile code. Strict mode replaces
  `datetime.datetime` and `datetime.date` with subclasses while a run executes, so
  `isinstance(x, datetime.datetime)` is False for instances built before the run.
- **Shrink-stable scheduling.** The scheduler's tie-breaks draw from one stream per run, so
  deleting a ticket while shrinking can reorder same-instant callbacks later in the run (ADR 0003).
  In this world that rarely matters (1 of 300 seeds changed its semantic trace between seeded and
  FIFO order); interleavings vary mostly through keyed, randomized latencies.
- **Live LLM evaluation.** The simulator measures harness safety under a seeded policy, not model
  quality. The Anthropic adapter is verified against a mocked client only; nothing here calls the
  API.
- **Asymmetric provenance.** Replay files use HMAC; anyone who can verify can sign.
- **The 10x shrinking target is not met.** The measured median is 8.05x (above). Failures that
  depend on state shared across tickets (a workload-wide budget, the worker pool, a crash fault
  that kills the whole process) keep the tickets that provide it, and failures found early have
  short original traces to begin with. Making the budget per ticket or scoping crashes differently
  would raise the number by changing what is being tested, so neither was done.
- **Cross-OS determinism is verified on Windows only.** The Linux side is the `determinism` CI
  matrix job, which has not run because this repository has no remote yet. The same holds for the
  per-push `sim-gate`, the nightly 50,000-seed swarm job and the container jobs: they are
  configured, not executed; the equivalent campaigns were run locally (above).

## Repository layout

| Path | Contents |
|---|---|
| `src/deterministic_agent_simulator/sim/` | event loop, SimRandom, traces, faults, hermetic guard, metrics |
| `src/deterministic_agent_simulator/world/` | services, ledger, inbox, store, humans, SimLLM, workload, port adapters |
| `src/deterministic_agent_simulator/agent/` | port contract, action grammar, bug catalog, reference agent, live adapter |
| `src/deterministic_agent_simulator/check/` | oracles, Wing-Gong checker, approval-ledger model |
| `src/deterministic_agent_simulator/` | runner, shrinker, replay files, test emission, reports, demo, CLI, results API |
| `src/deterministic_agent_simulator/agentic/`, `core/` | shared kit: providers, envelopes, audit log, budgets; hardened app, logging, config |
| `tests/` | unit, integration, the committed golden hashes and an emitted regression example |
| `bench/` | micro-benchmarks and the evaluation (`bench_sim.py`) with CI thresholds |
| `docs/` | SIMULATION.md, WRITING_SIM_TESTS.md, THREAT_MODEL.md, ADRs |

## References

- J. Zhou et al., "FoundationDB: A Distributed Unbundled Transactional Key Value Store", SIGMOD
  2021 (simulation testing section), and the FoundationDB simulation and testing documentation.
- M. Herlihy and J. Wing, "Linearizability: A Correctness Condition for Concurrent Objects", ACM
  TOPLAS 1990.
- J. Wing and C. Gong, "Testing and Verifying Concurrent Objects", JPDC 1993.
- G. Lowe, "Testing for Linearizability", Concurrency and Computation: Practice and Experience,
  2017.
- A. Zeller and R. Hildebrandt, "Simplifying and Isolating Failure-Inducing Input", IEEE TSE 2002.
- G. Misherghi and Z. Su, "HDD: Hierarchical Delta Debugging", ICSE 2006.
- A. Groce, C. Zhang, E. Eide, Y. Chen and J. Regehr, "Swarm Testing", ISSTA 2012.
- G. Steele, D. Lea and C. Flood, "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014.

## License

MIT, see [LICENSE](LICENSE).

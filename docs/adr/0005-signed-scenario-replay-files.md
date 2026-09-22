# 5. Replay files store a signed scenario, not a trace, and carry no ticket text

- Status: accepted
- Date: 2026-09-22

## Context

A shrunk failure must be shareable: attached to a CI run, committed next to a regression test,
replayed by someone else. Two formats were possible:

1. **A recorded trace** (every event, including ticket bodies and model outputs) replayed by
   feeding recorded responses back.
2. **The scenario**: seed, configuration, kept ticket indices, fault schedule and decision script.
   Everything else (ticket text, latencies, reviewer behavior) is regenerated from the seed.

A replay file is also an input that drives the simulator and generates Python source (the
regression test), so it needs integrity protection: a tampered file could claim a bug is fixed,
point a regression test at a different scenario, or try to inject code into the generated test.

## Decision

- Replay files (`das-repro/1`) store the scenario plus what the replay is expected to show (the
  violated invariant, the trace hash, trace lengths before and after shrinking) and a redacted
  one-line detail. They store no ticket text, no model prose beyond the recorded tool calls, and
  no canary tokens.
- Files are signed with HMAC-SHA256 over the canonical JSON of every field except the signature,
  key id included. Loading verifies the signature in constant time **before** parsing the
  document into a scenario; unknown key ids, unsigned files, oversized files (over 2 MB) and
  schema violations are rejected. Keys come from `APP_REPLAY_HMAC_KEY` (at least 32 characters);
  there is no default key.
- The generated regression test embeds only the seed, the invariant name (checked against the
  known list) and the replay file name (checked against a strict pattern); everything else is read
  from the signed file at test time.
- A regression test has two halves: reproduce the recorded invariant and trace hash with the bug
  flags it was found with, and hold on the current build (`APP_SIM_BUG_FLAGS`).

## Consequences

- Files are small (a few KB) and contain no customer data, so CI artifacts and committed repros
  are safe to share. The cost is that a replay depends on the simulator version: a behavior
  change changes trace hashes and can invalidate old repros, which is why
  `scripts/regenerate_fixtures.py` exists and why the trace hash is part of the expected block.
- HMAC is symmetric: anyone who can verify can also sign. That fits one team sharing a CI secret;
  cross-organization provenance would need asymmetric signatures (for example Ed25519), which is
  a non-goal.
- The regression test fails while the bug is present and passes once it is fixed, which is what a
  regression test should do; the "reproduces" half keeps the file honest.

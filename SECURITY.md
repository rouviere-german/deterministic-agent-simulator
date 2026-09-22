# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the simulator (kernel, world, oracles, checker, shrinker), the reference agent's
controls, replay-file signing and loading, the generated regression tests, the read-only results
API, the container image and the CI configuration.

Especially interesting reports:

- a way to make a campaign report "clean" for a build that violates one of the documented
  invariants (a false negative in an oracle or the linearizability checker);
- a replay file that loads without the key, or whose content changes what the generated test
  does;
- canary tokens, customer data or secrets reaching a report, trace, span or API response;
- code under simulation reaching the network or files outside tmp without being reported.

Out of scope: denial of service by traffic volume, findings that need a compromised host, bypasses
of the hermetic guard from C extensions or `ctypes` (it is a correctness aid, not a sandbox; see
docs/THREAT_MODEL.md), and issues in third-party dependencies that are already publicly tracked
(please report those upstream).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every README claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, a trivy image scan and a syft CycloneDX SBOM, plus a
  2,000-seed soundness campaign, planted-bug sensitivity and a cross-OS determinism check on every
  push.
- Secrets come only from the environment; only `.env.example` is committed. Replay files are
  HMAC-signed with `APP_REPLAY_HMAC_KEY`; there is no default key.
- Every artifact is JSON (replay files, reports, cassettes, audit chains); pickle-family
  serialization is banned.

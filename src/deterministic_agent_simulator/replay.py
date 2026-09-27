"""Signed replay files: a shrunk failing scenario that anyone with the key can re-run.

A replay file stores the seed, the configuration (including the bug flags the failure was found
with), the kept ticket indices, the fault schedule, the model decision script and what the replay
is expected to show (the violated invariant and the trace hash). It stores no ticket text: the
workload is regenerated from the seed, so the file carries no customer data and no canary
tokens.

The file is signed with HMAC-SHA256 over the canonical JSON of every field except the signature
(the key id included). Loading verifies the signature in constant time *before* the contents are
parsed into a scenario, so a tampered or forged file is rejected, not replayed. Keys come from
``APP_REPLAY_HMAC_KEY``; there is no default key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from deterministic_agent_simulator import __version__
from deterministic_agent_simulator.agent.bugs import parse_bug_flags
from deterministic_agent_simulator.check.oracles import INVARIANTS, Violation
from deterministic_agent_simulator.core.canonical import canonical_json
from deterministic_agent_simulator.core.config import Settings, get_settings
from deterministic_agent_simulator.report import redact_report_text
from deterministic_agent_simulator.runner import (
    Scenario,
    SimConfig,
    proposals_from_json,
    run_scenario,
)
from deterministic_agent_simulator.sim.faults import FaultKey

__all__ = [
    "MAX_REPRO_BYTES",
    "MAX_REPRO_DEPTH",
    "REPRO_FORMAT",
    "Keyring",
    "ReplayOutcome",
    "ReproError",
    "ReproSignatureError",
    "build_repro",
    "load_repro",
    "replay_file",
    "scenario_from_document",
    "scenario_to_document",
    "write_repro",
]

REPRO_FORMAT = "das-repro/1"
MAX_REPRO_BYTES = 2_000_000
MAX_REPRO_DEPTH = 24
"""Nesting limit, checked before the document is hashed or validated: a deeply nested file would
otherwise exhaust the stack inside canonical JSON or the schema instead of being rejected."""


class ReproError(ValueError):
    """A replay file is malformed, too large or cannot be handled with the configured keys."""


class ReproSignatureError(ReproError):
    """A replay file's signature is missing, from an unknown key or does not match."""


@dataclass(frozen=True, slots=True)
class Keyring:
    """HMAC keys by id; ``signing_key_id`` signs, every listed key may verify."""

    keys: Mapping[str, bytes]
    signing_key_id: str

    def __post_init__(self) -> None:
        if self.signing_key_id not in self.keys:
            raise ReproError("the signing key id is not in the keyring")
        if any(len(key) < 32 for key in self.keys.values()):
            raise ReproError("replay keys must be at least 32 bytes")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Keyring:
        cfg = settings if settings is not None else get_settings()
        if cfg.replay_hmac_key is None:
            raise ReproError("APP_REPLAY_HMAC_KEY is not set; replay files cannot be signed")
        key = cfg.replay_hmac_key.get_secret_value().encode("utf-8")
        return cls({cfg.replay_key_id: key}, cfg.replay_key_id)

    def _mac(self, key_id: str, body: Mapping[str, Any]) -> str:
        key = self.keys.get(key_id)
        if key is None:
            raise ReproSignatureError(f"replay file signed with unknown key id {key_id!r}")
        return hmac.new(key, canonical_json(dict(body)).encode("utf-8"), hashlib.sha256).hexdigest()

    def sign(self, body: Mapping[str, Any]) -> dict[str, Any]:
        document = {**dict(body), "key_id": self.signing_key_id}
        document.pop("signature", None)
        document["signature"] = self._mac(self.signing_key_id, document)
        return document

    def verify(self, document: Mapping[str, Any]) -> None:
        signature = document.get("signature")
        key_id = document.get("key_id")
        if not isinstance(signature, str) or not isinstance(key_id, str):
            raise ReproSignatureError("replay file is not signed")
        body = {name: value for name, value in document.items() if name != "signature"}
        expected = self._mac(key_id, body)
        if not hmac.compare_digest(expected, signature):
            raise ReproSignatureError("replay file signature does not match its contents")


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    plan: str = Field(pattern=r"^T-[0-9]{1,6}(~[0-9]{1,6})?$")
    n: int = Field(ge=1)
    purpose: Literal["next", "retry"]
    proposal: dict[str, JsonValue]


class _Expected(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    invariant: str
    trace_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_length: int = Field(ge=0)
    original_trace_length: int = Field(ge=0)
    detail: str


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["das-repro/1"]
    created_by: str
    seed: int = Field(ge=0)
    config: dict[str, JsonValue]
    tickets: list[int]
    faults: list[tuple[str, str, str, int]]
    decisions: list[_Decision]
    expected: _Expected
    key_id: str
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


def scenario_to_document(scenario: Scenario) -> dict[str, JsonValue]:
    """The replayable part of a scripted scenario, as JSON."""
    return {
        "seed": scenario.seed,
        "config": scenario.config.to_json(),
        "tickets": list(scenario.tickets or ()),
        "faults": [list(key) for key in scenario.faults or ()],
        "decisions": [
            {
                "plan": d.plan_id,
                "n": d.n,
                "purpose": d.purpose,
                "proposal": d.proposal.to_json(),
            }
            for d in scenario.decisions or ()
        ],
    }


def scenario_from_document(document: Mapping[str, Any]) -> Scenario:
    faults: tuple[FaultKey, ...] = tuple(
        (str(kind), str(point), str(scope), int(n)) for kind, point, scope, n in document["faults"]
    )
    return Scenario(
        seed=int(document["seed"]),
        config=SimConfig.from_json(dict(document["config"])),
        tickets=tuple(int(index) for index in document["tickets"]),
        faults=faults,
        decisions=proposals_from_json(document["decisions"]),
    )


def build_repro(
    scenario: Scenario,
    *,
    violation: Violation,
    trace_hash: str,
    trace_length: int,
    original_trace_length: int,
    keyring: Keyring,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": REPRO_FORMAT,
        "created_by": f"deterministic-agent-simulator {__version__}",
        **scenario_to_document(scenario),
        "expected": {
            "invariant": violation.invariant,
            "trace_hash": trace_hash,
            "trace_length": trace_length,
            "original_trace_length": original_trace_length,
            "detail": redact_report_text(violation.detail),
        },
    }
    return keyring.sign(body)


def write_repro(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(document), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_repro(path: Path, keyring: Keyring) -> tuple[Scenario, dict[str, Any]]:
    """Verify, then parse a replay file. Returns the scenario and its ``expected`` block."""
    raw = path.read_bytes()
    return parse_repro(raw, keyring)


def _depth(value: object) -> int:
    """Maximum nesting depth of a decoded JSON value, computed without recursion."""
    deepest = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, level = stack.pop()
        deepest = max(deepest, level)
        if isinstance(current, dict):
            stack.extend((item, level + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, level + 1) for item in current)
    return deepest


def parse_repro(raw: bytes, keyring: Keyring) -> tuple[Scenario, dict[str, Any]]:
    if len(raw) > MAX_REPRO_BYTES:
        raise ReproError("replay file is too large")
    try:
        # A decoding error, a number with more digits than int() accepts (ValueError) and nesting
        # deeper than the parser's stack (RecursionError) are all just malformed input.
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ReproError("replay file is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ReproError("replay file is not a JSON object")
    if _depth(document) > MAX_REPRO_DEPTH:
        raise ReproError("replay file is nested too deeply")
    keyring.verify(document)  # authenticate before trusting any field
    try:
        parsed = _Document.model_validate(document)
    except ValidationError as exc:
        raise ReproError("replay file does not match the das-repro/1 schema") from exc
    if parsed.expected.invariant not in INVARIANTS:
        raise ReproError("replay file names an unknown invariant")
    try:
        scenario = scenario_from_document(document)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReproError("replay file contents are inconsistent") from exc
    return scenario, parsed.expected.model_dump()


@dataclass
class ReplayOutcome:
    violations: list[Violation]
    trace_hash: str
    expected: dict[str, Any]
    build_flags: list[str] = field(default_factory=list)

    @property
    def invariant(self) -> str | None:
        return self.violations[0].invariant if self.violations else None

    @property
    def expected_trace_hash(self) -> str:
        return str(self.expected["trace_hash"])

    @property
    def report(self) -> str:
        if not self.violations:
            return "no invariant was violated"
        return redact_report_text("; ".join(v.describe() for v in self.violations))


def replay_file(
    path: Path,
    *,
    build: Literal["recorded", "current"] = "recorded",
    keyring: Keyring | None = None,
    settings: Settings | None = None,
) -> ReplayOutcome:
    """Replay a signed file with the flags it was found with, or with the current build's."""
    cfg = settings if settings is not None else get_settings()
    ring = keyring if keyring is not None else Keyring.from_settings(cfg)
    scenario, expected = load_repro(path, ring)
    if build == "current":
        flags = parse_bug_flags(cfg.sim_bug_flags)
        config = SimConfig.from_json({**scenario.config.to_json(), "bugs": sorted(flags)})
        scenario = Scenario(
            seed=scenario.seed,
            config=config,
            tickets=scenario.tickets,
            faults=scenario.faults,
            decisions=scenario.decisions,
        )
    result = run_scenario(scenario)
    return ReplayOutcome(
        violations=list(result.violations),
        trace_hash=result.trace_hash,
        expected=expected,
        build_flags=sorted(flag.value for flag in scenario.config.bugs),
    )

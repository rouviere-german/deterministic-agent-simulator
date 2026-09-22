"""The agent's local disk (a key-value store with crash semantics) and its audit chain.

``put`` writes into a buffer that a crash discards unless ``flush`` ran first; ``put_durable``
is an atomic, synced single-key write; ``put_many_durable`` writes several keys in order and can be
torn by a ``partial_write`` fault, leaving a durable prefix. Values are stored as canonical JSON
text, so the agent can never mutate stored state through an aliased object.

The audit chain uses the entry format and hash rule of the agentic kit's file-based audit log
(:mod:`deterministic_agent_simulator.agentic.audit`): an exported chain verifies with
``deterministic-agent-simulator audit verify``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from deterministic_agent_simulator.agentic.audit import GENESIS_HASH
from deterministic_agent_simulator.core.canonical import canonical_json, sha256_hex

__all__ = ["AgentStore", "AuditChain", "ChainCheck"]


def _encode(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AgentStore:
    """Durable map plus a volatile write buffer (lost on crash)."""

    def __init__(self) -> None:
        self._durable: dict[str, str] = {}
        self._buffer: dict[str, str] = {}
        self.crashes = 0
        self.torn_writes = 0

    def get(self, key: str) -> JsonValue | None:
        raw = self._buffer.get(key)
        if raw is None:
            raw = self._durable.get(key)
        return json.loads(raw) if raw is not None else None

    def scan(self, prefix: str) -> list[tuple[str, JsonValue]]:
        merged = {k: v for k, v in self._durable.items() if k.startswith(prefix)}
        merged.update({k: v for k, v in self._buffer.items() if k.startswith(prefix)})
        return [(key, json.loads(merged[key])) for key in sorted(merged)]

    def put(self, key: str, value: JsonValue) -> None:
        self._buffer[key] = _encode(value)

    def flush(self) -> None:
        self._durable.update(self._buffer)
        self._buffer.clear()

    def put_durable(self, key: str, value: JsonValue) -> None:
        self._buffer.pop(key, None)
        self._durable[key] = _encode(value)

    def put_prefix_durable(self, items: list[tuple[str, JsonValue]], count: int) -> None:
        """Durably apply the first ``count`` items (a torn batch when ``count < len(items)``)."""
        for key, value in items[:count]:
            self.put_durable(key, value)
        if count < len(items):
            self.torn_writes += 1

    def crash(self) -> None:
        self.crashes += 1
        self._buffer.clear()

    def durable_keys(self) -> list[str]:
        return sorted(self._durable)


@dataclass(frozen=True, slots=True)
class ChainCheck:
    ok: bool
    entries: int
    bad_index: int | None = None
    reason: str | None = None


class AuditChain:
    """In-memory hash chain with the audit kit's entry format."""

    def __init__(self) -> None:
        self.entries: list[dict[str, JsonValue]] = []
        self.head = GENESIS_HASH

    def append(
        self, t_us: int, actor: str, action: str, data: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        fields: dict[str, JsonValue] = {
            "seq": len(self.entries),
            "ts": f"t+{t_us}us",
            "actor": actor,
            "action": action,
            "data": data,
            "prev_hash": self.head,
        }
        fields["hash"] = sha256_hex(canonical_json(fields))
        self.entries.append(fields)
        self.head = str(fields["hash"])
        return fields

    @staticmethod
    def check_entry(entry: dict[str, JsonValue], index: int, prev_hash: str) -> str | None:
        """Why ``entry`` does not extend a chain whose head is ``prev_hash`` (None if it does)."""
        if entry.get("seq") != index:
            return "sequence gap or reordering"
        if entry.get("prev_hash") != prev_hash:
            return "broken hash chain"
        body = {key: value for key, value in entry.items() if key != "hash"}
        if sha256_hex(canonical_json(body)) != entry.get("hash"):
            return "entry content modified"
        return None

    def verify(self) -> ChainCheck:
        prev = GENESIS_HASH
        for index, entry in enumerate(self.entries):
            reason = self.check_entry(entry, index, prev)
            if reason is not None:
                return ChainCheck(False, len(self.entries), index, reason)
            prev = str(entry["hash"])
        return ChainCheck(True, len(self.entries))

    def export_jsonl(self, path: Path) -> None:
        """Write the chain in the audit kit's JSONL format (one canonical entry per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(canonical_json(entry) + "\n" for entry in self.entries)
        path.write_text(lines, encoding="utf-8", newline="\n")

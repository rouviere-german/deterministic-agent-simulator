"""Optional read-only results API: campaign summaries and replay-file metadata.

It is the core hardened app (API keys held as digests, rate limits, body-size limits, security
headers, non-leaking errors) plus four routes, each behind a *scope*:

- ``GET /v1/campaigns`` and ``GET /v1/campaigns/{name}`` (``results:read``): summaries written
  by ``campaign --out``;
- ``GET /v1/repros`` (``results:read``): seed, invariant and trace hash of every replay file,
  and whether its signature verifies;
- ``POST /v1/repros/verify`` (``repros:verify``): check a replay document's signature without
  storing it.

Scopes are granted per key with ``APP_API_KEY_SCOPES`` (``<sha256>=<scope>+<scope>``); a key
listed only in ``APP_API_KEY_SHA256`` authenticates but has no scope (403 on every route here).
Every scoped digest is also registered as an API key, so scoped keys get their own rate-limit
bucket. Nothing here writes, runs a simulation or returns ticket text: responses are redacted and
replay files carry no customer data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from deterministic_agent_simulator.core.app import create_app
from deterministic_agent_simulator.core.config import Settings, get_settings
from deterministic_agent_simulator.core.errors import AppError
from deterministic_agent_simulator.core.security import extract_api_key
from deterministic_agent_simulator.replay import Keyring, ReproError, parse_repro
from deterministic_agent_simulator.report import redact_report_value

__all__ = [
    "SCOPE_REPROS_VERIFY",
    "SCOPE_RESULTS_READ",
    "ScopedKeys",
    "VerifyRequest",
    "create_results_app",
]

SCOPE_RESULTS_READ = "results:read"
SCOPE_REPROS_VERIFY = "repros:verify"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_LISTED = 200


class ScopedKeys:
    """Key digests with their scopes; lookup hashes the key and compares every digest."""

    def __init__(self, entries: Iterable[str]) -> None:
        parsed: list[tuple[bytes, frozenset[str]]] = []
        for entry in entries:
            digest, _, scopes = entry.partition("=")
            parsed.append((bytes.fromhex(digest), frozenset(scopes.split("+"))))
        self._entries = tuple(parsed)

    @property
    def digests(self) -> list[str]:
        return [digest.hex() for digest, _ in self._entries]

    def scopes_for(self, presented: str | None) -> frozenset[str]:
        if not presented or len(presented) > 512:
            return frozenset()
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        granted: frozenset[str] = frozenset()
        for candidate, scopes in self._entries:  # no early exit: timing reveals nothing
            if hmac.compare_digest(digest, candidate):
                granted = scopes
        return granted


def require_scope(scope: str) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        keys: ScopedKeys = request.app.state.scoped_keys
        presented = extract_api_key(request.headers)
        if scope in keys.scopes_for(presented):
            return
        if request.app.state.api_key_verifier.verify(presented) is None:
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(
                401,
                "unauthorized",
                "A valid API key is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        raise AppError(403, "forbidden", "This API key lacks the required scope.")

    return dependency


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: dict[str, JsonValue] = Field(max_length=16)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _campaigns(root: Path) -> list[Path]:
    folder = root / "campaigns"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.json") if _NAME.fullmatch(p.stem))[:_MAX_LISTED]


def _repro_summary(path: Path, keyring: Keyring | None) -> dict[str, JsonValue] | None:
    data = _read_json(path)
    if data is None:
        return None
    expected = data.get("expected")
    valid = False
    if keyring is not None:
        try:
            parse_repro(path.read_bytes(), keyring)
            valid = True
        except (ReproError, OSError):
            valid = False
    return {
        "name": path.stem,
        "seed": data.get("seed") if isinstance(data.get("seed"), int) else None,
        "invariant": expected.get("invariant") if isinstance(expected, dict) else None,
        "trace_hash": expected.get("trace_hash") if isinstance(expected, dict) else None,
        "key_id": data.get("key_id") if isinstance(data.get("key_id"), str) else None,
        "signature_valid": valid,
    }


def create_results_app(
    settings: Settings | None = None, *, results_dir: Path | None = None
) -> FastAPI:
    cfg = settings if settings is not None else get_settings()
    root = results_dir if results_dir is not None else cfg.results_dir
    scoped = ScopedKeys(cfg.api_key_scopes)
    known = sorted({*cfg.api_key_sha256, *scoped.digests})
    app = create_app(cfg.model_copy(update={"api_key_sha256": known}))
    app.state.scoped_keys = scoped
    keyring: Keyring | None = None
    if cfg.replay_hmac_key is not None:
        keyring = Keyring.from_settings(cfg)
    router = APIRouter(prefix="/v1")
    read = [Depends(require_scope(SCOPE_RESULTS_READ))]

    @router.get("/campaigns", dependencies=read)
    async def list_campaigns() -> dict[str, JsonValue]:
        items: list[JsonValue] = []
        for path in _campaigns(root):
            data = _read_json(path)
            if data is None:
                continue
            items.append(
                {
                    "name": path.stem,
                    "seeds_run": data.get("seeds_run"),
                    "violations": data.get("violations"),
                    "stopped": data.get("stopped"),
                }
            )
        listed: dict[str, JsonValue] = redact_report_value({"campaigns": items})
        return listed

    @router.get("/campaigns/{name}", dependencies=read)
    async def get_campaign(name: str) -> dict[str, JsonValue]:
        if not _NAME.fullmatch(name):
            raise AppError(404, "not_found", "Not Found")
        data = _read_json(root / "campaigns" / f"{name}.json")
        if data is None:
            raise AppError(404, "not_found", "Not Found")
        redacted: dict[str, JsonValue] = redact_report_value(data)
        return redacted

    @router.get("/repros", dependencies=read)
    async def list_repros() -> dict[str, JsonValue]:
        folder = root / "repros"
        paths = sorted(folder.glob("*.json"))[:_MAX_LISTED] if folder.is_dir() else []
        summaries = [s for s in (_repro_summary(p, keyring) for p in paths) if s is not None]
        listed: dict[str, JsonValue] = redact_report_value({"repros": summaries})
        return listed

    @router.post("/repros/verify", dependencies=[Depends(require_scope(SCOPE_REPROS_VERIFY))])
    async def verify_repro(body: VerifyRequest) -> Mapping[str, JsonValue]:
        if keyring is None:
            raise AppError(503, "unavailable", "Replay verification is not configured.")
        raw = json.dumps(body.document).encode("utf-8")
        try:
            scenario, expected = parse_repro(raw, keyring)
        except ReproError:
            return {"valid": False}
        return {"valid": True, "seed": scenario.seed, "invariant": str(expected["invariant"])}

    app.include_router(router)
    return app

"""The optional read-only results API: scoped keys, redaction, signed-repro verification."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deterministic_agent_simulator.api import (
    SCOPE_REPROS_VERIFY,
    SCOPE_RESULTS_READ,
    ScopedKeys,
    create_results_app,
)
from deterministic_agent_simulator.core.config import Settings
from deterministic_agent_simulator.core.security import hash_api_key

pytestmark = pytest.mark.integration

READER = "reader-key-" + "r" * 30
VERIFIER = "verifier-key-" + "v" * 30
UNSCOPED = "unscoped-key-" + "u" * 30
REGRESSIONS = Path(__file__).resolve().parents[1] / "regressions"
# The example emitted by scripts/regenerate_fixtures.py (signed with the test-only key).
REPRO = sorted(REGRESSIONS.glob("repro_seed_*.json"))[0]


def settings(**overrides: Any) -> Settings:
    """Settings for the API; the replay key comes from the environment (``replay_key_env``)."""
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [hash_api_key(UNSCOPED)],
        "api_key_scopes": [
            f"{hash_api_key(READER)}={SCOPE_RESULTS_READ}",
            f"{hash_api_key(VERIFIER)}={SCOPE_REPROS_VERIFY}",
        ],
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def results_dir(tmp_path: Path, replay_key_env: str) -> Path:
    del replay_key_env  # configures APP_REPLAY_HMAC_KEY for every app built in these tests
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    summary = {
        "seeds_run": 2000,
        "violations": {},
        "stopped": "completed",
        "note": "contact ana.silva@customers.example about CANARY-0123456789abcdef",
    }
    (campaigns / "nightly-2026-09-22.json").write_text(json.dumps(summary), encoding="utf-8")
    (campaigns / "Bad Name.json").write_text("{}", encoding="utf-8")
    (campaigns / "broken.json").write_text("{not json", encoding="utf-8")
    repros = tmp_path / "repros"
    repros.mkdir()
    (repros / REPRO.name).write_bytes(REPRO.read_bytes())
    forged = json.loads(REPRO.read_text(encoding="utf-8"))
    forged["seed"] += 1
    (repros / "forged.json").write_text(json.dumps(forged), encoding="utf-8")
    (repros / "garbage.json").write_text("[]", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(results_dir: Path) -> Iterator[TestClient]:
    app = create_results_app(settings(), results_dir=results_dir)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def test_every_route_requires_its_scope(client: TestClient) -> None:
    assert client.get("/v1/campaigns").status_code == 401
    assert client.get("/v1/campaigns", headers=auth("made-up")).status_code == 401
    # An authenticated key without the scope is forbidden, not unauthorized.
    assert client.get("/v1/campaigns", headers=auth(UNSCOPED)).status_code == 403
    assert client.get("/v1/campaigns", headers=auth(VERIFIER)).status_code == 403
    assert client.get("/v1/campaigns", headers=auth(READER)).status_code == 200
    body: dict[str, Any] = {"document": {}}
    assert client.post("/v1/repros/verify", json=body, headers=auth(READER)).status_code == 403
    assert client.post("/v1/repros/verify", json=body, headers=auth(VERIFIER)).status_code == 200


def test_campaign_summaries_are_listed_and_redacted(client: TestClient) -> None:
    listed = client.get("/v1/campaigns", headers=auth(READER)).json()["campaigns"]
    assert [item["name"] for item in listed] == ["nightly-2026-09-22"]  # bad names are skipped
    detail = client.get("/v1/campaigns/nightly-2026-09-22", headers=auth(READER))
    assert detail.status_code == 200
    text = detail.text
    assert "ana.silva" not in text
    assert "CANARY-0123456789abcdef" not in text
    assert detail.json()["seeds_run"] == 2000


@pytest.mark.parametrize("name", ["..%2F..%2Fsecret", "Bad%20Name", "missing", "broken"])
def test_unknown_or_unsafe_campaign_names_are_not_found(client: TestClient, name: str) -> None:
    response = client.get(f"/v1/campaigns/{name}", headers=auth(READER))
    assert response.status_code == 404


def test_repro_listing_reports_signature_validity(client: TestClient) -> None:
    repros = client.get("/v1/repros", headers=auth(READER)).json()["repros"]
    by_name = {item["name"]: item for item in repros}
    assert by_name[REPRO.stem]["signature_valid"] is True
    expected = json.loads(REPRO.read_text(encoding="utf-8"))["expected"]["invariant"]
    assert by_name[REPRO.stem]["invariant"] == expected
    assert by_name["forged"]["signature_valid"] is False
    assert "garbage" not in by_name


def test_verify_accepts_signed_documents_and_rejects_forgeries(client: TestClient) -> None:
    document = json.loads(REPRO.read_text(encoding="utf-8"))
    valid = client.post("/v1/repros/verify", json={"document": document}, headers=auth(VERIFIER))
    assert valid.json() == {
        "valid": True,
        "seed": document["seed"],
        "invariant": document["expected"]["invariant"],
    }
    document["tickets"] = [0, 1, 2]
    forged = client.post("/v1/repros/verify", json={"document": document}, headers=auth(VERIFIER))
    assert forged.json() == {"valid": False}


def test_verify_is_unavailable_without_a_replay_key(results_dir: Path) -> None:
    app = create_results_app(settings(replay_hmac_key=None), results_dir=results_dir)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/v1/repros/verify", json={"document": {}}, headers=auth(VERIFIER)
        )
        assert response.status_code == 503
        listed = test_client.get("/v1/repros", headers=auth(READER)).json()["repros"]
        assert all(item["signature_valid"] is False for item in listed)


def test_empty_results_directory(tmp_path: Path) -> None:
    app = create_results_app(settings(), results_dir=tmp_path / "nothing-here")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        assert test_client.get("/v1/campaigns", headers=auth(READER)).json() == {"campaigns": []}
        assert test_client.get("/v1/repros", headers=auth(READER)).json() == {"repros": []}


def test_scoped_keys_ignore_oversized_or_missing_keys() -> None:
    keys = ScopedKeys([f"{hash_api_key(READER)}={SCOPE_RESULTS_READ}"])
    assert keys.scopes_for(READER) == frozenset({SCOPE_RESULTS_READ})
    assert keys.scopes_for(None) == frozenset()
    assert keys.scopes_for("x" * 600) == frozenset()
    assert keys.digests == [hash_api_key(READER)]

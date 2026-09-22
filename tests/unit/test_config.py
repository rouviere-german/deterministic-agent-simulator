from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from deterministic_agent_simulator.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.anthropic_api_key is None
    assert settings.llm_model == "claude-opus-5"


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("APP_LLM_EFFORT", "medium")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "not-a-real-key"
    assert "not-a-real-key" not in repr(settings)
    assert settings.llm_effort == "medium"


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {
        "ANTHROPIC_API_KEY" if name == "anthropic_api_key" else f"APP_{name.upper()}"
        for name in Settings.model_fields
    }
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_env_example_loads_and_simulator_settings_are_validated(repo_root: Path) -> None:
    example = Settings(_env_file=repo_root / ".env.example")
    assert example.replay_hmac_key is None  # an empty key means "not configured"
    assert example.sim_bug_flags == []
    assert example.api_key_scopes == []
    parsed = Settings(
        _env_file=None,
        replay_hmac_key="k" * 32,
        sim_bug_flags="missing_dedupe, non_idempotent_retry",
        api_key_scopes=f"{'a' * 64}=results:read+repros:verify",
    )
    assert parsed.sim_bug_flags == ["missing_dedupe", "non_idempotent_retry"]
    assert "k" * 32 not in repr(parsed)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, replay_hmac_key="too-short")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_scopes=["not-a-digest=results:read"])

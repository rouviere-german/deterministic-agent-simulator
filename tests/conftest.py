"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import hashlib
import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from hypothesis import HealthCheck, settings

from deterministic_agent_simulator.core.config import get_settings

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

# A replay-signing key for tests only. It is derived at runtime from a public label, so it is not
# a secret and no key material is committed; production keys come from APP_REPLAY_HMAC_KEY.
TEST_REPLAY_KEY = hashlib.sha256(b"deterministic-agent-simulator test-only replay key").hexdigest()
TEST_REPLAY_KEY_ID = "tests"


def _is_emitted_regression(path: Path) -> bool:
    return path.parent.name == "regressions"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Regression tests emitted by the simulator are integration tests (they run simulations)."""
    for item in items:
        if _is_emitted_regression(Path(item.path)):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _offline_and_clean_env(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)
    if _is_emitted_regression(Path(request.node.path)):
        # Emitted regression tests read the replay key from the environment, as in a deployment.
        monkeypatch.setenv("APP_REPLAY_HMAC_KEY", TEST_REPLAY_KEY)
        monkeypatch.setenv("APP_REPLAY_KEY_ID", TEST_REPLAY_KEY_ID)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def replay_key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure the test-only replay key through the environment, as a deployment would."""
    monkeypatch.setenv("APP_REPLAY_HMAC_KEY", TEST_REPLAY_KEY)
    monkeypatch.setenv("APP_REPLAY_KEY_ID", TEST_REPLAY_KEY_ID)
    get_settings.cache_clear()
    return TEST_REPLAY_KEY


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT

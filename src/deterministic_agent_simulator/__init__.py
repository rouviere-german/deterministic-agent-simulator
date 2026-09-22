"""Deterministic simulation testing for LLM agents.

Sub-packages:
- ``sim``: the kernel: a seeded single-threaded asyncio event loop with virtual time, SimRandom,
  trace hashing, fault profiles and the hermeticity guard.
- ``world``: simulated services, the approval ledger, humans, the at-least-once inbox, SimLLM and
  the port adapters the agent runs against.
- ``agent``: the port contract, the action grammar, the reference agent and its planted bugs,
  and the live (provider-backed) model port.
- ``check``: invariant oracles and the Wing-Gong linearizability checker.
- ``core`` and ``agentic``: the shared baseline (settings, redacting logs, tracing, metrics, the
  hardened app factory; providers, envelopes, tools, approvals, budgets, audit log).

Top-level modules: ``runner`` (scenarios and campaigns), ``shrink`` (ddmin), ``replay`` (signed
replay files), ``emit`` (regression tests), ``report``, ``demo``, ``cli`` and ``api``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

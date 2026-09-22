"""Human-facing reports: redacted failure summaries and OpenTelemetry export of traces.

Everything that leaves the simulator for a person or a collector passes through
:func:`redact_report_text`: canary tokens planted in injected tool output are replaced first, then
the core redactor removes e-mail addresses, tokens and other PII-shaped strings. Replay files are
different: they carry no ticket text at all (see :mod:`deterministic_agent_simulator.replay`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic import JsonValue

from deterministic_agent_simulator.core.redaction import redact_text
from deterministic_agent_simulator.sim.trace import Trace, TraceEvent

if TYPE_CHECKING:
    from deterministic_agent_simulator.runner import RunResult
    from deterministic_agent_simulator.shrink import ShrinkResult

__all__ = [
    "CANARY_PATTERN",
    "export_trace_spans",
    "failure_report",
    "redact_report_text",
    "redact_report_value",
    "trace_lines",
]

CANARY_PATTERN = re.compile(r"CANARY-[0-9a-f]{8,}")
_CANARY = "[REDACTED:CANARY]"
# Spans need absolute timestamps; virtual time starts at this fixed epoch (2026-01-01T00:00:00Z),
# so exported traces are identical from run to run.
_EPOCH_NS = 1_767_225_600_000_000_000


def redact_report_text(text: str) -> str:
    """Remove canary tokens, then PII and secrets (idempotent)."""
    return redact_text(CANARY_PATTERN.sub(_CANARY, text))


def redact_report_value(value: Any) -> Any:
    """Recursively redact strings inside JSON-like values."""
    if isinstance(value, str):
        return redact_report_text(value)
    if isinstance(value, dict):
        return {str(key): redact_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_report_value(item) for item in value]
    return value


def trace_lines(trace: Trace, *, limit: int = 400) -> list[str]:
    """Rendered trace events (redacted), at most ``limit`` of them."""
    return [redact_report_text(event.render()) for event in trace.events[:limit]]


def failure_report(result: RunResult, shrunk: ShrinkResult | None = None) -> dict[str, JsonValue]:
    """A redacted summary of a failing run, with the shrunk trace when there is one."""
    first = result.first
    report: dict[str, JsonValue] = {
        "seed": result.scenario.seed,
        "config": result.scenario.config.to_json(),
        "invariant": first.invariant if first else None,
        "detail": redact_report_text(first.detail) if first else None,
        "trace_hash": result.trace_hash,
        "trace_length": result.trace_length,
        "fault_profile": list(result.fault_profile),
    }
    if shrunk is not None:
        lines: list[JsonValue] = (
            list(trace_lines(shrunk.final.trace)) if shrunk.final.trace is not None else []
        )
        report["shrunk"] = {
            "trace_length": shrunk.shrunk_length,
            "reduction_ratio": round(shrunk.ratio, 2),
            "tests": shrunk.tests,
            "tickets": len(shrunk.shrunk.tickets or ()),
            "faults": [list(key) for key in shrunk.shrunk.faults or ()],
            "decisions": len(shrunk.shrunk.decisions or ()),
            "trace": lines,
        }
    redacted: dict[str, JsonValue] = redact_report_value(report)
    return redacted


def _span_attributes(event: TraceEvent) -> dict[str, str | int | bool]:
    attributes: dict[str, str | int | bool] = {"das.kind": event.kind, "das.t_us": event.t_us}
    for index, value in enumerate(event.fields):
        if value is None:
            continue
        attributes[f"das.field.{index}"] = (
            redact_report_text(value) if isinstance(value, str) else value
        )
    return attributes


def export_trace_spans(trace: Trace, provider: TracerProvider, *, name: str) -> int:
    """Export a kept trace as OpenTelemetry spans; returns the number of spans created.

    The run is one root span. Every RPC becomes a child span from its start to its completion
    (virtual time, anchored at a fixed epoch); every other event is a span event on the root.
    """
    if not trace.keep:
        raise ValueError("the trace was recorded with keep=False; nothing to export")
    tracer = provider.get_tracer("deterministic_agent_simulator.trace")
    end_us = trace.events[-1].t_us if trace.events else 0
    root = tracer.start_span(name, start_time=_EPOCH_NS, attributes={"das.events": trace.length})
    context = otel_trace.set_span_in_context(root)
    spans = 1
    for event in trace.events:
        timestamp = _EPOCH_NS + event.t_us * 1_000
        if event.kind == "rpc" and len(event.fields) >= 4 and isinstance(event.fields[3], int):
            start = _EPOCH_NS + event.fields[3] * 1_000
            child = tracer.start_span(
                str(event.fields[0]),
                context=context,
                start_time=start,
                attributes=_span_attributes(event),
            )
            child.end(end_time=timestamp)
            spans += 1
        else:
            root.add_event(event.kind, attributes=_span_attributes(event), timestamp=timestamp)
    root.end(end_time=_EPOCH_NS + end_us * 1_000)
    return spans

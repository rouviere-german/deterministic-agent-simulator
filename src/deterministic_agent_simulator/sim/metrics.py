"""Prometheus metrics for simulation campaigns (a per-campaign registry, no global state)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

__all__ = ["SimMetrics", "create_sim_metrics"]


@dataclass(frozen=True, slots=True)
class SimMetrics:
    registry: CollectorRegistry
    seeds_total: Counter
    invariant_violations_total: Counter
    seeds_per_second: Gauge
    trace_events_total: Counter

    def observe_run(self, mode: str, trace_length: int, invariant: str | None) -> None:
        self.seeds_total.labels(mode).inc()
        self.trace_events_total.inc(trace_length)
        if invariant is not None:
            self.invariant_violations_total.labels(invariant).inc()

    def set_throughput(self, seeds_per_second: float) -> None:
        self.seeds_per_second.set(seeds_per_second)

    def render(self) -> bytes:
        return generate_latest(self.registry)


def create_sim_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "das"
) -> SimMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return SimMetrics(
        registry=reg,
        seeds_total=Counter(
            "seeds_total",
            "Simulated seeds run, by fault mode.",
            ["mode"],
            namespace=namespace,
            registry=reg,
        ),
        invariant_violations_total=Counter(
            "invariant_violations_total",
            "Seeds whose first violation was this invariant.",
            ["invariant"],
            namespace=namespace,
            registry=reg,
        ),
        seeds_per_second=Gauge(
            "seeds_per_second", "Campaign throughput.", namespace=namespace, registry=reg
        ),
        trace_events_total=Counter(
            "trace_events_total",
            "Semantic trace events recorded.",
            namespace=namespace,
            registry=reg,
        ),
    )

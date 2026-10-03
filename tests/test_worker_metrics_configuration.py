"""#57: a SUPPORTED local metrics configuration, not a test monkeypatch.

At d0e271a the only way to get metrics out of this Worker was for a test to
assign ``Default.event_sink = aggregator``. That is not a configuration seam, it
is a way for the test suite to be the sole consumer of the instrumentation, and
it is the whole reason #205 REQ-CFOPS-016 could not honestly be called a source
pass: the shipped configuration dropped every event on the floor.

WHAT IS CLAIMED HERE, PRECISELY
-------------------------------
* Enablement is explicit, off by default, and read PER REQUEST from the
  Worker environment.
* The state is ISOLATE-scoped: one process, one aggregate, lost on recycle.
  It is not fleet-wide Cloudflare metrics and nothing here claims it is.
* A configuration change takes effect on the next request rather than being
  pinned to whatever the first request in this isolate happened to see.
* Retention is bounded by configuration and bounded again in code, so a large
  configured number cannot turn into unbounded memory.
* No secret, binding name or account identifier becomes a metric dimension.
"""

from __future__ import annotations

import asyncio

import pytest

from oai2.observability import MetricsAggregator, NullEventSink
from oai2.observability.isolate import ISOLATE_METRICS, IsolateMetrics
from tests._worker_entrypoint_fixtures import (
    Env,
    entrypoint,
    reset_admission,
    valid_body,
)

#: Stands in for a secret that is present in the environment but is not part
#: of the metrics configuration. Deliberately NOT credential-shaped: the
#: public-safety scan flags real-looking tokens wherever they appear, and a
#: test that trips the scan is a test people learn to ignore.
CANARY_ENV_SECRET = "CANARY_ENV_SECRET_7d41e0b3"


@pytest.fixture(autouse=True)
def _isolate_metrics_reset() -> None:
    ISOLATE_METRICS._reset_for_tests()
    reset_admission()


def test_the_default_configuration_drops_every_event() -> None:
    """Off by default, and off means NullEventSink rather than a live store.

    A default-on metrics store retains a bounded window of recent activity on
    every isolate for a deployment that never asked for it. That is not a
    default anyone should get for free.
    """
    sink = ISOLATE_METRICS.sink_for(enabled=False, max_recent_events=8)
    assert isinstance(sink, NullEventSink)
    assert ISOLATE_METRICS.aggregator() is None


def test_enabling_by_configuration_installs_a_real_aggregator() -> None:
    sink = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    assert isinstance(sink, MetricsAggregator)
    assert ISOLATE_METRICS.aggregator() is sink


def test_the_same_configuration_keeps_history_across_requests() -> None:
    """An aggregator is per-ISOLATE, so it must survive between requests.

    Rebuilding on every request would make every metric equal to the last
    request's metrics, which is indistinguishable from no metrics at all.
    """
    first = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    second = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    assert first is second


def test_disabling_after_enabling_stops_counting_on_the_next_request() -> None:
    """Cloudflare can reuse an isolate across a binding-only change.

    A sink pinned at first use would keep counting after an operator turned
    metrics off, forever, for the life of that isolate — which is the failure
    mode where turning telemetry off appears not to work.
    """
    ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    sink = ISOLATE_METRICS.sink_for(enabled=False, max_recent_events=8)
    assert isinstance(sink, NullEventSink)
    assert ISOLATE_METRICS.aggregator() is None


def test_re_enabling_after_disabling_works_without_an_isolate_recycle() -> None:
    ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    ISOLATE_METRICS.sink_for(enabled=False, max_recent_events=8)
    sink = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    assert isinstance(sink, MetricsAggregator)


def test_a_retention_change_rebuilds_rather_than_silently_keeping_the_old_bound() -> None:
    """A retention setting that does not take effect is worse than none.

    The retained window is part of the contract the operator configured, so a
    change rebuilds rather than being quietly ignored.
    """
    first = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    second = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=64)
    assert first is not second
    assert second is ISOLATE_METRICS.aggregator()
    assert ISOLATE_METRICS.rebuilds == 1, (
        "a configuration change must be visible, not silent"
    )


def test_a_rebuild_does_not_report_the_old_configuration_s_numbers() -> None:
    """Counters from a previous configuration must not be attributed to this one.

    Silently carrying history across a retention change would report a
    population that was never collected under the current bound.
    """
    first = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    first.emit(_event("knowledge.request.started"))
    second = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=64)
    assert second.counter("events_observed") == 0  # type: ignore[attr-defined]


def test_retention_is_clamped_so_configuration_cannot_make_it_unbounded() -> None:
    from oai2.observability.isolate import MAX_METRICS_RECENT_EVENTS

    sink = ISOLATE_METRICS.sink_for(
        enabled=True, max_recent_events=10**9
    )
    assert isinstance(sink, MetricsAggregator)
    assert sink.snapshot()["retained_event_capacity"] == MAX_METRICS_RECENT_EVENTS


def test_isolate_metrics_is_a_separate_instance_from_the_shared_singleton() -> None:
    """Two isolates are two processes, and the test proves it.

    Without this, a class-level cache would make the tests pass while the
    production shape is wrong — and the production shape is the thing nobody
    re-reads.
    """
    other = IsolateMetrics()
    a = other.sink_for(enabled=True, max_recent_events=8)
    b = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    assert a is not b
    assert other.aggregator() is a
    assert ISOLATE_METRICS.aggregator() is b


# ---------------------------------------------------------------------------
# The real Worker path
# ---------------------------------------------------------------------------


def test_the_worker_reads_enablement_from_its_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at d0e271a: normal configuration could not enable metrics at all.

    ``Default.event_sink`` was a hard-coded ``NullEventSink()`` and nothing in
    the shipped source could replace it, so every event the entrypoint emitted
    was discarded before it left the process.
    """
    from tests._worker_entrypoint_fixtures import drive

    with entrypoint() as entry:
        enabled = Env()
        enabled.KNOWLEDGE_METRICS_ENABLED = "true"

        response = asyncio.run(
            drive(monkeypatch, entry, enabled, valid_body())
        )

        assert response.status == 200
        agg = ISOLATE_METRICS.aggregator()
        assert agg is not None, (
            "enabling metrics through configuration produced no aggregator"
        )
        assert agg.counter("requests_started") == 1
        assert agg.counter("requests_completed") == 1


def test_the_worker_stays_silent_when_metrics_are_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests._worker_entrypoint_fixtures import drive

    with entrypoint() as entry:
        response = asyncio.run(
            drive(monkeypatch, entry, Env(), valid_body())
        )
        assert response.status == 200
        assert ISOLATE_METRICS.aggregator() is None


def test_enabling_metrics_does_not_change_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-OBS-025, driven through the real entrypoint.

    Same status, same body, with telemetry off and on. Only the trace and
    metric artifacts differ.
    """
    from tests._worker_entrypoint_fixtures import drive

    with entrypoint() as entry:
        off = asyncio.run(drive(monkeypatch, entry, Env(), valid_body("req-off")))

    ISOLATE_METRICS._reset_for_tests()
    reset_admission()

    with entrypoint() as entry:
        enabled = Env()
        enabled.KNOWLEDGE_METRICS_ENABLED = "true"
        on = asyncio.run(drive(monkeypatch, entry, enabled, valid_body("req-off")))

    assert off.status == on.status
    assert off.payload == on.payload
    assert ISOLATE_METRICS.aggregator().counter("requests_completed") == 1


@pytest.mark.asyncio
async def test_a_saturated_request_is_counted_when_metrics_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backpressure must be visible in the configured metrics, not just the 503.

    A saturation that is answered correctly but never counted is invisible to
    whoever is trying to size the isolate, which is the only person the
    saturation response is for.
    """
    from tests._worker_entrypoint_fixtures import Components, Gate, drive, wire

    with entrypoint() as entry:
        env = Env()
        env.KNOWLEDGE_METRICS_ENABLED = "1"
        env.KNOWLEDGE_MAX_IN_FLIGHT = "1"

        gate = Gate(hold=True)
        wire(monkeypatch, Components(gate))

        held = asyncio.create_task(
            entry.Default(env).fetch(_request())  # type: ignore[attr-defined]
        )
        await gate.entered.wait()

        # The single slot is held, so this one is refused.
        second = await drive(monkeypatch, entry, env, valid_body("req-2"))
        assert second.status == 503

        gate.release.set()
        await held

    agg = ISOLATE_METRICS.aggregator()
    assert agg is not None
    assert agg.counter("saturated_count") == 1
    assert agg.counter("requests_started") == 2, (
        "a refused request is still a request that was started"
    )


def test_an_invalid_enablement_value_is_read_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in configuration must not take the Worker down.

    Falling back to OFF is the safe direction: an unrecognised value cannot
    silently start retaining request data nobody asked to retain.
    """
    from tests._worker_entrypoint_fixtures import drive

    with entrypoint() as entry:
        env = Env()
        env.KNOWLEDGE_METRICS_ENABLED = "yes-please"

        response = asyncio.run(drive(monkeypatch, entry, env, valid_body()))
        assert response.status == 200
        assert ISOLATE_METRICS.aggregator() is None


def test_an_invalid_retention_value_is_read_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oai2.observability.isolate import DEFAULT_METRICS_RECENT_EVENTS
    from tests._worker_entrypoint_fixtures import drive

    with entrypoint() as entry:
        env = Env()
        env.KNOWLEDGE_METRICS_ENABLED = "true"
        env.KNOWLEDGE_METRICS_MAX_RECENT = "lots"

        response = asyncio.run(drive(monkeypatch, entry, env, valid_body()))
        assert response.status == 200
        agg = ISOLATE_METRICS.aggregator()
        assert agg is not None
        assert agg.snapshot()["retained_event_capacity"] == DEFAULT_METRICS_RECENT_EVENTS


def test_no_secret_or_binding_name_becomes_a_metric_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret in the environment is configuration, not telemetry.

    The aggregator's dimensions are drawn from closed vocabularies in the
    event schema, so the only way a token or a binding name could get in
    would be if configuration were being smuggled in as a label. Both the
    auth token and an unrelated secret are checked, because either one
    appearing would mean the whole configuration path was being retained
    rather than just read.
    """
    from tests._worker_entrypoint_fixtures import TOKEN, drive

    with entrypoint() as entry:
        env = Env()
        env.KNOWLEDGE_METRICS_ENABLED = "true"
        env.KNOWLEDGE_METRICS_MAX_RECENT = "64"
        # Present in the environment, not part of the metrics configuration.
        env.SOME_OTHER_SECRET = CANARY_ENV_SECRET
        env.KNOWLEDGE_KV = "production-kv-namespace"

        response = asyncio.run(drive(monkeypatch, entry, env, valid_body()))
        assert response.status == 200

        agg = ISOLATE_METRICS.aggregator()
        assert agg is not None
        snapshot = repr(agg.snapshot())
        recent = repr(
            [
                (r.timestamp, r.category, r.event_type, r.operation, r.outcome)
                for r in agg.recent_events
            ]
        )
        for secret in (TOKEN, CANARY_ENV_SECRET, "production-kv-namespace"):
            assert secret not in snapshot, f"{secret!r} reached a metric dimension"
            assert secret not in recent, f"{secret!r} reached a retained record"


# -- helpers ---------------------------------------------------------------


def _event(event_type: str) -> object:
    from oai2.observability import EventCategory, Trace

    trace = Trace()
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=event_type,
        timestamp=1.0,
    )


def _request() -> object:
    from tests._worker_entrypoint_fixtures import Request

    return Request(valid_body("req-1"))

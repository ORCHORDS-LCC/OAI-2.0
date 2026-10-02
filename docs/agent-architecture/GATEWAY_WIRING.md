# OAI-2.0 → public gateway wiring

This document describes the client-side contract between the OAI-2.0 runtime and
the public ORCHORDS inference gateway (`api.orchords.com`). It is the
source of truth for the env-var contract and the operational shape of
the path. No credentials or live tokens appear in this file.

## Data path

```
+------------------+      HTTP (HTTPS)      +-------------------+      +--------+
| OAI-2.0 runtime  |  ───────────────────▶  | api.orchords.com  |  ──▶ | oai-1.2 |
| GatewayRuntime   |  POST /v1/chat/        | ORCHORDS gateway  |      +--------+
| (InferenceRuntime|       completions       | (q-pipe)          |
|  subclass)       |  Authorization: Bearer |                   |
+------------------+                        +-------------------+
```

The runtime is :class:`oai2.runtime.gateway_runtime.GatewayRuntime`,
an :class:`oai2.runtime.InferenceRuntime` subclass that translates
:class:`oai2.runtime.InferenceRequest` into an OpenAI-compatible chat
completions request, POSTs it to the configured base URL, and parses
the response back into an :class:`oai2.runtime.InferenceResponse`.

The HTTP transport is `httpx`. For tests the transport is replaced
with `httpx.MockTransport` so the runner-free `scripts/verify.py`
cycle covers the wire path end-to-end with zero live calls.

## Configuration

The gateway runtime reads four environment variables. The first is
required; the others have documented defaults.

| Variable                      | Required | Default                     | Purpose                                         |
|-------------------------------|----------|------------------------------|--------------------------------------------------|
| `OAI2_GATEWAY_API_KEY`        | yes      | (empty)                      | Bearer token sent on every request.             |
| `OAI2_GATEWAY_BASE_URL`       | no       | `https://api.orchords.com`   | Base URL of the gateway.                        |
| `OAI2_GATEWAY_MODEL`          | no       | `oai-1.2`                    | Default model id sent in the request body.      |
| `OAI2_GATEWAY_TIMEOUT_SECONDS`| no       | `60`                         | Per-request HTTP timeout.                       |

The populated values must live in a gitignored `.env`. `.env.example`
documents the contract with empty values; real tokens are NEVER
committed. The `scripts/verify.py` `public-safety` scan enforces
this — it flags any committed string that looks like a hard-coded
`api_key=...` of 16+ characters.

## Operating the wire

A single chat completion probe (operator-side, after populating
`.env`):

```
uv run python scripts/gateway_smoke.py --prompt "ping"
uv run python scripts/gateway_smoke.py --prompt "ping" --json
```

A multi-prompt sweep with token/latency breakdown (planned,
see issue tracker): not in scope for this revision.

## Public runtime selector

`oai2.runtime.select_runtime_from_env()` is the env-aware public entry
point for callers that want the highest-fidelity runtime without
coupling to a specific backend. It returns
`oai2.runtime.GatewayRuntime` when `OAI2_GATEWAY_API_KEY` is set;
otherwise it returns `oai2.runtime.PlaceholderRuntime`. The selector
never raises on a missing or partial key — CI / offline / no-token
environments keep working with a deterministic offline runtime.

The static `oai2.runtime.default_runtime()` is **unchanged** and
always returns `PlaceholderRuntime`; the selector is additive and
does not flip the default. Tests pin both contracts.

## Local gate

`scripts/verify.py` runs four mandatory checks (sync / ruff / mypy /
pytest) plus three opt-in ones. The `gateway-reach` sub-check is
opt-in — it executes only when `OAI2_GATEWAY_API_KEY` is set, and
exits cleanly (SKIP) otherwise. When it runs, it sends one
`GET /v1/models` request and confirms the configured
`OAI2_GATEWAY_MODEL` is in the response.

```
PASS gateway-reach: https://api.orchords.com exposes oai-1.2 (of 1 models)
```

The check is intentionally live (no `MockTransport`) — it is the
single source of truth that the gateway is actually reachable from
the local host. Disable by leaving the env var unset.

## Errors

The runtime raises :class:`GatewayRuntimeError` on non-2xx
responses or transport failures. The error message is redacted
through :func:`_redact` so a bearer token that accidentally
appears in an upstream error body is not echoed to logs. The
HTTP status code is preserved on `exc.status_code`.

| Code | Meaning                                                    |
|------|-------------------------------------------------------------|
| 0    | Transport error (connect refused, DNS failure, timeout).    |
| 200  | Upstream returned 200 with malformed payload.               |
| 401  | Missing or invalid bearer token.                            |
| 5xx  | Upstream gateway fault.                                     |

## Status of the wire

| Layer                                                | Status      | Notes                                                                              |
|-------------------------------------------------------|-------------|------------------------------------------------------------------------------------|
| `oai2.runtime.GatewayRuntime`                        | IMPLEMENTED | 15 unit tests + 39 completion-contract tests, mock-transport only.                 |
| `oai2.runtime.GatewayConfig`                         | IMPLEMENTED | `repr()` redacts the API key.                                                      |
| `oai2.runtime.GatewayModelClient`                    | IMPLEMENTED | 14 unit tests; flattens messages, propagates `GatewayRuntimeError` unchanged.       |
| `scripts/gateway_smoke.py`                           | IMPLEMENTED | Single-shot end-to-end probe; 9 dedicated CLI tests covering every `main()` branch. |
| `scripts/bench.py --backend=gateway`                 | IMPLEMENTED | Drives `GatewayModelClient` end-to-end; 7 dedicated tests; clean SKIP without API key. |
| Cross-repo Protocol conformance (OAI-2.0 ↔ q-pipe)   | IMPLEMENTED | 10 tests in `tests/test_gateway_model_client_protocol.py`; `isinstance(client, qpipe.ModelClient)` is True. |
| `scripts/verify.py gateway-reach`                    | IMPLEMENTED | SKIP by default, live when key is set.                                             |
| `oai2.runtime.select_runtime_from_env()`             | IMPLEMENTED | Env-aware public entry point: returns `GatewayRuntime` when `OAI2_GATEWAY_API_KEY` is set, else `PlaceholderRuntime`. Never raises. 5 dedicated tests in `tests/test_runtime.py`. |
| `oai2.evals.run_eval_harness()` + `HarnessReport`    | IMPLEMENTED | Drives the builtin capability suites via `select_runtime_from_env()`; aggregates per-suite pass-rate; 12 dedicated tests in `tests/test_eval_harness.py` covering both placeholder and gateway (with MockTransport) paths plus `continue_on_error` fault isolation (`SuiteReport.error` carries `f"{ExcType}: {msg}"` for failing suites while surviving suites still complete). |
| `scripts/verify.py backend-smoke`                    | IMPLEMENTED | Hermetic cross-repo seam smoke: `scripts/backend_smoke.py` exercises `run_eval_harness()` end-to-end through `select_runtime_from_env()` + mocked `GatewayRuntime` (no live token); 10 dedicated tests in `tests/test_backend_smoke.py` pin the `PASS/FAIL` stdout contract and exit codes 0/1/2 (forced-failure paths via patched selector / patched harness / boundary-routed subprocess wrapper + the `_owns_runtime()` guard regression test for `runtime.close()`). |
| End-to-end from P50                                  | OPEN        | Not yet exercised — pending operator-supplied token.                                |
| api.orchords.com model list                          | OPEN        | Cloud currently exposes 4 models; local qpipe exposes 1.                           |
| Knowledge transport worker                           | PROPOSED    | Cloudflare Worker entrypoint in design phase.                                      |

## Test surface (cloud-touching layers)

All tests below pass under `httpx.MockTransport` and exercise the wire
path without a live endpoint. The runner-free `scripts/verify.py` cycle
runs the full set; live acceptance still requires
`OAI2_GATEWAY_API_KEY`.

| File                                              | Tests | Slice SHA  | Coverage                                                                 |
|---------------------------------------------------|-------|------------|--------------------------------------------------------------------------|
| `tests/test_gateway_runtime.py`                   | 15    | `2828a05`  | Request build, response parse, error mapping, lifecycle, key redaction.  |
| `tests/test_gateway_completion_contract.py`       | 39    | `2828a05`  | Malformed envelopes, finish-reason propagation, tool-call preservation.  |
| `tests/test_gateway_model_client.py`              | 14    | `56eab61`  | Structural adapter contract, message flattening, default substitution.    |
| `tests/test_gateway_smoke.py`                     |  9    | `6a4fb5f`  | CLI `main()` exit codes (0/1/2), JSON + human paths, `--model` override. |
| `tests/test_bench_gateway.py`                     |  7    | `53b546a`  | `bench.py --backend=gateway` end-to-end + key redaction in metrics.       |
| `tests/test_gateway_model_client_protocol.py`     | 10    | `33b76fb`  | Cross-repo `isinstance(client, qpipe.ModelClient)` + signature parity.    |
| `tests/test_runtime.py` (selector slice)           |  5    | (slice 12) | `select_runtime_from_env()` no-key / key-set / partial-config / re-export / default-unchanged. |
| `tests/test_eval_harness.py` (harness slice)       | 12    | (slice 13, 14) | `run_eval_harness()` placeholder path (5), gateway MockTransport path (1), suite-name filtering / pass-rate aggregation (2), and `continue_on_error` fault isolation (4) — runs-all-when-one-raises, surfaces-error-in-subreport, default-raises, default-propagates-subclass. |
| `tests/test_backend_smoke.py` (smoke gate)         | 10    | (slice 15, 16, 17) | `scripts/backend_smoke.py` exit-code / stdout contract: 6 happy-path + 3 forced-failure (selector mismatch → exit 1; construction failure → exit 2 via boundary wrapper; `scores != n_cases` → exit 1) + 1 regression (`runtime.close()` only called on `GatewayRuntime`, fixing the slice-16 `_CloseablePlaceholder` shim). |
| `tests/test_evidence_package_validation.py` (retrieval validation) | 49 | (slice 18) | Negative-path coverage for `oai2/knowledge/evidence_package.py` outer guard rails: `token_budget` / `max_entries` / `token_counter` callability / empty-package feasibility / `raw_source_tokens` / `_rate` rate-in-range. 49 parametrized cases pin every `ValueError` raise site not already covered by `test_evidence_package.py`. |
| **Total cloud-touching tests**                    | **170**|           |                                                                          |

Cross-repo Protocol tests discover q-pipe via the same convention
`scripts/verify.py` uses (`$OAI2_QPIPE_REPO` overrides;
`<repo-parent>/q-pipe` default) and `pytest.skip` cleanly when q-pipe
is unreachable.

## Status of the related q-pipe cloud

The local q-pipe checkout (`../q-pipe`) restricts `models_payload()`
to the single public model id `oai-1.2`. The deployed
`api.orchords.com` instance still exposes four models — that change
must be re-applied on the cloud deployment. Tracking is in the
issue tracker.

## Completion outcome contract (#238)

`GatewayRuntime` preserves the upstream `finish_reason` in `InferenceResponse`,
and `GatewayModelClient` forwards that value unchanged. `length` and
`content_filter` are not rewritten to `stop`; a missing or null reason remains
`None`. Callers must inspect the reason before treating a reply as complete.
Existing non-gateway runtimes default to an unknown reason rather than claiming
that upstream generation stopped normally.

Tool-only replies retain the ordered `tool_calls` objects, including their IDs,
names and argument strings. The adapter does not execute tools or validate their
arguments for execution; the authorized caller remains responsible for those
checks. Preserving response metadata does not implement outbound tool schemas,
structured conversation forwarding, streaming or a complete tool loop.

Invalid JSON, non-object envelopes, missing choices, invalid metadata types,
and a `tool_calls` finish reason without calls raise `GatewayRuntimeError`. Partial
text at a token limit remains available alongside its original termination
reason; it is not silently discarded or represented as a completed answer.

#238 evidence chain (slice-by-slice, latest first):

| SHA         | Subject                                                                        |
|-------------|--------------------------------------------------------------------------------|
| (slice 13)  | `oai2.evals.run_eval_harness()` + `HarnessReport` driving `select_runtime_from_env()`; 8 dedicated tests covering placeholder and gateway (MockTransport) paths. |
| `e645633`   | Env-aware runtime selector `oai2.runtime.select_runtime_from_env()`; 5 dedicated tests. |
| `69aac20`   | Doc refresh: wire-status table reflects post-`33b76fb` cloud-touching surface. |
| `33b76fb`   | Cross-repo `ModelClient` Protocol conformance (10 tests).                      |
| (slice 14)  | `continue_on_error` fault isolation on `run_eval_harness()` (4 tests; `SuiteReport.error` carries `f"{ExcType}: {msg}"` while surviving suites still complete). |
| (slice 15)  | Hermetic `scripts/verify.py backend-smoke` gate: `scripts/backend_smoke.py` drives `run_eval_harness()` end-to-end via mocked `GatewayRuntime`; 6 dedicated tests pin the PASS/FAIL/exit-code contract. |
| (slice 16)  | Full exit-code coverage of `scripts/backend_smoke.py`: 3 forced-failure tests pin FAIL/exit-1 (selector mismatch + `scores != n_cases`) and FAIL unexpected/exit-2 (boundary-routed construction failure via subprocess wrapper). Restores `OAI2_GATEWAY_API_KEY` env in `try/finally` so the leak doesn't reach `test_verify_script`. |
| (slice 17)  | Guard `runtime.close()` in `scripts/backend_smoke.main()` with `isinstance(runtime, GatewayRuntime)`. Root-causes the unconditional `.close()` call that masked contract-1 violations as `AttributeError` exit-2. Regression test (`test_backend_smoke_close_is_only_called_on_gateway_runtime`) fails before the fix, passes after. Removes the slice-16 `_CloseablePlaceholder` shim from the existing forced-failure test (now redundant). |
| (slice 18)  | Negative-path coverage for `oai2/knowledge/evidence_package.py` outer guard rails. 49 parametrized tests in `tests/test_evidence_package_validation.py` pin every `ValueError` raise site not already covered by `test_evidence_package.py`: `token_budget` (zero / negative / non-int / bool), `max_entries` (zero / negative / non-int / bool / callable shape), `token_counter` callability, empty-package feasibility (`token_budget cannot fit the empty package encoding`), `raw_source_tokens` (zero / negative / non-int / bool), empty `relevant_knowledge_ids`, and `_rate` out-of-range / non-numeric (parametrized over 8 edge values × 2 rate arguments). |
| `6a4fb5f`   | Smoke-CLI test coverage (9 tests on `scripts/gateway_smoke.py`).               |
| `2f29287`   | Mypy clean-up on `scripts/bench.py` (silences 2 long-standing errors).         |
| `53b546a`   | `bench.py --backend=gateway` integration + 7 dedicated tests.                   |
| `2828a05`   | Completion-outcome preservation (finish_reason, tool_calls).                   |
| `56eab61`   | `GatewayModelClient` adapter for q-pipe `ModelClient` seam.                     |

Focused verification in the supported project environment:

```bash
uv run pytest -W error \
    tests/test_gateway_runtime.py \
    tests/test_gateway_completion_contract.py \
    tests/test_gateway_model_client.py \
    tests/test_gateway_smoke.py \
    tests/test_bench_gateway.py \
    tests/test_gateway_model_client_protocol.py \
    tests/test_runtime.py \
    tests/test_eval_harness.py \
    tests/test_backend_smoke.py \
```

These tests use HTTPX mock transport. They are not live gateway, ZCode, model
quality, or cross-repository acceptance evidence. #238 remains open for live
integration evidence (operator-supplied `OAI2_GATEWAY_API_KEY`) and for the
q-pipe `HeldOutRunner` round-trip (operator-action item).

Protocol reference: [Chat Completions response contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/).

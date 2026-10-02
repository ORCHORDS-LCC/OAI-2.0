#!/usr/bin/env python3
"""OAI-2.0 cloud-model liveness probe — table view of every known cloud model.

Reports the status of every id in :data:`oai2.runtime.KNOWN_CLOUD_MODELS`
against the configured public gateway. ``KNOWN_CLOUD_MODELS`` is the
strict single-id allowlist this client treats as acceptable — only
``oai-2.0`` is in it. Designed for an operator with
``OAI2_GATEWAY_API_KEY`` in a gitignored ``.env`` to verify whether
``oai-2.0`` is reachable right now — independent of any configured
``OAI2_GATEWAY_MODEL`` default.

Usage::

    uv run python scripts/gateway_probe_models.py
    uv run python scripts/gateway_probe_models.py --json
    uv run python scripts/gateway_probe_models.py --candidates oai-2.0

The bearer token is read from ``OAI2_GATEWAY_API_KEY`` and never echoed.
The script is runner-free; it talks to the gateway directly via httpx.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from oai2.runtime import (  # noqa: E402
    DEFAULT_GATEWAY_BASE_URL,
    DEFAULT_GATEWAY_MODEL,
    KNOWN_CLOUD_MODELS,
    GatewayConfigError,
    ModelProbe,
    WorkingModelResolution,
    discover_cloud_models,
    load_gateway_config_from_env,
    resolve_working_model,
)

DEFAULT_PROBE_PROMPT = "ping"
DEFAULT_PROBE_MAX_TOKENS = 8


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe every known public-gateway model id and print a status table.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override OAI2_GATEWAY_BASE_URL (default: %(default)s).",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help=(
            "Comma-separated list of candidate model ids to test in order. "
            "Defaults to KNOWN_CLOUD_MODELS."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Configured primary model id (default: OAI2_GATEWAY_MODEL or oai-2.0).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROBE_PROMPT,
        help="Probe prompt text (default: %(default)r).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_PROBE_MAX_TOKENS,
        help="Max tokens for each probe (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object instead of a human-readable report.",
    )
    return parser


def _render_human(
    base_url: str,
    primary: str,
    probes: list[ModelProbe],
    selected: str | None,
    exposed: list[str] | None,
) -> str:
    lines: list[str] = []
    lines.append(f"OAI-2.0 gateway model probe — {base_url}")
    lines.append(f"primary model: {primary}")
    if exposed is not None:
        lines.append(f"GET /v1/models exposes: {exposed!r}")
    lines.append("")
    lines.append(f"{'model':<20} {'status':<10} {'reachable':<10} {'latency_ms':<14} error")
    lines.append("-" * 80)
    for probe in probes:
        latency = "-" if probe.latency_ms is None else f"{probe.latency_ms:.1f}"
        error = probe.error or "-"
        lines.append(
            f"{probe.model_id:<20} {probe.status_code:<10} "
            f"{str(probe.reachable):<10} {latency:<14} {error}",
        )
    lines.append("")
    if selected is None:
        lines.append("RESULT: no candidate answered successfully.")
    else:
        lines.append(f"RESULT: selected {selected} (first reachable in chain).")
    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    config = load_gateway_config_from_env()
    if config is None:
        raise GatewayConfigError(
            "OAI2_GATEWAY_API_KEY is not set; cannot probe the cloud.",
        )

    base_url = (args.base_url or config.base_url).rstrip("/") or DEFAULT_GATEWAY_BASE_URL
    primary = (args.model or config.model).strip() or DEFAULT_GATEWAY_MODEL
    if args.candidates:
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        candidates = sorted(KNOWN_CLOUD_MODELS)

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "User-Agent": "oai2-gateway-probe/0.1",
    }
    timeout_obj = httpx.Timeout(args.timeout)
    with httpx.Client(
        base_url=base_url,
        timeout=timeout_obj,
        headers=headers,
    ) as client:
        exposed: list[str] | None = None
        try:
            exposed = discover_cloud_models(client)
        except Exception:
            exposed = None

        order = [primary] + [c for c in candidates if c != primary]
        resolution: WorkingModelResolution = resolve_working_model(
            client,
            order,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        probes = list(resolution.probes)

    if args.json:
        payload = {
            "base_url": base_url,
            "primary_model": primary,
            "selected_model_id": resolution.selected_model_id,
            "exposed_models": exposed,
            "probes": [asdict(probe) for probe in probes],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            _render_human(
                base_url=base_url,
                primary=primary,
                probes=probes,
                selected=resolution.selected_model_id,
                exposed=exposed,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

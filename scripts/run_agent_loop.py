#!/usr/bin/env python3
"""CLI runner for the OAI-2.0 agent loop.

Sends a user prompt to oai-2.0 (via the canonical :class:`AgentLoop`)
with the six default tools (``Read`` / ``Edit`` / ``Write`` /
``Bash`` / ``Glob`` / ``Grep``) and prints the multi-step trace.

This is the harness that proves oai-2.0 actually uses tools when the
host provides tool scaffolding — the root cause of
``sess_10cbe33c-d83b-42ce-bf2c-e505173922c3`` producing zero tool
calls across 34 model turns was that the wire path did not advertise
the tools. With the agent loop, oai-2.0 emits ``tool_calls`` on the
first turn for any task that requires reading or modifying the host.

Run with::

    OAI2_GATEWAY_API_KEY=... uv run python scripts/run_agent_loop.py \
        --prompt 'Read /tmp/orchords-docs/LEARNED_SUMMARY.md and tell me how many sections it has.'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oai2.agents import (
    AgentLoop,
    default_dispatch_policy,
    default_system_prompt,
    default_tool_definitions,
)
from oai2.tools import to_openai_wire


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt sent to oai-2.0 with the default tool set.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Override the default system prompt.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Working directory for Read/Edit/Write/Bash/Glob/Grep.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--resource-scope",
        action="append",
        default=[],
        help="Allowed resource scope (path). Repeatable. Defaults to the cwd.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON trace instead of a human table.",
    )
    args = parser.parse_args(argv)

    scopes = args.resource_scope or [str(args.cwd)]
    policy = default_dispatch_policy(resource_scopes=scopes)
    loop = AgentLoop(
        policy=policy,
        cwd=args.cwd,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    run = loop.run(args.prompt, system_prompt=args.system or default_system_prompt())

    if args.json:
        payload = {
            "prompt": run.user_prompt,
            "final_text": run.final_text,
            "finished_reason": run.finished_reason,
            "total_tool_calls": run.total_tool_calls,
            "total_input_tokens": run.total_input_tokens,
            "total_output_tokens": run.total_output_tokens,
            "tools_advertised": [td.name for td in default_tool_definitions()],
            "tools_wire_count": len(to_openai_wire(default_tool_definitions())),
            "steps": [
                {
                    "step_index": s.step_index,
                    "finish_reason": s.response.finish_reason,
                    "text_excerpt": s.response.text[:200],
                    "tool_calls": [
                        {"id": tc.id, "name": tc.tool_id, "arguments": tc.arguments}
                        for tc in s.tool_calls
                    ],
                    "tool_results": [
                        {
                            "call_id": tr.call_id,
                            "ok": tr.ok,
                            "output_excerpt": str(tr.output)[:200],
                            "error": tr.error,
                        }
                        for tr in s.tool_results
                    ],
                }
                for s in run.steps
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("=== AgentLoop trace ===")
        print(f"prompt:        {run.user_prompt[:120]}")
        print(f"finished:      {run.finished_reason}")
        print(f"tool_calls:    {run.total_tool_calls}")
        print(f"steps:         {len(run.steps)}")
        print()
        for s in run.steps:
            print(f"--- step {s.step_index} (finish_reason={s.response.finish_reason}) ---")
            if s.response.text:
                print(f"  text:   {s.response.text[:160]}")
            for tc in s.tool_calls:
                print(f"  CALL   {tc.tool_id}({tc.arguments})")
            for tr in s.tool_results:
                excerpt = (str(tr.output)[:120] if tr.ok else f"ERR: {tr.error}")
                print(f"  RESULT ok={tr.ok} {excerpt}")
        print()
        print(f"=== final_text ({len(run.final_text)} chars) ===")
        print(run.final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

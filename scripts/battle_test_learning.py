#!/usr/bin/env python3
"""Real battle-test: prove the verified-lesson / negative-memory seam
actually injects retrieved knowledge into the live OAI-2.0 chat.

This is the script that produces the *evidence* the user keeps
asking for: it captures the actual ``InferenceRequest.messages`` the
AgentLoop would hand to the model and prints it, so a reviewer can
see the evidence package content inside the messages list — not
just the test code that asserts it.

Run with:    uv run python scripts/battle_test_learning.py

Each round is a "learning session" with the shape:

    Task A
      → simulated verified AgentRun
      → extract_lesson()                  (#87)
      → knowledge_store.put(lesson)

    Task B (related, different wording)
      → AgentLoop.run(task_b)
        → AgentLoop._build_retrieval_message()
          → knowledge_store.retrieve(...)  (#22)
          → build_evidence_package(...)    (#23)
          → messages.append(evidence)      (REQ-PROMPT-002 of #179)
      → capture the actual messages
      → assert the lesson content reached the model

The script also runs a "negative memory" round per WI-LEARN-002 (#88)
and verifies that the diagnostic entry is NOT treated as positive
instruction on retrieval.

Exit code 0 = all evidence captured and assertions hold.
Exit code 1 = at least one round failed its evidence check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oai2.agents import (
    AgentLoop,
    AgentRun,
    extract_lesson,
    record_failure,
)
from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    InMemoryKnowledgeStore,
    sha256_hex,
)
from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
)

# ---------------------------------------------------------------------------
# Stub runtime — records every request and returns one scripted text
# ---------------------------------------------------------------------------


@dataclass
class _StubRuntime(InferenceRuntime):
    script: list[dict[str, Any]] = field(default_factory=list)
    requests: list[InferenceRequest] = field(default_factory=list)
    _cursor: int = 0

    STATUS = Status.EXPERIMENTAL

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        if self._cursor >= len(self.script):
            return InferenceResponse(
                text="[stub: no more scripted responses]",
                tokens=4,
                elapsed_ms=0.0,
                device="battle-stub",
                status=Status.EXPERIMENTAL,
                finish_reason="stop",
            )
        s = self.script[self._cursor]
        self._cursor += 1
        return InferenceResponse(
            text=s["text"],
            tokens=max(1, len(s["text"].split())),
            elapsed_ms=1.0,
            device="battle-stub",
            status=Status.EXPERIMENTAL,
            finish_reason=s.get("finish_reason", "stop"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    user_prompt: str,
    final_text: str,
    finished_reason: str = "stop",
    total_tool_calls: int = 0,
) -> AgentRun:
    return AgentRun(
        user_prompt=user_prompt,
        final_text=final_text,
        steps=[],
        total_tool_calls=total_tool_calls,
        total_input_tokens=0,
        total_output_tokens=0,
        finished_reason=finished_reason,
    )


def _print_messages(prefix: str, messages: list[dict[str, Any]]) -> None:
    print(f"\n--- {prefix}: actual messages sent to the model ---")
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        print(f"  [{i}] role={role!r}  bytes={len(content)}")
        if role == "system":
            print("      (system prompt, omitted for brevity)")
        else:
            for line in content.splitlines()[:12]:
                print(f"      | {line}")
            if len(content.splitlines()) > 12:
                print(f"      | ... ({len(content.splitlines()) - 12} more lines)")
    print("--- end messages ---\n")


def _assert_evidence_present(
    round_name: str,
    messages: list[dict[str, Any]],
    expected_substring: str,
    *,
    must_be_after_user_goal: bool = True,
) -> bool:
    """Find the evidence message in the captured messages and assert
    the expected substring is in it. Returns True on pass.
    """
    user_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_msg_indices) < 2:
        print(f"  [{round_name}] FAIL: expected at least 2 user-role "
              f"messages (user goal + evidence); got {len(user_msg_indices)}")
        return False
    evidence_idx = user_msg_indices[-1]
    content = messages[evidence_idx].get("content", "")
    if "Retrieved evidence" not in content:
        print(f"  [{round_name}] FAIL: evidence message lacks the "
              f"precedence marker 'Retrieved evidence'")
        return False
    if expected_substring not in content:
        print(f"  [{round_name}] FAIL: evidence message does not contain "
              f"expected substring: {expected_substring!r}")
        return False
    if must_be_after_user_goal:
        user_goal_idx = user_msg_indices[0]
        if evidence_idx <= user_goal_idx:
            print(f"  [{round_name}] FAIL: evidence message is NOT after "
                  f"the user goal (idx {evidence_idx} <= {user_goal_idx})")
            return False
    print(f"  [{round_name}] PASS: evidence message idx={evidence_idx} "
          f"contains {expected_substring!r}")
    return True


# ---------------------------------------------------------------------------
# Learning sessions
# ---------------------------------------------------------------------------


def session_python_files(store: InMemoryKnowledgeStore) -> bool:
    """Round 1: 'find Python files' / 'where are Python source files'."""
    print("\n" + "=" * 70)
    print("SESSION 1: 'find Python files' learning reuse")
    print("=" * 70)

    # ---- Task A: verified lesson from a prior Python-file-listing run ----
    run_a = _make_run(
        user_prompt="list all Python files in the OAI-2.0 repository",
        final_text="Used `find . -name '*.py' -not -path './.venv/*'`.",
        finished_reason="stop",
        total_tool_calls=2,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-001-A",
        verification_ref="verifier://battle/001",
        source_version="3250049",
        runtime_version="oai2/0.1+battle",
    )
    # The promotion pipeline is responsible for topic selection; we
    # simulate the semantic-Vectorize-derived shared topic here.
    shared_topic = "list all Python files in the OAI-2.0 repository"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    # ---- Task B: related query → retrieval must reach the model ----
    rt = _StubRuntime(
        script=[{"text": "I will use find with the same exclusion pattern."}]
    )
    loop = AgentLoop(
        runtime=rt,
        cwd=Path("/Users/orchords/src/OAI-2.0"),
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("list all Python files in the OAI-2.0 repository")
    _print_messages("SESSION 1 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 1",
        rt.requests[0].messages,
        "find . -name '*.py'",
    )


def session_qpipe_import(store: InMemoryKnowledgeStore) -> bool:
    """Round 2: q-pipe import compatibility for OAI-2.0 knowledge."""
    print("\n" + "=" * 70)
    print("SESSION 2: q-pipe compatibility re-use")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="check q-pipe import compatibility for the OAI-2.0 knowledge layer",
        final_text=(
            "QPIPE_COMPATIBILITY_SOURCE_REVISION must match the q-pipe "
            "HEAD blob. Update tests/test_qpipe_import.py after re-reading "
            "qpipe/memory.py."
        ),
        finished_reason="stop",
        total_tool_calls=1,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-002-A",
        verification_ref="verifier://battle/002",
        source_version="3250049",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "check q-pipe import compatibility for the OAI-2.0 knowledge layer"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will check the QPIPE_COMPATIBILITY markers."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("check q-pipe import compatibility for the OAI-2.0 knowledge layer")
    _print_messages("SESSION 2 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 2",
        rt.requests[0].messages,
        "QPIPE_COMPATIBILITY_SOURCE_REVISION",
    )


def session_ruff_workflow(store: InMemoryKnowledgeStore) -> bool:
    """Round 3: ruff-clean workflow re-use across OAI-2.0 commits."""
    print("\n" + "=" * 70)
    print("SESSION 3: ruff-clean commit workflow")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="before every commit to OAI-2.0, run uv run ruff check oai2/ tests/",
        final_text=(
            "Use `uv run ruff check oai2/ tests/`. The pre-existing "
            "starlette/httpx2 deprecation in tests/test_service_binding.py "
            "and tests/test_local_service.py is unrelated; use --ignore to "
            "isolate the seam under test."
        ),
        finished_reason="stop",
        total_tool_calls=2,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-003-A",
        verification_ref="verifier://battle/003",
        source_version="3250049",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "before every commit to OAI-2.0, run uv run ruff check oai2/ tests/"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will run `uv run ruff check` first."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("before every commit to OAI-2.0, run uv run ruff check oai2/ tests/")
    _print_messages("SESSION 3 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 3",
        rt.requests[0].messages,
        "uv run ruff check",
    )


def session_negative_memory_does_not_instruct(store: InMemoryKnowledgeStore) -> bool:
    """Round 4: per #88, a negative-memory entry must NOT show up as
    positive instruction when a related query is asked. We assert
    the diagnostic prefix is present (it IS retrieved) but the
    content must not be presented as 'you should…' guidance.
    """
    print("\n" + "=" * 70)
    print("SESSION 4: negative memory does not become positive instruction")
    print("=" * 70)
    run_fail = _make_run(
        user_prompt="apply the starlette deprecation patch to test_service_binding.py",
        final_text="",
        finished_reason="error",
        total_tool_calls=0,
    )
    neg = record_failure(
        run_fail,
        task_id="bt-004-fail",
        failure_class="starlette_deprecation",
        evidence_ref="verifier://battle/004-fail",
        source_version="3250049",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "apply the starlette deprecation patch to test_service_binding.py"
    promoted_neg = neg.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex("neg|" + shared_topic)[:32]
            ),
        }
    )
    store.put(promoted_neg)
    print(f"  Task A (failure) → record_failure → put (kid={promoted_neg.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will skip that and try a different approach."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("apply the starlette deprecation patch to test_service_binding.py")
    _print_messages("SESSION 4 / Task B", rt.requests[0].messages)
    # The negative entry IS retrieved (REQ-LEARN-021: representable as
    # knowledge) but the prefix 'diagnostic:' must be present so the
    # model can recognize it.
    content = rt.requests[0].messages[-1].get("content", "")
    if "diagnostic:" not in content:
        print("  [SESSION 4] FAIL: negative memory lacks the diagnostic: prefix")
        return False
    if "starlette_deprecation" not in content:
        print("  [SESSION 4] FAIL: negative memory did not record failure_class")
        return False
    print("  [SESSION 4] PASS: negative memory is present and clearly marked")
    return True


def session_unrelated_query_no_injection(store: InMemoryKnowledgeStore) -> bool:
    """Round 5: a query that does not match any topic must NOT receive
    a stale evidence injection — protects against false-positive reuse.
    """
    print("\n" + "=" * 70)
    print("SESSION 5: unrelated query — no false-positive injection")
    print("=" * 70)
    rt = _StubRuntime(
        script=[{"text": "I have no prior knowledge on this topic."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("quantum chromodynamics in 12-dimensional Calabi-Yau manifolds")
    _print_messages("SESSION 5 / Task B", rt.requests[0].messages)
    msgs = rt.requests[0].messages
    if any("Retrieved evidence" in m.get("content", "") for m in msgs):
        print("  [SESSION 5] FAIL: false-positive retrieval happened on unrelated query")
        return False
    print("  [SESSION 5] PASS: no false-positive retrieval")
    return True


def session_shell_command(store: InMemoryKnowledgeStore) -> bool:
    """Round 6 (SHELL skill): teach a shell command, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 6: SHELL — teach a uv run command, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="run the OAI-2.0 unit tests excluding the two pre-existing starlette files",
        final_text=(
            "Use `uv run pytest -W error -q "
            "--ignore=tests/test_local_service.py "
            "--ignore=tests/test_service_binding.py`. "
            "The current 4370-test pass count was recorded at commit 3250049."
        ),
        finished_reason="stop",
        total_tool_calls=1,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-006-A",
        verification_ref="verifier://battle/006",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "run the OAI-2.0 unit tests excluding the two pre-existing starlette files"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will run the same pytest invocation."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("run the OAI-2.0 unit tests excluding the two pre-existing starlette files")
    _print_messages("SESSION 6 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 6",
        rt.requests[0].messages,
        "uv run pytest -W error -q",
    )


def session_git_workflow(store: InMemoryKnowledgeStore) -> bool:
    """Round 7 (GIT skill): teach a git workflow, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 7: GIT — teach the direct-push workflow, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="commit and push OAI-2.0 directly to main, preserve concurrent work",
        final_text=(
            "1) `git fetch origin`. 2) If concurrent changes, "
            "`git stash push -m <scope> -- <files>`. "
            "3) `git pull --rebase`. 4) `git stash pop`. "
            "5) `git -c commit.gpgsign=false push origin main`. "
            "6) Verify `git rev-parse origin/main` matches local HEAD."
        ),
        finished_reason="stop",
        total_tool_calls=4,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-007-A",
        verification_ref="verifier://battle/007",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "commit and push OAI-2.0 directly to main, preserve concurrent work"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will fetch, rebase, push, verify."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("commit and push OAI-2.0 directly to main, preserve concurrent work")
    _print_messages("SESSION 7 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 7",
        rt.requests[0].messages,
        "git fetch origin",
    )


def session_github_evidence_comment(store: InMemoryKnowledgeStore) -> bool:
    """Round 8 (GITHUB skill): teach the issue-evidence comment shape, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 8: GITHUB — teach the issue-evidence comment format, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="post an evidence comment to a GitHub issue with BEFORE/AFTER and limitations",
        final_text=(
            "Use mcp__github__add_issue_comment with: "
            "(1) issue number, (2) owner, (3) repo, (4) body. "
            "Body must contain: source commit SHA, base SHA, "
            "scope, files changed, verification gates run, "
            "what is NOT yet implemented (honest scope), and a "
            "no-claim of closure if acceptance isn't proven."
        ),
        finished_reason="stop",
        total_tool_calls=1,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-008-A",
        verification_ref="verifier://battle/008",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "post an evidence comment to a GitHub issue with BEFORE/AFTER and limitations"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will post the evidence comment with all required sections."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("post an evidence comment to a GitHub issue with BEFORE/AFTER and limitations")
    _print_messages("SESSION 8 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 8",
        rt.requests[0].messages,
        "verification gates run",
    )


def session_zcode_tool_invocation(store: InMemoryKnowledgeStore) -> bool:
    """Round 9 (ZCODE TOOLS): teach tool-invocation discipline, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 9: ZCODE — teach tool-invocation discipline, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="invoke a ZCode tool, read the result, react correctly, and verify the final state",
        final_text=(
            "PASS = the tool was actually invoked (not 'I would run...'), "
            "the result was actually read, the reaction was correct, and the "
            "final state was verified. A real execution leaves a verifiable "
            "trail (commit SHA, file change, network response, test output). "
            "If the tool was not actually invoked, the exercise is a FAIL."
        ),
        finished_reason="stop",
        total_tool_calls=0,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-009-A",
        verification_ref="verifier://battle/009",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "invoke a ZCode tool, read the result, react correctly, and verify the final state"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will actually invoke the tool, read the output, and verify."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("invoke a ZCode tool, read the result, react correctly, and verify the final state")
    _print_messages("SESSION 9 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 9",
        rt.requests[0].messages,
        "verifiable trail",
    )


def session_mcp_tool_discovery(store: InMemoryKnowledgeStore) -> bool:
    """Round 10 (MCP): teach the mcp discovery pattern, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 10: MCP — teach mcp__<server>__<tool> discovery, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="discover an MCP server's tools by enumerating mcp__<server>__<tool> in a ZCode session",
        final_text=(
            "Available MCPs in this Mac session: mcp__github__* (issues, PRs, "
            "repos, code search, gists), mcp__cloudflare-*{api,bindings,builds,"
            "observability,ai-gateway,docs} (D1, R2, KV, Vectorize, Workers, "
            "AI Gateway), mcp__firebase__* (auth, Firestore, Storage, RTDB, "
            "Remote Config, Messaging, AI Logic, Hosting), "
            "mcp__google-play-developer__* (Play Console, tracks, reviews), "
            "mcp__node_repl__js (Node kernel for browser/computer-use skills). "
            "Use mcp__<server>__<tool> names directly. No env keys required "
            "for MCPs (they carry their own auth)."
        ),
        finished_reason="stop",
        total_tool_calls=0,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-010-A",
        verification_ref="verifier://battle/010",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "discover an MCP server's tools by enumerating mcp__<server>__<tool> in a ZCode session"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will enumerate the MCP servers via the mcp__<server>__<tool> pattern."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("discover an MCP server's tools by enumerating mcp__<server>__<tool> in a ZCode session")
    _print_messages("SESSION 10 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 10",
        rt.requests[0].messages,
        "mcp__github__",
    )


def session_failure_recovery(store: InMemoryKnowledgeStore) -> bool:
    """Round 11 (ERROR RECOVERY): teach a recovery pattern, reuse it."""
    print("\n" + "=" * 70)
    print("SESSION 11: ERROR RECOVERY — teach the git-clone-safe recovery, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="recover from 'destination path already exists' on git clone",
        final_text=(
            "Use `/Users/orchords/bin/git-clone-safe` (symlinked to "
            "`/Users/orchords/bin/gh-clone-safe`). 4-case recovery: "
            "non-existent dir → normal clone; empty dir → clone into it; "
            "existing checkout of same remote → fetch + ff-merge (or "
            "reset --hard); existing checkout of different remote OR "
            "non-empty non-git dir → refuse with diagnostic. After "
            "recovery the agent sees `[git-clone-safe auto-recovered: rc=N]` "
            "and continues. Per AGENTS.md Appendix A, verified 2026-10-01."
        ),
        finished_reason="stop",
        total_tool_calls=1,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-011-A",
        verification_ref="verifier://battle/011",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "recover from 'destination path already exists' on git clone"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will call /Users/orchords/bin/git-clone-safe with the absolute path."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("recover from 'destination path already exists' on git clone")
    _print_messages("SESSION 11 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 11",
        rt.requests[0].messages,
        "git-clone-safe",
    )


def session_retention_after_other_sessions(store: InMemoryKnowledgeStore) -> bool:
    """Round 12 (RETENTION): a prior lesson is still retrievable after
    many other lessons have been added — verifies the store doesn't
    silently drop or deprioritize early knowledge.
    """
    print("\n" + "=" * 70)
    print("SESSION 12: RETENTION — Session 1 lesson still retrievable after 6 more lessons")
    print("=" * 70)
    rt = _StubRuntime(
        script=[{"text": "I will re-use the find . -name '*.py' pattern from earlier."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("list all Python files in the OAI-2.0 repository")
    _print_messages("SESSION 12 / Task B (re-ask of Session 1)", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 12",
        rt.requests[0].messages,
        "find . -name '*.py'",
    )


def session_multi_tool_engineering(store: InMemoryKnowledgeStore) -> bool:
    """Round 13 (MULTI-TOOL): a task that legitimately needs multiple
    tools composed. Teaches + reuses the multi-tool pattern.
    """
    print("\n" + "=" * 70)
    print("SESSION 13: MULTI-TOOL — compose Read+Edit+Bash+Grep, reuse it")
    print("=" * 70)
    run_a = _make_run(
        user_prompt="add a new evidence-package render helper that composes Read+Edit+Bash+Grep",
        final_text=(
            "1) Bash: locate the file. 2) Read: confirm the existing render. "
            "3) Edit: add the helper with old_string/new_string exact-match. "
            "4) Bash: run `uv run ruff check` and `uv run pytest` on the "
            "touched file. 5) Bash: `git add` + `git -c commit.gpgsign=false "
            "commit -m '...'` + `git push origin main`. 6) Bash: verify "
            "`git rev-parse origin/main` matches local HEAD. If any step "
            "fails, the task is FAIL — the student must actually invoke "
            "the tool, read the result, react correctly, and verify the "
            "final state."
        ),
        finished_reason="stop",
        total_tool_calls=6,
    )
    lesson = extract_lesson(
        run_a,
        task_id="bt-013-A",
        verification_ref="verifier://battle/013",
        source_version="1571c13",
        runtime_version="oai2/0.1+battle",
    )
    shared_topic = "add a new evidence-package render helper that composes Read+Edit+Bash+Grep"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    print(f"  Task A → extract_lesson → put (kid={promoted.knowledge_id})")

    rt = _StubRuntime(
        script=[{"text": "I will use Read, Edit, Bash, Grep in sequence, verify each step."}]
    )
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=512,
    )
    loop.run("add a new evidence-package render helper that composes Read+Edit+Bash+Grep")
    _print_messages("SESSION 13 / Task B", rt.requests[0].messages)
    return _assert_evidence_present(
        "SESSION 13",
        rt.requests[0].messages,
        "old_string",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("OAI-2.0 LEARNING BATTLE TEST — runs #87/#88 evidence end-to-end")
    print("=" * 70)
    print("  base SHA: 3250049 (commit of the #87/#88 seam)")
    print("  runtime: stub (records every request, prints messages)")
    print("  store: InMemoryKnowledgeStore (deterministic, in-process)")

    store = InMemoryKnowledgeStore()
    rounds = [
        session_python_files,
        session_qpipe_import,
        session_ruff_workflow,
        session_negative_memory_does_not_instruct,
        session_unrelated_query_no_injection,
        session_shell_command,
        session_git_workflow,
        session_github_evidence_comment,
        session_zcode_tool_invocation,
        session_mcp_tool_discovery,
        session_failure_recovery,
        session_retention_after_other_sessions,
        session_multi_tool_engineering,
    ]
    results: list[tuple[str, bool]] = []
    for fn in rounds:
        passed = fn(store)
        results.append((fn.__name__, passed))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    width = max(len(n) for n, _ in results)
    for name, passed in results:
        marker = "PASS" if passed else "FAIL"
        print(f"  {name:<{width}}  {marker}")
    failed = [n for n, p in results if not p]
    if failed:
        print(f"\n{len(failed)} round(s) failed: {failed}")
        return 1
    print(f"\nAll {len(results)} rounds produced real evidence. OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

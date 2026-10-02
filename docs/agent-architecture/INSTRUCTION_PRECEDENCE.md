# Instruction Precedence and Trust Boundaries (WI-PROMPT-001)

_Owner issue: [#185](https://github.com/ORCHORDS-LCC/OAI-2.0/issues/185) · Parent: #179 · Traceability: `#1 → #179 → WI-PROMPT-001`_
_Status: `IMPLEMENTED` for the model and resolver. Not yet wired into `AgentLoop` — that is #186 / `WI-PROMPT-002`._

## Purpose

Give every instruction entering a prompt an explicit precedence and trust class,
and decide precedence in code rather than by asking a model to behave.

The module is `oai2/agents/instructions.py`. It is **pure**: no model calls, no
I/O, no clock, no global state. The same input list always produces the same
output, which is what makes REQ-PROMPT-014 satisfiable at all.

## Precedence classes

REQ-PROMPT-011 names the sources but not their order. This module orders them:

| Rank | Class | Typical source | Trust |
| --- | --- | --- | --- |
| 0 | `SYSTEM` | runtime/host policy | operator-owned, never overridable |
| 1 | `USER` | the user's goal this turn | operator-owned, outranks the repo |
| 2 | `REPOSITORY` | `AGENTS.md` and equivalents | operator-authored, overridable |
| 3 | `TOOL_METADATA` | tool descriptions and schemas | semantic input, **not** a grant |
| 4 | `UNTRUSTED` | web pages, file contents, tool output | evidence only, never policy |

**`USER` outranks `REPOSITORY` deliberately.** If repository policy outranked
the operator, a hostile checkout could permanently override the person running
it. Repository content is still treated as more trustworthy than tool metadata
and web output, so the ordering is a trust gradient, not a flat demotion.

## Two fields that carry the weight

### `directive` / `value` — how conflict is detected

Detecting *semantic* conflict between two instructions cannot be done
deterministically, and REQ-PROMPT-014 demands determinism. So an instruction may
declare a normalised `(directive, value)` pair — for example
`directive="network_access"`, `value="deny"`.

Two instructions conflict when they are at the **same precedence**, name the
**same directive**, and assert **different values**. Instructions without a
directive never conflict; they are prose, and this module does not pretend
otherwise. The tie-break is arrival order, and it is recorded rather than
applied silently (REQ-PROMPT-013).

### `asserted_precedence` — how override is detected

Every instruction also carries the precedence that the *content claims for
itself*. This is the load-bearing idea, and it is deliberately structural
rather than textual: the resolver never scans for phrases like "ignore previous
instructions", because phrase matching is **evadable and incomplete**. It misses
paraphrase, non-English text, and — the important case — instructions that
simply do not announce themselves. Deterministic phrase matching is entirely
possible; it is just not a sound boundary.

Three rejections follow from the structure:

1. `privilege_escalation` — content at a lower class claims a higher
   `asserted_precedence` (REQ-PROMPT-012). A **claim to evaluate, never a
   grant**: it can only cause a rejection, never a promotion.
2. `override_attempt` — content re-asserts a `directive` already bound by a
   strictly higher class (REQ-PROMPT-012).
3. `permission_grant_claimed` — `TOOL_METADATA` declares
   `grants_permission=True`. A tool description is semantic input; it is never a
   permission grant (REQ-PROMPT-015).

## Traceability

Every resolution returns a `ResolutionTrace` carrying
`schema_version`, the accepted instructions, the suppressed ones, every
conflict, and every override attempt. Nothing is dropped silently — that is the
issue's acceptance wording, and it is also the only way a later reader can tell
"the resolver rejected this" apart from "the resolver never saw it"
(REQ-PROMPT-014).

`INSTRUCTION_SCHEMA_VERSION` is a module constant carried onto every trace, so a
stored trace identifies the rules that produced it (REQ-PROMPT-016).

## Deliberate limits

- **This is not a security boundary.** It is a host-side control that makes
  precedence explicit and auditable. The real boundary for tool access remains
  the host-side tool policy, matching the existing statement in
  `oai2/agents/agent_loop.py` that the note "is identity branding only".
- **An attacker who never claims elevated precedence** is the expected case,
  not an edge case. It is classified `UNTRUSTED` and cannot bind a directive
  already held by a higher class, cannot revise anything, and cannot grant a
  permission. `asserted_precedence` is defence in depth on top of that, not the
  mechanism that carries it. `tests/test_instructions.py` covers the
  no-metadata case explicitly.
- **Trust is assigned by the host, never by the content.** `Instruction` is not
  the sanctioned constructor: `trusted_instruction()` is, and it forces
  `authenticated=False` for `UNTRUSTED` and `TOOL_METADATA` so evidence cannot
  claim authority by passing a flag.
- **Revisions are an operator capability.** `revises` on an authenticated
  `SYSTEM`/`USER`/`REPOSITORY` instruction replaces the earlier binding and is
  recorded; `revises` from anything else is refused as
  `revision_not_authorised`. This is what stops "use repository B instead of A"
  from being discarded because an earlier instruction arrived first — and what
  stops a retrieved paragraph imitating that correction.
- **No NLP.** Nothing here infers intent, sentiment, or meaning. If a future
  requirement needs that, it belongs in a different component with its own
  determinism story.
- **Not wired into `AgentLoop`.** `_build_retrieval_message()` in
  `oai2/agents/agent_loop.py:263` still emits its own hardcoded precedence
  marker. Replacing it with a generated, traceable one is #186's job; doing it
  here would couple two issues and touch the hot agent path.

## Verification

`tests/test_instructions.py` carries the prompt-injection and conflict fixtures
the acceptance criterion asks for:

| Fixture | Proves |
| --- | --- |
| web page claiming developer mode | `privilege_escalation` recorded, not accepted |
| web page re-asserting a bound directive | `override_attempt` recorded, not accepted |
| tool description claiming a grant | `permission_grant_claimed`, not accepted |
| repo policy vs user goal on one directive | user wins, conflict recorded |
| two same-level instructions, one directive | deterministic tie-break, both recorded |
| no conflicts at all | full list accepted, no invented conflicts |

Determinism is asserted directly: resolving the same input twice yields equal
traces, and resolving a permuted-but-equivalent input yields the same
precedence outcome.

## Composer (WI-PROMPT-002 / #186)

`oai2/agents/composer.py` builds on this model. Layout, always:

```
[system policy]        stable, versioned, cacheable
[project guidance]     stable, versioned, cacheable
[user goal]            current turn
[state delta]          only against compatible retained state
[fenced evidence]      tail, data not instruction
```

- The prefix identity is a digest over the exact head text **and** every version
  axis (policy, tools, project, composer). It is forwarded as
  `InferenceRequest.prefix_digest`, which is what #240's `PrefixKVCache` keys
  on — the composer supplies identity, the runtime owns KV state, and neither
  duplicates the other. `SessionCompatibilityKey` is reused rather than a
  second identity scheme invented.
- A delta is emitted **only** when retained state matches the exact prefix
  digest and composer version. Otherwise the composer rehydrates in full. It
  never assumes a server kept an earlier request's context.
- Compact references replace a body only when the resolver can return it; a
  missing or version-mismatched target falls back to the inline body, because a
  dangling `[ref:...]` would be silent context loss.
- Evidence is a `user`-role tail message inside an explicit fence, never a
  `system` message. `assert_template_safe()` runs on every composition and
  raises on a system message after the first non-system turn — the shape that
  took `api.orchords.com` down once already.

Measured by `scripts/bench_composer_prefix.py`. That script reports composition
shape and prefix identity only; TTFT, prefill and end-to-end rows are emitted as
`null` with a `pending_reason`, because they need the target Mac and the #240
owner is measuring there. A shorter wire payload is not by itself evidence of
less prefill.

**ORCHORDS — BUILD DIFFERENT.**

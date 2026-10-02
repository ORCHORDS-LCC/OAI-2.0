"""Concrete Apple-Silicon MLX inference runtime with hot model residency.

:class:`MLXHotRuntime` is the local counterpart of
:class:`~oai2.runtime.GatewayRuntime` for WI-PERF-003 / #240. It loads
``mlx_lm`` weights once, keeps them resident across calls, and emits
``prefill_seconds`` / ``decode_seconds`` / ``decode_tps`` in the
:class:`~oai2.runtime.InferenceResponse` notes so callers can report
TTFT, decode tok/s, and total latency independently.

The runtime deliberately does NOT implement MLX-level batched inference.
Concurrent requests are served by independent
:func:`mlx_lm.stream_generate` streams. MLX 0.32 supports concurrent
streams on the Metal device, so the benefit of "hot residency" appears
as soon as the load cost is amortized across requests.

Status: EXPERIMENTAL (added for WI-PERF-003 / #240).
"""

from __future__ import annotations

import time
from typing import Any

from mlx_lm import load as _mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..core import Status
from .inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    TemplateRenderError,
)
from .model import (
    ModelSpec,
    _prefill_tokens,
    _reset_peak_memory,
    _sampler,
    _seed_random,
    discover_default_device,
)
from .prefix_kv_cache import PrefixKVCache
from .tool_calls import (
    ToolCallParseError,
    normalise_tool_choice,
    parse_tool_calls,
)


def _has_structured_tool_history(messages: list[dict[str, Any]]) -> bool:
    """True when flattening ``messages`` to prose would destroy tool state.

    A ``tool`` role message, a ``tool_calls`` array, or a ``tool_call_id``
    all carry meaning that a ``### role:`` rendering drops. When any of them
    is present the caller must not silently degrade to that rendering.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            return True
        if message.get("tool_calls"):
            return True
        if message.get("tool_call_id"):
            return True
    return False


def _clear_layers(layers: Any) -> None:
    """Reset every KV layer to empty.

    The runtime keeps ONE set of cache layers alive across requests so a
    stored prefix can be restored into it. That makes them carry the
    previous request's state forward, including whatever was appended while
    decoding. On a cache miss nothing is restored, so without this reset the
    new prompt would be prefilled *on top of the last request's context* and
    the model would answer about a conversation that never happened.

    ``trim(n)`` returns the number actually trimmed and is a no-op on layers
    that cannot trim, so this is safe to call unconditionally.
    """
    for layer in layers:
        state = getattr(layer, "state", None)
        offset = 0
        if isinstance(state, tuple) and len(state) == 3:
            try:
                offset = int(state[2] or 0)
            except (TypeError, ValueError):
                offset = 0
        if offset > 0:
            trim = getattr(layer, "trim", None)
            if callable(trim):
                trim(offset)


class MLXHotRuntime(InferenceRuntime):
    """MLX runtime that keeps the model resident between calls.

    On :meth:`load` (idempotent) the runtime loads the model and
    tokenizer through :func:`mlx_lm.load`. :meth:`generate` streams
    tokens one at a time and records the per-stage wall time
    (prefill, decode, end-to-end) so callers can report TTFT, decode
    tok/s and total latency independently.
    """

    STATUS = Status.EXPERIMENTAL

    def __init__(
        self,
        spec: ModelSpec,
        *,
        model_id: str,
        prefix_cache: PrefixKVCache | None = None,
        gate_digest: bool = False,
    ) -> None:
        super().__init__(spec)
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: TokenizerWrapper | None = None
        self._load_seconds: float | None = None
        # Opt-in digest-keyed prefix KV-state reuse (WI-PERF-003 config D).
        # When enabled, repeated prompts whose token sequence extends a
        # stored prefix prefill only the divergent remainder.
        self._prefix_cache = prefix_cache
        # Serving paths set this: the request digest carries tool schema and
        # world state, which token-prefix matching cannot see, so reuse must
        # be gated on it. Training loops leave it off (see PrefixKVCache).
        self._gate_digest = gate_digest
        self._kv_layers: Any | None = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        t0 = time.perf_counter()
        # ``mlx_lm.load`` returns ``Union[(model, tokenizer),
        # (model, tokenizer, config)]``; default path is the 2-tuple
        # but mypy cannot narrow it.
        result: tuple[Any, ...] = _mlx_lm_load(self._model_id)
        self._model = result[0]
        self._tokenizer = result[1]
        self._load_seconds = time.perf_counter() - t0
        if self._prefix_cache is not None and self._kv_layers is None:
            from mlx_lm.models.cache import make_prompt_cache

            self._kv_layers = make_prompt_cache(self._model)

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(self, tokenizer: TokenizerWrapper, request: InferenceRequest) -> tuple[str, bool]:
        """Render the request for the serving tokenizer.

        Returns ``(prompt, used_fallback)``. Tools declared on the request
        are handed to the chat template so the model can actually see the
        signatures it is allowed to call — without this the advertised tool
        set never reaches the rendered input at all.

        A template failure falls back to a plain role-delimited rendering
        *only* when doing so cannot lose structured tool history; when it
        can, this raises :class:`TemplateRenderError` instead of returning a
        prompt that has quietly discarded the conversation's tool calls.
        """
        if not request.messages:
            return request.prompt, False

        messages = [dict(m) for m in request.messages]
        mode = normalise_tool_choice(request.tool_choice)
        # tool_choice="none" means "do not call tools", so the signatures are
        # withheld rather than advertised and then ignored.
        tools: list[dict[str, Any]] = [] if mode == "none" else list(request.tools)
        kwargs: dict[str, Any] = {"tools": tools} if tools else {}

        try:
            return (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                ),
                False,
            )
        except Exception as exc:
            if tools:
                raise TemplateRenderError(
                    f"chat template rejected the request while tools were advertised: {exc}"
                ) from exc
            if _has_structured_tool_history(messages):
                raise TemplateRenderError(
                    "chat template failed on structured tool history; refusing to flatten it "
                    f"into prose: {exc}"
                ) from exc
            parts = [f"### {m.get('role', 'user')}:\n{m.get('content', '')}\n" for m in messages]
            parts.append("### assistant:\n")
            return "\n".join(parts), True

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        tokenizer: TokenizerWrapper = self._tokenizer
        model: Any = self._model

        prompt_str, used_fallback = self._render_prompt(tokenizer, request)
        mode = normalise_tool_choice(request.tool_choice)

        _reset_peak_memory()
        if request.seed is not None:
            _seed_random(request.seed)
        ids = tokenizer.encode(prompt_str)
        prompt_tokens = len(ids)
        t_prefill_start = time.perf_counter()
        prefix_matched: int | None = None
        gen_input: str | list[int] = prompt_str
        cache_arg: Any = None
        if self._prefix_cache is not None and self._kv_layers is not None:
            # Always start from empty: on a miss nothing is restored below,
            # and prefilling onto the previous request's state would silently
            # answer from a conversation that never happened.
            _clear_layers(self._kv_layers)
            # STRICT-PREFIX reuse only. ``lookup_common_and_trim`` matches a
            # shared leading token run and trims the stored state back to it,
            # which is unsound against installed mlx_lm 0.32: after a trim the
            # cache still holds the full key/value arrays, so the next direct
            # model call builds its attention mask from a stale sequence
            # length and the model call raises
            # "too many values to unpack (expected 3, got 4)".
            # Truncating arrays is not available from here, so the runtime
            # reuses only when a stored entry is a strict prefix of this
            # query — the case where no trim is needed and the restored state
            # is exactly valid.
            prefix_matched = self._prefix_cache.lookup_and_apply(
                ids,
                self._kv_layers,
                digest=request.prefix_digest,
                gate_digest=self._gate_digest,
            )
            remainder = ids[prefix_matched:]
            if len(remainder) > 1:
                _prefill_tokens(model, self._kv_layers, remainder[:-1])
                self._prefix_cache.store(
                    ids[:-1],
                    [layer.state for layer in self._kv_layers],
                    digest=request.prefix_digest,
                )
            gen_input = remainder[-1:]
            cache_arg = self._kv_layers

        # Sampling settings must reach the backend; without a sampler
        # mlx_lm generates greedily regardless of what the caller asked for.
        gen_kwargs: dict[str, Any] = {}
        sampler = _sampler(temp=request.temperature, top_p=request.top_p)
        if sampler is not None:
            gen_kwargs["sampler"] = sampler

        text = ""
        tokens_generated = 0
        first_token_at: float | None = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=gen_input,
            max_tokens=request.max_tokens,
            prompt_cache=cache_arg,
            **gen_kwargs,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text += response.text
            tokens_generated = response.generation_tokens
        t_end = time.perf_counter()

        tool_calls: tuple[dict[str, Any], ...] = ()
        parse_error: str | None = None
        tool_call_source = "none"
        if mode != "none":
            tools_rendered = 0 if mode == "none" else len(request.tools)
            try:
                tool_calls = parse_tool_calls(text)
                if tool_calls:
                    tool_call_source = "tagged"
                elif tools_rendered:
                    # The request advertised tools, so the model was asked
                    # for a call. Small instruct models often answer with a
                    # fenced JSON object instead of the template's tag;
                    # refusing to read that would discard a real call.
                    tool_calls = parse_tool_calls(text, allow_bare_json=True)
                    if tool_calls:
                        tool_call_source = "fenced_json"
            except ToolCallParseError as exc:
                # The model *tried* to call a tool and produced something
                # unusable. Reporting this beats silently returning prose,
                # which the client would read as "no tool needed".
                parse_error = f"{type(exc).__name__}: {exc}"

        if first_token_at is None:
            prefill_seconds = t_end - t_prefill_start
            decode_seconds = 0.0
        else:
            prefill_seconds = first_token_at - t_prefill_start
            decode_seconds = t_end - first_token_at
        decode_tps = tokens_generated / decode_seconds if decode_seconds > 0 else None

        finish_reason = "tool_calls" if tool_calls else "stop"
        if parse_error is not None:
            finish_reason = "tool_call_parse_error"

        notes: list[str] = [
            f"model_id={self._model_id}",
            f"prompt_tokens={prompt_tokens}",
            f"prefill_seconds={prefill_seconds:.4f}",
            f"decode_seconds={decode_seconds:.4f}",
            f"tools_rendered={0 if mode == 'none' else len(request.tools)}",
            f"tool_call_source={tool_call_source}",
        ]
        if decode_tps is not None:
            notes.append(f"decode_tps={decode_tps:.2f}")
        if prefix_matched is not None:
            notes.append(f"prefix_matched={prefix_matched}")
            notes.append(f"prefix_avoided_tokens={prefix_matched}")
        if used_fallback:
            notes.append("chat_template_fallback=1")
        if parse_error is not None:
            notes.append(f"tool_call_parse_error={parse_error}")

        return InferenceResponse(
            text=text,
            tokens=tokens_generated,
            elapsed_ms=(t_end - t_prefill_start) * 1000.0,
            device=f"mlx:{discover_default_device()}",
            status=Status.EXPERIMENTAL,
            notes=notes,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            prefix_matched=prefix_matched,
            prefix_prompt_tokens=prompt_tokens,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    @property
    def prefix_cache(self) -> PrefixKVCache | None:
        """The opt-in digest-keyed prefix KV-state cache, if enabled."""
        return self._prefix_cache

    @property
    def gate_digest(self) -> bool:
        """Whether prefix reuse is restricted to a matching compatibility digest."""
        return self._gate_digest

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._kv_layers = None


__all__ = ["MLXHotRuntime"]

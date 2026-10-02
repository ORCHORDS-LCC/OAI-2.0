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
)
from .model import ModelSpec, _reset_peak_memory, discover_default_device


class MLXHotRuntime(InferenceRuntime):
    """MLX runtime that keeps the model resident between calls.

    On :meth:`load` (idempotent) the runtime loads the model and
    tokenizer through :func:`mlx_lm.load`. :meth:`generate` streams
    tokens one at a time and records the per-stage wall time
    (prefill, decode, end-to-end) so callers can report TTFT, decode
    tok/s and total latency independently.
    """

    STATUS = Status.EXPERIMENTAL

    def __init__(self, spec: ModelSpec, *, model_id: str) -> None:
        super().__init__(spec)
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: TokenizerWrapper | None = None
        self._load_seconds: float | None = None

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

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        tokenizer: TokenizerWrapper = self._tokenizer
        model: Any = self._model

        _reset_peak_memory()
        prompt_tokens = len(tokenizer.encode(request.prompt))
        t_prefill_start = time.perf_counter()
        text = ""
        tokens_generated = 0
        first_token_at: float | None = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text += response.text
            tokens_generated = response.generation_tokens
        t_end = time.perf_counter()

        if first_token_at is None:
            prefill_seconds = t_end - t_prefill_start
            decode_seconds = 0.0
        else:
            prefill_seconds = first_token_at - t_prefill_start
            decode_seconds = t_end - first_token_at
        decode_tps = tokens_generated / decode_seconds if decode_seconds > 0 else None

        notes: list[str] = [
            f"model_id={self._model_id}",
            f"prompt_tokens={prompt_tokens}",
            f"prefill_seconds={prefill_seconds:.4f}",
            f"decode_seconds={decode_seconds:.4f}",
        ]
        if decode_tps is not None:
            notes.append(f"decode_tps={decode_tps:.2f}")

        return InferenceResponse(
            text=text,
            tokens=tokens_generated,
            elapsed_ms=(t_end - t_prefill_start) * 1000.0,
            device=f"mlx:{discover_default_device()}",
            status=Status.EXPERIMENTAL,
            notes=notes,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


__all__ = ["MLXHotRuntime"]

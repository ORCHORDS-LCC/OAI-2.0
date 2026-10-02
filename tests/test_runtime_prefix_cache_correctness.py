"""Prefix KV reuse must not change what the model says.

This file records a measured defect rather than asserting a hoped-for one.

What was measured, on the real endpoint serving ZCode's configured model:
with reuse enabled, a request that followed an unrelated request returned a
continuation of the PREVIOUS prompt. The response was fluent and plausible,
so nothing in the payload said anything was wrong.

Root cause, at the source of installed mlx_lm 0.32: reusing KV state means
driving mlx_lm's prompt cache outside its own prefill path.
``KVCache.trim(n)`` decrements ``offset`` but leaves the key/value arrays at
their grown length, so the next ``model(tokens, cache=...)`` builds its
attention mask from a stale sequence length and raises::

    ValueError: too many values to unpack (expected 3, got 4)

A second, independent bug lived on top of it: the runtime keeps ONE set of
cache layers alive so a stored prefix can be restored into them, and on a
cache miss it never cleared them — so a miss prefilled the new prompt on top
of the previous request's entire context. ``_clear_layers`` fixes that one,
and the helper is covered below because it is the part that is reusable.
"""

from __future__ import annotations

import inspect

import pytest

from oai2.runtime.mlx_hot_runtime import _clear_layers


def test_reuse_is_opt_in_on_the_serving_path() -> None:
    """Correctness first: the endpoint must not reuse KV state by default."""
    from oai2.server.openai_compat_app import create_app

    default = inspect.signature(create_app).parameters["enable_prefix_cache"].default
    assert default is False, (
        "prefix reuse is wired but not sound against mlx_lm 0.32; enabling it "
        "by default makes the endpoint answer about the previous request"
    )


def test_the_cli_offers_an_explicit_opt_in() -> None:
    from oai2.server.openai_compat_app import main

    source = inspect.getsource(main)
    assert '"--prefix-cache"' in source


class _Layer:
    """Stand-in for mlx_lm's KVCache: state carries (keys, values, offset)."""

    def __init__(self, offset: int, trimmable: bool = True) -> None:
        self.state = (["k"], ["v"], offset)
        self._trimmable = trimmable
        self.trimmed = 0

    def is_trimmable(self) -> bool:
        return self._trimmable

    def trim(self, n: int) -> None:
        self.trimmed += n
        self.state = (self.state[0], self.state[1], max(0, self.state[2] - n))


def test_clear_layers_empties_every_layer() -> None:
    """A miss must not inherit the previous request's context."""
    layers = [_Layer(120), _Layer(7)]
    _clear_layers(layers)
    assert [layer.state[2] for layer in layers] == [0, 0]
    assert [layer.trimmed for layer in layers] == [120, 7]


def test_clear_layers_leaves_an_already_empty_layer_alone() -> None:
    layers = [_Layer(0)]
    _clear_layers(layers)
    assert layers[0].trimmed == 0


def test_clear_layers_tolerates_layers_without_state() -> None:
    _clear_layers([object()])


def test_clear_layers_ignores_a_malformed_offset() -> None:
    class Bad:
        state = (["k"], ["v"], "not-an-int")

        def trim(self, n: int) -> None:  # pragma: no cover - must not be reached
            raise AssertionError("trim must not be called for a malformed offset")

    _clear_layers([Bad()])


@pytest.mark.parametrize("offset", [0, 1, 4096])
def test_clear_layers_is_idempotent(offset: int) -> None:
    layer = _Layer(offset)
    _clear_layers([layer])
    first = layer.state[2]
    _clear_layers([layer])
    assert first == 0 and layer.state[2] == 0

"""Structural + validation contract for ``oai2.runtime.inference``.

Pin the outer guard-rail contract of the inference runtime scaffold —
the public request/response shape, the abstract ``InferenceRuntime``
base, the deterministic ``PlaceholderRuntime`` used by tests / offline
mode, and the environment-driven ``select_runtime_from_env`` selector.

Sections:
  - module docstring + module-level imports
  - public names (6-name re-export through oai2.runtime.__init__)
  - InferenceRequest (Pydantic v2 BaseModel, extra="forbid")
  - InferenceResponse (stdlib @dataclass, slots=True, mutable list field)
  - InferenceRuntime abstract base
  - PlaceholderRuntime deterministic offline runtime
  - default_runtime() + select_runtime_from_env() factory functions
  - estimate_tokens() + stop_sequences() helper functions
  - public-safety + structural counts
"""

from __future__ import annotations

import ast
import inspect
import re
import unittest.mock as mock

import pytest
from pydantic import ValidationError

from oai2.core import Status
from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
    default_runtime,
    select_runtime_from_env,
)
from oai2.runtime import inference as inference_module
from oai2.runtime.model import ModelSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_spec() -> ModelSpec:
    return ModelSpec(name="placeholder-spec")


# ---------------------------------------------------------------------------
# 1. Module docstring + module-level imports
# ---------------------------------------------------------------------------


class TestModuleDocstringAndImports:
    def test_module_docstring_mentions_runtime(self) -> None:
        doc = inference_module.__doc__
        assert isinstance(doc, str)
        lower = doc.lower()
        assert "inference" in lower
        assert "runtime" in lower

    def test_module_docstring_mentions_request_response_and_base(self) -> None:
        lower = inference_module.__doc__.lower()
        assert "request" in lower
        assert "response" in lower
        assert "abstract" in lower

    def test_module_docstring_mentions_status_proposed(self) -> None:
        lower = inference_module.__doc__.lower()
        assert "proposed" in lower

    def test_module_has_future_annotations(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            head = f.read(1024)
        assert "from __future__ import annotations" in head

    def test_module_imports_stdlib_time(self) -> None:
        assert hasattr(inference_module, "time")

    def test_module_imports_abc_ABC_and_abstractmethod(self) -> None:
        assert hasattr(inference_module, "ABC")
        assert hasattr(inference_module, "abstractmethod")

    def test_module_imports_collections_abc_sequence(self) -> None:
        assert hasattr(inference_module, "Sequence")

    def test_module_imports_dataclass_and_field(self) -> None:
        assert hasattr(inference_module, "dataclass")
        assert hasattr(inference_module, "field")

    def test_module_imports_pydantic_basemodel_configdict_field(self) -> None:
        assert hasattr(inference_module, "BaseModel")
        assert hasattr(inference_module, "ConfigDict")
        assert hasattr(inference_module, "Field")

    def test_module_imports_typing_any(self) -> None:
        assert hasattr(inference_module, "Any")

    def test_module_relative_imports_core_status(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "from ..core import Status" in src

    def test_module_relative_imports_runtime_model(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "from .model import" in src

    def test_module_has_no_absolute_import_oai2(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert not re.search(r"^import oai2\b", src, re.MULTILINE)
        assert not re.search(r"^from oai2\b", src, re.MULTILINE)

    def test_module_has_no_wildcard_imports(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "import *" not in src

    def test_module_does_not_import_mlx_at_module_scope(self) -> None:
        # MLX is defer-imported only inside model.discover_default_device()
        # and select_runtime_from_env() (via local-import of gateway_runtime)
        # — NOT at module top
        assert not hasattr(inference_module, "mlx")
        assert not hasattr(inference_module, "mx")


# ---------------------------------------------------------------------------
# 2. Public names (6-name re-export through oai2.runtime.__init__)
# ---------------------------------------------------------------------------


class TestPublicNames:
    def test_top_level_import_resolves_same_object_as_module_attr(self) -> None:
        assert InferenceRequest is inference_module.InferenceRequest
        assert InferenceResponse is inference_module.InferenceResponse
        assert InferenceRuntime is inference_module.InferenceRuntime
        assert PlaceholderRuntime is inference_module.PlaceholderRuntime
        assert default_runtime is inference_module.default_runtime
        assert select_runtime_from_env is inference_module.select_runtime_from_env

    def test_oai2_runtime_package_reexports_all_six(self) -> None:
        import oai2.runtime

        for name in (
            "InferenceRequest",
            "InferenceResponse",
            "InferenceRuntime",
            "PlaceholderRuntime",
            "default_runtime",
            "select_runtime_from_env",
        ):
            assert name in oai2.runtime.__all__

    def test_oai2_runtime_package_identity_equal_resolve(self) -> None:
        import oai2.runtime

        assert oai2.runtime.InferenceRequest is inference_module.InferenceRequest
        assert oai2.runtime.InferenceResponse is inference_module.InferenceResponse
        assert oai2.runtime.InferenceRuntime is inference_module.InferenceRuntime
        assert oai2.runtime.PlaceholderRuntime is inference_module.PlaceholderRuntime
        assert oai2.runtime.default_runtime is inference_module.default_runtime
        assert oai2.runtime.select_runtime_from_env is inference_module.select_runtime_from_env

    def test_direct_module_import_matches_top_level(self) -> None:
        from oai2.runtime.inference import (
            InferenceRequest as DirectReq,
        )
        from oai2.runtime.inference import (
            InferenceResponse as DirectRes,
        )
        from oai2.runtime.inference import (
            InferenceRuntime as DirectRt,
        )
        from oai2.runtime.inference import (
            PlaceholderRuntime as DirectPh,
        )
        from oai2.runtime.inference import (
            default_runtime as DirectDefault,
        )
        from oai2.runtime.inference import (
            select_runtime_from_env as DirectSelect,
        )

        assert DirectReq is InferenceRequest
        assert DirectRes is InferenceResponse
        assert DirectRt is InferenceRuntime
        assert DirectPh is PlaceholderRuntime
        assert DirectDefault is default_runtime
        assert DirectSelect is select_runtime_from_env

    def test_helper_functions_importable_directly_from_module(self) -> None:
        # estimate_tokens and stop_sequences are helpers; they are NOT
        # re-exported through oai2.runtime but ARE available from the
        # inference module itself
        from oai2.runtime.inference import (
            estimate_tokens,
            stop_sequences,
        )

        assert callable(estimate_tokens)
        assert callable(stop_sequences)


# ---------------------------------------------------------------------------
# 3. InferenceRequest
# ---------------------------------------------------------------------------


class TestInferenceRequest:
    def test_subclasses_basemodel(self) -> None:
        from pydantic import BaseModel

        assert issubclass(InferenceRequest, BaseModel)

    def test_model_config_extra_forbid(self) -> None:
        assert InferenceRequest.model_config["extra"] == "forbid"

    def test_unknown_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", not_a_real_field="y")  # type: ignore[call-arg]

    def test_field_set_pinned_to_nine_names(self) -> None:
        assert set(InferenceRequest.model_fields.keys()) == {
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "model",
            "speculative",
            "messages",
            "tools",
            "tool_choice",
        }

    def test_prompt_is_mandatory(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest()  # type: ignore[call-arg]

    def test_prompt_min_length_one(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="")

    def test_prompt_accepts_nonempty_string(self) -> None:
        req = InferenceRequest(prompt="hello")
        assert req.prompt == "hello"

    def test_max_tokens_default_is_256(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.max_tokens == 256

    def test_max_tokens_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", max_tokens=0)

    def test_max_tokens_le_32768(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", max_tokens=32_769)

    def test_max_tokens_accepts_in_range_values(self) -> None:
        for n in (1, 100, 32_768):
            req = InferenceRequest(prompt="x", max_tokens=n)
            assert req.max_tokens == n

    def test_temperature_default_is_0_7(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.temperature == 0.7

    def test_temperature_ge_0(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", temperature=-0.01)

    def test_temperature_le_2(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", temperature=2.01)

    def test_temperature_accepts_boundary_values(self) -> None:
        for t in (0.0, 0.7, 2.0):
            req = InferenceRequest(prompt="x", temperature=t)
            assert req.temperature == t

    def test_top_p_default_is_0_95(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.top_p == 0.95

    def test_top_p_ge_0(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", top_p=-0.01)

    def test_top_p_le_1(self) -> None:
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x", top_p=1.01)

    def test_top_p_accepts_boundary_values(self) -> None:
        for p in (0.0, 0.95, 1.0):
            req = InferenceRequest(prompt="x", top_p=p)
            assert req.top_p == p

    def test_stop_default_is_empty_tuple(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.stop == ()

    def test_stop_accepts_tuple(self) -> None:
        req = InferenceRequest(prompt="x", stop=("a", "b"))
        assert req.stop == ("a", "b")

    def test_stop_accepts_empty_tuple(self) -> None:
        req = InferenceRequest(prompt="x", stop=())
        assert req.stop == ()

    def test_seed_default_is_none(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.seed is None

    def test_seed_accepts_int(self) -> None:
        req = InferenceRequest(prompt="x", seed=42)
        assert req.seed == 42

    def test_seed_accepts_none(self) -> None:
        req = InferenceRequest(prompt="x", seed=None)
        assert req.seed is None

    def test_model_default_is_none(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.model is None

    def test_model_accepts_ModelSpec(self) -> None:
        req = InferenceRequest(prompt="x", model=_model_spec())
        assert req.model is not None
        assert req.model.name == "placeholder-spec"

    def test_speculative_default_is_false(self) -> None:
        req = InferenceRequest(prompt="x")
        assert req.speculative is False

    def test_speculative_accepts_true(self) -> None:
        req = InferenceRequest(prompt="x", speculative=True)
        assert req.speculative is True

    def test_speculative_accepts_false(self) -> None:
        req = InferenceRequest(prompt="x", speculative=False)
        assert req.speculative is False

    def test_model_dump_round_trips(self) -> None:
        spec = _model_spec()
        req = InferenceRequest(
            prompt="hello",
            max_tokens=128,
            temperature=0.5,
            top_p=0.9,
            stop=("a", "b"),
            seed=7,
            model=spec,
            speculative=True,
        )
        dumped = req.model_dump()
        assert dumped["prompt"] == "hello"
        assert dumped["max_tokens"] == 128
        assert dumped["temperature"] == 0.5
        assert dumped["top_p"] == 0.9
        assert dumped["stop"] == ("a", "b")
        assert dumped["seed"] == 7
        assert dumped["speculative"] is True


# ---------------------------------------------------------------------------
# 4. InferenceResponse (mutable @dataclass)
# ---------------------------------------------------------------------------


class TestInferenceResponse:
    def test_is_a_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(InferenceResponse)

    def test_is_slotted(self) -> None:

        params = getattr(InferenceResponse, "__dataclass_params__", None)
        assert params is not None
        assert params.slots is True

    def test_is_NOT_frozen(self) -> None:
        # notes is a list — the dataclass must allow post-init mutation

        params = getattr(InferenceResponse, "__dataclass_params__", None)
        assert params is not None
        assert params.frozen is False

    def test_constructs_with_all_required_fields(self) -> None:
        resp = InferenceResponse(
            text="hi",
            tokens=1,
            elapsed_ms=1.5,
            device="cpu",
        )
        assert resp.text == "hi"
        assert resp.tokens == 1
        assert resp.elapsed_ms == 1.5
        assert resp.device == "cpu"

    def test_status_defaults_to_EXPERIMENTAL(self) -> None:
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        assert resp.status is Status.EXPERIMENTAL

    def test_notes_default_is_empty_list(self) -> None:
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        assert resp.notes == []

    def test_notes_can_be_appended_post_init(self) -> None:
        # the dataclass is NOT frozen — mutation is allowed
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        resp.notes.append("hello")
        assert resp.notes == ["hello"]

    def test_finish_reason_default_is_none(self) -> None:
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        assert resp.finish_reason is None

    def test_tool_calls_default_is_empty_tuple(self) -> None:
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        assert resp.tool_calls == ()

    def test_field_set_pinned_to_nine_names(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(InferenceResponse)}
        assert field_names == {
            "text",
            "tokens",
            "elapsed_ms",
            "device",
            "status",
            "notes",
            "finish_reason",
            "tool_calls",
        }

    def test_no_dict_on_instance(self) -> None:
        resp = InferenceResponse(text="hi", tokens=1, elapsed_ms=1.5, device="cpu")
        assert not hasattr(resp, "__dict__")

    def test_status_accepts_other_status_values(self) -> None:
        resp = InferenceResponse(
            text="hi",
            tokens=1,
            elapsed_ms=1.5,
            device="cpu",
            status=Status.PROPOSED,
        )
        assert resp.status is Status.PROPOSED

    def test_tool_calls_accepts_tuple_of_dicts(self) -> None:
        calls = ({"name": "tool_a", "args": {}},)
        resp = InferenceResponse(
            text="hi",
            tokens=1,
            elapsed_ms=1.5,
            device="cpu",
            tool_calls=calls,
        )
        assert resp.tool_calls == calls

    def test_finish_reason_accepts_string(self) -> None:
        resp = InferenceResponse(
            text="hi",
            tokens=1,
            elapsed_ms=1.5,
            device="cpu",
            finish_reason="stop",
        )
        assert resp.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 5. InferenceRuntime abstract base
# ---------------------------------------------------------------------------


class TestInferenceRuntime:
    def test_is_a_class(self) -> None:
        assert inspect.isclass(InferenceRuntime)

    def test_subclasses_ABC(self) -> None:
        from abc import ABC

        assert issubclass(InferenceRuntime, ABC)

    def test_class_constant_STATUS_is_Status_PROPOSED(self) -> None:
        assert InferenceRuntime.STATUS is Status.PROPOSED

    def test_generate_is_abstractmethod(self) -> None:
        method = InferenceRuntime.__dict__["generate"]
        assert getattr(method, "__isabstractmethod__", False) is True

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            InferenceRuntime()  # type: ignore[abstract]

    def test_init_with_none_spec_uses_default(self) -> None:
        rt = PlaceholderRuntime(spec=None)
        # default ModelSpec(name="placeholder")
        assert rt.spec.name == "placeholder"

    def test_init_with_ModelSpec_stores_identity(self) -> None:
        spec = _model_spec()
        rt = PlaceholderRuntime(spec=spec)
        assert rt.spec is spec

    def test_device_is_a_property(self) -> None:
        assert isinstance(InferenceRuntime.__dict__["device"], property)

    def test_device_returns_string(self) -> None:
        rt = PlaceholderRuntime()
        d = rt.device
        assert isinstance(d, str)


# ---------------------------------------------------------------------------
# 6. PlaceholderRuntime
# ---------------------------------------------------------------------------


class TestPlaceholderRuntime:
    def test_subclasses_InferenceRuntime(self) -> None:
        assert issubclass(PlaceholderRuntime, InferenceRuntime)

    def test_class_constant_STATUS_is_Status_EXPERIMENTAL(self) -> None:
        assert PlaceholderRuntime.STATUS is Status.EXPERIMENTAL

    def test_generate_is_overridden(self) -> None:
        method = PlaceholderRuntime.__dict__["generate"]
        # NOT an abstractmethod on this class
        assert getattr(method, "__isabstractmethod__", False) is False

    def test_default_init_uses_placeholder_spec(self) -> None:
        rt = PlaceholderRuntime()
        assert rt.spec.name == "placeholder"

    def test_generate_returns_InferenceResponse(self) -> None:
        rt = PlaceholderRuntime()
        req = InferenceRequest(prompt="hello world")
        resp = rt.generate(req)
        assert isinstance(resp, InferenceResponse)

    def test_generate_text_includes_placeholder_prefix(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hello"))
        assert resp.text.startswith("[placeholder:")
        assert "] prompt_len=" in resp.text
        assert "prompt_len=" in resp.text

    def test_generate_text_includes_prompt_length(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hello world"))
        # 11 chars
        assert "prompt_len=11" in resp.text

    def test_generate_text_includes_device(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        # device substring is between [placeholder: and ] — must appear
        # as the brace-content of the prefix
        import re as _re

        m = _re.match(r"^\[placeholder:([^\]]+)\] prompt_len=", resp.text)
        assert m is not None

    def test_generate_tokens_is_positive_int(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hello"))
        assert isinstance(resp.tokens, int)
        assert resp.tokens >= 1

    def test_generate_elapsed_ms_is_float(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hello"))
        assert isinstance(resp.elapsed_ms, float)
        assert resp.elapsed_ms >= 0.0

    def test_generate_device_is_string(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        assert isinstance(resp.device, str)

    def test_generate_status_is_EXPERIMENTAL(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        assert resp.status is Status.EXPERIMENTAL

    def test_generate_notes_default_to_placeholder_marker(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        assert len(resp.notes) >= 1
        assert "placeholder" in resp.notes[0].lower()

    def test_generate_finish_reason_is_none_by_default(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        assert resp.finish_reason is None

    def test_generate_tool_calls_empty_by_default(self) -> None:
        rt = PlaceholderRuntime()
        resp = rt.generate(InferenceRequest(prompt="hi"))
        assert resp.tool_calls == ()

    def test_generate_does_not_load_any_model(self) -> None:
        # the placeholder runtime is pure, no MLX/weights — its response
        # is deterministic regardless of prompt content (modulo prompt
        # length)
        rt = PlaceholderRuntime()
        resp_a = rt.generate(InferenceRequest(prompt="abc"))
        resp_b = rt.generate(InferenceRequest(prompt="abc"))
        # token counts and elapsed_ms can differ by milliseconds; text
        # must be identical (modulo device string and token count)
        assert "[placeholder:" in resp_a.text
        assert "[placeholder:" in resp_b.text


# ---------------------------------------------------------------------------
# 7. default_runtime() + select_runtime_from_env()
# ---------------------------------------------------------------------------


class TestDefaultRuntime:
    def test_default_runtime_returns_PlaceholderRuntime_instance(self) -> None:
        rt = default_runtime()
        assert isinstance(rt, PlaceholderRuntime)

    def test_default_runtime_does_not_use_gateway(self) -> None:
        # regardless of env, default_runtime() is always PlaceholderRuntime
        with mock.patch.dict("os.environ", {"OAI2_GATEWAY_API_KEY": "anything"}):
            assert isinstance(default_runtime(), PlaceholderRuntime)

    def test_default_runtime_is_callable_multiple_times(self) -> None:
        a = default_runtime()
        b = default_runtime()
        assert isinstance(a, PlaceholderRuntime)
        assert isinstance(b, PlaceholderRuntime)


class TestSelectRuntimeFromEnv:
    def test_returns_PlaceholderRuntime_when_key_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            # ensure OAI2_GATEWAY_API_KEY is unset
            os_environ = mock.patch.dict(
                "os.environ",
                {},
                clear=False,
            )
            with os_environ:
                import os as _os

                _os.environ.pop("OAI2_GATEWAY_API_KEY", None)
                rt = select_runtime_from_env()
        assert isinstance(rt, PlaceholderRuntime)

    def test_returns_something_inference_runtime_instance(self) -> None:
        rt = select_runtime_from_env()
        assert isinstance(rt, InferenceRuntime)

    def test_does_not_raise_when_gateway_key_missing(self) -> None:
        # must never raise on missing config — the docstring is explicit
        with mock.patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("OAI2_GATEWAY_API_KEY", None)
            # should not raise
            select_runtime_from_env()

    def test_catches_GatewayConfigError(self) -> None:
        # simulate GatewayConfigError raised from GatewayRuntime.from_env()
        # — select_runtime must catch and return PlaceholderRuntime
        from oai2.runtime.gateway_runtime import GatewayConfigError

        # Patch from_env on the real class to raise
        with mock.patch(
            "oai2.runtime.gateway_runtime.GatewayRuntime.from_env",
            side_effect=GatewayConfigError("simulated"),
        ):
            rt = select_runtime_from_env()
        assert isinstance(rt, PlaceholderRuntime)

    def test_local_import_of_gateway_runtime(self) -> None:
        # select_runtime_from_env uses a local from_gateway_runtime — the
        # module-level inference file must defer-import gateway_runtime
        # inside the function, not at module top
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # check the gateway_runtime import is inside the select_runtime
        # function (locally), not at module scope
        # the local import is inside the function body
        assert "from .gateway_runtime import GatewayConfigError, GatewayRuntime" in src


# ---------------------------------------------------------------------------
# 8. estimate_tokens + stop_sequences
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_at_least_one(self) -> None:
        from oai2.runtime.inference import estimate_tokens

        assert estimate_tokens("") == 1

    def test_single_word_returns_one(self) -> None:
        from oai2.runtime.inference import estimate_tokens

        assert estimate_tokens("hello") == 1

    def test_multi_word_returns_word_count(self) -> None:
        from oai2.runtime.inference import estimate_tokens

        assert estimate_tokens("hello world") == 2

    def test_returns_int(self) -> None:
        from oai2.runtime.inference import estimate_tokens

        assert isinstance(estimate_tokens("a b c"), int)

    def test_returns_at_least_one_for_any_input(self) -> None:
        from oai2.runtime.inference import estimate_tokens

        # empty input still produces >=1 (placeholder floor)
        for s in ("", " ", "a", "a b c d"):
            n = estimate_tokens(s)
            assert n >= 1


class TestStopSequences:
    def test_returns_Sequence(self) -> None:
        from collections.abc import Sequence

        from oai2.runtime.inference import stop_sequences

        assert isinstance(stop_sequences(), Sequence)

    def test_returns_three_strings(self) -> None:
        from oai2.runtime.inference import stop_sequences

        seqs = stop_sequences()
        assert len(seqs) == 3
        for s in seqs:
            assert isinstance(s, str)

    def test_pinned_three_sequences(self) -> None:
        from oai2.runtime.inference import stop_sequences

        assert stop_sequences() == ("\n\n<", "<|end|>", "</s>")

    def test_returns_tuple(self) -> None:
        from oai2.runtime.inference import stop_sequences

        seqs = stop_sequences()
        assert isinstance(seqs, tuple)

    def test_distinct_sequences(self) -> None:
        from oai2.runtime.inference import stop_sequences

        seqs = stop_sequences()
        assert len(set(seqs)) == len(seqs)


# ---------------------------------------------------------------------------
# 9. Public-safety + structural counts
# ---------------------------------------------------------------------------


class TestPublicSafetyAndStructuralCounts:
    def test_no_cloud_runtime_imports(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        for banned in (
            "boto3",
            "azure",
            "google.cloud",
            "kubernetes",
            "docker",
            "fabric",
        ):
            assert banned not in src

    def test_no_hardcoded_credentials(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert 'api_key="sk-' not in src
        assert "BEGIN PRIVATE KEY" not in src
        assert "AKIA" not in src
        assert "AIza" not in src
        assert re.search(r"\bxox[abprs]-", src) is None

    def test_no_print_or_pprint(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "print(" not in src
        assert "pprint(" not in src

    def test_no_subprocess_or_shell(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "subprocess" not in src
        assert "shell=True" not in src
        assert "os.system" not in src

    def test_no_eval_or_exec(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "eval(" not in src
        assert "exec(" not in src
        assert "compile(" not in src

    def test_no_os_environ_or_getenv(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "os.environ" not in src
        assert "os.getenv" not in src

    def test_no_todo_or_fixme_markers(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        assert "TODO" not in src
        assert "FIXME" not in src
        assert "XXX" not in src

    def test_source_ends_with_single_trailing_newline(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, "rb") as f:
            data = f.read()
        assert data.endswith(b"\n")
        assert not data.endswith(b"\n\n")

    def test_pins_exactly_one_basemodel_subclass(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        matches = re.findall(r"^class\s+(\w+)\s*\(\s*BaseModel\s*\)\s*:", src, re.MULTILINE)
        assert matches == ["InferenceRequest"]

    def test_pins_exactly_one_dataclass_decorator(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # match the full decorator including its (multi-line) arguments
        decorators = re.findall(r"@dataclass\([^)]*\)", src, re.DOTALL)
        assert len(decorators) == 1
        assert "slots=True" in decorators[0]
        assert "frozen=True" not in decorators[0]

    def test_pins_exactly_one_abstract_class(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly one `class Foo(ABC):` block
        matches = re.findall(r"^class\s+(\w+)\s*\(\s*ABC\s*\)\s*:", src, re.MULTILINE)
        assert matches == ["InferenceRuntime"]

    def test_pins_exactly_one_abstractmethod(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        # exactly one `@abstractmethod` decorator at module scope
        abstracts = re.findall(r"@abstractmethod", src)
        assert len(abstracts) == 1

    def test_class_set_pinned(self) -> None:
        # 3 classes total — InferenceRequest (BaseModel),
        # InferenceRuntime (ABC), PlaceholderRuntime (InferenceRuntime)
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        classes = re.findall(r"^class\s+(\w+)[\(:]", src, re.MULTILINE)
        assert set(classes) == {
            "InferenceRequest",
            "InferenceResponse",
            "InferenceRuntime",
            "PlaceholderRuntime",
        }
        assert len(classes) == 4

    def test_function_defs_pinned(self) -> None:
        # 4 module-level functions: default_runtime, select_runtime_from_env,
        # estimate_tokens, stop_sequences
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            src = f.read()
        funcs = re.findall(r"^def\s+(\w+)\(", src, re.MULTILINE)
        assert funcs == [
            "default_runtime",
            "select_runtime_from_env",
            "estimate_tokens",
            "stop_sequences",
        ]

    def test_no_unexpected_top_level_statement_kinds(self) -> None:
        source_path = inspect.getsourcefile(inference_module)
        assert source_path is not None
        with open(source_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        allowed = {
            ast.Module,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.Expr,
            ast.If,
            ast.Try,
            ast.With,
            ast.Pass,
        }
        for node in tree.body:
            assert type(node) in allowed, f"unexpected: {type(node).__name__}"

    def test_status_constant_pins_default_value(self) -> None:
        # InferenceResponse defaults status to EXPERIMENTAL, not PROPOSED —
        # this is the structural pin (placeholder returns observed runtime
        # results, not proposed)
        resp = InferenceResponse(text="x", tokens=1, elapsed_ms=1.0, device="cpu")
        assert resp.status is Status.EXPERIMENTAL

    def test_class_STATUS_CSTATUS_constants_pinned(self) -> None:
        # InferenceRuntime.STATUS is PROPOSED, PlaceholderRuntime.STATUS
        # is EXPERIMENTAL — distinct values
        assert InferenceRuntime.STATUS is Status.PROPOSED
        assert PlaceholderRuntime.STATUS is Status.EXPERIMENTAL
        assert InferenceRuntime.STATUS is not PlaceholderRuntime.STATUS

# -*- coding: utf-8 -*-

"""
Unit tests for native reasoning effort (``additionalModelRequestFields``).

Covers:
- Schema parsing (Kiro IDE parity) and live-vs-static capability lookup
- Effort mapping onto each model's allowed enum
- ``-<effort>`` model-name suffixes
- OpenAI / Anthropic / Responses payload integration (fake reasoning skipped)
- Native ``reasoningContentEvent`` parsing into thinking events
"""

from unittest.mock import patch

import pytest

from kiro import native_effort
from kiro.converters_anthropic import anthropic_to_kiro
from kiro.converters_openai import build_kiro_payload
from kiro.converters_responses import build_kiro_payload_from_responses
from kiro.models_anthropic import AnthropicMessage, AnthropicMessagesRequest
from kiro.models_openai import ChatCompletionRequest, ChatMessage
from kiro.models_responses import ResponsesRequest
from kiro.native_effort import (
    NativeReasoningCapability,
    build_additional_model_request_fields as build_fields,
    capability_from_schema,
    get_native_capability,
    register_model_schemas,
    split_effort_suffix,
)
from kiro.parsers import AwsEventStreamParser
from kiro.streaming_core import _process_chunk


# Real schemas returned by ListAvailableModels (2026-10-22).
OPUS_5_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adaptive", "disabled"]},
                "display": {"type": "string", "enum": ["summarized", "omitted"]},
            },
            "required": ["type"],
        },
        "output_config": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                }
            },
        },
        "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 128000},
    },
    "additionalProperties": False,
}
GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                }
            },
        }
    },
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def isolated_registry():
    """Keep live schema registrations from leaking between tests."""
    native_effort.clear_live_capabilities()
    yield
    native_effort.clear_live_capabilities()


@pytest.fixture
def fake_reasoning_on(monkeypatch):
    """Force fake reasoning on so we can assert it is skipped for native requests."""
    monkeypatch.setattr("kiro.converters_core.FAKE_REASONING_ENABLED", True)


def _user_content(payload):
    return payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]


def _model_id(payload):
    return payload["conversationState"]["currentMessage"]["userInputMessage"]["modelId"]


# ==================================================================================================
# Schema parsing / capability lookup
# ==================================================================================================

class TestCapabilityFromSchema:
    def test_parses_claude_output_config_and_thinking(self):
        cap = capability_from_schema(OPUS_5_SCHEMA)
        assert cap.effort_path == "output_config"
        assert cap.effort_levels == ("low", "medium", "high", "xhigh", "max")
        assert cap.default_effort == "high"
        assert cap.thinking_types == ("adaptive", "disabled")
        assert cap.thinking_displays == ("summarized", "omitted")

    def test_parses_gpt_reasoning_effort(self):
        cap = capability_from_schema(GPT_SCHEMA)
        assert cap.effort_path == "reasoning"
        assert "none" in cap.effort_levels
        assert cap.thinking_types == ()

    @pytest.mark.parametrize("schema", [None, {}, {"properties": {"max_tokens": {"type": "integer"}}}, "x"])
    def test_schema_without_reasoning_controls(self, schema):
        assert capability_from_schema(schema) is None


class TestCapabilityLookup:
    def test_static_snapshot_used_without_live_data(self):
        assert get_native_capability("claude-opus-5").effort_path == "output_config"
        assert get_native_capability("claude-haiku-4.5") is None

    def test_live_schema_overrides_static(self):
        register_model_schemas([
            {"modelId": "claude-opus-5"},  # listed live without schema
            {"modelId": "brand-new-model", "additionalModelRequestFieldsSchema": GPT_SCHEMA},
        ])
        assert get_native_capability("claude-opus-5") is None
        assert get_native_capability("brand-new-model").effort_path == "reasoning"

    def test_register_ignores_malformed_entries(self):
        register_model_schemas([None, {"modelId": ""}, {"noId": 1}])
        assert get_native_capability("claude-opus-5") is not None


# ==================================================================================================
# Effort mapping
# ==================================================================================================

class TestBuildAdditionalModelRequestFields:
    def test_no_preference_sends_nothing(self):
        assert build_fields("claude-opus-5") is None

    def test_unsupported_model_sends_nothing(self):
        assert build_fields("claude-haiku-4.5", effort="high") is None
        assert build_fields("unknown-model", effort="high") is None

    def test_claude_effort_enables_adaptive_thinking(self):
        assert build_fields("claude-opus-5", effort="xhigh") == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        }

    def test_gpt_effort_uses_reasoning_path(self):
        assert build_fields("gpt-5.6-luna", effort="low") == {"reasoning": {"effort": "low"}}

    def test_effort_is_case_insensitive(self):
        assert build_fields("gpt-5.6-luna", effort="HIGH") == {"reasoning": {"effort": "high"}}

    def test_unknown_effort_string_is_ignored(self):
        assert build_fields("gpt-5.6-luna", effort="turbo") is None

    @pytest.mark.parametrize(
        "model,effort,expected",
        [
            ("claude-opus-4.6", "xhigh", "high"),  # 4.6 has no xhigh; tie -> cheaper
            ("claude-opus-5", "minimal", "low"),
            ("kimi-k3", "medium", "low"),
            ("kimi-k3", "xhigh", "high"),  # tie between high/max -> cheaper
            ("kimi-k3", "max", "max"),
        ],
    )
    def test_maps_to_nearest_allowed_level(self, model, effort, expected):
        fields = build_fields(model, effort=effort)
        path = get_native_capability(model).effort_path
        assert fields[path] == {"effort": expected}

    def test_none_disables_claude_thinking(self):
        assert build_fields("claude-sonnet-4.6", effort="none") == {"thinking": {"type": "disabled"}}

    def test_none_maps_to_gpt_none(self):
        assert build_fields("gpt-5.6-terra", effort="none") == {"reasoning": {"effort": "none"}}

    def test_none_on_model_without_disable_uses_cheapest(self):
        # claude-opus-5.5 only allows thinking.type=adaptive
        assert build_fields("claude-opus-5.5", effort="none") == {"output_config": {"effort": "low"}}

    def test_thinking_disabled_caps_thinking_only_levels(self):
        assert build_fields("claude-opus-5", effort="max", thinking="disabled") == {
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        }

    def test_thinking_enabled_without_effort(self):
        assert build_fields("claude-opus-5", thinking="enabled") == {"thinking": {"type": "adaptive"}}

    def test_display_passthrough_only_when_allowed(self):
        assert build_fields("claude-opus-5", thinking="enabled", thinking_display="summarized") == {
            "thinking": {"type": "adaptive", "display": "summarized"}
        }
        assert build_fields("claude-opus-5", thinking="enabled", thinking_display="bogus") == {
            "thinking": {"type": "adaptive"}
        }

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setattr("kiro.config.NATIVE_EFFORT_ENABLED", False)
        assert build_fields("claude-opus-5", effort="high") is None


class TestSplitEffortSuffix:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("kiro-o-5-high", ("kiro-o-5", "high")),
            ("gpt-5.6-luna-MAX", ("gpt-5.6-luna", "max")),
            ("claude-opus-4.8-xhigh", ("claude-opus-4.8", "xhigh")),
            ("claude-opus-5", ("claude-opus-5", None)),
            ("minimax-m2.5", ("minimax-m2.5", None)),
            ("high", ("high", None)),
        ],
    )
    def test_split(self, model, expected):
        assert split_effort_suffix(model) == expected

    def test_known_model_name_is_not_split(self):
        register_model_schemas([{"modelId": "future-model-max"}])
        assert split_effort_suffix("future-model-max") == ("future-model-max", None)


# ==================================================================================================
# Converter integration
# ==================================================================================================

class TestOpenAIIntegration:
    def test_reasoning_effort_sent_natively(self, fake_reasoning_on):
        request = ChatCompletionRequest(
            model="claude-opus-5",
            messages=[ChatMessage(role="user", content="Hi")],
            reasoning_effort="max",
        )
        payload = build_kiro_payload(request, "conv-1", "arn:test")
        assert payload["additionalModelRequestFields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        }
        assert "<thinking_mode>" not in _user_content(payload)

    def test_model_suffix_sets_effort_and_resolves_alias(self, fake_reasoning_on):
        request = ChatCompletionRequest(
            model="kiro-o-5-low",
            messages=[ChatMessage(role="user", content="Hi")],
            reasoning_effort="high",  # suffix wins
        )
        payload = build_kiro_payload(request, "conv-1", "arn:test")
        assert _model_id(payload) == "claude-opus-5"
        assert payload["additionalModelRequestFields"]["output_config"] == {"effort": "low"}

    def test_unsupported_model_keeps_fake_reasoning(self, fake_reasoning_on):
        request = ChatCompletionRequest(
            model="claude-haiku-4.5",
            messages=[ChatMessage(role="user", content="Hi")],
            reasoning_effort="high",
        )
        payload = build_kiro_payload(request, "conv-1", "arn:test")
        assert "additionalModelRequestFields" not in payload
        assert "<thinking_mode>enabled</thinking_mode>" in _user_content(payload)

    def test_no_effort_keeps_legacy_payload(self, fake_reasoning_on):
        request = ChatCompletionRequest(
            model="claude-opus-5", messages=[ChatMessage(role="user", content="Hi")]
        )
        payload = build_kiro_payload(request, "conv-1", "arn:test")
        assert "additionalModelRequestFields" not in payload
        assert "<thinking_mode>enabled</thinking_mode>" in _user_content(payload)


class TestAnthropicIntegration:
    def _request(self, **kwargs):
        return AnthropicMessagesRequest(
            model=kwargs.pop("model", "claude-opus-4.8"),
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=32000,
            **kwargs,
        )

    def test_claude_code_adaptive_xhigh(self, fake_reasoning_on):
        payload = anthropic_to_kiro(
            self._request(thinking={"type": "adaptive"}, output_config={"effort": "xhigh"}),
            "conv-1",
            "arn:test",
        )
        assert payload["additionalModelRequestFields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        }
        assert "<thinking_mode>" not in _user_content(payload)

    def test_budget_thinking_becomes_adaptive(self):
        payload = anthropic_to_kiro(
            self._request(thinking={"type": "enabled", "budget_tokens": 8000}), "conv-1", "arn:test"
        )
        assert payload["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}

    def test_thinking_disabled(self):
        payload = anthropic_to_kiro(
            self._request(thinking={"type": "disabled"}), "conv-1", "arn:test"
        )
        assert payload["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}

    def test_unsupported_model_keeps_legacy(self):
        payload = anthropic_to_kiro(
            self._request(model="claude-haiku-4.5", thinking={"type": "adaptive"},
                          output_config={"effort": "high"}),
            "conv-1",
            "arn:test",
        )
        assert "additionalModelRequestFields" not in payload


class TestResponsesIntegration:
    def test_reasoning_max_accepted_and_forwarded(self, fake_reasoning_on):
        request = ResponsesRequest(model="gpt-5.6-luna", input="Hi", reasoning={"effort": "max"})
        payload = build_kiro_payload_from_responses(request, "conv-1", "arn:test").payload
        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": "max"}}
        assert "<thinking_mode>" not in _user_content(payload)

    def test_model_suffix(self):
        request = ResponsesRequest(model="gpt-5.6-terra-none", input="Hi")
        payload = build_kiro_payload_from_responses(request, "conv-1", "arn:test").payload
        assert _model_id(payload) == "gpt-5.6-terra"
        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": "none"}}


# ==================================================================================================
# Native reasoning stream events
# ==================================================================================================

class TestReasoningContentEvents:
    def test_parser_emits_reasoning_and_ignores_signature(self):
        parser = AwsEventStreamParser()
        events = parser.feed(
            b'{"text":"Let me "}{"text":"Let me "}{"signature":"EpASCnw"}{"content":"391"}'
        )
        assert events == [
            {"type": "reasoning", "data": "Let me "},
            {"type": "reasoning", "data": "Let me "},  # not deduplicated
            {"type": "content", "data": "391"},
        ]

    @pytest.mark.asyncio
    async def test_reasoning_becomes_thinking_event(self):
        parser = AwsEventStreamParser()
        events = [e async for e in _process_chunk(parser, b'{"text":"hmm"}{"content":"ok"}', None)]
        assert [(e.type, e.thinking_content or e.content) for e in events] == [
            ("thinking", "hmm"),
            ("content", "ok"),
        ]

    @pytest.mark.asyncio
    async def test_remove_mode_drops_native_reasoning(self):
        parser = AwsEventStreamParser()
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            events = [e async for e in _process_chunk(parser, b'{"text":"hmm"}{"content":"ok"}', None)]
        assert [e.type for e in events] == ["content"]

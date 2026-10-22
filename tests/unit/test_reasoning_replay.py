# -*- coding: utf-8 -*-

"""
Unit tests for native reasoning signature round-trip (``kiro.reasoning_replay``).

Covers:
- Placeholder vs real signatures (placeholders must never be replayed)
- Responses ``encrypted_content`` carrier encode/decode, including hostile input
- Parser / streaming core surfacing of the upstream signature
- Anthropic: real signature emitted (stream + non-stream) and replayed into history
- Responses: signature carried in encrypted_content and replayed into history
- History replay gating (models without native schema, NATIVE_EFFORT=false)
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro import native_effort
from kiro.converters_anthropic import anthropic_to_kiro, extract_reasoning_from_anthropic_content
from kiro.converters_core import UnifiedMessage, build_kiro_history, merge_adjacent_messages
from kiro.converters_responses import build_kiro_payload_from_responses, convert_responses_input_to_unified
from kiro.models_anthropic import AnthropicMessage, AnthropicMessagesRequest
from kiro.models_responses import ResponsesRequest
from kiro.parsers import AwsEventStreamParser
from kiro.reasoning_replay import (
    RESPONSES_ENCRYPTED_PREFIX,
    decode_responses_encrypted_content,
    encode_responses_encrypted_content,
    generate_placeholder_signature,
    is_placeholder_signature,
    make_reasoning,
    to_history_reasoning_content,
)
from kiro.streaming_anthropic import collect_anthropic_response, stream_kiro_to_anthropic
from kiro.streaming_core import KiroEvent, StreamResult, _process_chunk
from kiro.streaming_responses import collect_stream_response

REAL_SIG = "EukHCnwIExABGAIqQAXmHXmncFUnUH5SFqLFxK+kvLDBTFp2tW23j+LHsh9"


@pytest.fixture(autouse=True)
def isolated_registry():
    native_effort.clear_live_capabilities()
    yield
    native_effort.clear_live_capabilities()


def _history(payload):
    return payload["conversationState"].get("history", [])


def _assistant_entries(payload):
    return [h["assistantResponseMessage"] for h in _history(payload) if "assistantResponseMessage" in h]


def _sse(chunks):
    events = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# ==================================================================================================
# Signature primitives
# ==================================================================================================

class TestSignaturePrimitives:
    def test_placeholder_round_trip(self):
        assert is_placeholder_signature(generate_placeholder_signature())

    @pytest.mark.parametrize("value", [REAL_SIG, "sig_short", "sig_" + "z" * 32, "", None, 123, ["sig_"]])
    def test_non_placeholders(self, value):
        assert not is_placeholder_signature(value)

    def test_make_reasoning_requires_real_signature(self):
        assert make_reasoning("t", REAL_SIG) == {"text": "t", "signature": REAL_SIG}
        assert make_reasoning("t", generate_placeholder_signature()) is None
        assert make_reasoning("t", "") is None
        assert make_reasoning("t", "   ") is None
        assert make_reasoning("t", None) is None
        assert make_reasoning("t", 42) is None

    def test_make_reasoning_tolerates_missing_text(self):
        # Kiro accepts an empty reasoning text with a valid signature.
        assert make_reasoning(None, REAL_SIG) == {"text": "", "signature": REAL_SIG}

    def test_history_shape_matches_kiro_ide(self):
        assert to_history_reasoning_content({"text": "a", "signature": "b"}) == {
            "reasoningText": {"text": "a", "signature": "b"}
        }


class TestResponsesCarrier:
    def test_round_trip_preserves_unicode(self):
        encoded = encode_responses_encrypted_content("思考 ✓", REAL_SIG)
        assert encoded.startswith(RESPONSES_ENCRYPTED_PREFIX)
        assert decode_responses_encrypted_content(encoded) == {"text": "思考 ✓", "signature": REAL_SIG}

    def test_no_carrier_without_real_signature(self):
        assert encode_responses_encrypted_content("t", None) is None
        assert encode_responses_encrypted_content("t", generate_placeholder_signature()) is None

    @pytest.mark.parametrize(
        "value",
        [
            None,
            123,
            "",
            "gAAAAABreal-openai-blob",  # foreign carrier
            "plain compaction summary",
            RESPONSES_ENCRYPTED_PREFIX,  # empty payload
            RESPONSES_ENCRYPTED_PREFIX + "!!!not-base64!!!",
            RESPONSES_ENCRYPTED_PREFIX + base64.b64encode(b"not json").decode(),
            RESPONSES_ENCRYPTED_PREFIX + base64.b64encode(b"[1,2]").decode(),
            RESPONSES_ENCRYPTED_PREFIX + base64.b64encode(b'{"t":"x"}').decode(),  # no signature
            RESPONSES_ENCRYPTED_PREFIX + base64.b64encode(b"\xff\xfe").decode(),  # bad utf-8
            RESPONSES_ENCRYPTED_PREFIX
            + base64.b64encode(json.dumps({"t": "x", "s": generate_placeholder_signature()}).encode()).decode(),
        ],
    )
    def test_decode_ignores_invalid_values(self, value):
        assert decode_responses_encrypted_content(value) is None

    def test_decode_rejects_oversized_input(self):
        huge = RESPONSES_ENCRYPTED_PREFIX + "A" * (5 * 1024 * 1024)
        assert decode_responses_encrypted_content(huge) is None


# ==================================================================================================
# Parser / streaming core
# ==================================================================================================

class TestSignatureEvents:
    def test_parser_ignores_empty_or_non_string_signature(self):
        events = AwsEventStreamParser().feed(b'{"signature":""}{"signature":5}{"content":"x"}')
        assert events == [{"type": "content", "data": "x"}]

    @pytest.mark.asyncio
    async def test_core_emits_signature_event(self):
        parser = AwsEventStreamParser()
        events = [e async for e in _process_chunk(parser, b'{"text":"a"}{"signature":"S1"}', None)]
        assert [(e.type, e.thinking_content or e.thinking_signature) for e in events] == [
            ("thinking", "a"),
            ("thinking_signature", "S1"),
        ]

    @pytest.mark.asyncio
    async def test_remove_mode_drops_signature(self):
        parser = AwsEventStreamParser()
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            events = [e async for e in _process_chunk(parser, b'{"text":"a"}{"signature":"S1"}', None)]
        assert events == []


# ==================================================================================================
# Anthropic output
# ==================================================================================================

class TestAnthropicStreamingSignature:
    async def _run(self, kiro_events):
        async def fake_parse(*args, **kwargs):
            for event in kiro_events:
                yield event

        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        response = AsyncMock()
        with patch("kiro.streaming_anthropic.parse_kiro_stream", fake_parse):
            chunks = [c async for c in stream_kiro_to_anthropic(response, "m", cache, MagicMock())]
        return _sse(chunks)

    @pytest.mark.asyncio
    async def test_signature_delta_before_block_stop(self):
        events = await self._run([
            KiroEvent(type="thinking", thinking_content="hmm"),
            KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG),
            KiroEvent(type="content", content="391"),
            KiroEvent(type="context_usage", context_usage_percentage=1.0),
        ])
        thinking_index = next(
            e["index"] for e in events
            if e["type"] == "content_block_start" and e["content_block"]["type"] == "thinking"
        )
        block_events = [e for e in events if e.get("index") == thinking_index]
        kinds = [(e["type"], e.get("delta", {}).get("type")) for e in block_events]
        assert kinds == [
            ("content_block_start", None),
            ("content_block_delta", "thinking_delta"),
            ("content_block_delta", "signature_delta"),
            ("content_block_stop", None),
        ]
        assert block_events[2]["delta"]["signature"] == REAL_SIG

    @pytest.mark.asyncio
    async def test_signature_emitted_when_tool_call_closes_thinking(self):
        events = await self._run([
            KiroEvent(type="thinking", thinking_content="plan"),
            KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG),
            KiroEvent(type="tool_start", tool_use={"id": "t1", "function": {"name": "f", "arguments": ""}}),
            KiroEvent(type="tool_stop", tool_use={"id": "t1", "function": {"name": "f", "arguments": "{}"}}),
            KiroEvent(type="context_usage", context_usage_percentage=1.0),
        ])
        signature_deltas = [
            e for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "signature_delta"
        ]
        assert [d["delta"]["signature"] for d in signature_deltas] == [REAL_SIG]

    @pytest.mark.asyncio
    async def test_signature_emitted_when_stream_ends_in_thinking(self):
        events = await self._run([
            KiroEvent(type="thinking", thinking_content="only thinking"),
            KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG),
        ])
        assert any(
            e["type"] == "content_block_delta" and e["delta"].get("signature") == REAL_SIG for e in events
        )

    @pytest.mark.asyncio
    async def test_fake_reasoning_has_no_signature_delta(self):
        events = await self._run([
            KiroEvent(type="thinking", thinking_content="fake"),
            KiroEvent(type="content", content="x"),
        ])
        assert not any(
            e["type"] == "content_block_delta" and e["delta"]["type"] == "signature_delta" for e in events
        )

    @pytest.mark.asyncio
    async def test_signature_without_open_block_is_ignored(self):
        events = await self._run([
            KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG),
            KiroEvent(type="content", content="x"),
        ])
        assert not any(e.get("delta", {}).get("type") == "signature_delta" for e in events)

    @pytest.mark.asyncio
    async def test_signature_delta_sent_once(self):
        events = await self._run([
            KiroEvent(type="thinking", thinking_content="a"),
            KiroEvent(type="thinking_signature", thinking_signature="S-old"),
            KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG),
            KiroEvent(type="content", content="x"),
        ])
        deltas = [e["delta"]["signature"] for e in events if e.get("delta", {}).get("type") == "signature_delta"]
        assert deltas == [REAL_SIG]


class TestAnthropicNonStreamingSignature:
    @pytest.mark.asyncio
    async def test_real_signature_used(self):
        result = StreamResult(content="391", thinking_content="hmm", thinking_signature=REAL_SIG,
                              context_usage_percentage=1.0)
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        with patch("kiro.streaming_anthropic.collect_stream_to_result", return_value=result):
            response = await collect_anthropic_response(AsyncMock(), "m", cache, MagicMock())
        assert response["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": REAL_SIG}

    @pytest.mark.asyncio
    async def test_placeholder_for_fake_reasoning(self):
        result = StreamResult(content="x", thinking_content="fake", context_usage_percentage=1.0)
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        with patch("kiro.streaming_anthropic.collect_stream_to_result", return_value=result):
            response = await collect_anthropic_response(AsyncMock(), "m", cache, MagicMock())
        assert is_placeholder_signature(response["content"][0]["signature"])


# ==================================================================================================
# Anthropic input → history replay
# ==================================================================================================

class TestAnthropicReplay:
    def _request(self, assistant_content, model="claude-sonnet-4.6"):
        return AnthropicMessagesRequest(
            model=model,
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Q1"),
                AnthropicMessage(role="assistant", content=assistant_content),
                AnthropicMessage(role="user", content="Q2"),
            ],
        )

    def test_real_signature_replayed(self):
        payload = anthropic_to_kiro(self._request([
            {"type": "thinking", "thinking": "reason", "signature": REAL_SIG},
            {"type": "text", "text": "A1"},
        ]), "c", "arn")
        (assistant,) = _assistant_entries(payload)
        assert assistant["content"] == "A1"
        assert assistant["reasoningContent"] == {"reasoningText": {"text": "reason", "signature": REAL_SIG}}

    def test_placeholder_signature_not_replayed(self):
        payload = anthropic_to_kiro(self._request([
            {"type": "thinking", "thinking": "fake", "signature": generate_placeholder_signature()},
            {"type": "text", "text": "A1"},
        ]), "c", "arn")
        assert "reasoningContent" not in _assistant_entries(payload)[0]

    def test_thinking_text_never_leaks_into_content(self):
        payload = anthropic_to_kiro(self._request([
            {"type": "thinking", "thinking": "SECRET-REASONING", "signature": REAL_SIG},
            {"type": "text", "text": "A1"},
        ]), "c", "arn")
        assert "SECRET-REASONING" not in _assistant_entries(payload)[0]["content"]

    def test_redacted_thinking_not_replayed(self):
        payload = anthropic_to_kiro(self._request([
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "A1"},
        ]), "c", "arn")
        assert "reasoningContent" not in _assistant_entries(payload)[0]

    def test_not_replayed_for_models_without_native_schema(self):
        payload = anthropic_to_kiro(self._request([
            {"type": "thinking", "thinking": "r", "signature": REAL_SIG},
            {"type": "text", "text": "A1"},
        ], model="deepseek-3.2"), "c", "arn")
        assert "reasoningContent" not in _assistant_entries(payload)[0]

    def test_not_replayed_when_native_effort_disabled(self, monkeypatch):
        monkeypatch.setattr("kiro.config.NATIVE_EFFORT_ENABLED", False)
        payload = anthropic_to_kiro(self._request([
            {"type": "thinking", "thinking": "r", "signature": REAL_SIG},
            {"type": "text", "text": "A1"},
        ]), "c", "arn")
        assert "reasoningContent" not in _assistant_entries(payload)[0]

    def test_tool_loop_reasoning_kept_with_tool_use(self):
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4.6",
            max_tokens=1024,
            tools=[{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}],
            messages=[
                AnthropicMessage(role="user", content="Weather?"),
                AnthropicMessage(role="assistant", content=[
                    {"type": "thinking", "thinking": "need tool", "signature": REAL_SIG},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                ]),
                AnthropicMessage(role="user", content=[
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"},
                ]),
            ],
        )
        (assistant,) = _assistant_entries(anthropic_to_kiro(request, "c", "arn"))
        assert assistant["toolUses"][0]["toolUseId"] == "toolu_1"
        assert assistant["reasoningContent"]["reasoningText"]["signature"] == REAL_SIG

    def test_first_real_block_wins(self):
        reasoning = extract_reasoning_from_anthropic_content([
            {"type": "thinking", "thinking": "fake", "signature": generate_placeholder_signature()},
            {"type": "thinking", "thinking": "one", "signature": "S1"},
            {"type": "thinking", "thinking": "two", "signature": "S2"},
        ])
        assert reasoning == {"text": "one", "signature": "S1"}

    @pytest.mark.parametrize("content", ["plain string", None, [], [{"type": "text", "text": "x"}],
                                         [{"type": "thinking", "thinking": "x"}]])
    def test_extract_returns_none_when_nothing_replayable(self, content):
        assert extract_reasoning_from_anthropic_content(content) is None


class TestCoreReplayBehaviour:
    def test_merge_keeps_first_reasoning(self):
        merged = merge_adjacent_messages([
            UnifiedMessage(role="assistant", content="a", reasoning={"text": "1", "signature": "S1"}),
            UnifiedMessage(role="assistant", content="b", reasoning={"text": "2", "signature": "S2"}),
        ])
        assert len(merged) == 1 and merged[0].reasoning == {"text": "1", "signature": "S1"}

    def test_merge_adopts_later_reasoning_when_first_has_none(self):
        merged = merge_adjacent_messages([
            UnifiedMessage(role="assistant", content="a"),
            UnifiedMessage(role="assistant", content="b", reasoning={"text": "2", "signature": "S2"}),
        ])
        assert merged[0].reasoning == {"text": "2", "signature": "S2"}

    def test_user_messages_never_carry_reasoning(self):
        history = build_kiro_history(
            [UnifiedMessage(role="user", content="q", reasoning={"text": "x", "signature": "S"})],
            "claude-sonnet-4.6",
        )
        assert "reasoningContent" not in json.dumps(history)


# ==================================================================================================
# Responses output + input
# ==================================================================================================

class TestResponsesSignature:
    @pytest.mark.asyncio
    async def test_reasoning_item_carries_encrypted_content(self):
        async def fake_parse(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="deep")
            yield KiroEvent(type="thinking_signature", thinking_signature=REAL_SIG)
            yield KiroEvent(type="content", content="391")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        with patch("kiro.streaming_responses.parse_kiro_stream", fake_parse):
            result = await collect_stream_response(AsyncMock(), AsyncMock(), "m", cache, MagicMock())
        reasoning = next(o for o in result["output"] if o["type"] == "reasoning")
        assert decode_responses_encrypted_content(reasoning["encrypted_content"]) == {
            "text": "deep", "signature": REAL_SIG
        }

    @pytest.mark.asyncio
    async def test_fake_reasoning_item_has_no_encrypted_content(self):
        async def fake_parse(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="fake")
            yield KiroEvent(type="content", content="x")
            yield KiroEvent(type="context_usage", context_usage_percentage=1.0)

        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        with patch("kiro.streaming_responses.parse_kiro_stream", fake_parse):
            result = await collect_stream_response(AsyncMock(), AsyncMock(), "m", cache, MagicMock())
        reasoning = next(o for o in result["output"] if o["type"] == "reasoning")
        assert "encrypted_content" not in reasoning

    def _carrier(self, text="r", sig=REAL_SIG):
        return {"type": "reasoning", "id": "rs_1", "summary": [],
                "encrypted_content": encode_responses_encrypted_content(text, sig)}

    def test_replayed_on_following_message(self):
        _, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "user", "content": "Q1"},
            self._carrier(),
            {"type": "message", "role": "assistant", "content": "A1"},
            {"type": "message", "role": "user", "content": "Q2"},
        ], None)
        assistant = next(m for m in messages if m.role == "assistant")
        assert assistant.reasoning == {"text": "r", "signature": REAL_SIG}

    def test_replayed_on_following_function_call(self):
        _, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "user", "content": "Q1"},
            self._carrier(),
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ], None)
        assistant = next(m for m in messages if m.role == "assistant")
        assert assistant.tool_calls and assistant.reasoning == {"text": "r", "signature": REAL_SIG}

    def test_dropped_when_user_turn_comes_first(self):
        _, messages = convert_responses_input_to_unified([
            self._carrier(),
            {"type": "message", "role": "user", "content": "Q"},
            {"type": "message", "role": "assistant", "content": "A"},
        ], None)
        assert all(m.reasoning is None for m in messages)

    def test_trailing_reasoning_without_assistant_is_dropped(self):
        _, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "user", "content": "Q"},
            self._carrier(),
        ], None)
        assert all(m.reasoning is None for m in messages)

    def test_foreign_encrypted_content_ignored(self):
        _, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "user", "content": "Q1"},
            {"type": "reasoning", "id": "rs_x", "summary": [], "encrypted_content": "gAAAAAB-openai"},
            {"type": "message", "role": "assistant", "content": "A1"},
        ], None)
        assert all(m.reasoning is None for m in messages)

    def test_payload_contains_history_reasoning(self):
        request = ResponsesRequest(model="claude-sonnet-4.6", input=[
            {"type": "message", "role": "user", "content": "Q1"},
            self._carrier(),
            {"type": "message", "role": "assistant", "content": "A1"},
            {"type": "message", "role": "user", "content": "Q2"},
        ])
        payload = build_kiro_payload_from_responses(request, "c", "arn").payload
        (assistant,) = _assistant_entries(payload)
        assert assistant["reasoningContent"]["reasoningText"] == {"text": "r", "signature": REAL_SIG}

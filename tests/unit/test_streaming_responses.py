# -*- coding: utf-8 -*-

"""
Unit tests for streaming_responses module.

Covers minimal Codex-oriented Responses SSE sequences:
- text: response.created → output_text.delta → output_item.done(message) → response.completed
- tool_call: response.created → output_item.added(function_call) → output_item.done → response.completed
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro.streaming_core import KiroEvent
from kiro.streaming_responses import (
    FirstTokenTimeoutError,
    collect_stream_response,
    format_sse_event,
    generate_response_id,
    stream_kiro_to_responses,
    stream_with_first_token_retry,
)


# ==================================================================================================
# Fixtures
# ==================================================================================================

@pytest.fixture
def mock_model_cache():
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    return MagicMock()


@pytest.fixture
def mock_http_client():
    return AsyncMock()


@pytest.fixture
def mock_response():
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


def _parse_sse_events(chunks):
    """Parse list of SSE strings into list of (event_name, data_dict)."""
    events = []
    for chunk in chunks:
        event_name = None
        data_str = None
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        if data_str and data_str != "[DONE]":
            data = json.loads(data_str)
            events.append((event_name or data.get("type"), data))
    return events


def _event_types(events):
    return [e[0] for e in events]


# ==================================================================================================
# Helpers
# ==================================================================================================

class TestFormatAndIds:
    def test_generate_response_id_prefix(self):
        rid = generate_response_id()
        assert rid.startswith("resp_")
        assert len(rid) > 10

    def test_format_sse_event_has_event_and_type(self):
        text = format_sse_event("response.created", {"response": {"id": "resp_1"}, "sequence_number": 0})
        assert text.startswith("event: response.created\n")
        assert '"type": "response.created"' in text
        assert text.endswith("\n\n")


# ==================================================================================================
# Text streaming
# ==================================================================================================

class TestStreamKiroToResponsesText:
    @pytest.mark.asyncio
    async def test_text_minimal_sse_sequence(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        chunks = []
        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    chunks.append(chunk)

        events = _parse_sse_events(chunks)
        types = _event_types(events)

        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert "response.output_item.done" in types
        assert types[-1] == "response.completed"

        deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
        assert deltas == ["Hello", " World"]

        done_items = [
            e[1]["item"]
            for e in events
            if e[0] == "response.output_item.done"
        ]
        assert any(i.get("type") == "message" for i in done_items)
        message_done = next(i for i in done_items if i.get("type") == "message")
        assert message_done["content"][0]["text"] == "Hello World"

        completed = events[-1][1]["response"]
        assert completed["id"].startswith("resp_")
        assert completed["status"] == "completed"
        assert "usage" in completed
        assert "input_tokens" in completed["usage"]
        assert "output_tokens" in completed["usage"]
        assert "total_tokens" in completed["usage"]
        assert any(o.get("type") == "message" for o in completed["output"])

    @pytest.mark.asyncio
    async def test_empty_content_events_produce_no_delta(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="")
            yield KiroEvent(type="content", content="Hi")
            yield KiroEvent(type="content", content="")

        chunks = []
        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    chunks.append(chunk)

        events = _parse_sse_events(chunks)
        deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
        assert deltas == ["Hi"]

    @pytest.mark.asyncio
    async def test_closes_response_on_completion(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="x")

        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for _ in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    pass

        mock_response.aclose.assert_called()


# ==================================================================================================
# Tool call streaming
# ==================================================================================================

class TestStreamKiroToResponsesToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_minimal_sse_sequence(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        tool_use = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'},
        }

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check")
            yield KiroEvent(type="tool_use", tool_use=tool_use)
            yield KiroEvent(type="usage", usage={"credits": 0.001})

        chunks = []
        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    chunks.append(chunk)

        events = _parse_sse_events(chunks)
        types = _event_types(events)

        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert "response.output_item.added" in types
        assert "response.output_item.done" in types
        assert types[-1] == "response.completed"

        # function_call added + done
        fc_added = [
            e[1]["item"]
            for e in events
            if e[0] == "response.output_item.added" and e[1]["item"].get("type") == "function_call"
        ]
        fc_done = [
            e[1]["item"]
            for e in events
            if e[0] == "response.output_item.done" and e[1]["item"].get("type") == "function_call"
        ]
        assert len(fc_added) == 1
        assert len(fc_done) == 1
        assert fc_added[0]["name"] == "get_weather"
        assert fc_added[0]["call_id"] == "call_123"
        assert fc_done[0]["arguments"] == '{"city": "Moscow"}'
        assert fc_done[0]["status"] == "completed"

        completed = events[-1][1]["response"]
        assert completed["id"].startswith("resp_")
        assert completed["status"] == "completed"
        assert "usage" in completed
        assert "input_tokens" in completed["usage"]
        assert any(o.get("type") == "function_call" for o in completed["output"])
        assert any(o.get("type") == "message" for o in completed["output"])

    @pytest.mark.asyncio
    async def test_tool_only_no_text(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "func1", "arguments": "{}"},
                },
            )

        chunks = []
        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    chunks.append(chunk)

        events = _parse_sse_events(chunks)
        types = _event_types(events)
        assert types[0] == "response.created"
        assert "response.output_text.delta" not in types
        assert any(
            e[0] == "response.output_item.added"
            and e[1]["item"].get("type") == "function_call"
            for e in events
        )
        assert types[-1] == "response.completed"


# ==================================================================================================
# collect_stream_response
# ==================================================================================================

class TestCollectStreamResponse:
    @pytest.mark.asyncio
    async def test_collects_completed_response_object(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)

        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                result = await collect_stream_response(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                )

        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["id"].startswith("resp_")
        assert result["model"] == "claude-sonnet-4"
        assert "usage" in result
        assert result["usage"]["input_tokens"] >= 0
        assert result["usage"]["output_tokens"] >= 0
        assert result["usage"]["total_tokens"] >= 0
        message = next(o for o in result["output"] if o["type"] == "message")
        assert message["content"][0]["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_collects_tool_calls(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "func1", "arguments": '{"a": 1}'},
                },
            )

        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                result = await collect_stream_response(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                )

        fc = next(o for o in result["output"] if o["type"] == "function_call")
        assert fc["name"] == "func1"
        assert fc["call_id"] == "call_1"
        assert fc["arguments"] == '{"a": 1}'


# ==================================================================================================
# Errors / retry
# ==================================================================================================

class TestStreamingResponsesErrors:
    @pytest.mark.asyncio
    async def test_propagates_first_token_timeout(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # pragma: no cover

        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with pytest.raises(FirstTokenTimeoutError):
                async for _ in stream_kiro_to_responses(
                    mock_http_client,
                    mock_response,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                ):
                    pass

    @pytest.mark.asyncio
    async def test_emits_response_failed_on_error(
        self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager
    ):
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise RuntimeError("boom")

        chunks = []
        with patch("kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                with pytest.raises(RuntimeError):
                    async for chunk in stream_kiro_to_responses(
                        mock_http_client,
                        mock_response,
                        "claude-sonnet-4",
                        mock_model_cache,
                        mock_auth_manager,
                    ):
                        chunks.append(chunk)

        events = _parse_sse_events(chunks)
        types = _event_types(events)
        assert types[0] == "response.created"
        assert "response.failed" in types
        failed = next(e[1] for e in events if e[0] == "response.failed")
        assert failed["response"]["status"] == "failed"
        assert failed["response"]["error"]["message"] == "boom"
        mock_response.aclose.assert_called()

    @pytest.mark.asyncio
    async def test_retries_on_first_token_timeout(
        self, mock_http_client, mock_model_cache, mock_auth_manager
    ):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()

        call_count = 0

        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            return mock_response

        timeout_raised = False

        async def mock_parse_kiro_stream_with_retry(*args, **kwargs):
            nonlocal timeout_raised
            if not timeout_raised:
                timeout_raised = True
                raise FirstTokenTimeoutError("Timeout!")
            yield KiroEvent(type="content", content="Success")

        chunks = []
        with patch(
            "kiro.streaming_responses.parse_kiro_stream", mock_parse_kiro_stream_with_retry
        ):
            with patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[]):
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15,
                ):
                    chunks.append(chunk)

        assert call_count == 2
        events = _parse_sse_events(chunks)
        assert _event_types(events)[0] == "response.created"
        assert any(e[0] == "response.output_text.delta" for e in events)
        assert _event_types(events)[-1] == "response.completed"

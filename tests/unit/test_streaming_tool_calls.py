"""Tests for incremental (OpenAI-spec) tool_call streaming.

Regression coverage for the Cursor-hang bug: a single very large tool_call
`arguments` payload (e.g. a ~33KB Write) used to be emitted as one giant SSE
chunk. We now stream the arguments incrementally. These tests assert the wire
format is a proper incremental tool_calls sequence and that both the streaming
and non-streaming collectors reconstruct the original arguments exactly.
"""

import json

import pytest

from kiro import streaming_anthropic, streaming_openai
from kiro.parsers import AwsEventStreamParser
from kiro.streaming_core import KiroEvent


class _FakeResponse:
    """Minimal stand-in for httpx.Response (only aclose is exercised)."""

    async def aclose(self):
        return None


class _FakeModelCache:
    def get_max_input_tokens(self, model):
        return 200_000


def _make_events(tool_args_json: str):
    """Build a Kiro event sequence: a bit of content + one big tool_use."""
    return [
        KiroEvent(type="content", content="Working on it"),
        KiroEvent(
            type="tool_use",
            tool_use={
                "id": "toolu_123",
                "type": "function",
                "function": {"name": "Write", "arguments": tool_args_json},
            },
        ),
    ]


def _patch_stream(monkeypatch, events):
    async def fake_parse_kiro_stream(response, first_token_timeout, *args, **kwargs):
        for ev in events:
            yield ev

    monkeypatch.setattr(streaming_openai, "parse_kiro_stream", fake_parse_kiro_stream)


def _big_write_args():
    # ~33KB markdown plan, like the real failure case. Includes characters that
    # require JSON escaping (quotes, newlines, unicode) to exercise escaping.
    body = ("# Plan\n\n" + 'Step "one": do the thing.\n' + "中文内容 \u2713\n") * 900
    return json.dumps({"path": "PLAN.md", "contents": body}, ensure_ascii=False)


async def _collect_sse(gen):
    """Drain an SSE generator into a list of parsed (raw, json|None) chunks."""
    chunks = []
    async for raw in gen:
        assert raw.startswith("data: ")
        payload = raw[len("data: "):].strip()
        if payload == "[DONE]":
            chunks.append((raw, None))
        else:
            chunks.append((raw, json.loads(payload)))
    return chunks


@pytest.mark.asyncio
async def test_large_tool_call_streams_incrementally(monkeypatch):
    args_json = _big_write_args()
    assert len(args_json) > 30_000  # mirror the real ~33KB case

    _patch_stream(monkeypatch, _make_events(args_json))

    gen = streaming_openai.stream_kiro_to_openai_internal(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    )
    chunks = await _collect_sse(gen)

    # Last two wire items must be the final chunk then [DONE].
    assert chunks[-1][0] == "data: [DONE]\n\n"
    final_chunk = chunks[-2][1]
    assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"
    assert final_chunk["choices"][0]["delta"] == {}
    assert "usage" in final_chunk

    # Gather all tool_call deltas (in order).
    tool_deltas = []
    for _raw, data in chunks:
        if data is None:
            continue
        delta = data["choices"][0]["delta"]
        if "tool_calls" in delta:
            # Each chunk we emit carries exactly one tool_call delta.
            assert len(delta["tool_calls"]) == 1
            tool_deltas.append(delta["tool_calls"][0])

    # First tool delta is the "opening" one: identity + empty arguments.
    opening = tool_deltas[0]
    assert opening["index"] == 0
    assert opening["id"] == "toolu_123"
    assert opening["type"] == "function"
    assert opening["function"]["name"] == "Write"
    assert opening["function"]["arguments"] == ""

    # Must be split into MULTIPLE argument deltas (not one giant chunk).
    arg_deltas = tool_deltas[1:]
    assert len(arg_deltas) > 1
    for d in arg_deltas:
        assert d["index"] == 0
        # Argument deltas carry only the arguments fragment.
        assert set(d["function"].keys()) == {"arguments"}
        # No single SSE line should be oversized.
        assert len(d["function"]["arguments"]) <= streaming_openai.TOOL_CALL_ARG_CHUNK_SIZE

    # Reassembled arguments must equal the original JSON exactly.
    reassembled = "".join(d["function"]["arguments"] for d in arg_deltas)
    assert reassembled == args_json
    assert json.loads(reassembled)["path"] == "PLAN.md"


@pytest.mark.asyncio
async def test_collect_stream_response_reassembles_arguments(monkeypatch):
    args_json = _big_write_args()
    _patch_stream(monkeypatch, _make_events(args_json))

    result = await streaming_openai.collect_stream_response(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    )

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["id"] == "toolu_123"
    assert tc["function"]["name"] == "Write"
    # Non-streaming collection must rebuild the full arguments string.
    assert tc["function"]["arguments"] == args_json
    assert json.loads(tc["function"]["arguments"])["contents"]


@pytest.mark.asyncio
async def test_multiple_tool_calls_keep_separate_indices(monkeypatch):
    events = [
        KiroEvent(
            type="tool_use",
            tool_use={
                "id": "toolu_a",
                "type": "function",
                "function": {"name": "Read", "arguments": json.dumps({"path": "a.txt"})},
            },
        ),
        KiroEvent(
            type="tool_use",
            tool_use={
                "id": "toolu_b",
                "type": "function",
                "function": {"name": "Glob", "arguments": json.dumps({"glob": "*.py"})},
            },
        ),
    ]
    _patch_stream(monkeypatch, events)

    result = await streaming_openai.collect_stream_response(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    )

    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert [tc["id"] for tc in tool_calls] == ["toolu_a", "toolu_b"]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "a.txt"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"glob": "*.py"}


@pytest.mark.asyncio
async def test_upstream_tool_lifecycle_is_forwarded_before_stream_end(monkeypatch):
    """Tool start and argument fragments must be emitted in arrival order."""
    events = [
        KiroEvent(
            type="tool_start",
            tool_use={
                "id": "toolu_live",
                "type": "function",
                "function": {"name": "Write", "arguments": ""},
            },
        ),
        KiroEvent(
            type="tool_input",
            tool_call_id="toolu_live",
            tool_input_delta='{"path":',
        ),
        KiroEvent(
            type="tool_input",
            tool_call_id="toolu_live",
            tool_input_delta='"live.txt"}',
        ),
        KiroEvent(
            type="tool_stop",
            tool_use={
                "id": "toolu_live",
                "type": "function",
                "function": {
                    "name": "Write",
                    "arguments": '{"path":"live.txt"}',
                },
            },
        ),
        KiroEvent(type="usage", usage={"credits": 0.1}),
    ]
    _patch_stream(monkeypatch, events)

    chunks = await _collect_sse(streaming_openai.stream_kiro_to_openai_internal(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    ))

    tool_deltas = [
        data["choices"][0]["delta"]["tool_calls"][0]
        for _raw, data in chunks
        if data is not None
        and "tool_calls" in data["choices"][0]["delta"]
    ]
    assert tool_deltas[0]["function"]["name"] == "Write"
    assert "".join(
        delta["function"].get("arguments", "")
        for delta in tool_deltas[1:]
    ) == '{"path":"live.txt"}'

    final_index = next(
        index
        for index, (_raw, data) in enumerate(chunks)
        if data is not None and data["choices"][0]["finish_reason"] == "tool_calls"
    )
    last_tool_index = max(
        index
        for index, (_raw, data) in enumerate(chunks)
        if data is not None and "tool_calls" in data["choices"][0]["delta"]
    )
    assert last_tool_index < final_index


@pytest.mark.asyncio
async def test_xml_fallback_is_returned_as_streamed_tool_call(monkeypatch):
    """Legacy XML tool output must still become OpenAI tool_call SSE deltas."""
    xml = (
        '<invoke name="Shell">'
        '<parameter name="command">pwd</parameter>'
        '</invoke>'
    )
    _patch_stream(monkeypatch, [
        KiroEvent(type="content", content=xml),
        KiroEvent(type="usage", usage={"credits": 0.1}),
    ])

    chunks = await _collect_sse(streaming_openai.stream_kiro_to_openai_internal(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    ))
    tool_deltas = [
        data["choices"][0]["delta"]["tool_calls"][0]
        for _raw, data in chunks
        if data is not None
        and "tool_calls" in data["choices"][0]["delta"]
    ]

    assert tool_deltas[0]["function"]["name"] == "Shell"
    arguments = "".join(
        delta["function"].get("arguments", "")
        for delta in tool_deltas[1:]
    )
    assert json.loads(arguments) == {"command": "pwd"}


# ==================================================================================================
# Duplicate tool_use id regression (Anthropic + OpenAI + Responses)
# ==================================================================================================
#
# Upstream Kiro emits the *same* toolUseId twice for a single tool call: once
# carrying the real arguments, then a second, empty repetition. The
# non-streaming path collapses these via deduplicate_tool_calls(), but the
# streaming paths used to forward both, so clients that key tool input by id
# saw the empty repetition overwrite the real arguments (Claude Code rendered
# an empty Bash description).


DUP_TOOL_ARGS = {"command": "ls -la", "description": "列出当前目录内容"}


def _duplicate_id_upstream_chunks():
    """Raw Kiro chunks that emit one toolUseId twice.

    Mirrors the observed gateway log: a full lifecycle carrying the real
    arguments, immediately followed by a second lifecycle for the same id whose
    accumulated argument string is empty.
    """
    args_json = json.dumps(DUP_TOOL_ARGS, ensure_ascii=False)
    return [
        b'{"name":"Bash","toolUseId":"toolu_dup"}',
        json.dumps({"input": args_json}, ensure_ascii=False).encode("utf-8"),
        b'{"stop":true}',
        # Second, empty emission of the SAME id.
        b'{"name":"Bash","toolUseId":"toolu_dup"}',
        b'{"stop":true}',
        b'{"usage":0.1}',
    ]


def _duplicate_id_events():
    """Drive the real AwsEventStreamParser and collect the KiroEvents it emits.

    Going through the parser (rather than hand-building KiroEvents) is what makes
    these tests cover the actual de-duplication path.
    """
    parser = AwsEventStreamParser()
    raw_events = []
    for chunk in _duplicate_id_upstream_chunks():
        raw_events.extend(parser.feed(chunk))

    events = []
    for ev in raw_events:
        etype = ev["type"]
        data = ev["data"]
        if etype == "tool_start":
            events.append(KiroEvent(type="tool_start", tool_use=data))
        elif etype == "tool_input":
            events.append(KiroEvent(
                type="tool_input",
                tool_call_id=data["tool_call_id"],
                tool_input_delta=data["arguments_delta"],
            ))
        elif etype == "tool_stop":
            events.append(KiroEvent(type="tool_stop", tool_use=data))
        elif etype == "usage":
            events.append(KiroEvent(type="usage", usage={"credits": data}))
    return events


def test_parser_suppresses_replayed_tool_id():
    """The parser must not surface a second lifecycle for a finalized id."""
    parser = AwsEventStreamParser()
    events = []
    for chunk in _duplicate_id_upstream_chunks():
        events.extend(parser.feed(chunk))

    starts = [e for e in events if e["type"] == "tool_start"]
    stops = [e for e in events if e["type"] == "tool_stop"]
    assert len(starts) == 1, f"expected one tool_start, got {len(starts)}"
    assert len(stops) == 1, f"expected one tool_stop, got {len(stops)}"

    deltas = "".join(
        e["data"]["arguments_delta"] for e in events if e["type"] == "tool_input"
    )
    assert json.loads(deltas) == DUP_TOOL_ARGS

    # The non-streaming collector must also keep exactly one, correct copy.
    assert len(parser.tool_calls) == 1
    assert json.loads(parser.tool_calls[0]["function"]["arguments"]) == DUP_TOOL_ARGS


def test_parser_keeps_real_arguments_when_empty_emission_comes_first():
    """If the empty emission arrives first, the real arguments must still win."""
    parser = AwsEventStreamParser()
    args_json = json.dumps(DUP_TOOL_ARGS, ensure_ascii=False)
    chunks = [
        b'{"name":"Bash","toolUseId":"toolu_dup"}',
        b'{"stop":true}',
        b'{"name":"Bash","toolUseId":"toolu_dup"}',
        json.dumps({"input": args_json}, ensure_ascii=False).encode("utf-8"),
        b'{"stop":true}',
    ]
    for chunk in chunks:
        parser.feed(chunk)

    assert len(parser.tool_calls) == 1
    assert json.loads(parser.tool_calls[0]["function"]["arguments"]) == DUP_TOOL_ARGS


def test_parser_keeps_distinct_tool_ids_separate():
    """De-duplication must key on id, not collapse genuinely different calls."""
    parser = AwsEventStreamParser()
    chunks = [
        b'{"name":"Bash","toolUseId":"toolu_a"}',
        json.dumps({"input": '{"command": "pwd"}'}).encode("utf-8"),
        b'{"stop":true}',
        b'{"name":"Bash","toolUseId":"toolu_b"}',
        json.dumps({"input": '{"command": "whoami"}'}).encode("utf-8"),
        b'{"stop":true}',
    ]
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))

    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 2
    assert [s["data"]["id"] for s in starts] == ["toolu_a", "toolu_b"]
    assert len(parser.tool_calls) == 2
    assert json.loads(parser.tool_calls[0]["function"]["arguments"]) == {"command": "pwd"}
    assert json.loads(parser.tool_calls[1]["function"]["arguments"]) == {"command": "whoami"}


async def _collect_anthropic_sse(gen):
    """Drain an Anthropic SSE generator into parsed data payloads."""
    events = []
    async for raw in gen:
        for line in raw.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


@pytest.mark.asyncio
async def test_duplicate_tool_id_emits_single_anthropic_block(monkeypatch):
    """A repeated toolUseId must not open a second, empty tool_use block."""
    async def fake_parse_kiro_stream(response, *args, **kwargs):
        for ev in _duplicate_id_events():
            yield ev

    monkeypatch.setattr(
        streaming_anthropic, "parse_kiro_stream", fake_parse_kiro_stream
    )
    monkeypatch.setattr(
        streaming_anthropic, "parse_bracket_tool_calls", lambda _content: []
    )
    monkeypatch.setattr(
        streaming_anthropic, "parse_xml_tool_calls", lambda _content: []
    )

    events = await _collect_anthropic_sse(
        streaming_anthropic.stream_kiro_to_anthropic(
            _FakeResponse(),
            "claude-sonnet-4",
            _FakeModelCache(),
            None,
        )
    )

    starts = [
        e for e in events
        if e.get("type") == "content_block_start"
        and e["content_block"].get("type") == "tool_use"
    ]
    assert len(starts) == 1, (
        f"expected one tool_use block, got {len(starts)}: "
        f"{[s['content_block']['id'] for s in starts]}"
    )

    index = starts[0]["index"]
    reassembled = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("index") == index
        and e["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(reassembled) == {
        "command": "ls -la",
        "description": "列出当前目录内容",
    }

    stops = [
        e for e in events
        if e.get("type") == "content_block_stop" and e.get("index") == index
    ]
    assert len(stops) == 1, f"expected one stop for block {index}, got {len(stops)}"


@pytest.mark.asyncio
async def test_duplicate_tool_id_emits_single_openai_tool_call(monkeypatch):
    """The OpenAI path must not append an empty duplicate tool_call."""
    _patch_stream(monkeypatch, _duplicate_id_events())

    chunks = await _collect_sse(streaming_openai.stream_kiro_to_openai_internal(
        client=None,
        response=_FakeResponse(),
        model="claude-sonnet-4",
        model_cache=_FakeModelCache(),
        auth_manager=None,
    ))

    tool_deltas = [
        data["choices"][0]["delta"]["tool_calls"][0]
        for _raw, data in chunks
        if data is not None
        and "tool_calls" in data["choices"][0]["delta"]
    ]
    named = [d for d in tool_deltas if d.get("function", {}).get("name")]
    assert len(named) == 1, f"expected one tool_call opener, got {len(named)}"

    arguments = "".join(
        d["function"].get("arguments", "")
        for d in tool_deltas
        if d["index"] == named[0]["index"]
    )
    assert json.loads(arguments) == {
        "command": "ls -la",
        "description": "列出当前目录内容",
    }

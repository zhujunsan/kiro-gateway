# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Streaming logic for converting Kiro stream to OpenAI Responses API format.

Consumes parse_kiro_stream() directly (not chat.completion chunks) and emits
Responses SSE events that Codex (wire_api=responses) consumes:

- response.created
- response.in_progress
- response.output_item.added (reasoning / message / function_call)
- response.reasoning_summary_part.added / response.reasoning_summary_text.delta|done /
  response.reasoning_summary_part.done (when FAKE_REASONING_HANDLING=as_reasoning_content)
- response.output_text.delta / response.output_text.done
- response.function_call_arguments.delta / response.function_call_arguments.done
- response.output_item.done (reasoning / message / function_call)
- response.completed
- response.failed

Kiro usually delivers complete tool arguments in one tool_use event; we still
chunk them into function_call_arguments.delta then .done for Codex compatibility.

TODO(models_responses / converters_responses): routes may later pass typed
request helpers; this module stays self-contained on response id helpers and
optional list-shaped request_messages/request_tools for token fallback.
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    FAKE_REASONING_HANDLING,
    FIRST_TOKEN_MAX_RETRIES,
    FIRST_TOKEN_TIMEOUT,
)
from kiro.output_tokens import count_generated_output_tokens
from kiro.parsers import deduplicate_tool_calls, parse_bracket_tool_calls, parse_xml_tool_calls
from kiro.streaming_core import (
    FirstTokenTimeoutError,
    calculate_tokens_from_context_usage,
    parse_kiro_stream,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)
from kiro.tokenizer import count_message_tokens, count_tokens, count_tools_tokens

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# Slice complete tool-argument JSON into bounded deltas (same rationale as
# streaming_openai.TOOL_CALL_ARG_CHUNK_SIZE: avoid oversized SSE lines).
FUNCTION_CALL_ARG_CHUNK_SIZE = 1024


__all__ = [
    "FirstTokenTimeoutError",
    "FUNCTION_CALL_ARG_CHUNK_SIZE",
    "generate_response_id",
    "format_sse_event",
    "stream_kiro_to_responses",
    "stream_kiro_to_responses_internal",
    "stream_with_first_token_retry",
    "collect_stream_response",
]


def generate_response_id() -> str:
    """Generate a unique Responses API id (resp_...)."""
    return f"resp_{uuid.uuid4().hex}"


def generate_message_item_id() -> str:
    """Generate a unique message output item id (msg_...)."""
    return f"msg_{uuid.uuid4().hex}"


def generate_function_call_item_id() -> str:
    """Generate a unique function_call output item id (fc_...)."""
    return f"fc_{uuid.uuid4().hex}"


def generate_reasoning_item_id() -> str:
    """Generate a unique reasoning output item id (rs_...)."""
    return f"rs_{uuid.uuid4().hex}"


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format one Responses API SSE event.

    Wire format (OpenAI Responses):
        event: {event_type}
        data: {json with type=event_type}

    Args:
        event_type: Event name (e.g. response.created)
        data: Event payload (type is set/overwritten to event_type)

    Returns:
        SSE string ending with blank line
    """
    payload = dict(data)
    payload["type"] = event_type
    text = f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    if debug_logger:
        debug_logger.log_modified_chunk(text.encode("utf-8"))
    return text


def _base_response(
    response_id: str,
    model: str,
    created_at: int,
    status: str,
    output: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Minimal response object snapshot for created/completed/failed events."""
    obj: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output if output is not None else [],
        "error": error,
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def _message_item(item_id: str, text: str, status: str = "completed") -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _function_call_item(
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _reasoning_item(
    item_id: str,
    summary: Optional[List[Dict[str, Any]]] = None,
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": summary if summary is not None else [],
    }


def _summary_text_part(text: str = "") -> Dict[str, Any]:
    return {"type": "summary_text", "text": text}


def _normalize_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Kiro/OpenAI-shaped tool_use into name/arguments/call_id."""
    from kiro.converters_core import get_original_tool_name

    func = tc.get("function") or {}
    name = func.get("name") or tc.get("name") or ""
    name = get_original_tool_name(name)
    arguments = func.get("arguments")
    if arguments is None:
        arguments = tc.get("input")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    if not arguments:
        arguments = "{}"
    call_id = tc.get("id") or tc.get("call_id") or f"call_{uuid.uuid4().hex[:8]}"
    return {
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "raw": tc,
    }


async def stream_kiro_to_responses_internal(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    conversation_id: Optional[str] = None,
    echo_reasoning_items: Optional[List[Dict[str, Any]]] = None,
    emit_reasoning_summary: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Internal generator: Kiro stream → Responses API SSE.

    Raises FirstTokenTimeoutError if the first token is not received in time
    (for stream_with_first_token_retry).

    Args:
        echo_reasoning_items: Input ``type:reasoning`` stubs to echo at the
            start of ``output`` (id/summary preserved; no encrypted_content).
        emit_reasoning_summary: When True and FAKE_REASONING_HANDLING is
            ``as_reasoning_content``, stream thinking as reasoning summary
            text. When False, still emit a reasoning item with empty summary.
    """
    response_id = generate_response_id()
    message_item_id = generate_message_item_id()
    reasoning_item_id = generate_reasoning_item_id()
    created_at = int(time.time())
    sequence_number = 0

    metering_data = None
    context_usage_percentage = None
    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream: List[Dict[str, Any]] = []
    tool_streams: Dict[str, Dict[str, Any]] = {}
    streamed_tool_ids = set()
    streamed_function_items: List[Tuple[int, Dict[str, Any]]] = []
    message_item_started = False
    reasoning_item_started = False
    reasoning_item_closed = False
    reasoning_summary_part_started = False
    streaming_error_occurred = False
    created_emitted = False
    # Live output_index for the next new item (echo stubs consume 0..n-1 first).
    next_output_index = 0
    message_output_index = 0
    reasoning_output_index = 0

    echo_items = list(echo_reasoning_items or [])

    def next_seq() -> int:
        nonlocal sequence_number
        seq = sequence_number
        sequence_number += 1
        return seq

    def emit_created() -> str:
        """Emit response.created once (deferred until first Kiro event for retry)."""
        nonlocal created_emitted
        created_emitted = True
        return format_sse_event(
            "response.created",
            {
                "response": _base_response(
                    response_id, model, created_at, "in_progress"
                ),
                "sequence_number": next_seq(),
            },
        )

    def emit_in_progress() -> str:
        """Emit response.in_progress immediately after created."""
        return format_sse_event(
            "response.in_progress",
            {
                "response": _base_response(
                    response_id, model, created_at, "in_progress"
                ),
                "sequence_number": next_seq(),
            },
        )

    def emit_echo_reasoning_events() -> List[str]:
        """Emit completed echo stubs for input reasoning items."""
        nonlocal next_output_index
        events: List[str] = []
        for stub in echo_items:
            idx = next_output_index
            next_output_index += 1
            item = {
                "id": stub.get("id") or generate_reasoning_item_id(),
                "type": "reasoning",
                "status": stub.get("status") or "completed",
                "summary": stub.get("summary") if isinstance(stub.get("summary"), list) else [],
            }
            events.append(
                format_sse_event(
                    "response.output_item.added",
                    {
                        "output_index": idx,
                        "item": {**item, "status": "completed"},
                        "sequence_number": next_seq(),
                    },
                )
            )
            events.append(
                format_sse_event(
                    "response.output_item.done",
                    {
                        "output_index": idx,
                        "item": item,
                        "sequence_number": next_seq(),
                    },
                )
            )
        return events

    def start_reasoning_item_events() -> List[str]:
        nonlocal reasoning_item_started, reasoning_output_index, next_output_index
        nonlocal reasoning_summary_part_started
        if reasoning_item_started:
            return []
        reasoning_item_started = True
        reasoning_output_index = next_output_index
        next_output_index += 1
        events = [
            format_sse_event(
                "response.output_item.added",
                {
                    "output_index": reasoning_output_index,
                    "item": _reasoning_item(
                        reasoning_item_id, summary=[], status="in_progress"
                    ),
                    "sequence_number": next_seq(),
                },
            )
        ]
        if emit_reasoning_summary:
            reasoning_summary_part_started = True
            events.append(
                format_sse_event(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": reasoning_item_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "part": _summary_text_part(""),
                        "sequence_number": next_seq(),
                    },
                )
            )
        return events

    def close_reasoning_item_events() -> List[str]:
        nonlocal reasoning_item_closed, reasoning_summary_part_started
        if not reasoning_item_started or reasoning_item_closed:
            return []
        reasoning_item_closed = True
        events: List[str] = []
        summary_parts: List[Dict[str, Any]] = []
        if emit_reasoning_summary and full_thinking_content:
            if reasoning_summary_part_started:
                events.append(
                    format_sse_event(
                        "response.reasoning_summary_text.done",
                        {
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "summary_index": 0,
                            "text": full_thinking_content,
                            "sequence_number": next_seq(),
                        },
                    )
                )
                part = _summary_text_part(full_thinking_content)
                events.append(
                    format_sse_event(
                        "response.reasoning_summary_part.done",
                        {
                            "item_id": reasoning_item_id,
                            "output_index": reasoning_output_index,
                            "summary_index": 0,
                            "part": part,
                            "sequence_number": next_seq(),
                        },
                    )
                )
                summary_parts = [part]
            else:
                summary_parts = [_summary_text_part(full_thinking_content)]
        done_item = _reasoning_item(
            reasoning_item_id, summary=summary_parts, status="completed"
        )
        events.append(
            format_sse_event(
                "response.output_item.done",
                {
                    "output_index": reasoning_output_index,
                    "item": done_item,
                    "sequence_number": next_seq(),
                },
            )
        )
        return events

    try:
        # Defer response.created until the first Kiro event so first-token
        # timeout can retry before any SSE bytes reach the client.
        async for event in parse_kiro_stream(response, first_token_timeout):
            if not created_emitted:
                yield emit_created()
                yield emit_in_progress()
                for echo_evt in emit_echo_reasoning_events():
                    yield echo_evt

            if event.type == "content" and event.content:
                # Close open reasoning before starting message text.
                for close_evt in close_reasoning_item_events():
                    yield close_evt

                full_content += event.content

                if not message_item_started:
                    message_output_index = next_output_index
                    next_output_index += 1
                    yield format_sse_event(
                        "response.output_item.added",
                        {
                            "output_index": message_output_index,
                            "item": _message_item(
                                message_item_id, "", status="in_progress"
                            ),
                            "sequence_number": next_seq(),
                        },
                    )
                    message_item_started = True

                yield format_sse_event(
                    "response.output_text.delta",
                    {
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "delta": event.content,
                        "sequence_number": next_seq(),
                    },
                )

            elif event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content

                if FAKE_REASONING_HANDLING == "as_reasoning_content":
                    # Native Responses reasoning item — do not dump into output_text.
                    for start_evt in start_reasoning_item_events():
                        yield start_evt
                    if emit_reasoning_summary:
                        yield format_sse_event(
                            "response.reasoning_summary_text.delta",
                            {
                                "item_id": reasoning_item_id,
                                "output_index": reasoning_output_index,
                                "summary_index": 0,
                                "delta": event.thinking_content,
                                "sequence_number": next_seq(),
                            },
                        )
                elif FAKE_REASONING_HANDLING in (
                    "pass",
                    "strip_tags",
                    "include_as_text",
                ):
                    # Include thinking in regular output_text (do not also emit
                    # a reasoning item — avoids double-dumping).
                    full_content += event.thinking_content
                    if not message_item_started:
                        for close_evt in close_reasoning_item_events():
                            yield close_evt
                        message_output_index = next_output_index
                        next_output_index += 1
                        yield format_sse_event(
                            "response.output_item.added",
                            {
                                "output_index": message_output_index,
                                "item": _message_item(
                                    message_item_id, "", status="in_progress"
                                ),
                                "sequence_number": next_seq(),
                            },
                        )
                        message_item_started = True
                    yield format_sse_event(
                        "response.output_text.delta",
                        {
                            "item_id": message_item_id,
                            "output_index": message_output_index,
                            "content_index": 0,
                            "delta": event.thinking_content,
                            "sequence_number": next_seq(),
                        },
                    )
                # else (remove): keep full_thinking_content for token counting only

            elif event.type == "tool_start" and event.tool_use:
                for close_evt in close_reasoning_item_events():
                    yield close_evt

                from kiro.converters_core import get_original_tool_name

                tool = event.tool_use
                function = tool.get("function") or {}
                tool_id = tool.get("id") or generate_function_call_item_id()
                tool_name = get_original_tool_name(
                    function.get("name") or tool.get("name") or ""
                )
                suppress = tool_name == "web_search"
                output_index = next_output_index
                next_output_index += 1
                item_id = generate_function_call_item_id()
                tool_streams[tool_id] = {
                    "item_id": item_id,
                    "output_index": output_index,
                    "name": tool_name,
                    "suppress": suppress,
                }
                if suppress:
                    # The emulated web search becomes message text, so it must
                    # not reserve a function-call output slot.
                    next_output_index -= 1
                    continue

                yield format_sse_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": _function_call_item(
                            item_id,
                            tool_id,
                            tool_name,
                            "",
                            status="in_progress",
                        ),
                        "sequence_number": next_seq(),
                    },
                )
                streamed_tool_ids.add(tool_id)

            elif event.type == "tool_input":
                state = tool_streams.get(event.tool_call_id or "")
                if not state or state["suppress"] or not event.tool_input_delta:
                    continue
                for start in range(
                    0, len(event.tool_input_delta), FUNCTION_CALL_ARG_CHUNK_SIZE
                ):
                    yield format_sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": state["item_id"],
                            "output_index": state["output_index"],
                            "delta": event.tool_input_delta[
                                start:start + FUNCTION_CALL_ARG_CHUNK_SIZE
                            ],
                            "sequence_number": next_seq(),
                        },
                    )

            elif event.type in ("tool_stop", "tool_use") and event.tool_use:
                tool_id_from_event = event.tool_use.get("id")
                stream_state = tool_streams.pop(tool_id_from_event, None)
                if stream_state and not stream_state["suppress"]:
                    tool_calls_from_stream.append(event.tool_use)
                    normalized = _normalize_tool_call(event.tool_use)
                    yield format_sse_event(
                        "response.function_call_arguments.done",
                        {
                            "item_id": stream_state["item_id"],
                            "output_index": stream_state["output_index"],
                            "name": stream_state["name"],
                            "arguments": normalized["arguments"],
                            "sequence_number": next_seq(),
                        },
                    )
                    done_item = _function_call_item(
                        stream_state["item_id"],
                        normalized["call_id"],
                        stream_state["name"],
                        normalized["arguments"],
                        status="completed",
                    )
                    yield format_sse_event(
                        "response.output_item.done",
                        {
                            "output_index": stream_state["output_index"],
                            "item": done_item,
                            "sequence_number": next_seq(),
                        },
                    )
                    streamed_function_items.append(
                        (stream_state["output_index"], done_item)
                    )
                    continue

                for close_evt in close_reasoning_item_events():
                    yield close_evt

                tool = event.tool_use
                tool_name = ""
                if tool:
                    tool_name = (tool.get("function") or {}).get("name", "") or tool.get(
                        "name", ""
                    )

                from kiro.converters_core import get_original_tool_name

                tool_name = get_original_tool_name(tool_name)

                # WebSearch Path B: MCP tool emulation (same as streaming_openai)
                if tool_name == "web_search":
                    from kiro.mcp_tools import call_kiro_mcp_api, generate_search_summary

                    logger.info(
                        "Intercepted web_search tool call (Path B - MCP emulation, Responses)"
                    )
                    tool_input = (tool.get("function") or {}).get("arguments", {}) or tool.get(
                        "input", {}
                    )
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            tool_input = {}

                    query = tool_input.get("query", "") if isinstance(tool_input, dict) else ""
                    if query:
                        mcp_tool_use_id, results = await call_kiro_mcp_api(query, auth_manager)
                        if results is not None:
                            summary = generate_search_summary(query, results)
                            if not message_item_started:
                                message_output_index = next_output_index
                                next_output_index += 1
                                yield format_sse_event(
                                    "response.output_item.added",
                                    {
                                        "output_index": message_output_index,
                                        "item": _message_item(
                                            message_item_id, "", status="in_progress"
                                        ),
                                        "sequence_number": next_seq(),
                                    },
                                )
                                message_item_started = True
                            chunk_size = 100
                            for i in range(0, len(summary), chunk_size):
                                piece = summary[i : i + chunk_size]
                                full_content += piece
                                yield format_sse_event(
                                    "response.output_text.delta",
                                    {
                                        "item_id": message_item_id,
                                        "output_index": message_output_index,
                                        "content_index": 0,
                                        "delta": piece,
                                        "sequence_number": next_seq(),
                                    },
                                )
                            continue

                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "usage" and event.usage:
                metering_data = event.usage

            elif event.type == "context_usage" and event.context_usage_percentage is not None:
                context_usage_percentage = event.context_usage_percentage

        if not created_emitted:
            # Empty stream still needs a created → in_progress → completed handshake.
            yield emit_created()
            yield emit_in_progress()
            for echo_evt in emit_echo_reasoning_events():
                yield echo_evt

        # Close reasoning if stream ended with thinking and no content/tools yet.
        for close_evt in close_reasoning_item_events():
            yield close_evt

        received_usage = metering_data is not None
        received_context_usage = context_usage_percentage is not None
        stream_completed_normally = received_usage or received_context_usage

        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        xml_tool_calls = parse_xml_tool_calls(full_content)
        all_tool_calls = deduplicate_tool_calls(
            tool_calls_from_stream + bracket_tool_calls + xml_tool_calls
        )

        # Kiro has no hard max_output_tokens API. Abrupt stream end without
        # usage/context_usage (and with content, no tools) is treated as
        # truncation → Responses status=incomplete / reason=max_output_tokens.
        content_was_truncated = (
            not stream_completed_normally
            and len(full_content) > 0
            and not all_tool_calls
        )
        if content_was_truncated:
            from kiro.config import TRUNCATION_RECOVERY

            logger.error(
                f"Content truncated by Kiro API (Responses): stream ended without "
                f"completion signals, length={len(full_content)} chars. "
                f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
            )

        output_tokens = count_generated_output_tokens(
            full_content,
            full_thinking_content,
            all_tool_calls,
        )
        input_tokens, total_tokens, prompt_source, total_source = (
            calculate_tokens_from_context_usage(
                context_usage_percentage, output_tokens, model_cache, model
            )
        )

        if prompt_source == "unknown" and request_messages:
            # TODO(converters_responses): prefer unified→openai-shaped messages
            # for accurate tiktoken fallback once converters land.
            input_tokens = count_message_tokens(
                request_messages, apply_claude_correction=False
            )
            if request_tools:
                input_tokens += count_tools_tokens(
                    request_tools, apply_claude_correction=False
                )
            total_tokens = input_tokens + output_tokens
            prompt_source = "tiktoken"
            total_source = "tiktoken"

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if metering_data:
            usage["credits_used"] = metering_data

        logger.debug(
            f"[Usage/Responses] {model}: "
            f"input_tokens={input_tokens} ({prompt_source}), "
            f"output_tokens={output_tokens} (tiktoken), "
            f"total_tokens={total_tokens} ({total_source})"
        )

        # Truncation recovery bookkeeping (same as OpenAI path)
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import save_content_truncation, save_tool_truncation

        if should_inject_recovery():
            truncated_count = 0
            for tc in all_tool_calls:
                if tc.get("_truncation_detected"):
                    save_tool_truncation(
                        tool_call_id=tc["id"],
                        tool_name=tc["function"]["name"],
                        truncation_info=tc["_truncation_info"],
                    )
                    truncated_count += 1
            if content_was_truncated:
                save_content_truncation(full_content)
            if truncated_count > 0 or content_was_truncated:
                logger.info(
                    f"Truncation detected (Responses): {truncated_count} tool(s), "
                    f"content={content_was_truncated}."
                )

        output_items: List[Dict[str, Any]] = []

        # Echoed input reasoning stubs (already streamed as added/done).
        for stub in echo_items:
            output_items.append({
                "id": stub.get("id") or generate_reasoning_item_id(),
                "type": "reasoning",
                "status": stub.get("status") or "completed",
                "summary": stub.get("summary") if isinstance(stub.get("summary"), list) else [],
            })

        # New reasoning item from this turn's thinking.
        if reasoning_item_started:
            summary_parts: List[Dict[str, Any]] = []
            if emit_reasoning_summary and full_thinking_content:
                summary_parts = [_summary_text_part(full_thinking_content)]
            output_items.append(
                _reasoning_item(
                    reasoning_item_id, summary=summary_parts, status="completed"
                )
            )

        output_index = next_output_index
        # Close message item if we streamed any text
        if message_item_started or (full_content and not all_tool_calls):
            if not message_item_started and full_content:
                message_output_index = output_index
                output_index += 1
                yield format_sse_event(
                    "response.output_item.added",
                    {
                        "output_index": message_output_index,
                        "item": _message_item(
                            message_item_id, "", status="in_progress"
                        ),
                        "sequence_number": next_seq(),
                    },
                )
            yield format_sse_event(
                "response.output_text.done",
                {
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": full_content,
                    "sequence_number": next_seq(),
                },
            )
            msg_status = "incomplete" if content_was_truncated else "completed"
            msg_item = _message_item(
                message_item_id, full_content, status=msg_status
            )
            yield format_sse_event(
                "response.output_item.done",
                {
                    "output_index": message_output_index,
                    "item": msg_item,
                    "sequence_number": next_seq(),
                },
            )
            output_items.append(msg_item)

        # Incrementally streamed function items have already completed on the
        # wire; include them in the final response output in index order.
        for _, item in sorted(streamed_function_items, key=lambda entry: entry[0]):
            output_items.append(item)

        # Text-embedded fallback and legacy complete events still arrive as full
        # calls. Emit only those that did not have a live start/input lifecycle.
        fc_output_index = next_output_index

        for tc in all_tool_calls:
            normalized = _normalize_tool_call(tc)
            if normalized["call_id"] in streamed_tool_ids:
                continue
            fc_item_id = generate_function_call_item_id()
            added_item = _function_call_item(
                fc_item_id,
                normalized["call_id"],
                normalized["name"],
                "",
                status="in_progress",
            )
            yield format_sse_event(
                "response.output_item.added",
                {
                    "output_index": fc_output_index,
                    "item": added_item,
                    "sequence_number": next_seq(),
                },
            )

            tool_args = normalized["arguments"]
            # Always emit at least one delta so Codex sees a complete arg stream
            # even when arguments are "{}" or empty.
            if tool_args:
                for start in range(0, len(tool_args), FUNCTION_CALL_ARG_CHUNK_SIZE):
                    yield format_sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": fc_item_id,
                            "output_index": fc_output_index,
                            "delta": tool_args[start : start + FUNCTION_CALL_ARG_CHUNK_SIZE],
                            "sequence_number": next_seq(),
                        },
                    )
            else:
                yield format_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": fc_item_id,
                        "output_index": fc_output_index,
                        "delta": "",
                        "sequence_number": next_seq(),
                    },
                )

            yield format_sse_event(
                "response.function_call_arguments.done",
                {
                    "item_id": fc_item_id,
                    "output_index": fc_output_index,
                    "name": normalized["name"],
                    "arguments": tool_args,
                    "sequence_number": next_seq(),
                },
            )

            done_item = _function_call_item(
                fc_item_id,
                normalized["call_id"],
                normalized["name"],
                tool_args,
                status="completed",
            )
            yield format_sse_event(
                "response.output_item.done",
                {
                    "output_index": fc_output_index,
                    "item": done_item,
                    "sequence_number": next_seq(),
                },
            )
            output_items.append(done_item)
            fc_output_index += 1

        completed_status = "incomplete" if content_was_truncated else "completed"
        completed_response = _base_response(
            response_id,
            model,
            created_at,
            completed_status,
            output=output_items,
            usage=usage,
        )
        if content_was_truncated:
            completed_response["incomplete_details"] = {"reason": "max_output_tokens"}

        yield format_sse_event(
            "response.completed",
            {
                "response": completed_response,
                "sequence_number": next_seq(),
            },
        )

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit) on Responses stream")
        streaming_error_occurred = True
    except Exception as e:
        streaming_error_occurred = True
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(
            f"Error during Responses streaming: [{error_type}] {error_msg}",
            exc_info=True,
        )
        if created_emitted:
            try:
                yield format_sse_event(
                    "response.failed",
                    {
                        "response": _base_response(
                            response_id,
                            model,
                            created_at,
                            "failed",
                            error={"code": error_type, "message": error_msg},
                        ),
                        "sequence_number": next_seq(),
                    },
                )
            except Exception:
                logger.debug("Failed to emit response.failed after stream error")
        raise
    finally:
        try:
            await response.aclose()
        except Exception as close_error:
            logger.debug(f"Error closing response: {close_error}")

        if streaming_error_occurred:
            logger.debug("Responses streaming completed with error")
        else:
            logger.debug("Responses streaming completed successfully")


async def stream_kiro_to_responses(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    echo_reasoning_items: Optional[List[Dict[str, Any]]] = None,
    emit_reasoning_summary: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Convert Kiro stream to Responses API SSE (no first-token retry).

    Retry is handled by stream_with_first_token_retry.
    """
    async for chunk in stream_kiro_to_responses_internal(
        client,
        response,
        model,
        model_cache,
        auth_manager,
        request_messages=request_messages,
        request_tools=request_tools,
        echo_reasoning_items=echo_reasoning_items,
        emit_reasoning_summary=emit_reasoning_summary,
    ):
        yield chunk


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    echo_reasoning_items: Optional[List[Dict[str, Any]]] = None,
    emit_reasoning_summary: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Responses streaming with automatic retry on first-token timeout.

    Same signature shape as streaming_openai.stream_with_first_token_retry
    so routes_responses can mirror the OpenAI account loop.
    """

    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        return HTTPException(
            status_code=status_code,
            detail=f"Upstream API error: {error_text}",
        )

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        return HTTPException(
            status_code=504,
            detail=(
                f"Model did not respond within {timeout}s after {retries} attempts. "
                "Please try again."
            ),
        )

    async def stream_processor(response: httpx.Response) -> AsyncGenerator[str, None]:
        async for chunk in stream_kiro_to_responses_internal(
            client,
            response,
            model,
            model_cache,
            auth_manager,
            first_token_timeout=first_token_timeout,
            request_messages=request_messages,
            request_tools=request_tools,
            echo_reasoning_items=echo_reasoning_items,
            emit_reasoning_summary=emit_reasoning_summary,
        ):
            yield chunk

    async for chunk in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        initial_response=initial_response,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield chunk


async def collect_stream_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    echo_reasoning_items: Optional[List[Dict[str, Any]]] = None,
    emit_reasoning_summary: bool = True,
) -> dict:
    """
    Collect a full non-streaming Responses JSON object from the Kiro stream.

    Consumes stream_kiro_to_responses and returns the `response` payload from
    response.completed (or a reconstructed object if that event is missing).
    Useful for stream=false and unit tests.
    """
    final_response: Optional[dict] = None
    failed_response: Optional[dict] = None

    async for chunk_str in stream_kiro_to_responses(
        client,
        response,
        model,
        model_cache,
        auth_manager,
        request_messages=request_messages,
        request_tools=request_tools,
        echo_reasoning_items=echo_reasoning_items,
        emit_reasoning_summary=emit_reasoning_summary,
    ):
        if "data:" not in chunk_str:
            continue
        # Support both "event: ...\ndata: ..." and bare "data: ..."
        data_line = None
        for line in chunk_str.splitlines():
            if line.startswith("data:"):
                data_line = line[len("data:") :].strip()
                break
        if not data_line or data_line == "[DONE]":
            continue
        try:
            event = json.loads(data_line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "response.completed" and "response" in event:
            final_response = event["response"]
        elif event_type == "response.failed" and "response" in event:
            failed_response = event["response"]

    if final_response is not None:
        return final_response
    if failed_response is not None:
        return failed_response

    # Fallback empty completed response (should not happen on happy path)
    return _base_response(
        generate_response_id(),
        model,
        int(time.time()),
        "completed",
        output=[],
        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )

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
- response.output_item.added (message / function_call, status=in_progress)
- response.output_text.delta / response.output_text.done
- response.function_call_arguments.delta / response.function_call_arguments.done
- response.output_item.done (message / function_call)
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
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    FAKE_REASONING_HANDLING,
    FIRST_TOKEN_MAX_RETRIES,
    FIRST_TOKEN_TIMEOUT,
)
from kiro.parsers import deduplicate_tool_calls, parse_bracket_tool_calls
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
) -> AsyncGenerator[str, None]:
    """
    Internal generator: Kiro stream → Responses API SSE.

    Raises FirstTokenTimeoutError if the first token is not received in time
    (for stream_with_first_token_retry).
    """
    response_id = generate_response_id()
    message_item_id = generate_message_item_id()
    created_at = int(time.time())
    sequence_number = 0

    metering_data = None
    context_usage_percentage = None
    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream: List[Dict[str, Any]] = []
    message_item_started = False
    streaming_error_occurred = False
    created_emitted = False

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

    try:
        # Defer response.created until the first Kiro event so first-token
        # timeout can retry before any SSE bytes reach the client.
        async for event in parse_kiro_stream(response, first_token_timeout):
            if not created_emitted:
                yield emit_created()
                yield emit_in_progress()

            if event.type == "content" and event.content:
                full_content += event.content

                if not message_item_started:
                    # Optional handshake for message item; Codex primarily
                    # needs deltas + output_item.done for text.
                    yield format_sse_event(
                        "response.output_item.added",
                        {
                            "output_index": 0,
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
                        "output_index": 0,
                        "content_index": 0,
                        "delta": event.content,
                        "sequence_number": next_seq(),
                    },
                )

            elif event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content
                # Responses API has native reasoning items; for fake reasoning
                # we only surface thinking as text when configured that way.
                if FAKE_REASONING_HANDLING == "include_as_text":
                    full_content += event.thinking_content
                    if not message_item_started:
                        yield format_sse_event(
                            "response.output_item.added",
                            {
                                "output_index": 0,
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
                            "output_index": 0,
                            "content_index": 0,
                            "delta": event.thinking_content,
                            "sequence_number": next_seq(),
                        },
                    )
                # else: accumulate for token counting only (as_reasoning_content)

            elif event.type == "tool_use" and event.tool_use:
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
                                yield format_sse_event(
                                    "response.output_item.added",
                                    {
                                        "output_index": 0,
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
                                        "output_index": 0,
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

        received_usage = metering_data is not None
        received_context_usage = context_usage_percentage is not None
        stream_completed_normally = received_usage or received_context_usage

        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = deduplicate_tool_calls(tool_calls_from_stream + bracket_tool_calls)

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

        if content_was_truncated:
            status = "incomplete"
        elif all_tool_calls:
            status = "completed"  # function calls are a normal completed turn
        else:
            status = "completed"

        output_tokens = count_tokens(full_content + full_thinking_content)
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
        output_index = 0

        # Close message item if we streamed any text
        if message_item_started or (full_content and not all_tool_calls):
            if not message_item_started and full_content:
                yield format_sse_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
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
                    "output_index": output_index,
                    "content_index": 0,
                    "text": full_content,
                    "sequence_number": next_seq(),
                },
            )
            msg_item = _message_item(message_item_id, full_content, status="completed")
            yield format_sse_event(
                "response.output_item.done",
                {
                    "output_index": output_index,
                    "item": msg_item,
                    "sequence_number": next_seq(),
                },
            )
            output_items.append(msg_item)
            output_index += 1

        # Emit function_call items (added → arg deltas → args done → item done)
        for tc in all_tool_calls:
            normalized = _normalize_tool_call(tc)
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
                    "output_index": output_index,
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
                            "output_index": output_index,
                            "delta": tool_args[start : start + FUNCTION_CALL_ARG_CHUNK_SIZE],
                            "sequence_number": next_seq(),
                        },
                    )
            else:
                yield format_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": fc_item_id,
                        "output_index": output_index,
                        "delta": "",
                        "sequence_number": next_seq(),
                    },
                )

            yield format_sse_event(
                "response.function_call_arguments.done",
                {
                    "item_id": fc_item_id,
                    "output_index": output_index,
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
                    "output_index": output_index,
                    "item": done_item,
                    "sequence_number": next_seq(),
                },
            )
            output_items.append(done_item)
            output_index += 1

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

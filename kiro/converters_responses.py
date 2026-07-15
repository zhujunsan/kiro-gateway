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
Converters for transforming OpenAI Responses API format to Kiro format.

Adapter layer: Responses input/tools/instructions/reasoning → unified format →
converters_core.build_kiro_payload.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import (
    ResponsesFunctionTool,
    ResponsesRequest,
    validate_responses_request,
)
from kiro.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload as core_build_kiro_payload,
    prepare_tool_name_for_kiro,
    sanitize_tool_use_id,
)
from kiro.converters_openai import reasoning_effort_to_budget


# ==================================================================================================
# Content helpers
# ==================================================================================================

def _extract_responses_text(content: Any) -> str:
    """
    Extract text from Responses message content.

    Supports:
    - plain string
    - list of parts with type input_text / output_text / text
    - None
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("input_text", "output_text", "text"):
                parts.append(item.get("text", "") or "")
            elif "text" in item and item_type not in (
                "input_image",
                "input_file",
                "refusal",
            ):
                parts.append(item.get("text", "") or "")
        return "".join(parts)
    return str(content)


def _extract_function_call_output_text(output: Any) -> str:
    """Extract text from function_call_output.output (string or content list)."""
    text = _extract_responses_text(output)
    return text if text else "(empty result)"


# ==================================================================================================
# Input → UnifiedMessage
# ==================================================================================================

def _flush_pending_tool_calls(
    pending_calls: List[Dict[str, Any]],
    processed: List[UnifiedMessage],
) -> None:
    if not pending_calls:
        return
    processed.append(UnifiedMessage(
        role="assistant",
        content="",
        tool_calls=pending_calls.copy(),
    ))
    pending_calls.clear()


def _flush_pending_tool_results(
    pending_results: List[Dict[str, Any]],
    processed: List[UnifiedMessage],
) -> None:
    if not pending_results:
        return
    processed.append(UnifiedMessage(
        role="user",
        content="",
        tool_results=pending_results.copy(),
    ))
    pending_results.clear()


def convert_responses_input_to_unified(
    input_data: Union[str, List[Any]],
    instructions: Optional[str] = None,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert Responses ``instructions`` + ``input`` to unified messages.

    Mapping:
    - ``instructions`` → system prompt (plus system/developer message roles)
    - ``message`` / EasyInputMessage → user or assistant
    - ``function_call`` → assistant ``tool_calls`` (``call_id`` preserved as id)
    - ``function_call_output`` → user ``tool_results`` (``call_id`` → tool_use_id)
    - ``reasoning`` → ignored (accepted for multi-turn continuity)

    Args:
        input_data: Responses input (string or list of items)
        instructions: Optional top-level instructions (system prompt)

    Returns:
        Tuple of (system_prompt, unified_messages)

    Raises:
        ValueError: On unsupported input items (via validate helpers)
    """
    from kiro.models_responses import validate_responses_input

    validate_responses_input(input_data)

    system_parts: List[str] = []
    if instructions:
        system_parts.append(instructions)

    # Normalize string input to a single user message item
    if isinstance(input_data, str):
        items: List[Any] = [{"type": "message", "role": "user", "content": input_data}]
    else:
        items = input_data

    processed: List[UnifiedMessage] = []
    pending_tool_calls: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []
    total_tool_calls = 0
    total_tool_results = 0

    for item in items:
        if isinstance(item, str):
            _flush_pending_tool_calls(pending_tool_calls, processed)
            _flush_pending_tool_results(pending_tool_results, processed)
            processed.append(UnifiedMessage(role="user", content=item))
            continue

        if not isinstance(item, dict):
            # validate_responses_input should have caught this
            raise ValueError(f"Unsupported input item type: {type(item).__name__}")

        item_type = item.get("type")

        # EasyInputMessage without type
        if item_type is None and "role" in item:
            item_type = "message"

        if item_type == "reasoning":
            # Accept and ignore (summary / encrypted_content not forwarded)
            continue

        if item_type == "function_call":
            _flush_pending_tool_results(pending_tool_results, processed)
            call_id = sanitize_tool_use_id(item.get("call_id") or item.get("id") or "")
            pending_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
            total_tool_calls += 1
            continue

        if item_type == "function_call_output":
            _flush_pending_tool_calls(pending_tool_calls, processed)
            call_id = sanitize_tool_use_id(item.get("call_id") or "")
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": _extract_function_call_output_text(item.get("output")),
            })
            total_tool_results += 1
            continue

        if item_type == "message":
            _flush_pending_tool_calls(pending_tool_calls, processed)
            _flush_pending_tool_results(pending_tool_results, processed)

            role = item.get("role") or "user"
            content_text = _extract_responses_text(item.get("content"))

            if role in ("system", "developer"):
                if content_text:
                    system_parts.append(content_text)
                continue

            if role not in ("user", "assistant"):
                # Treat unknown roles as user text for forward compatibility
                role = "user"

            processed.append(UnifiedMessage(role=role, content=content_text))
            continue

        raise ValueError(
            f"Unsupported input item type '{item_type}'. "
            f"Supported: message, function_call, function_call_output, reasoning."
        )

    _flush_pending_tool_calls(pending_tool_calls, processed)
    _flush_pending_tool_results(pending_tool_results, processed)

    system_prompt = "\n".join(p.strip() for p in system_parts if p and str(p).strip()).strip()

    if total_tool_calls > 0 or total_tool_results > 0:
        logger.debug(
            f"Converted Responses input: {len(items)} items → "
            f"{len(processed)} messages, {total_tool_calls} tool_calls, "
            f"{total_tool_results} tool_results"
        )

    return system_prompt, processed


# ==================================================================================================
# Tools → UnifiedTool
# ==================================================================================================

def _unified_tool_from_function_dict(tool_dict: Dict[str, Any]) -> Optional[UnifiedTool]:
    """Build UnifiedTool from a flat or Chat Completions nested function dict."""
    from kiro.models_responses import _function_tool_name

    function = tool_dict.get("function")
    if isinstance(function, dict) and function.get("name"):
        return UnifiedTool(
            name=function["name"],
            description=function.get("description"),
            input_schema=function.get("parameters"),
        )

    name = _function_tool_name(tool_dict)
    if not name:
        return None

    return UnifiedTool(
        name=name,
        description=tool_dict.get("description"),
        input_schema=tool_dict.get("parameters"),
    )


def convert_responses_tools_to_unified(
    tools: Optional[List[ResponsesFunctionTool]],
) -> Optional[List[UnifiedTool]]:
    """
    Convert Responses tools to unified format.

    Supports:
    1. Responses flat: ``{"type": "function", "name": "...", "parameters": {...}}``
    2. Chat Completions nested: ``{"type": "function", "function": {"name": ...}}``
    3. Codex ``type: namespace`` wrappers — nested function tools are expanded
       with lossless qualified names ``{namespace}__{local_name}`` (truncated to
       Kiro's 64-char limit with reverse mapping for streaming restore)
    4. Built-in / unknown wrapper types (``web_search``, etc.) — skipped with log

    Exact duplicate tool names (same final Kiro name) are still collapsed: the
    first occurrence is kept and later ones are skipped with a log. Same local
    names across different namespaces are kept as distinct qualified names.

    Raises:
        ValueError: Malformed function / namespace entries.
    """
    from kiro.models_responses import (
        NAMESPACE_TOOL_TYPE,
        UNSUPPORTED_BUILTIN_TOOL_TYPES,
        _tool_as_dict,
        validate_responses_tools,
    )

    validate_responses_tools(tools)

    if not tools:
        return None

    unified_tools: List[UnifiedTool] = []
    seen_names: set[str] = set()

    def _append_unique(unified: UnifiedTool, source: str) -> bool:
        """Keep first tool per name; skip later duplicates (Bedrock TOOL_DUPLICATE)."""
        if unified.name in seen_names:
            logger.info(
                f"Skipping duplicate tool name {unified.name!r} from {source} "
                f"(keeping first occurrence)"
            )
            return False
        seen_names.add(unified.name)
        unified_tools.append(unified)
        return True

    for i, tool in enumerate(tools):
        tool_dict = _tool_as_dict(tool)
        tool_type = tool_dict.get("type") or "function"

        if tool_type == NAMESPACE_TOOL_TYPE:
            ns_name = tool_dict.get("name") or f"tools[{i}]"
            nested = tool_dict.get("tools") or []
            if not isinstance(nested, list):
                continue
            expanded = 0
            for nested_tool in nested:
                if not isinstance(nested_tool, dict):
                    if hasattr(nested_tool, "model_dump"):
                        nested_tool = nested_tool.model_dump()
                    else:
                        continue
                nested_type = nested_tool.get("type") or "function"
                if nested_type != "function":
                    logger.debug(
                        f"Skipping non-function tool inside namespace "
                        f"'{ns_name}': type={nested_type!r}"
                    )
                    continue
                unified = _unified_tool_from_function_dict(nested_tool)
                if unified is None:
                    logger.warning(
                        f"Skipping invalid function inside namespace '{ns_name}': "
                        f"no name found"
                    )
                    continue
                # Lossless qualification: keep all namespace tools even when
                # local names collide (e.g. MCP apps each expose _get_profile).
                qualified = f"{ns_name}__{unified.name}"
                unified.name = prepare_tool_name_for_kiro(qualified)
                if _append_unique(unified, f"namespace '{ns_name}'"):
                    expanded += 1
            logger.debug(
                f"Expanded namespace tool '{ns_name}' → {expanded} function tool(s)"
            )
            continue

        if tool_type == "function":
            unified = _unified_tool_from_function_dict(tool_dict)
            if unified is None:
                logger.warning("Skipping invalid Responses tool: no name found")
                continue
            _append_unique(unified, f"tools[{i}] function")
            continue

        # Built-ins and other Codex-only wrappers: strip, do not fail the request.
        reason = (
            "built-in"
            if tool_type in UNSUPPORTED_BUILTIN_TOOL_TYPES
            else "unknown wrapper"
        )
        logger.info(
            f"Ignoring Responses {reason} tool type={tool_type!r} at tools[{i}] "
            f"(name={tool_dict.get('name')!r}); only function tools are forwarded"
        )

    return unified_tools if unified_tools else None


# ==================================================================================================
# Thinking configuration
# ==================================================================================================

def extract_thinking_config_from_responses(request: ResponsesRequest) -> ThinkingConfig:
    """
    Extract thinking configuration from Responses ``reasoning.effort``.

    Mirrors :func:`kiro.converters_openai.extract_thinking_config_from_openai`:
    - missing effort → enabled with default budget
    - ``none`` → disabled
    - otherwise → percentage budget from ``max_output_tokens`` (fallback 4096)
    """
    effort = request.reasoning.effort if request.reasoning else None

    if not effort:
        return ThinkingConfig(enabled=True, budget_tokens=None)

    if effort == "none":
        return ThinkingConfig(enabled=False, budget_tokens=None)

    max_tokens = request.max_output_tokens or 4096
    budget = reasoning_effort_to_budget(max_tokens, effort)

    logger.debug(
        f"Extracted thinking config from Responses: reasoning.effort='{effort}', "
        f"max_output_tokens={max_tokens}, budget={budget}"
    )

    return ThinkingConfig(enabled=True, budget_tokens=budget)


# ==================================================================================================
# Main entry point
# ==================================================================================================

def build_kiro_payload_from_responses(
    request_data: ResponsesRequest,
    conversation_id: str,
    profile_arn: str,
) -> dict:
    """
    Build complete Kiro API payload from a Responses API request.

    Validates unsupported input items (ValueError → HTTP 400), expands
    ``namespace`` tools / strips built-ins, converts to unified messages/tools,
    then calls converters_core.build_kiro_payload.

    Args:
        request_data: Parsed ResponsesRequest
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Payload dictionary for POST to Kiro API

    Raises:
        ValueError: Unsupported input/tools, or no messages to send
    """
    validate_responses_request(request_data)

    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        instructions=request_data.instructions,
    )
    unified_tools = convert_responses_tools_to_unified(request_data.tools)
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"Converting Responses request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, "
        f"tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, "
        f"thinking_budget={thinking_config.budget_tokens}"
    )

    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )
    return result.payload

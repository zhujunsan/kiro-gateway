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
Pydantic models for OpenAI Responses API (POST /v1/responses).

Uses a loose input shape (string or list of dicts) plus explicit validation
helpers that raise ValueError for route handlers to map to HTTP 400.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# Input item types we accept (reasoning content/summary is ignored downstream).
SUPPORTED_INPUT_ITEM_TYPES = frozenset({
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
})

# Built-in / hosted tools that are not client function tools.
UNSUPPORTED_BUILTIN_TOOL_TYPES = frozenset({
    "web_search",
    "web_search_preview",
    "web_search_2025_08_26",
    "web_search_preview_2025_03_11",
    "file_search",
    "computer",
    "computer_use",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "local_shell",
    "shell",
    "mcp",
    "custom",
})


class ResponsesReasoning(BaseModel):
    """
    Reasoning configuration for Responses API.

    Attributes:
        effort: Reasoning effort level (maps to thinking budget)
        summary: Optional summary preference (accepted, ignored)
    """
    effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    summary: Optional[Any] = None

    model_config = {"extra": "allow"}


class ResponsesFunctionTool(BaseModel):
    """
    Function tool in Responses API flat format.

    Responses uses top-level name/description/parameters (unlike Chat Completions
    nested ``function`` object). Nested Chat Completions shape is also accepted
    for convenience.
    """
    type: str = "function"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    # Chat Completions nested shape compatibility
    function: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    """
    Request body for OpenAI Responses API (``POST /v1/responses``).

    ``input`` is intentionally loose (string or list of dicts). Call
    :func:`validate_responses_request` (or the converter entry point) to
    reject unsupported items / built-in tools with a 400-ready ValueError.
    """
    model: str
    input: Union[str, List[Any]]
    instructions: Optional[str] = None
    tools: Optional[List[ResponsesFunctionTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    stream: bool = False
    reasoning: Optional[ResponsesReasoning] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Accepted for compatibility; store / previous_response_id are not implemented.
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


def _describe_item_type(item: Any) -> str:
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type:
            return str(item_type)
        if "role" in item:
            return "message (role-only EasyInputMessage)"
        return "dict without type/role"
    return type(item).__name__


def validate_responses_input_item(item: Any, index: Optional[int] = None) -> None:
    """
    Validate a single Responses ``input`` item.

    Raises:
        ValueError: If the item type is unsupported (400-ready message).
    """
    loc = f"input[{index}]" if index is not None else "input item"

    if isinstance(item, str):
        return

    if not isinstance(item, dict):
        raise ValueError(
            f"Unsupported {loc}: expected object or string, got {_describe_item_type(item)}"
        )

    item_type = item.get("type")

    # EasyInputMessage / message without explicit type — treat as message when role present.
    if item_type is None:
        if "role" in item:
            return
        raise ValueError(
            f"Unsupported {loc}: missing 'type' (and no 'role'). "
            f"Supported item types: {', '.join(sorted(SUPPORTED_INPUT_ITEM_TYPES))}."
        )

    if item_type in SUPPORTED_INPUT_ITEM_TYPES:
        return

    raise ValueError(
        f"Unsupported {loc} type '{item_type}'. "
        f"Supported item types: {', '.join(sorted(SUPPORTED_INPUT_ITEM_TYPES))} "
        f"(message, function_call, function_call_output, reasoning). "
        f"Built-in tool items are not supported."
    )


def validate_responses_input(input_data: Union[str, List[Any], None]) -> None:
    """
    Validate Responses ``input`` field (string or list of items).

    Raises:
        ValueError: On empty list or unsupported items.
    """
    if input_data is None:
        raise ValueError("input is required")

    if isinstance(input_data, str):
        return

    if not isinstance(input_data, list):
        raise ValueError(
            f"input must be a string or array of items, got {type(input_data).__name__}"
        )

    if len(input_data) == 0:
        raise ValueError("input array must not be empty")

    for i, item in enumerate(input_data):
        validate_responses_input_item(item, index=i)


def validate_responses_tools(tools: Optional[List[Any]]) -> None:
    """
    Validate Responses ``tools`` — only ``type: function`` is supported.

    Raises:
        ValueError: If a built-in / non-function tool is present (400-ready).
    """
    if not tools:
        return

    for i, tool in enumerate(tools):
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()
        elif isinstance(tool, dict):
            tool_dict = tool
        else:
            raise ValueError(
                f"tools[{i}]: expected object, got {type(tool).__name__}"
            )

        tool_type = tool_dict.get("type") or "function"

        if tool_type == "function":
            # Flat Responses shape or nested Chat Completions shape
            name = tool_dict.get("name")
            function = tool_dict.get("function")
            if not name and isinstance(function, dict):
                name = function.get("name")
            if not name:
                raise ValueError(
                    f"tools[{i}]: function tool requires 'name' "
                    f"(or function.name in Chat Completions shape)"
                )
            continue

        if tool_type in UNSUPPORTED_BUILTIN_TOOL_TYPES or tool_type != "function":
            raise ValueError(
                f"Unsupported tool type '{tool_type}' at tools[{i}]. "
                f"Only function tools are supported "
                f"(built-in tools such as web_search, local_shell are not available)."
            )


def validate_responses_request(request: ResponsesRequest) -> None:
    """
    Run all Responses request checks that should map to HTTP 400.

    Raises:
        ValueError: On unsupported input items or non-function tools.
    """
    validate_responses_input(request.input)
    validate_responses_tools(request.tools)

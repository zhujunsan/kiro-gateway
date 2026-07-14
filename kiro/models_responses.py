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
# These are ignored (stripped) during conversion rather than rejecting the
# whole request — Codex and other clients often send them alongside functions.
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

# Codex / Responses wrapper that groups nested function tools.
# Nested ``tools`` are expanded into flat function tools for Kiro.
NAMESPACE_TOOL_TYPE = "namespace"


def _tool_as_dict(tool: Any) -> Dict[str, Any]:
    """Normalize a tool model or dict to a plain dict."""
    if hasattr(tool, "model_dump"):
        return tool.model_dump()
    if isinstance(tool, dict):
        return tool
    raise TypeError(f"expected tool object or dict, got {type(tool).__name__}")


def _function_tool_name(tool_dict: Dict[str, Any]) -> Optional[str]:
    """Resolve function tool name from flat or Chat Completions nested shape."""
    name = tool_dict.get("name")
    if name:
        return str(name)
    function = tool_dict.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return None


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
    reject unsupported input items with a 400-ready ValueError. Built-in
    tools are stripped during conversion; ``namespace`` tools are expanded.
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


def _validate_function_tool_dict(tool_dict: Dict[str, Any], loc: str) -> None:
    """Ensure a function tool has a usable name (flat or nested shape)."""
    if not _function_tool_name(tool_dict):
        raise ValueError(
            f"{loc}: function tool requires 'name' "
            f"(or function.name in Chat Completions shape)"
        )


def validate_responses_tools(tools: Optional[List[Any]]) -> None:
    """
    Validate Responses ``tools`` for conversion readiness.

    Accepted:
    - ``type: function`` (must have a name)
    - ``type: namespace`` (Codex wrapper; nested function tools are validated)

    Built-in tools (``web_search``, ``local_shell``, …) and other unknown
    wrapper types are **not** rejected here — converters strip them so Codex
    requests still succeed as long as function tools remain.

    Raises:
        ValueError: Malformed function / namespace entries (400-ready).
    """
    if not tools:
        return

    for i, tool in enumerate(tools):
        try:
            tool_dict = _tool_as_dict(tool)
        except TypeError:
            raise ValueError(
                f"tools[{i}]: expected object, got {type(tool).__name__}"
            ) from None

        tool_type = tool_dict.get("type") or "function"

        if tool_type == "function":
            _validate_function_tool_dict(tool_dict, f"tools[{i}]")
            continue

        if tool_type == NAMESPACE_TOOL_TYPE:
            nested = tool_dict.get("tools")
            if nested is None:
                continue
            if not isinstance(nested, list):
                raise ValueError(
                    f"tools[{i}]: namespace tool 'tools' must be an array"
                )
            for j, nested_tool in enumerate(nested):
                try:
                    nested_dict = _tool_as_dict(nested_tool)
                except TypeError:
                    raise ValueError(
                        f"tools[{i}].tools[{j}]: expected object, "
                        f"got {type(nested_tool).__name__}"
                    ) from None
                nested_type = nested_dict.get("type") or "function"
                if nested_type == "function":
                    _validate_function_tool_dict(
                        nested_dict, f"tools[{i}].tools[{j}]"
                    )
            continue

        # Built-ins / unknown wrappers: tolerate (stripped during conversion).
        continue


def validate_responses_request(request: ResponsesRequest) -> None:
    """
    Run all Responses request checks that should map to HTTP 400.

    Raises:
        ValueError: On unsupported input items or malformed function tools.
    """
    validate_responses_input(request.input)
    validate_responses_tools(request.tools)

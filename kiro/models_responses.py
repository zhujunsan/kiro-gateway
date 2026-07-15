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
helpers that raise ValueError / ResponsesRequestError for route handlers to
map to HTTP 400 or 422.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


# Input item types we accept.
# ``reasoning`` is kept for multi-turn shape: converters echo id+summary stubs
# and never forward encrypted_content to Kiro.
SUPPORTED_INPUT_ITEM_TYPES = frozenset({
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    # From POST /v1/responses/compact — plaintext stub in encrypted_content.
    "compaction",
    # Codex remote-compaction v2 transient marker (ignored by converters).
    "compaction_trigger",
})

# Values that disable reasoning summary text emission (reasoning item may still exist).
REASONING_SUMMARY_DISABLED = frozenset({False, "none", "null"})

# Built-in / hosted tools that are not client function tools.
# Mixed with function tools → stripped (reported via unsupported_features).
# Hosted-only → HTTP 422 hosted_tools_not_supported (do not silently succeed).
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

# tool_choice string modes accepted by Responses API.
TOOL_CHOICE_STRING_MODES = frozenset({"none", "auto", "required"})


class ResponsesRequestError(ValueError):
    """
    Responses request error with HTTP status mapping for route handlers.

    Default status is 400 (bad request). Use :class:`ResponsesUnprocessableError`
    (or status_code=422) for unsupported-but-recognized features.
    """

    status_code: int = 400
    code: str = "invalid_request"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class ResponsesUnprocessableError(ResponsesRequestError):
    """Recognized but unsupported Responses feature → HTTP 422."""

    status_code: int = 422
    code: str = "unprocessable_entity"


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
        summary: Summary preference. ``none``/false disables summary text
            emission; ``auto``/``concise``/``detailed``/omitted emit summary
            from Kiro thinking when available. Concise/detailed also scale
            the thinking budget proportion.
    """
    effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    summary: Optional[Any] = None

    model_config = {"extra": "allow"}


def should_emit_reasoning_summary(summary: Optional[Any]) -> bool:
    """
    Whether to emit reasoning summary text for this request.

    ``none`` / false / ``\"null\"`` → False. Missing / auto / concise /
    detailed → True (thinking text is reused as summary).
    """
    if summary is None:
        return True
    if isinstance(summary, str) and summary.strip().lower() in REASONING_SUMMARY_DISABLED:
        return False
    if summary in REASONING_SUMMARY_DISABLED:
        return False
    return True


def reasoning_summary_budget_factor(summary: Optional[Any]) -> float:
    """
    Scale factor applied to thinking budget when ``reasoning.summary`` is set.

    ``concise`` → 0.5, ``detailed`` → 1.25, ``auto``/other/missing → 1.0.
    """
    if not isinstance(summary, str):
        return 1.0
    key = summary.strip().lower()
    if key == "concise":
        return 0.5
    if key == "detailed":
        return 1.25
    return 1.0


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



class ResponsesTextFormat(BaseModel):
    """
    Structured output format for Responses ``text.format``.

    Supported:
    - ``text`` — free-form (default / no-op)
    - ``json_object`` — require a JSON object
    - ``json_schema`` — require JSON conforming to ``schema``
    """
    type: str = "text"
    name: Optional[str] = None
    description: Optional[str] = None
    # OpenAI wire key is ``schema``; alias avoids BaseModel.schema clash.
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None

    model_config = {"extra": "allow", "populate_by_name": True}


class ResponsesText(BaseModel):
    """Responses ``text`` configuration (structured outputs)."""
    format: Optional[ResponsesTextFormat] = None

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
    # Thinking-budget basis only. Kiro has no hard output-token cap API;
    # see extract_thinking_config_from_responses / streaming incomplete_details.
    max_output_tokens: Optional[int] = None
    # Explicit values → 400 sampling_not_supported. Omit / null = OK.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Structured outputs (json_schema / json_object).
    text: Optional[ResponsesText] = None
    # Store / previous_response_id: in-memory chaining (see response_store.py).
    # OpenAI-ish: store when ``store`` is not explicitly false (omit/true → store).
    # Missing previous_response_id → HTTP 400. background=true → 400.
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None
    # background mode is not supported (no async job runner).
    background: Optional[bool] = None
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
        f"(message, function_call, function_call_output, reasoning, "
        f"compaction, compaction_trigger). "
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

    Built-in / hosted tools (``web_search``, ``local_shell``, …) and other
    unknown wrappers are **not** rejected here — converters apply hosted
    policy (422 if hosted-only; strip + report if mixed with functions).

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

        # Built-ins / unknown wrappers: tolerate (hosted policy in converters).
        continue


def is_hosted_tool_type(tool_type: Optional[str]) -> bool:
    """Return True if ``tool_type`` is a known OpenAI hosted / built-in tool."""
    if not tool_type:
        return False
    return tool_type in UNSUPPORTED_BUILTIN_TOOL_TYPES


def _namespace_has_function_tools(tool_dict: Dict[str, Any]) -> bool:
    nested = tool_dict.get("tools")
    if not isinstance(nested, list):
        return False
    for nested_tool in nested:
        try:
            nested_dict = _tool_as_dict(nested_tool)
        except TypeError:
            continue
        nested_type = nested_dict.get("type") or "function"
        if nested_type == "function" and _function_tool_name(nested_dict):
            return True
    return False


def classify_responses_tools(
    tools: Optional[Sequence[Any]],
) -> Tuple[List[Any], List[str], List[str]]:
    """
    Split Responses ``tools`` into keepable vs hosted/unsupported.

    Returns:
        Tuple of:
        - tools to keep (``function`` + ``namespace`` entries)
        - hosted / built-in tool types found (for unsupported_features)
        - unknown non-function wrapper types found
    """
    if not tools:
        return [], [], []

    keep: List[Any] = []
    hosted_types: List[str] = []
    unknown_types: List[str] = []

    for tool in tools:
        try:
            tool_dict = _tool_as_dict(tool)
        except TypeError:
            continue

        tool_type = tool_dict.get("type") or "function"

        if tool_type == "function" or tool_type == NAMESPACE_TOOL_TYPE:
            keep.append(tool)
            continue

        if is_hosted_tool_type(tool_type):
            if tool_type not in hosted_types:
                hosted_types.append(str(tool_type))
            continue

        type_name = str(tool_type)
        if type_name not in unknown_types:
            unknown_types.append(type_name)

    return keep, hosted_types, unknown_types


def collect_function_tool_names(tools: Optional[Sequence[Any]]) -> List[str]:
    """
    Collect client-facing function tool names from flat + namespace tools.

    Does not apply namespace qualification — that is owned by converters.
    Order follows first-seen occurrence; duplicates are skipped.
    """
    if not tools:
        return []

    names: List[str] = []
    seen: set[str] = set()

    def _add(name: Optional[str]) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        names.append(name)

    for tool in tools:
        try:
            tool_dict = _tool_as_dict(tool)
        except TypeError:
            continue

        tool_type = tool_dict.get("type") or "function"

        if tool_type == "function":
            _add(_function_tool_name(tool_dict))
            continue

        if tool_type != NAMESPACE_TOOL_TYPE:
            continue

        nested = tool_dict.get("tools")
        if not isinstance(nested, list):
            continue
        for nested_tool in nested:
            try:
                nested_dict = _tool_as_dict(nested_tool)
            except TypeError:
                continue
            nested_type = nested_dict.get("type") or "function"
            if nested_type == "function":
                _add(_function_tool_name(nested_dict))

    return names


def collect_hosted_tool_names(tools: Optional[Sequence[Any]]) -> List[str]:
    """Collect names declared on hosted / built-in tool entries."""
    if not tools:
        return []

    names: List[str] = []
    seen: set[str] = set()
    for tool in tools:
        try:
            tool_dict = _tool_as_dict(tool)
        except TypeError:
            continue
        tool_type = tool_dict.get("type") or "function"
        if not is_hosted_tool_type(tool_type):
            continue
        name = tool_dict.get("name")
        if name is None:
            # Hosted tools often use type as the selectable name.
            name = tool_type
        name_str = str(name)
        if name_str not in seen:
            seen.add(name_str)
            names.append(name_str)
    return names


def parse_responses_tool_choice(
    tool_choice: Optional[Union[str, Dict[str, Any]]],
) -> Tuple[str, Optional[str]]:
    """
    Normalize ``tool_choice`` to ``(mode, function_name)``.

    Modes: ``none`` | ``auto`` | ``required`` | ``function``.

    Raises:
        ResponsesRequestError: Invalid shape (400).
        ResponsesUnprocessableError: Hosted tool selected (422).
    """
    if tool_choice is None:
        return "auto", None

    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in TOOL_CHOICE_STRING_MODES:
            return mode, None
        raise ResponsesRequestError(
            f"Unsupported tool_choice string '{tool_choice}'. "
            f"Expected one of: {', '.join(sorted(TOOL_CHOICE_STRING_MODES))}.",
            code="invalid_tool_choice",
        )

    if not isinstance(tool_choice, dict):
        raise ResponsesRequestError(
            f"tool_choice must be a string or object, got {type(tool_choice).__name__}",
            code="invalid_tool_choice",
        )

    choice_type = tool_choice.get("type")
    if not choice_type:
        raise ResponsesRequestError(
            "tool_choice object requires 'type'",
            code="invalid_tool_choice",
        )

    choice_type_str = str(choice_type)

    if choice_type_str == "function":
        name = tool_choice.get("name")
        if not name and isinstance(tool_choice.get("function"), dict):
            name = tool_choice["function"].get("name")
        if not name:
            raise ResponsesRequestError(
                "tool_choice type=function requires 'name'",
                code="invalid_tool_choice",
            )
        return "function", str(name)

    if choice_type_str in TOOL_CHOICE_STRING_MODES:
        # Some clients send {"type": "auto"} etc.
        return choice_type_str, None

    if is_hosted_tool_type(choice_type_str):
        raise ResponsesUnprocessableError(
            f"tool_choice type '{choice_type_str}' selects a hosted tool, "
            f"which is not supported by this gateway.",
            code="hosted_tools_not_supported",
        )

    raise ResponsesRequestError(
        f"Unsupported tool_choice type '{choice_type_str}'. "
        f"Expected none/auto/required or type=function.",
        code="invalid_tool_choice",
    )


def validate_responses_tool_choice(
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    tools: Optional[Sequence[Any]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Validate ``tool_choice`` and that a named function exists among tools.

    Raises:
        ResponsesRequestError: Invalid / unknown function name (400).
        ResponsesUnprocessableError: Hosted tool selected by type or name (422).
    """
    mode, function_name = parse_responses_tool_choice(tool_choice)

    if mode != "function" or not function_name:
        return mode, function_name

    hosted_names = set(collect_hosted_tool_names(tools))
    if function_name in hosted_names or is_hosted_tool_type(function_name):
        raise ResponsesUnprocessableError(
            f"tool_choice function '{function_name}' refers to a hosted tool, "
            f"which is not supported by this gateway.",
            code="hosted_tools_not_supported",
        )

    function_names = collect_function_tool_names(tools)
    if function_name not in function_names:
        available = ", ".join(function_names) if function_names else "(none)"
        raise ResponsesRequestError(
            f"tool_choice function '{function_name}' not found in tools. "
            f"Available function tools: {available}.",
            code="invalid_tool_choice",
        )

    return mode, function_name


def validate_responses_sampling_params(request: ResponsesRequest) -> None:
    """
    Reject explicit ``temperature`` / ``top_p`` (Kiro cannot honor sampling).

    Omit or JSON ``null`` is allowed. Any concrete numeric value → HTTP 400
    with code ``sampling_not_supported`` (do not silently ignore).

    Raises:
        ResponsesRequestError: When temperature or top_p is explicitly set.
    """
    bad: List[str] = []
    if request.temperature is not None:
        bad.append("temperature")
    if request.top_p is not None:
        bad.append("top_p")
    if not bad:
        return
    names = " and ".join(bad)
    raise ResponsesRequestError(
        f"Sampling parameter(s) not supported by this gateway: {names}. "
        f"Omit them (or pass null); Kiro does not expose temperature/top_p controls.",
        code="sampling_not_supported",
        status_code=400,
    )



def resolve_responses_text_format(
    text: Optional[Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Normalize ``text.format`` to ``(type, schema, name)``.

    Returns ``(None, None, None)`` when absent or type=text.
    Accepts ResponsesText model or a raw dict.
    """
    if text is None:
        return None, None, None

    if hasattr(text, "model_dump"):
        text_dict = text.model_dump(by_alias=True)
    elif isinstance(text, dict):
        text_dict = text
    else:
        return None, None, None

    fmt = text_dict.get("format")
    if fmt is None:
        return None, None, None
    if hasattr(fmt, "model_dump"):
        fmt = fmt.model_dump(by_alias=True)
    if not isinstance(fmt, dict):
        return None, None, None

    fmt_type = fmt.get("type") or "text"
    if isinstance(fmt_type, str):
        fmt_type = fmt_type.strip().lower()
    else:
        fmt_type = "text"

    if fmt_type in ("", "text"):
        return None, None, None

    if fmt_type not in ("json_object", "json_schema"):
        raise ResponsesRequestError(
            f"Unsupported text.format.type '{fmt_type}'. "
            f"Supported: text, json_object, json_schema.",
            code="invalid_text_format",
        )

    schema = fmt.get("schema")
    if schema is None:
        schema = fmt.get("schema_")
    if fmt_type == "json_schema" and not isinstance(schema, dict):
        raise ResponsesRequestError(
            "text.format.type=json_schema requires a 'schema' object",
            code="invalid_text_format",
        )

    name = fmt.get("name")
    return fmt_type, (schema if isinstance(schema, dict) else None), (
        str(name) if name else None
    )


def validate_store_and_previous(
    store: Optional[bool],
    previous_response_id: Optional[str],
    *,
    background: Optional[bool] = None,
) -> None:
    """
    Validate store / previous_response_id / background fields.

    Raises:
        ResponsesRequestError: Unsupported background=true.
    """
    if background is True:
        raise ResponsesRequestError(
            "background=true is not supported by this gateway "
            "(responses are processed synchronously).",
            code="not_supported",
        )
    # store=false + previous_response_id is allowed (OpenAI-ish): look up prior,
    # but do not persist the new response.
    _ = store
    _ = previous_response_id


def validate_responses_request(request: ResponsesRequest) -> None:
    """
    Run Responses request checks that map to HTTP 400 / 422.

    Raises:
        ValueError / ResponsesRequestError: Unsupported input, sampling, or bad tools.
        ResponsesUnprocessableError: Hosted tool_choice (422).
    """
    validate_store_and_previous(
        request.store,
        request.previous_response_id,
        background=request.background,
    )
    validate_responses_input(request.input)
    validate_responses_tools(request.tools)
    validate_responses_tool_choice(request.tool_choice, request.tools)
    resolve_responses_text_format(request.text)
    validate_responses_sampling_params(request)

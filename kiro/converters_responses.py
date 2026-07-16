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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import (
    ResponsesFunctionTool,
    ResponsesRequest,
    ResponsesRequestError,
    ResponsesUnprocessableError,
    classify_responses_tools,
    reasoning_summary_budget_factor,
    resolve_responses_text_format,
    should_emit_reasoning_summary,
    validate_responses_request,
    validate_responses_tool_choice,
)
from kiro.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload as core_build_kiro_payload,
    extract_images_from_content,
    prepare_tool_name_for_kiro,
    sanitize_tool_use_id,
)
from kiro.converters_openai import reasoning_effort_to_budget


# System constraints used because Kiro has no native tool_choice /
# parallel_tool_calls controls.
_TOOL_CHOICE_REQUIRED_PROMPT = (
    "You must call at least one tool/function before responding with plain text."
)
_TOOL_CHOICE_NAMED_PROMPT = (
    "You must call the tool/function named '{name}' before responding with plain text. "
    "Do not call any other tool."
)
_TOOL_CHOICE_NONE_PROMPT = (
    "Do not call any tools or functions. Respond with plain text only."
)
_PARALLEL_TOOL_CALLS_FALSE_PROMPT = (
    "You must call at most one tool/function in this turn. "
    "Do not make parallel or multiple tool calls."
)
_JSON_OBJECT_PROMPT = (
    "You must respond with a single valid JSON object only. "
    "Do not wrap it in markdown code fences or add commentary."
)
_JSON_SCHEMA_PROMPT = (
    "You must respond with a single valid JSON value that conforms to this JSON Schema"
    "{name_part}. Do not wrap it in markdown code fences or add commentary.\n"
    "JSON Schema:\n{schema_json}"
)


@dataclass
class ResponsesBuildResult:
    """Kiro payload plus Responses-side metadata for the route layer."""

    payload: dict
    unsupported_features: List[str] = field(default_factory=list)
    parallel_tool_calls: Optional[bool] = None
    tool_choice_mode: str = "auto"
    # Structured outputs: json_object / json_schema (for non-stream validate).
    text_format_type: Optional[str] = None
    text_format_schema: Optional[Dict[str, Any]] = None
    text_format_name: Optional[str] = None
    # Input reasoning items stubbed for multi-turn echo (no encrypted_content).
    echo_reasoning_items: List[Dict[str, Any]] = field(default_factory=list)
    # Whether to emit reasoning summary text from thinking (request.reasoning.summary).
    emit_reasoning_summary: bool = True


# ==================================================================================================
# Reasoning input stubs (multi-turn echo)
# ==================================================================================================

def stub_reasoning_item_from_input(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an echo/stub reasoning item for multi-turn continuity.

    Preserves ``id`` and ``summary`` shape. Never copies ``encrypted_content``
    (must not be forwarded to Kiro or re-emitted).
    """
    import uuid

    item_id = item.get("id")
    if not item_id:
        item_id = f"rs_{uuid.uuid4().hex}"

    summary = item.get("summary")
    if not isinstance(summary, list):
        summary = []

    stub: Dict[str, Any] = {
        "id": str(item_id),
        "type": "reasoning",
        "summary": summary,
    }
    status = item.get("status")
    if status is not None:
        stub["status"] = status
    return stub


def extract_reasoning_input_stubs(
    input_data: Union[str, List[Any], None],
) -> List[Dict[str, Any]]:
    """
    Collect reasoning stubs from Responses ``input`` (for response output echo).

    Does not forward anything to Kiro — callers still skip reasoning in
    :func:`convert_responses_input_to_unified`.
    """
    if not isinstance(input_data, list):
        return []

    stubs: List[Dict[str, Any]] = []
    for item in input_data:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            stubs.append(stub_reasoning_item_from_input(item))
    return stubs


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


def _responses_image_url(item: Dict[str, Any]) -> str:
    """Resolve image URL string from an input_image content part."""
    image_url = item.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "")
    return ""


def _extract_responses_images(content: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract base64 images from Responses message content into unified format.

    Uses converters_core.extract_images_from_content after normalizing
    ``input_image`` parts to OpenAI ``image_url`` / Anthropic ``image`` shapes.

    Raises:
        ResponsesRequestError: URL images, file_id, or input_file (HTTP 400).
    """
    if not isinstance(content, list):
        return None

    normalized: List[Dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "input_file":
            raise ResponsesRequestError(
                "input_file content parts are not supported by this gateway. "
                "Inline text or base64 input_image data URLs instead.",
                code="input_file_not_supported",
            )

        if item_type != "input_image":
            # Pass through image_url / image blocks for extract_images_from_content.
            if item_type in ("image_url", "image"):
                normalized.append(item)
            continue

        if item.get("file_id"):
            raise ResponsesRequestError(
                "input_image with file_id is not supported; "
                "send a base64 data URL in image_url.",
                code="input_image_file_id_not_supported",
            )

        url = _responses_image_url(item)
        source = item.get("source")

        if url.startswith(("http://", "https://")):
            raise ResponsesRequestError(
                "input_image URL references are not supported; "
                "send a base64 data URL (data:image/...;base64,...) in image_url.",
                code="input_image_url_not_supported",
            )

        if url.startswith("data:"):
            normalized.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
            continue

        # Anthropic-like inline source on input_image
        if isinstance(source, dict) and source.get("type") == "base64" and source.get("data"):
            normalized.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source.get("media_type") or "image/jpeg",
                    "data": source["data"],
                },
            })
            continue

        # Raw base64 in image_url without data: prefix (treat as jpeg)
        if url and not url.startswith(("http://", "https://", "data:")):
            normalized.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.get("media_type") or "image/jpeg",
                    "data": url,
                },
            })
            continue

        raise ResponsesRequestError(
            "input_image requires a base64 data URL in image_url "
            "(URL fetch and file_id are not supported).",
            code="input_image_invalid",
        )

    if not normalized:
        return None

    images = extract_images_from_content(normalized)
    return images or None


def apply_text_format_constraint(
    system_prompt: str,
    text: Any = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Inject system instructions for ``text.format`` json_object / json_schema.

    Returns:
        (system_prompt, format_type, schema, name)
    """
    fmt_type, schema, name = resolve_responses_text_format(text)
    if not fmt_type:
        return system_prompt or "", None, None, None

    if fmt_type == "json_object":
        prompt = _append_system_constraint(system_prompt, _JSON_OBJECT_PROMPT)
        return prompt, fmt_type, None, name

    # json_schema
    import json as _json
    schema_json = _json.dumps(schema or {}, ensure_ascii=False, indent=2)
    name_part = f" named '{name}'" if name else ""
    constraint = _JSON_SCHEMA_PROMPT.format(
        name_part=name_part,
        schema_json=schema_json,
    )
    prompt = _append_system_constraint(system_prompt, constraint)
    return prompt, fmt_type, schema, name


def _strip_json_fences(text: str) -> str:
    """Strip optional markdown ```json fences around model output."""
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def collect_responses_output_text(response: Dict[str, Any]) -> str:
    """Collect concatenated output_text from a Responses JSON object."""
    parts: List[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(part.get("text") or "")
    return "".join(parts)


def validate_responses_json_output(
    output_text: str,
    format_type: Optional[str],
    schema: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Validate non-stream model text against text.format.

    Returns an error message string if invalid, else None.
    Full JSON Schema validation is not required — parseability (+ object/array
    top-level type when declared) is enough without adding jsonschema dep.
    """
    if not format_type or format_type not in ("json_object", "json_schema"):
        return None

    import json as _json

    cleaned = _strip_json_fences(output_text)
    if not cleaned:
        return "Model output was empty; expected JSON for text.format"

    try:
        parsed = _json.loads(cleaned)
    except _json.JSONDecodeError as exc:
        return f"Model output is not valid JSON required by text.format: {exc}"

    if format_type == "json_object" and not isinstance(parsed, dict):
        return "Model output must be a JSON object (text.format.type=json_object)"

    if format_type == "json_schema" and isinstance(schema, dict):
        expected = schema.get("type")
        if expected == "object" and not isinstance(parsed, dict):
            return "Model output must be a JSON object per text.format.schema"
        if expected == "array" and not isinstance(parsed, list):
            return "Model output must be a JSON array per text.format.schema"

    return None


def apply_text_format_validation_to_response(
    response: Dict[str, Any],
    format_type: Optional[str],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    For non-stream responses: if text.format requires JSON and output is
    invalid, set ``status=failed`` with an error object. Tool-call turns
    (function_call in output) are skipped.
    """
    if not format_type or format_type not in ("json_object", "json_schema"):
        return response

    output = response.get("output") or []
    if any(isinstance(i, dict) and i.get("type") == "function_call" for i in output):
        return response

    err = validate_responses_json_output(
        collect_responses_output_text(response),
        format_type,
        schema,
    )
    if not err:
        return response

    updated = dict(response)
    updated["status"] = "failed"
    updated["error"] = {
        "code": "invalid_json_output",
        "message": err,
    }
    return updated



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


def _tool_arguments_score(arguments: Any) -> int:
    """Score function-call arguments when choosing between duplicate calls.

    Args:
        arguments: Raw Responses function-call arguments.

    Returns:
        Zero for empty arguments, otherwise the serialized argument length.
    """
    if arguments is None:
        return 0
    if isinstance(arguments, str):
        stripped = arguments.strip()
        return 0 if stripped in ("", "{}") else len(stripped)
    if isinstance(arguments, dict):
        return 0 if not arguments else len(json.dumps(arguments, sort_keys=True))
    return len(str(arguments))


def _deduplicate_responses_tool_items(items: List[Any]) -> List[Any]:
    """Remove duplicate Responses calls and outputs by sanitized call ID.

    Kiro requires every toolUseId and matching toolResult to be unique. Some
    upstream streams repeat a completed call with empty arguments; Responses
    clients then replay both copies on the next turn. Calls retain the copy
    with the most complete arguments, while outputs retain the first result.

    Args:
        items: Validated Responses input items.

    Returns:
        Input items with duplicate function calls and outputs removed.
    """
    result: List[Any] = []
    call_positions: Dict[str, int] = {}
    output_ids: set[str] = set()
    duplicate_calls = 0
    duplicate_outputs = 0

    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue

        item_type = item.get("type")
        if item_type not in ("function_call", "function_call_output"):
            result.append(item)
            continue

        raw_call_id = item.get("call_id") or (
            item.get("id") if item_type == "function_call" else ""
        )
        call_id = sanitize_tool_use_id(raw_call_id or "")
        if not call_id:
            result.append(item)
            continue

        if item_type == "function_call":
            existing_position = call_positions.get(call_id)
            if existing_position is None:
                call_positions[call_id] = len(result)
                result.append(item)
                continue

            duplicate_calls += 1
            existing = result[existing_position]
            if _tool_arguments_score(item.get("arguments")) > _tool_arguments_score(
                existing.get("arguments")
            ):
                result[existing_position] = item
            continue

        if call_id in output_ids:
            duplicate_outputs += 1
            continue
        output_ids.add(call_id)
        result.append(item)

    if duplicate_calls or duplicate_outputs:
        logger.warning(
            "Deduplicated Responses replay items: "
            f"{duplicate_calls} function_call(s), "
            f"{duplicate_outputs} function_call_output(s)"
        )

    return result


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
    - ``reasoning`` → skipped for Kiro (use :func:`extract_reasoning_input_stubs`
      for multi-turn echo; encrypted_content is never forwarded)

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
        items = _deduplicate_responses_tool_items(input_data)

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
            # Do not forward to Kiro (including encrypted_content).
            # Echo stubs are collected separately via extract_reasoning_input_stubs.
            continue

        if item_type == "compaction_trigger":
            # Codex remote-compaction v2 transient marker — not forwarded.
            continue

        if item_type == "compaction":
            # Local compact stub: surface plaintext encrypted_content as a
            # synthetic user summary so Kiro still sees prior context.
            _flush_pending_tool_calls(pending_tool_calls, processed)
            _flush_pending_tool_results(pending_tool_results, processed)
            stub_text = item.get("encrypted_content") or ""
            if stub_text:
                processed.append(
                    UnifiedMessage(
                        role="user",
                        content=f"[compacted prior context]\n{stub_text}",
                    )
                )
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
            _flush_pending_tool_results(pending_tool_results, processed)

            role = item.get("role") or "user"
            raw_content = item.get("content")
            content_text = _extract_responses_text(raw_content)
            images = _extract_responses_images(raw_content)

            if role in ("system", "developer"):
                if content_text:
                    system_parts.append(content_text)
                continue

            if role not in ("user", "assistant"):
                # Treat unknown roles as user text for forward compatibility
                role = "user"

            if role == "assistant" and pending_tool_calls:
                processed.append(UnifiedMessage(
                    role="assistant",
                    content=content_text,
                    tool_calls=pending_tool_calls.copy(),
                    images=images,
                ))
                pending_tool_calls.clear()
                continue

            _flush_pending_tool_calls(pending_tool_calls, processed)
            processed.append(UnifiedMessage(
                role=role,
                content=content_text,
                images=images,
            ))
            continue

        raise ValueError(
            f"Unsupported input item type '{item_type}'. "
            f"Supported: message, function_call, function_call_output, "
            f"reasoning, compaction, compaction_trigger."
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


def _append_system_constraint(system_prompt: str, constraint: str) -> str:
    """Append a unique system constraint paragraph."""
    constraint = (constraint or "").strip()
    if not constraint:
        return system_prompt or ""
    existing = (system_prompt or "").strip()
    if constraint in existing:
        return existing
    if not existing:
        return constraint
    return f"{existing}\n\n{constraint}"


def prepare_responses_tools_policy(
    tools: Optional[Sequence[Any]],
) -> Tuple[Optional[List[Any]], List[str]]:
    """
    Hosted-tools policy for Responses ``tools``.

    - Hosted-only (no function / namespace functions) → 422
      ``hosted_tools_not_supported`` (never silent success).
    - Mixed hosted + function → keep function/namespace tools; report stripped
      types via ``unsupported_features``.
    - Function-only / empty → unchanged.

    Returns:
        (tools_for_conversion, unsupported_features)
    """
    if not tools:
        return None, []

    keep, hosted_types, unknown_types = classify_responses_tools(tools)
    stripped_types = list(hosted_types) + list(unknown_types)
    unsupported = [f"tool:{t}" for t in stripped_types]

    # Detect whether any keepable entry can yield function tools.
    from kiro.models_responses import (
        NAMESPACE_TOOL_TYPE,
        _namespace_has_function_tools,
        _tool_as_dict,
        collect_function_tool_names,
    )

    has_functions = bool(collect_function_tool_names(keep))
    if not has_functions:
        # Namespace entries without nested functions still count as "no functions".
        for tool in keep or []:
            try:
                tool_dict = _tool_as_dict(tool)
            except TypeError:
                continue
            if (tool_dict.get("type") or "function") == NAMESPACE_TOOL_TYPE:
                if _namespace_has_function_tools(tool_dict):
                    has_functions = True
                    break

    if stripped_types and not has_functions:
        types_label = ", ".join(stripped_types)
        raise ResponsesUnprocessableError(
            f"Hosted/built-in tools are not supported by this gateway "
            f"(received only: {types_label}). Provide at least one function tool.",
            code="hosted_tools_not_supported",
        )

    if stripped_types:
        logger.info(
            f"Stripping unsupported Responses tools {stripped_types}; "
            f"forwarding {len(keep or [])} function/namespace tool entr(y/ies)"
        )

    return (list(keep) if keep else None), unsupported


def apply_responses_tool_choice(
    system_prompt: str,
    unified_tools: Optional[List[UnifiedTool]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    *,
    raw_tools: Optional[Sequence[Any]] = None,
) -> Tuple[str, Optional[List[UnifiedTool]], str]:
    """
    Apply Responses ``tool_choice`` without native Kiro support.

    - ``none``: omit tools + system constraint (no tool calls).
    - ``auto``: unchanged.
    - ``required``: system constraint to call at least one tool.
    - named function: system constraint to call that tool (validated first).

    Returns:
        (system_prompt, tools, mode)
    """
    mode, function_name = validate_responses_tool_choice(tool_choice, raw_tools)

    if mode == "none":
        prompt = _append_system_constraint(system_prompt, _TOOL_CHOICE_NONE_PROMPT)
        return prompt, None, mode

    if mode == "auto":
        return system_prompt or "", unified_tools, mode

    if not unified_tools:
        # required / named without tools cannot be satisfied meaningfully.
        from kiro.models_responses import ResponsesRequestError

        raise ResponsesRequestError(
            f"tool_choice={mode!r} requires at least one function tool",
            code="invalid_tool_choice",
        )

    if mode == "required":
        prompt = _append_system_constraint(system_prompt, _TOOL_CHOICE_REQUIRED_PROMPT)
        return prompt, unified_tools, mode

    # named function — accept client local name or namespace-qualified Kiro name
    assert function_name is not None
    matched = next(
        (
            t for t in unified_tools
            if t.name == function_name or t.name.endswith(f"__{function_name}")
        ),
        None,
    )
    if matched is None:
        from kiro.models_responses import ResponsesRequestError

        available = ", ".join(sorted(t.name for t in unified_tools)) or "(none)"
        raise ResponsesRequestError(
            f"tool_choice function '{function_name}' not found in converted tools. "
            f"Available: {available}.",
            code="invalid_tool_choice",
        )

    # Prompt uses the Kiro-facing name the model actually sees.
    prompt = _append_system_constraint(
        system_prompt,
        _TOOL_CHOICE_NAMED_PROMPT.format(name=matched.name),
    )
    return prompt, unified_tools, mode


def apply_parallel_tool_calls_constraint(
    system_prompt: str,
    parallel_tool_calls: Optional[bool],
) -> str:
    """
    When ``parallel_tool_calls is False``, add a system constraint to call
    at most one tool. ``True`` / ``None`` leave the prompt unchanged.
    """
    if parallel_tool_calls is False:
        return _append_system_constraint(
            system_prompt, _PARALLEL_TOOL_CALLS_FALSE_PROMPT
        )
    return system_prompt or ""


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
    4. Built-in / unknown wrapper types (``web_search``, etc.) — hosted policy
       at the start (422 if hosted-only; otherwise stripped)

    Exact duplicate tool names (same final Kiro name) are still collapsed: the
    first occurrence is kept and later ones are skipped with a log. Same local
    names across different namespaces are kept as distinct qualified names.

    Raises:
        ValueError: Malformed function / namespace entries.
        ResponsesUnprocessableError: Hosted-only tools (422).
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

    # Hosted policy hook only — do not alter namespace naming below.
    tools, _unsupported = prepare_responses_tools_policy(tools)
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

        # Defense in depth: hosted should already be stripped by policy hook.
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
    Extract thinking configuration from Responses ``reasoning.effort`` /
    ``reasoning.summary``.

    Mirrors :func:`kiro.converters_openai.extract_thinking_config_from_openai`:
    - missing effort → enabled with default budget
    - ``none`` → disabled
    - otherwise → percentage budget from ``max_output_tokens`` (fallback 4096)

    When ``reasoning.summary`` is ``concise`` / ``detailed``, the budget is
    scaled by :func:`reasoning_summary_budget_factor`. Budgets are capped by
    ``FAKE_REASONING_BUDGET_CAP``.

    Important: Kiro's generateAssistantResponse API has **no hard output-token
    cap**. ``max_output_tokens`` therefore only drives thinking-budget sizing
    here; it cannot be forwarded as an absolute stop to Kiro. Truncation is
    detected heuristically in streaming and surfaced as
    ``status=incomplete`` + ``incomplete_details.reason=max_output_tokens``.
    """
    effort = request.reasoning.effort if request.reasoning else None
    summary = request.reasoning.summary if request.reasoning else None
    summary_factor = reasoning_summary_budget_factor(summary)

    if not effort:
        # Summary alone can still size a budget when concise/detailed.
        if summary_factor != 1.0:
            max_tokens = request.max_output_tokens or 4096
            budget = int(reasoning_effort_to_budget(max_tokens, "medium") * summary_factor)
            budget = max(budget, 1)
            from kiro.config import FAKE_REASONING_BUDGET_CAP
            if FAKE_REASONING_BUDGET_CAP > 0 and budget > FAKE_REASONING_BUDGET_CAP:
                budget = FAKE_REASONING_BUDGET_CAP
            return ThinkingConfig(enabled=True, budget_tokens=budget)
        return ThinkingConfig(enabled=True, budget_tokens=None)

    if effort == "none":
        return ThinkingConfig(enabled=False, budget_tokens=None)

    max_tokens = request.max_output_tokens or 4096
    budget = reasoning_effort_to_budget(max_tokens, effort)
    if summary_factor != 1.0:
        budget = max(int(budget * summary_factor), 1)

    from kiro.config import FAKE_REASONING_BUDGET_CAP

    if FAKE_REASONING_BUDGET_CAP > 0 and budget > FAKE_REASONING_BUDGET_CAP:
        logger.debug(
            f"Responses thinking budget {budget} exceeds cap "
            f"{FAKE_REASONING_BUDGET_CAP}; using capped value"
        )
        budget = FAKE_REASONING_BUDGET_CAP

    logger.debug(
        f"Extracted thinking config from Responses: reasoning.effort='{effort}', "
        f"reasoning.summary={summary!r}, factor={summary_factor}, "
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
) -> ResponsesBuildResult:
    """
    Build complete Kiro API payload from a Responses API request.

    Validates unsupported input items (ValueError → HTTP 400), applies
    hosted-tools policy (422 if hosted-only), applies ``tool_choice`` /
    ``parallel_tool_calls`` constraints, expands ``namespace`` tools,
    converts to unified messages/tools, then calls
    converters_core.build_kiro_payload.

    Args:
        request_data: Parsed ResponsesRequest
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        ResponsesBuildResult with Kiro payload and response metadata

    Raises:
        ValueError / ResponsesRequestError: Unsupported input/tools (400)
        ResponsesUnprocessableError: Hosted-only tools / hosted tool_choice (422)
    """
    validate_responses_request(request_data)

    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        instructions=request_data.instructions,
    )
    echo_reasoning_items = extract_reasoning_input_stubs(request_data.input)
    emit_summary = should_emit_reasoning_summary(
        request_data.reasoning.summary if request_data.reasoning else None
    )

    # Hosted policy first so unsupported_features is available to routes.
    filtered_tools, unsupported_features = prepare_responses_tools_policy(
        request_data.tools
    )
    unified_tools = convert_responses_tools_to_unified(filtered_tools)

    system_prompt, unified_tools, tool_choice_mode = apply_responses_tool_choice(
        system_prompt,
        unified_tools,
        request_data.tool_choice,
        raw_tools=request_data.tools,
    )
    system_prompt = apply_parallel_tool_calls_constraint(
        system_prompt,
        request_data.parallel_tool_calls,
    )
    system_prompt, text_format_type, text_format_schema, text_format_name = (
        apply_text_format_constraint(system_prompt, request_data.text)
    )

    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"Converting Responses request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, "
        f"tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"tool_choice={tool_choice_mode}, "
        f"parallel_tool_calls={request_data.parallel_tool_calls}, "
        f"unsupported_features={unsupported_features}, "
        f"echo_reasoning_items={len(echo_reasoning_items)}, "
        f"emit_reasoning_summary={emit_summary}, "
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
    return ResponsesBuildResult(
        payload=result.payload,
        unsupported_features=unsupported_features,
        parallel_tool_calls=request_data.parallel_tool_calls,
        tool_choice_mode=tool_choice_mode,
        echo_reasoning_items=echo_reasoning_items,
        emit_reasoning_summary=emit_summary,
        text_format_type=text_format_type,
        text_format_schema=text_format_schema,
        text_format_name=text_format_name,
    )

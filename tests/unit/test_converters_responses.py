# -*- coding: utf-8 -*-

"""
Unit tests for Responses API models and converters.

Covers:
- Loose input validation (unknown items / non-function tools → ValueError)
- instructions / input / tools / reasoning.effort → unified → Kiro payload
- call_id round-trip between function_call and function_call_output
"""

import pytest

from kiro.converters_core import sanitize_tool_use_id, TOOL_USE_ID_MAX_LENGTH
from kiro.converters_responses import (
    build_kiro_payload_from_responses,
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    extract_thinking_config_from_responses,
    _extract_responses_text,
)
from kiro.models_responses import (
    ResponsesFunctionTool,
    ResponsesReasoning,
    ResponsesRequest,
    validate_responses_input,
    validate_responses_request,
    validate_responses_tools,
)


# ==================================================================================================
# Models / validation
# ==================================================================================================

class TestResponsesRequestModel:
    """Basic pydantic parsing for ResponsesRequest."""

    def test_accepts_string_input(self):
        req = ResponsesRequest(model="claude-sonnet-4-5", input="Hello")
        assert req.model == "claude-sonnet-4-5"
        assert req.input == "Hello"
        assert req.stream is False

    def test_accepts_list_input(self):
        req = ResponsesRequest(
            model="claude-sonnet-4-5",
            input=[{"type": "message", "role": "user", "content": "Hi"}],
        )
        assert isinstance(req.input, list)
        assert len(req.input) == 1

    def test_accepts_instructions_and_reasoning(self):
        req = ResponsesRequest(
            model="m",
            input="x",
            instructions="Be brief",
            reasoning=ResponsesReasoning(effort="high"),
            max_output_tokens=2048,
        )
        assert req.instructions == "Be brief"
        assert req.reasoning.effort == "high"
        assert req.max_output_tokens == 2048

    def test_accepts_flat_function_tool(self):
        req = ResponsesRequest(
            model="m",
            input="x",
            tools=[ResponsesFunctionTool(
                type="function",
                name="get_weather",
                description="Weather",
                parameters={"type": "object", "properties": {}},
            )],
        )
        assert req.tools[0].name == "get_weather"


class TestValidateResponsesInput:
    """400-ready validation for input items."""

    def test_string_input_ok(self):
        validate_responses_input("hello")

    def test_known_item_types_ok(self):
        validate_responses_input([
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "summary": []},
        ])

    def test_easy_input_message_without_type_ok(self):
        validate_responses_input([{"role": "user", "content": "hi"}])

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_responses_input([])

    def test_unknown_item_type_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported input\\[0\\] type 'web_search_call'"):
            validate_responses_input([{"type": "web_search_call", "id": "ws1"}])

    def test_unknown_item_mentions_supported_types(self):
        with pytest.raises(ValueError, match="function_call_output"):
            validate_responses_input([{"type": "local_shell_call"}])

    def test_dict_without_type_or_role_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            validate_responses_input([{"foo": "bar"}])


class TestValidateResponsesTools:
    """Function + namespace accepted; built-ins ignored (not 400)."""

    def test_function_tool_ok(self):
        validate_responses_tools([
            ResponsesFunctionTool(type="function", name="Read", parameters={}),
        ])

    def test_web_search_ignored(self):
        validate_responses_tools([{"type": "web_search"}])

    def test_local_shell_ignored(self):
        validate_responses_tools([{"type": "local_shell"}])

    def test_namespace_with_nested_functions_ok(self):
        validate_responses_tools([
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "sub-agents",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {"type": "object"},
                    },
                ],
            }
        ])

    def test_namespace_nested_function_without_name_raises(self):
        with pytest.raises(ValueError, match=r"tools\[0\]\.tools\[0\].*requires 'name'"):
            validate_responses_tools([
                {
                    "type": "namespace",
                    "name": "ns",
                    "tools": [{"type": "function"}],
                }
            ])

    def test_function_without_name_raises(self):
        with pytest.raises(ValueError, match="requires 'name'"):
            validate_responses_tools([{"type": "function"}])


# ==================================================================================================
# Content extraction
# ==================================================================================================

class TestExtractResponsesText:
    def test_string(self):
        assert _extract_responses_text("hello") == "hello"

    def test_input_text_parts(self):
        assert _extract_responses_text([
            {"type": "input_text", "text": "Hello "},
            {"type": "input_text", "text": "world"},
        ]) == "Hello world"

    def test_output_text_parts(self):
        assert _extract_responses_text([
            {"type": "output_text", "text": "Done"},
        ]) == "Done"

    def test_none(self):
        assert _extract_responses_text(None) == ""


# ==================================================================================================
# convert_responses_input_to_unified
# ==================================================================================================

class TestConvertResponsesInputToUnified:
    def test_string_input_becomes_user_message(self):
        system, msgs = convert_responses_input_to_unified("Hello")
        assert system == ""
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello"

    def test_instructions_become_system_prompt(self):
        system, msgs = convert_responses_input_to_unified(
            [{"type": "message", "role": "user", "content": "Hi"}],
            instructions="You are helpful.",
        )
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_system_and_developer_roles_merge_into_system_prompt(self):
        system, msgs = convert_responses_input_to_unified([
            {"type": "message", "role": "system", "content": "Sys."},
            {"type": "message", "role": "developer", "content": "Dev."},
            {"type": "message", "role": "user", "content": "Hi"},
        ], instructions="Top.")
        assert "Top." in system
        assert "Sys." in system
        assert "Dev." in system
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_message_with_input_text_content(self):
        system, msgs = convert_responses_input_to_unified([
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Ping"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Pong"}],
            },
        ])
        assert system == ""
        assert msgs[0].role == "user" and msgs[0].content == "Ping"
        assert msgs[1].role == "assistant" and msgs[1].content == "Pong"

    def test_easy_input_message_without_type(self):
        _, msgs = convert_responses_input_to_unified([
            {"role": "user", "content": "Hi"},
        ])
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hi"

    def test_function_call_to_assistant_tool_calls(self):
        _, msgs = convert_responses_input_to_unified([
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        ])
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].tool_calls is not None
        assert msgs[0].tool_calls[0]["id"] == "call_abc"
        assert msgs[0].tool_calls[0]["function"]["name"] == "get_weather"
        assert msgs[0].tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'

    def test_function_call_output_to_user_tool_results(self):
        _, msgs = convert_responses_input_to_unified([
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "22C",
            },
        ])
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].tool_results[0]["tool_use_id"] == "call_abc"
        assert msgs[0].tool_results[0]["content"] == "22C"

    def test_call_id_roundtrip_between_call_and_output(self):
        _, msgs = convert_responses_input_to_unified([
            {
                "type": "message",
                "role": "user",
                "content": "weather?",
            },
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_xyz",
                "output": "sunny",
            },
            {
                "type": "message",
                "role": "user",
                "content": "thanks",
            },
        ])
        assert msgs[1].role == "assistant"
        assert msgs[1].tool_calls[0]["id"] == "call_xyz"
        assert msgs[2].role == "user"
        assert msgs[2].tool_results[0]["tool_use_id"] == "call_xyz"
        assert msgs[3].role == "user"
        assert msgs[3].content == "thanks"

    def test_parallel_function_calls_merged(self):
        _, msgs = convert_responses_input_to_unified([
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "b",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "ra",
            },
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": "rb",
            },
        ])
        assert len(msgs) == 2
        assert len(msgs[0].tool_calls) == 2
        assert len(msgs[1].tool_results) == 2

    def test_reasoning_items_ignored(self):
        _, msgs = convert_responses_input_to_unified([
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "role": "user", "content": "hi"},
        ])
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    def test_unknown_item_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            convert_responses_input_to_unified([{"type": "computer_call"}])

    def test_sanitizes_call_ids(self):
        raw_id = "call_ba9Q96rddtkJMtrrtZQXHWDr\nfc_08a8627642d75eb1016a182009cd9481a2bdbf6c0ae2e7d"
        clean_id = sanitize_tool_use_id(raw_id)
        assert len(clean_id) <= TOOL_USE_ID_MAX_LENGTH
        _, msgs = convert_responses_input_to_unified([
            {
                "type": "function_call",
                "call_id": raw_id,
                "name": "Read",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": raw_id,
                "output": "file",
            },
        ])
        assert msgs[0].tool_calls[0]["id"] == clean_id
        assert msgs[1].tool_results[0]["tool_use_id"] == clean_id


# ==================================================================================================
# Tools conversion
# ==================================================================================================

class TestConvertResponsesToolsToUnified:
    def test_flat_function_tool(self):
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(
                type="function",
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"},
            ),
        ])
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].description == "Get weather"
        assert tools[0].input_schema == {"type": "object"}

    def test_nested_chat_completions_shape(self):
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(
                type="function",
                function={
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            ),
        ])
        assert tools[0].name == "Read"
        assert tools[0].description == "Read a file"

    def test_none_returns_none(self):
        assert convert_responses_tools_to_unified(None) is None

    def test_builtin_tool_stripped(self):
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(type="function", name="Read", parameters={}),
            ResponsesFunctionTool(type="web_search", name="web_search"),
        ])
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "Read"

    def test_builtin_only_returns_none(self):
        assert convert_responses_tools_to_unified([
            ResponsesFunctionTool(type="file_search", name="file_search"),
        ]) is None

    def test_namespace_expands_nested_functions(self):
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(
                type="function",
                name="exec_command",
                parameters={"type": "object"},
            ),
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Tools for spawning and managing sub-agents.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string"}},
                        },
                    },
                    {
                        "type": "function",
                        "name": "close_agent",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
            ResponsesFunctionTool(type="web_search"),
        ])
        assert tools is not None
        names = [t.name for t in tools]
        assert names == [
            "exec_command",
            "multi_agent_v1__spawn_agent",
            "multi_agent_v1__close_agent",
        ]
        assert tools[1].description == "Spawn a sub-agent"
        assert tools[1].input_schema["properties"]["prompt"]["type"] == "string"

    def test_unknown_wrapper_type_stripped(self):
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(type="function", name="f", parameters={}),
            ResponsesFunctionTool.model_validate({
                "type": "codex_mystery_wrapper",
                "name": "x",
            }),
        ])
        assert [t.name for t in tools] == ["f"]

    def test_dedupe_same_local_name_across_namespaces(self):
        """Same local name in different namespaces → distinct qualified names."""
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "mcp__codex_apps__github",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "description": "github profile",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "_search",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "mcp__codex_apps__gmail",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "description": "gmail profile (duplicate local name)",
                        "parameters": {"type": "object", "properties": {"x": {}}},
                    },
                    {
                        "type": "function",
                        "name": "_fetch",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "mcp__codex_apps__google_drive",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "description": "drive profile",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
        ])
        assert tools is not None
        names = [t.name for t in tools]
        assert names == [
            "mcp__codex_apps__github___get_profile",
            "mcp__codex_apps__github___search",
            "mcp__codex_apps__gmail___get_profile",
            "mcp__codex_apps__gmail___fetch",
            "mcp__codex_apps__google_drive___get_profile",
        ]
        assert names.count("mcp__codex_apps__github___get_profile") == 1
        assert tools[0].description == "github profile"
        assert tools[2].description == "gmail profile (duplicate local name)"
        assert tools[4].description == "drive profile"

    def test_namespace_vs_flat_function_both_kept(self):
        """Flat function and namespace-qualified name are distinct — both ship."""
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool(
                type="function",
                name="_get_profile",
                description="flat first",
                parameters={"type": "object"},
            ),
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "mcp__codex_apps__github",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "description": "ns variant",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "_list_repos",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
        ])
        assert [t.name for t in tools] == [
            "_get_profile",
            "mcp__codex_apps__github___get_profile",
            "mcp__codex_apps__github___list_repos",
        ]
        assert tools[0].description == "flat first"
        assert tools[1].description == "ns variant"

    def test_flat_after_namespace_both_kept(self):
        """Qualified namespace name and flat local name do not collide."""
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": "ns_a",
                "tools": [
                    {
                        "type": "function",
                        "name": "shared",
                        "description": "from namespace",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
            ResponsesFunctionTool(
                type="function",
                name="shared",
                description="flat later",
                parameters={"type": "object"},
            ),
        ])
        assert [t.name for t in tools] == ["ns_a__shared", "shared"]
        assert tools[0].description == "from namespace"
        assert tools[1].description == "flat later"

    def test_namespace_long_qualified_name_round_trip(self):
        """Qualified names over 64 chars are shortened with reverse mapping."""
        from kiro.converters_core import get_original_tool_name

        long_ns = "mcp__codex_apps__" + ("very_long_server_name_" * 3)
        assert len(f"{long_ns}___get_profile") > 64
        tools = convert_responses_tools_to_unified([
            ResponsesFunctionTool.model_validate({
                "type": "namespace",
                "name": long_ns,
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "parameters": {"type": "object"},
                    },
                ],
            }),
        ])
        assert tools is not None
        assert len(tools) == 1
        assert len(tools[0].name) <= 64
        expected = f"{long_ns}___get_profile"
        assert get_original_tool_name(tools[0].name) == expected


# ==================================================================================================
# Thinking config
# ==================================================================================================

class TestExtractThinkingConfigFromResponses:
    def test_no_reasoning_uses_defaults(self):
        req = ResponsesRequest(model="m", input="x")
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is True
        assert cfg.budget_tokens is None

    def test_effort_none_disables(self):
        req = ResponsesRequest(
            model="m",
            input="x",
            reasoning=ResponsesReasoning(effort="none"),
        )
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is False

    def test_effort_high_uses_max_output_tokens(self):
        req = ResponsesRequest(
            model="m",
            input="x",
            reasoning=ResponsesReasoning(effort="high"),
            max_output_tokens=4096,
        )
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is True
        assert cfg.budget_tokens == int(4096 * 0.80)

    def test_effort_without_max_output_tokens_falls_back(self):
        req = ResponsesRequest(
            model="m",
            input="x",
            reasoning=ResponsesReasoning(effort="medium"),
        )
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.budget_tokens == int(4096 * 0.50)


# ==================================================================================================
# build_kiro_payload_from_responses
# ==================================================================================================

class TestBuildKiroPayloadFromResponses:
    def test_builds_simple_payload(self):
        req = ResponsesRequest(
            model="claude-sonnet-4-5",
            input="Hello",
            instructions="Be nice",
        )
        payload = build_kiro_payload_from_responses(req, "conv-1", "arn:aws:test")
        assert "conversationState" in payload
        current = payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert "Hello" in current["content"]
        assert current["modelId"]

    def test_includes_tools_and_call_id_in_history(self):
        req = ResponsesRequest(
            model="claude-sonnet-4-5",
            input=[
                {"type": "message", "role": "user", "content": "list files"},
                {
                    "type": "function_call",
                    "call_id": "call_list",
                    "name": "list_dir",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_list",
                    "output": "a.txt\nb.txt",
                },
                {"type": "message", "role": "user", "content": "continue"},
            ],
            tools=[ResponsesFunctionTool(
                type="function",
                name="list_dir",
                description="List a directory",
                parameters={"type": "object", "properties": {}},
            )],
        )
        payload = build_kiro_payload_from_responses(req, "conv-2", "arn:aws:test")
        history = payload["conversationState"]["history"]
        tool_use_ids = [
            tu["toolUseId"]
            for turn in history
            for tu in (turn.get("assistantResponseMessage") or {}).get("toolUses") or []
        ]
        tool_result_ids = [
            tr["toolUseId"]
            for turn in history
            for tr in ((turn.get("userInputMessage") or {}).get("userInputMessageContext") or {}).get("toolResults") or []
        ]
        current_ctx = (
            payload["conversationState"]["currentMessage"]["userInputMessage"]
            .get("userInputMessageContext") or {}
        )
        tool_result_ids.extend(tr["toolUseId"] for tr in current_ctx.get("toolResults") or [])

        assert "call_list" in tool_use_ids
        assert "call_list" in tool_result_ids

    def test_rejects_unknown_input_item(self):
        req = ResponsesRequest(
            model="m",
            input=[{"type": "web_search_call", "id": "w1"}],
        )
        with pytest.raises(ValueError, match="Unsupported"):
            build_kiro_payload_from_responses(req, "c", "arn:aws:test")

    def test_strips_builtin_and_expands_namespace_tools(self):
        req = ResponsesRequest(
            model="m",
            input="hi",
            tools=[
                ResponsesFunctionTool(type="function", name="Read", parameters={}),
                ResponsesFunctionTool.model_validate({
                    "type": "namespace",
                    "name": "mcp__node_repl",
                    "tools": [
                        {
                            "type": "function",
                            "name": "js",
                            "parameters": {"type": "object"},
                        },
                    ],
                }),
                ResponsesFunctionTool(type="file_search", name="file_search"),
            ],
        )
        payload = build_kiro_payload_from_responses(req, "c", "arn:aws:test")
        ctx = (
            payload["conversationState"]["currentMessage"]["userInputMessage"]
            .get("userInputMessageContext") or {}
        )
        names = [
            t.get("toolSpecification", {}).get("name")
            for t in (ctx.get("tools") or [])
        ]
        assert "Read" in names
        assert "mcp__node_repl__js" in names
        assert "js" not in names
        assert "file_search" not in names
        assert "mcp__node_repl" not in names

    def test_validate_responses_request_wires_both_checks(self):
        req = ResponsesRequest(
            model="m",
            input=[{"type": "message", "role": "user", "content": "ok"}],
            tools=[ResponsesFunctionTool(type="function", name="f", parameters={})],
        )
        validate_responses_request(req)  # should not raise

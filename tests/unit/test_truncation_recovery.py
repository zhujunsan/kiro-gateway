# -*- coding: utf-8 -*-

"""
Unit tests for truncation_recovery.py - Synthetic message generation.

Tests cover:
- Tool truncation message generation
- Content truncation message generation
- Recovery enabled/disabled check
- Message format validation
"""

import os
from unittest.mock import patch

import pytest

from kiro.truncation_recovery import (
    should_inject_recovery,
    generate_truncation_tool_result,
    generate_truncation_user_message
)


class TestRecoveryEnabledCheck:
    """Test suite for recovery enabled/disabled check."""
    
    def test_should_inject_recovery_when_enabled(self):
        """
        What it does: Verify should_inject_recovery() returns True when enabled.
        Goal: Ensure config is respected.
        """
        with patch.dict(os.environ, {"TRUNCATION_RECOVERY": "true"}):
            from importlib import reload
            from kiro import config
            reload(config)
            
            result = should_inject_recovery()
        
        assert result is True, "Should return True when TRUNCATION_RECOVERY=true"
    
    def test_should_inject_recovery_when_disabled(self):
        """
        What it does: Verify should_inject_recovery() returns False when disabled.
        Goal: Ensure config is respected.
        """
        with patch.dict(os.environ, {"TRUNCATION_RECOVERY": "false"}):
            from importlib import reload
            from kiro import config
            reload(config)
            
            result = should_inject_recovery()
        
        assert result is False, "Should return False when TRUNCATION_RECOVERY=false"


class TestToolTruncationMessage:
    """Test suite for tool truncation message generation."""
    
    def test_generate_truncation_tool_result_format(self):
        """
        What it does: Verify synthetic tool_result message format.
        Goal: Ensure message structure is correct for both APIs.
        """
        tool_name = "write_to_file"
        tool_use_id = "tooluse_xyz123"
        truncation_info = {"size_bytes": 5000, "reason": "missing 1 closing brace"}
        
        result = generate_truncation_tool_result(tool_name, tool_use_id, truncation_info)
        
        assert isinstance(result, dict), "Should return dict"
        assert result["type"] == "tool_result", "Type should be 'tool_result'"
        assert result["tool_use_id"] == tool_use_id
        assert result["is_error"] is True, "is_error should be True"
        
        content = result["content"]
        assert isinstance(content, str), "Content should be string"
        assert len(content) > 0, "Content should not be empty"
        
        assert "[API Limitation]" in content, "Should contain [API Limitation] marker"
        assert "discarded" in content.lower(), "Should say the call was discarded"
        assert "upstream api" in content.lower(), "Should mention upstream API"
        assert "size limit" in content.lower(), "Should mention size limits"
        
        assert "if" in content.lower() or "likely" in content.lower(), "Should use conditional language"
        assert "consequence" in content.lower(), "Should explain error is consequence"
        
        assert "retrying" in content.lower(), "Should warn against retrying as-is"
        assert "split" in content.lower(), "Should advise splitting the content"
        assert "kb" in content.lower(), "Should name the concrete size limit"
    
    def test_generate_truncation_tool_result_different_tools(self):
        """
        What it does: Verify message generation works for various tool names.
        Goal: Ensure no tool-specific hardcoding.
        """
        tools = [
            ("write_to_file", "tooluse_1"),
            ("read_file", "tooluse_2"),
            ("execute_command", "tooluse_3"),
            ("search_files", "tooluse_4")
        ]
        
        for tool_name, tool_id in tools:
            result = generate_truncation_tool_result(
                tool_name=tool_name,
                tool_use_id=tool_id,
                truncation_info={"size_bytes": 1000, "reason": "test"}
            )
            
            assert result["type"] == "tool_result", f"Should work for {tool_name}"
            assert result["tool_use_id"] == tool_id, f"Should preserve tool_id for {tool_name}"
            assert "[API Limitation]" in result["content"], f"Should have marker for {tool_name}"
    
    def test_generate_truncation_tool_result_no_micro_management(self):
        """
        What it does: Verify the message names the ceiling and suggests splitting
                      without dictating how many lines or bytes per call.
        Goal: Name the limit, leave the granularity to the model.
        """
        result = generate_truncation_tool_result(
            tool_name="write_to_file",
            tool_use_id="test",
            truncation_info={"size_bytes": 5000, "reason": "test"}
        )
        
        content = result["content"].lower()
        
        forbidden_phrases = [
            "one line at a time",
            "line by line",
            "a few lines",
            "10 lines",
            "100 lines",
            "one function at a time",
        ]
        
        for phrase in forbidden_phrases:
            assert phrase not in content, f"Should NOT prescribe granularity: '{phrase}'"
        
        assert "split" in content, "Should suggest splitting"
        assert "kb" in content, "Should name the concrete limit"


class TestContentTruncationMessage:
    """Test suite for content truncation message generation."""
    
    def test_generate_truncation_user_message_format(self):
        """
        What it does: Verify synthetic user message format.
        Goal: Ensure message is appropriate for content truncation.
        """
        message = generate_truncation_user_message()
        
        assert isinstance(message, str), "Should return string"
        assert len(message) > 0, "Should not be empty"
        assert "[System Notice]" in message, "Should contain [System Notice] marker"
        assert "truncated" in message.lower(), "Should mention truncation"
        assert "api" in message.lower(), "Should mention API"
        assert "output size" in message.lower() or "size limit" in message.lower(), "Should mention size limits"
        assert "not an error on your part" in message.lower() or "not your fault" in message.lower(), \
            "Should clarify it's not model's fault"
        assert "adapt" in message.lower(), "Should suggest adaptation"
    
    def test_generate_truncation_user_message_no_micro_steps(self):
        """
        What it does: Verify message doesn't tell model to "break into steps".
        Goal: Prevent micro-step behavior that was problematic in earlier iterations.
        """
        message = generate_truncation_user_message()
        content = message.lower()
        
        forbidden_phrases = [
            "break into steps",
            "step by step",
            "one step at a time",
            "smaller steps",
            "incremental"
        ]
        
        for phrase in forbidden_phrases:
            assert phrase not in content, f"Should NOT contain micro-step trigger: '{phrase}'"
    


class TestToolResultAdvertisesSizeLimit:
    """Test suite for the concrete size limit in tool truncation messages."""

    TRUNCATION_INFO = {"size_bytes": 30, "reason": "missing 1 closing brace(s)"}

    def test_message_states_concrete_limit(self):
        """
        What it does: Verifies the advertised limit appears in the message.
        Goal: The model should resize in one step instead of guessing blindly.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        assert "50 KB" in result["content"]

    def test_message_follows_configured_limit(self):
        """
        What it does: Verifies the number tracks the config value.
        Goal: A tuned limit must not be contradicted by a stale message.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 40000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        assert "40 KB" in result["content"]
        assert "50 KB" not in result["content"]

    def test_message_states_argument_dropped_entirely(self):
        """
        What it does: Verifies the message says the argument was dropped whole.
        Goal: Prevent the model from trying to continue a partial write that
              never landed - Kiro discards the argument rather than cutting it.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        content = result["content"]
        assert "ENTIRELY" in content
        assert "nothing to continue from" in content

    def test_message_advises_splitting(self):
        """
        What it does: Verifies the message tells the model to split the content.
        Goal: Give an actionable path forward, not just a rejection.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        content = result["content"]
        assert "Split the content" in content
        assert "sequential calls" in content

    def test_message_retains_api_limitation_prefix(self):
        """
        What it does: Verifies the [API Limitation] prefix is preserved.
        Goal: The system prompt legitimizes this exact prefix; changing it would
              make the model treat the notice as prompt injection.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        assert result["content"].startswith("[API Limitation]")

    def test_message_still_flags_error_and_id(self):
        """
        What it does: Verifies structural fields survive the message rewrite.
        Goal: The tool_result must stay pairable and marked as an error.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_xyz", self.TRUNCATION_INFO
            )

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_xyz"
        assert result["is_error"] is True

    def test_zero_limit_does_not_crash(self):
        """
        What it does: Verifies a limit of 0 still produces a usable message.
        Goal: Message generation must not divide by or depend on a positive
              limit - recovery is orthogonal to whether the hint is enabled.
        """
        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 0):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        assert result["content"].startswith("[API Limitation]")
        assert result["is_error"] is True

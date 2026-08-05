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
        Test Case 2.3: Recovery enabled check (enabled)
        
        What it does: Verify should_inject_recovery() returns True when enabled
        Goal: Ensure config is respected
        """
        print("\n=== Test: Recovery enabled check (enabled) ===")
        
        # Arrange & Act
        with patch.dict(os.environ, {"TRUNCATION_RECOVERY": "true"}):
            # Need to reload config to pick up env change
            from importlib import reload
            from kiro import config
            reload(config)
            
            result = should_inject_recovery()
            print(f"TRUNCATION_RECOVERY=true → should_inject_recovery() = {result}")
        
        # Assert
        assert result is True, "Should return True when TRUNCATION_RECOVERY=true"
        
        print("✅ Test passed: Recovery enabled check works")
    
    def test_should_inject_recovery_when_disabled(self):
        """
        Test Case 2.3: Recovery enabled check (disabled)
        
        What it does: Verify should_inject_recovery() returns False when disabled
        Goal: Ensure config is respected
        """
        print("\n=== Test: Recovery enabled check (disabled) ===")
        
        # Arrange & Act
        with patch.dict(os.environ, {"TRUNCATION_RECOVERY": "false"}):
            # Need to reload config to pick up env change
            from importlib import reload
            from kiro import config
            reload(config)
            
            result = should_inject_recovery()
            print(f"TRUNCATION_RECOVERY=false → should_inject_recovery() = {result}")
        
        # Assert
        assert result is False, "Should return False when TRUNCATION_RECOVERY=false"
        
        print("✅ Test passed: Recovery disabled check works")


class TestToolTruncationMessage:
    """Test suite for tool truncation message generation."""
    
    def test_generate_truncation_tool_result_format(self):
        """
        Test Case 2.1: Generate tool truncation message
        
        What it does: Verify synthetic tool_result message format
        Goal: Ensure message structure is correct for both APIs
        """
        print("\n=== Test: Generate tool truncation message ===")
        
        # Arrange
        tool_name = "write_to_file"
        tool_use_id = "tooluse_xyz123"
        truncation_info = {"size_bytes": 5000, "reason": "missing 1 closing brace"}
        
        print(f"Generating message for tool={tool_name}, id={tool_use_id}")
        
        # Act
        result = generate_truncation_tool_result(tool_name, tool_use_id, truncation_info)
        print(f"Generated result: {result}")
        
        # Assert - Structure
        assert isinstance(result, dict), "Should return dict"
        assert result["type"] == "tool_result", "Type should be 'tool_result'"
        assert result["tool_use_id"] == tool_use_id, f"tool_use_id should be {tool_use_id}"
        assert result["is_error"] is True, "is_error should be True"
        
        # Assert - Content
        content = result["content"]
        assert isinstance(content, str), "Content should be string"
        assert len(content) > 0, "Content should not be empty"
        
        # Assert - Key phrases present
        assert "[API Limitation]" in content, "Should contain [API Limitation] marker"
        assert "discarded" in content.lower(), "Should say the call was discarded"
        assert "upstream api" in content.lower(), "Should mention upstream API"
        assert "size limit" in content.lower(), "Should mention size limits"
        
        # Assert - Universal formulation (conditional language)
        assert "if" in content.lower() or "likely" in content.lower(), "Should use conditional language"
        assert "consequence" in content.lower(), "Should explain error is consequence"
        
        # Assert - Warning against a naive retry, plus an actionable next step.
        # Kiro drops oversized arguments whole, so "adapt somehow" is not enough:
        # the model needs the concrete ceiling to resize in a single attempt.
        assert "retrying" in content.lower(), "Should warn against retrying as-is"
        assert "split" in content.lower(), "Should advise splitting the content"
        assert "kb" in content.lower(), "Should name the concrete size limit"
        
        print("✅ Test passed: Tool truncation message format correct")
    
    def test_generate_truncation_tool_result_different_tools(self):
        """
        Test Case: Generate messages for different tools
        
        What it does: Verify message generation works for various tool names
        Goal: Ensure no tool-specific hardcoding
        """
        print("\n=== Test: Generate messages for different tools ===")
        
        # Arrange
        tools = [
            ("write_to_file", "tooluse_1"),
            ("read_file", "tooluse_2"),
            ("execute_command", "tooluse_3"),
            ("search_files", "tooluse_4")
        ]
        
        # Act & Assert
        for tool_name, tool_id in tools:
            print(f"Testing tool: {tool_name}")
            result = generate_truncation_tool_result(
                tool_name=tool_name,
                tool_use_id=tool_id,
                truncation_info={"size_bytes": 1000, "reason": "test"}
            )
            
            assert result["type"] == "tool_result", f"Should work for {tool_name}"
            assert result["tool_use_id"] == tool_id, f"Should preserve tool_id for {tool_name}"
            assert "[API Limitation]" in result["content"], f"Should have marker for {tool_name}"
        
        print("✅ Test passed: Works for all tool types")
    
    def test_generate_truncation_tool_result_no_micro_management(self):
        """
        Test Case: Message avoids prescribing a chunk size
        
        What it does: Verify the message names the ceiling and suggests splitting
                      without dictating how many lines or bytes per call.
        Goal: Prescribing a step size makes the model fall into micro-steps, but
              withholding the ceiling entirely makes it guess. Name the limit,
              leave the granularity to the model.
        """
        print("\n=== Test: Message avoids micro-management ===")
        
        # Arrange
        result = generate_truncation_tool_result(
            tool_name="write_to_file",
            tool_use_id="test",
            truncation_info={"size_bytes": 5000, "reason": "test"}
        )
        
        content = result["content"].lower()
        print(f"Checking content for specific instructions...")
        
        # Assert - Must not prescribe a granularity (causes micro-steps)
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
        
        # Assert - Must still give an actionable direction with the real ceiling
        assert "split" in content, "Should suggest splitting"
        assert "kb" in content, "Should name the concrete limit"
        
        print("✅ Test passed: Names the limit without prescribing granularity")


class TestContentTruncationMessage:
    """Test suite for content truncation message generation."""
    
    def test_generate_truncation_user_message_format(self):
        """
        Test Case 2.2: Generate content truncation message
        
        What it does: Verify synthetic user message format
        Goal: Ensure message is appropriate for content truncation
        """
        print("\n=== Test: Generate content truncation message ===")
        
        # Act
        message = generate_truncation_user_message()
        print(f"Generated message: {message}")
        
        # Assert - Basic structure
        assert isinstance(message, str), "Should return string"
        assert len(message) > 0, "Should not be empty"
        
        # Assert - Key markers
        assert "[System Notice]" in message, "Should contain [System Notice] marker"
        
        # Assert - Key phrases
        assert "truncated" in message.lower(), "Should mention truncation"
        assert "api" in message.lower(), "Should mention API"
        assert "output size" in message.lower() or "size limit" in message.lower(), "Should mention size limits"
        
        # Assert - Not model's fault
        assert "not an error on your part" in message.lower() or "not your fault" in message.lower(), \
            "Should clarify it's not model's fault"
        
        # Assert - Adaptation suggestion
        assert "adapt" in message.lower(), "Should suggest adaptation"
        
        print("✅ Test passed: Content truncation message format correct")
    
    def test_generate_truncation_user_message_no_micro_steps(self):
        """
        Test Case: Message doesn't cause micro-steps
        
        What it does: Verify message doesn't tell model to "break into steps"
        Goal: Prevent micro-step behavior that was problematic in earlier iterations
        """
        print("\n=== Test: Message doesn't cause micro-steps ===")
        
        # Act
        message = generate_truncation_user_message()
        content = message.lower()
        print(f"Checking for micro-step triggers...")
        
        # Assert - Should NOT contain phrases that cause micro-steps
        forbidden_phrases = [
            "break into steps",
            "step by step",
            "one step at a time",
            "smaller steps",
            "incremental"
        ]
        
        for phrase in forbidden_phrases:
            assert phrase not in content, f"Should NOT contain micro-step trigger: '{phrase}'"
        
        print("✅ Test passed: No micro-step triggers")
    
    def test_generate_truncation_user_message_consistency(self):
        """
        Test Case: Message is consistent across calls
        
        What it does: Verify same message is generated each time
        Goal: Ensure deterministic behavior
        """
        print("\n=== Test: Message consistency ===")
        
        # Act
        message1 = generate_truncation_user_message()
        message2 = generate_truncation_user_message()
        message3 = generate_truncation_user_message()
        
        print(f"Message 1: {message1[:50]}...")
        print(f"Message 2: {message2[:50]}...")
        print(f"Message 3: {message3[:50]}...")
        
        # Assert
        assert message1 == message2 == message3, "Should generate same message each time (deterministic)"
        
        print("✅ Test passed: Message is consistent")


class TestMessageIntegration:
    """Test suite for message integration scenarios."""
    
    def test_tool_result_can_be_prepended_to_original(self):
        """
        Test Case: Tool result can be prepended to original content
        
        What it does: Verify synthetic message can be combined with original tool_result
        Goal: Ensure message format works with prepending pattern
        """
        print("\n=== Test: Tool result prepending ===")
        
        # Arrange
        synthetic = generate_truncation_tool_result(
            tool_name="write_to_file",
            tool_use_id="test",
            truncation_info={"size_bytes": 5000, "reason": "test"}
        )
        original_content = "Error: Missing required parameter 'content'"
        
        # Act - Simulate prepending (as done in routes)
        combined = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
        
        print(f"Combined length: {len(combined)} chars")
        print(f"Combined preview: {combined[:100]}...")
        
        # Assert
        assert "[API Limitation]" in combined, "Should contain synthetic message"
        assert original_content in combined, "Should contain original content"
        assert combined.index("[API Limitation]") < combined.index(original_content), \
            "Synthetic message should come before original"
        
        print("✅ Test passed: Prepending works correctly")
    
    def test_user_message_can_be_inserted_after_assistant(self):
        """
        Test Case: User message can be inserted after assistant message
        
        What it does: Verify synthetic user message works in conversation flow
        Goal: Ensure message format works with insertion pattern
        """
        print("\n=== Test: User message insertion ===")
        
        # Arrange
        assistant_message = "Here's the code you requested:\n\n```python\n# This is a very long file..."
        synthetic_user_message = generate_truncation_user_message()
        
        # Act - Simulate conversation flow
        conversation = [
            {"role": "user", "content": "Write a large file"},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": synthetic_user_message}  # Inserted
        ]
        
        print(f"Conversation has {len(conversation)} messages")
        print(f"Last message: {conversation[-1]['content'][:50]}...")
        
        # Assert
        assert len(conversation) == 3, "Should have 3 messages"
        assert conversation[-1]["role"] == "user", "Last message should be user"
        assert "[System Notice]" in conversation[-1]["content"], "Should contain system notice"
        
        print("✅ Test passed: Insertion works correctly")


class TestToolResultAdvertisesSizeLimit:
    """Test suite for the concrete size limit in tool truncation messages."""

    TRUNCATION_INFO = {"size_bytes": 30, "reason": "missing 1 closing brace(s)"}

    def test_message_states_concrete_limit(self):
        """
        What it does: Verifies the advertised limit appears in the message.
        Goal: The model should resize in one step instead of guessing blindly.
        """
        print("\n=== Test: Tool result advertises concrete limit ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        print(f"Message: {result['content'][:120]}...")
        assert "50 KB" in result["content"]

    def test_message_follows_configured_limit(self):
        """
        What it does: Verifies the number tracks the config value.
        Goal: A tuned limit must not be contradicted by a stale message.
        """
        print("\n=== Test: Tool result follows configured limit ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 40000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        print(f"Message mentions: {'40 KB' in result['content']}")
        assert "40 KB" in result["content"]
        assert "50 KB" not in result["content"]

    def test_message_states_argument_dropped_entirely(self):
        """
        What it does: Verifies the message says the argument was dropped whole.
        Goal: Prevent the model from trying to continue a partial write that
              never landed - Kiro discards the argument rather than cutting it.
        """
        print("\n=== Test: Tool result says argument dropped entirely ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        content = result["content"]
        print(f"Contains ENTIRELY: {'ENTIRELY' in content}")
        assert "ENTIRELY" in content
        assert "nothing to continue from" in content

    def test_message_advises_splitting(self):
        """
        What it does: Verifies the message tells the model to split the content.
        Goal: Give an actionable path forward, not just a rejection.
        """
        print("\n=== Test: Tool result advises splitting ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        content = result["content"]
        print(f"Contains split advice: {'Split the content' in content}")
        assert "Split the content" in content
        assert "sequential calls" in content

    def test_message_retains_api_limitation_prefix(self):
        """
        What it does: Verifies the [API Limitation] prefix is preserved.
        Goal: The system prompt legitimizes this exact prefix; changing it would
              make the model treat the notice as prompt injection.
        """
        print("\n=== Test: Tool result retains [API Limitation] prefix ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        print(f"Prefix: {result['content'][:20]}")
        assert result["content"].startswith("[API Limitation]")

    def test_message_still_flags_error_and_id(self):
        """
        What it does: Verifies structural fields survive the message rewrite.
        Goal: The tool_result must stay pairable and marked as an error.
        """
        print("\n=== Test: Tool result structure intact ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 50000):
            result = generate_truncation_tool_result(
                "Write", "call_xyz", self.TRUNCATION_INFO
            )

        print(f"Result keys: {sorted(result)}")
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_xyz"
        assert result["is_error"] is True

    def test_zero_limit_does_not_crash(self):
        """
        What it does: Verifies a limit of 0 still produces a usable message.
        Goal: Message generation must not divide by or depend on a positive
              limit - recovery is orthogonal to whether the hint is enabled.
        """
        print("\n=== Test: Tool result with zero limit ===")

        with patch("kiro.config.TOOL_ARGS_SIZE_LIMIT_BYTES", 0):
            result = generate_truncation_tool_result(
                "Write", "call_abc", self.TRUNCATION_INFO
            )

        print(f"Message generated, length={len(result['content'])}")
        assert result["content"].startswith("[API Limitation]")
        assert result["is_error"] is True

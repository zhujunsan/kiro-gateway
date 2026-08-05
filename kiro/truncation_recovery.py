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
Truncation recovery system for handling upstream Kiro API limitations.

Generates synthetic messages to inform the model about truncation.
ONLY activates when truncation is actually detected.

This module addresses Issue #56 - Kiro API truncates large tool call payloads
and content mid-stream. Since this is an upstream limitation that cannot be
prevented, we inform the model about the truncation so it can adapt its approach.
"""

from typing import Dict, Any

from loguru import logger


def should_inject_recovery() -> bool:
    """
    Check if truncation recovery is enabled.
    
    Returns:
        True if recovery should be injected, False otherwise
    """
    from kiro.config import TRUNCATION_RECOVERY
    return TRUNCATION_RECOVERY


def generate_truncation_tool_result(
    tool_name: str,
    tool_use_id: str,
    truncation_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate synthetic tool_result for truncated tool call.
    
    Message is carefully worded to:
    - Acknowledge API limitation (not model's fault)
    - State that the argument was dropped whole, so nothing can be continued
    - Name the concrete size limit, so the model resizes in one step
    - Suggest splitting without prescribing a chunk size (avoids micro-steps)
    
    Args:
        tool_name: Name of the truncated tool
        tool_use_id: ID of the truncated tool call
        truncation_info: Diagnostic information about truncation
    
    Returns:
        Synthetic tool_result in unified format
    
    Example:
        >>> generate_truncation_tool_result("Write", "call_123", {"size_bytes": 5000, "reason": "missing 2 closing braces"})
        {'type': 'tool_result', 'tool_use_id': 'call_123', 'content': '[API Limitation] ...', 'is_error': True}
    """
    from kiro.config import TOOL_ARGS_SIZE_LIMIT_BYTES

    limit_kb = TOOL_ARGS_SIZE_LIMIT_BYTES // 1000

    content = (
        "[API Limitation] Your tool call was discarded by the upstream API because its "
        f"arguments exceeded the size limit of roughly {limit_kb} KB.\n\n"
        "Note that the argument was dropped ENTIRELY rather than cut short, so no partial "
        "result was applied and there is nothing to continue from. If the tool result below "
        "shows an error or unexpected behavior, that is a CONSEQUENCE of this, not the root cause.\n\n"
        f"Retrying the same call will fail the same way. Split the content into portions under "
        f"{limit_kb} KB and send them as several sequential calls."
    )

    logger.debug(
        f"Generated synthetic tool_result for truncated tool '{tool_name}' "
        f"(id={tool_use_id}, {truncation_info['size_bytes']} bytes, {truncation_info['reason']}, "
        f"advertised limit {limit_kb} KB)"
    )
    
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": True
    }


def generate_truncation_user_message() -> str:
    """
    Generate synthetic user message for content truncation.
    
    Message is carefully worded to:
    - Acknowledge it's not model's fault
    - Suggest adaptation without specific instructions
    - NOT tell model to "break into steps" (causes micro-steps)
    
    Returns:
        Synthetic user message text
    
    Example:
        >>> generate_truncation_user_message()
        '[System Notice] Your previous response was truncated...'
    """
    return (
        "[System Notice] Your previous response was truncated by the API due to "
        "output size limitations. This is not an error on your part. "
        "If you need to continue, please adapt your approach rather than repeating the same output."
    )

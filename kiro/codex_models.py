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
Codex-shaped model stubs for ``GET /v1/models`` dual-compat.

Codex deserializes ``ModelsResponse { models: Vec<ModelInfo> }`` and fails
with ``missing field `models` `` on the standard OpenAI ``{object,data}``
list. Stub fields below are the minimum required for decode success
(see openai/codex ``protocol/src/openai_models.rs`` ModelInfo).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_codex_model_info(
    model_id: str,
    *,
    priority: int = 0,
    description: str = "Model via Kiro Gateway",
) -> Dict[str, Any]:
    """
    Build a minimal Codex ``ModelInfo`` dict for ``model_id``.

    Required (no serde default) fields are always set. Optional / defaulted
    fields are included with conservative stubs so older/newer Codex builds
    that tighten decoding still succeed.
    """
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": description,
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": (
            "You are a coding agent running through Kiro Gateway. "
            "Follow the user's instructions carefully."
        ),
        "model_messages": None,
        "include_skills_usage_instructions": False,
        # Codex 0.144 required field name:
        "supports_reasoning_summaries": False,
        # Newer Codex builds renamed this; harmless extra for forward compat.
        "supports_reasoning_summary_parameter": True,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "context_window": None,
        "max_context_window": None,
        "auto_compact_token_limit": None,
        "comp_hash": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
        "use_responses_lite": False,
        "auto_review_model_override": None,
        "tool_mode": None,
        "multi_agent_version": None,
    }


def build_codex_models_list(model_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Build Codex ``models`` array; earlier ids get higher priority (lower number)."""
    return [
        build_codex_model_info(model_id, priority=index)
        for index, model_id in enumerate(model_ids)
    ]

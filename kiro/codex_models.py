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

Capability fields (``input_modalities``, context windows, reasoning levels,
image detail) are filled per canonical model id after alias resolution via
``MODEL_ALIASES``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from kiro.config import MODEL_ALIASES

# Codex ReasoningLevelInfo shape: {"effort": str, "description": str}
_REASONING_DESCRIPTIONS: Dict[str, str] = {
    "low": "Low reasoning effort — faster, lighter deliberation.",
    "medium": "Medium reasoning effort — balanced depth and latency.",
    "high": "High reasoning effort — deeper analysis for harder tasks.",
    "xhigh": "Extra-high reasoning effort — extended deliberation.",
    "max": "Maximum reasoning effort — fullest available deliberation.",
}


def _reasoning_levels(*efforts: str) -> List[Dict[str, str]]:
    return [
        {"effort": effort, "description": _REASONING_DESCRIPTIONS[effort]}
        for effort in efforts
    ]


_TEXT = ["text"]
_TEXT_IMAGE = ["text", "image"]

_LEVELS_OPUS_NEW = _reasoning_levels("low", "medium", "high", "xhigh", "max")
_LEVELS_OPUS_46 = _reasoning_levels("low", "medium", "high", "max")
_LEVELS_SONNET_46 = _reasoning_levels("low", "medium", "high", "max")
_LEVELS_GPT_56 = _reasoning_levels("low", "medium", "high", "xhigh", "max")

# Per canonical modelId (after alias resolve). Values override stub defaults.
# Keys: input_modalities, supports_image_detail_original,
#       supported_reasoning_levels, default_reasoning_level,
#       context_window / max_context_window (same value when set).
MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "auto": {
        "input_modalities": list(_TEXT),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": None,
        "max_context_window": None,
    },
    "claude-opus-4.8": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": list(_LEVELS_OPUS_NEW),
        "default_reasoning_level": "medium",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
    },
    "claude-opus-4.7": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": list(_LEVELS_OPUS_NEW),
        "default_reasoning_level": "medium",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
    },
    "claude-opus-4.6": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": list(_LEVELS_OPUS_46),
        "default_reasoning_level": "medium",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
    },
    "claude-sonnet-5": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
    },
    "claude-sonnet-4.6": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": list(_LEVELS_SONNET_46),
        "default_reasoning_level": "medium",
        "context_window": 1_000_000,
        "max_context_window": 1_000_000,
    },
    "claude-haiku-4.5": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 200_000,
        "max_context_window": 200_000,
    },
    "gpt-5.6-sol": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": list(_LEVELS_GPT_56),
        "default_reasoning_level": "low",
        "context_window": 272_000,
        "max_context_window": 272_000,
    },
    "gpt-5.6-terra": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": list(_LEVELS_GPT_56),
        "default_reasoning_level": "medium",
        "context_window": 272_000,
        "max_context_window": 272_000,
    },
    "gpt-5.6-luna": {
        "input_modalities": list(_TEXT_IMAGE),
        "supports_image_detail_original": True,
        "supported_reasoning_levels": list(_LEVELS_GPT_56),
        "default_reasoning_level": "medium",
        "context_window": 272_000,
        "max_context_window": 272_000,
    },
    "deepseek-3.2": {
        "input_modalities": list(_TEXT),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 128_000,
        "max_context_window": 128_000,
    },
    "glm-5": {
        "input_modalities": list(_TEXT),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 200_000,
        "max_context_window": 200_000,
    },
    "minimax-m2.5": {
        "input_modalities": list(_TEXT),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 200_000,
        "max_context_window": 200_000,
    },
    "qwen3-coder-next": {
        "input_modalities": list(_TEXT),
        "supports_image_detail_original": False,
        "supported_reasoning_levels": [],
        "default_reasoning_level": None,
        "context_window": 256_000,
        "max_context_window": 256_000,
    },
}

# Conservative fallback: text-only, no reasoning, no context, no image detail.
# Unknown claude-* get multimodal (Kiro Claude family is vision-capable) but
# still no fabricated reasoning/context; unknown open-weight stay text-only.
_FALLBACK_TEXT: Dict[str, Any] = {
    "input_modalities": list(_TEXT),
    "supports_image_detail_original": False,
    "supported_reasoning_levels": [],
    "default_reasoning_level": None,
    "context_window": None,
    "max_context_window": None,
}

_FALLBACK_CLAUDE_HEURISTIC: Dict[str, Any] = {
    "input_modalities": list(_TEXT_IMAGE),
    "supports_image_detail_original": False,
    "supported_reasoning_levels": [],
    "default_reasoning_level": None,
    "context_window": None,
    "max_context_window": None,
}


def resolve_canonical_model_id(model_id: str) -> str:
    """Map list slug / alias to the real modelId used for capability lookup."""
    return MODEL_ALIASES.get(model_id, model_id)


def lookup_model_capabilities(model_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve aliases then return ``(canonical_id, capability_overrides)``.

    Unknown ids fall back conservatively (text-only); bare ``claude-*``
    unknowns may get text+image without inventing context/reasoning.
    """
    canonical = resolve_canonical_model_id(model_id)
    caps = MODEL_CAPABILITIES.get(canonical)
    if caps is not None:
        return canonical, caps
    if canonical.startswith("claude-"):
        return canonical, _FALLBACK_CLAUDE_HEURISTIC
    return canonical, _FALLBACK_TEXT


def build_codex_model_info(
    model_id: str,
    *,
    priority: int = 0,
    description: str = "Model via Kiro Gateway",
) -> Dict[str, Any]:
    """
    Build a Codex ``ModelInfo`` dict for ``model_id``.

    Required (no serde default) fields are always set. Optional / defaulted
    fields are included with conservative stubs so older/newer Codex builds
    that tighten decoding still succeed. Multimodal / context / reasoning
    fields come from ``MODEL_CAPABILITIES`` after alias resolution; the
    ``slug`` remains the list id (alias or real) as clients requested it.
    """
    _canonical, caps = lookup_model_capabilities(model_id)

    return {
        "slug": model_id,
        "display_name": model_id,
        "description": description,
        "default_reasoning_level": caps["default_reasoning_level"],
        "supported_reasoning_levels": list(caps["supported_reasoning_levels"]),
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
        "supports_image_detail_original": caps["supports_image_detail_original"],
        "context_window": caps["context_window"],
        "max_context_window": caps["max_context_window"],
        "auto_compact_token_limit": None,
        "comp_hash": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": list(caps["input_modalities"]),
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

# -*- coding: utf-8 -*-

"""Unit tests for per-model Codex ModelInfo capability stubs."""

from __future__ import annotations

import pytest

from kiro.codex_models import (
    MODEL_CAPABILITIES,
    build_codex_model_info,
    build_codex_models_list,
    lookup_model_capabilities,
    resolve_canonical_model_id,
)
from kiro.config import MODEL_ALIASES


def _efforts(info: dict) -> list[str]:
    return [level["effort"] for level in info["supported_reasoning_levels"]]


# ---------------------------------------------------------------------------
# Canonical models — modalities / context / reasoning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_id,modalities,image_detail,efforts,default_effort,context",
    [
        ("auto", ["text"], False, [], None, None),
        (
            "claude-opus-4.8",
            ["text", "image"],
            True,
            ["low", "medium", "high", "xhigh", "max"],
            "medium",
            1_000_000,
        ),
        (
            "claude-opus-4.7",
            ["text", "image"],
            True,
            ["low", "medium", "high", "xhigh", "max"],
            "medium",
            1_000_000,
        ),
        (
            "claude-opus-4.6",
            ["text", "image"],
            False,
            ["low", "medium", "high", "max"],
            "medium",
            1_000_000,
        ),
        (
            "claude-sonnet-5",
            ["text", "image"],
            True,
            [],
            None,
            1_000_000,
        ),
        (
            "claude-sonnet-4.6",
            ["text", "image"],
            False,
            ["low", "medium", "high", "max"],
            "medium",
            1_000_000,
        ),
        (
            "claude-haiku-4.5",
            ["text", "image"],
            False,
            [],
            None,
            200_000,
        ),
        (
            "gpt-5.6-sol",
            ["text", "image"],
            True,
            ["low", "medium", "high", "xhigh", "max"],
            "low",
            272_000,
        ),
        (
            "gpt-5.6-terra",
            ["text", "image"],
            True,
            ["low", "medium", "high", "xhigh", "max"],
            "medium",
            272_000,
        ),
        (
            "gpt-5.6-luna",
            ["text", "image"],
            True,
            ["low", "medium", "high", "xhigh", "max"],
            "medium",
            272_000,
        ),
        ("deepseek-3.2", ["text"], False, [], None, 128_000),
        ("glm-5", ["text"], False, [], None, 200_000),
        ("minimax-m2.5", ["text"], False, [], None, 200_000),
        ("qwen3-coder-next", ["text"], False, [], None, 256_000),
    ],
)
def test_canonical_model_capabilities(
    model_id,
    modalities,
    image_detail,
    efforts,
    default_effort,
    context,
):
    info = build_codex_model_info(model_id)
    assert info["slug"] == model_id
    assert info["input_modalities"] == modalities
    assert info["supports_image_detail_original"] is image_detail
    assert _efforts(info) == efforts
    assert info["default_reasoning_level"] == default_effort
    assert info["context_window"] == context
    assert info["max_context_window"] == context
    # Shared defaults
    assert info["supports_parallel_tool_calls"] is True
    assert info["apply_patch_tool_type"] is None
    assert info["supports_search_tool"] is False
    assert info["support_verbosity"] is False
    assert info["web_search_tool_type"] == "text"
    assert info["use_responses_lite"] is False
    assert info["shell_type"] == "shell_command"
    assert info["tool_mode"] is None
    assert info["multi_agent_version"] is None
    assert info["effective_context_window_percent"] == 95
    for level in info["supported_reasoning_levels"]:
        assert set(level.keys()) == {"effort", "description"}
        assert isinstance(level["description"], str) and level["description"]


# ---------------------------------------------------------------------------
# Aliases inherit capabilities of the resolved real modelId
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("kiro-o-4.8", "claude-opus-4.8"),
        ("kiro-o-4.7", "claude-opus-4.7"),
        ("kiro-o-4.6", "claude-opus-4.6"),
        ("kiro-s-5", "claude-sonnet-5"),
        ("kiro-s-4.6", "claude-sonnet-4.6"),
        ("kiro-h-4.5", "claude-haiku-4.5"),
        ("kiro-5.6-sol", "gpt-5.6-sol"),
        ("kiro-5.6-terra", "gpt-5.6-terra"),
        ("kiro-5.6-luna", "gpt-5.6-luna"),
        ("kiro-deepseek-3.2", "deepseek-3.2"),
        ("kiro-glm-5", "glm-5"),
        ("kiro-minimax-m2.5", "minimax-m2.5"),
        ("kiro-qwen3-coder-next", "qwen3-coder-next"),
        ("auto-kiro", "auto"),
    ],
)
def test_alias_inherits_canonical_capabilities(alias, canonical):
    assert MODEL_ALIASES[alias] == canonical
    assert resolve_canonical_model_id(alias) == canonical

    alias_info = build_codex_model_info(alias)
    canon_info = build_codex_model_info(canonical)

    # Slug stays as the list/alias id clients see
    assert alias_info["slug"] == alias
    assert canon_info["slug"] == canonical

    capability_keys = (
        "input_modalities",
        "supports_image_detail_original",
        "supported_reasoning_levels",
        "default_reasoning_level",
        "context_window",
        "max_context_window",
    )
    for key in capability_keys:
        assert alias_info[key] == canon_info[key], f"{alias} vs {canonical}: {key}"


def test_kiro_o_48_matches_opus_vision_and_reasoning():
    info = build_codex_model_info("kiro-o-4.8")
    assert "image" in info["input_modalities"]
    assert info["supports_image_detail_original"] is True
    assert info["context_window"] == 1_000_000
    assert _efforts(info) == ["low", "medium", "high", "xhigh", "max"]


def test_kiro_glm_5_is_text_only():
    info = build_codex_model_info("kiro-glm-5")
    assert info["input_modalities"] == ["text"]
    assert info["supports_image_detail_original"] is False
    assert info["context_window"] == 200_000
    assert info["supported_reasoning_levels"] == []


def test_kiro_56_sol_has_image():
    info = build_codex_model_info("kiro-5.6-sol")
    assert info["input_modalities"] == ["text", "image"]
    assert info["supports_image_detail_original"] is True
    assert info["default_reasoning_level"] == "low"
    assert "ultra" not in _efforts(info)


# ---------------------------------------------------------------------------
# Unknown / fallback
# ---------------------------------------------------------------------------

def test_unknown_open_weight_is_text_only():
    info = build_codex_model_info("some-unknown-oss-model")
    assert info["input_modalities"] == ["text"]
    assert info["supports_image_detail_original"] is False
    assert info["supported_reasoning_levels"] == []
    assert info["default_reasoning_level"] is None
    assert info["context_window"] is None
    assert info["max_context_window"] is None


def test_unknown_claude_heuristic_gets_image_without_fabricated_context():
    info = build_codex_model_info("claude-future-9.9")
    assert info["input_modalities"] == ["text", "image"]
    assert info["supports_image_detail_original"] is False
    assert info["supported_reasoning_levels"] == []
    assert info["context_window"] is None


def test_gpt_levels_exclude_ultra():
    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert "ultra" not in _efforts(build_codex_model_info(model_id))


def test_lookup_returns_canonical_and_caps():
    canonical, caps = lookup_model_capabilities("kiro-o-4.8")
    assert canonical == "claude-opus-4.8"
    assert caps is MODEL_CAPABILITIES["claude-opus-4.8"]


def test_build_codex_models_list_preserves_order_and_priority():
    models = build_codex_models_list(["kiro-glm-5", "claude-opus-4.8"])
    assert [m["slug"] for m in models] == ["kiro-glm-5", "claude-opus-4.8"]
    assert models[0]["priority"] == 0
    assert models[1]["priority"] == 1
    assert models[0]["input_modalities"] == ["text"]
    assert "image" in models[1]["input_modalities"]

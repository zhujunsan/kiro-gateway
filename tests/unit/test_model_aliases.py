# -*- coding: utf-8 -*-

"""Unit tests for Cursor-safe model alias generation and resolution."""

from __future__ import annotations

import pytest

from kiro.cache import ModelInfoCache
from kiro.config import FALLBACK_MODELS, HIDDEN_FROM_LIST, MODEL_ALIASES
from kiro.model_aliases import generate_model_alias, resolve_model_alias
from kiro.model_resolver import ModelResolver, get_model_id_for_kiro


class TestGenerateModelAlias:
    """Tests for generate_model_alias()."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("claude-opus-5", "kiro-o-5"),
            ("claude-opus-4.6", "kiro-o-4.6"),
            ("claude-sonnet-5", "kiro-s-5"),
            ("claude-haiku-4.5", "kiro-h-4.5"),
            ("deepseek-3.2", "kiro-deepseek-3.2"),
            ("glm-5", "kiro-glm-5"),
            ("minimax-m2.5", "kiro-minimax-m2.5"),
            ("qwen3-coder-next", "kiro-qwen3-coder-next"),
        ],
    )
    def test_generates_expected_aliases(self, model_id: str, expected: str) -> None:
        assert generate_model_alias(model_id) == expected

    @pytest.mark.parametrize(
        "model_id",
        ["auto", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "GPT-5.6-SOL"],
    )
    def test_auto_and_gpt_have_no_alias(self, model_id: str) -> None:
        assert generate_model_alias(model_id) is None

    @pytest.mark.parametrize("model_id", ["", "   ", None])
    def test_empty_or_none_returns_none(self, model_id: str | None) -> None:
        assert generate_model_alias(model_id) is None  # type: ignore[arg-type]


class TestResolveModelAlias:
    """Tests for resolve_model_alias()."""

    def test_explicit_table_takes_precedence(self) -> None:
        aliases = {"my-auto": "auto", "kiro-o-5": "claude-opus-4.6"}
        assert resolve_model_alias("my-auto", aliases) == "auto"
        # Explicit override wins over syntactic reverse
        assert resolve_model_alias("kiro-o-5", aliases) == "claude-opus-4.6"

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("kiro-o-5", "claude-opus-5"),
            ("kiro-o-4.6", "claude-opus-4.6"),
            ("kiro-s-5", "claude-sonnet-5"),
            ("kiro-h-4.5", "claude-haiku-4.5"),
            ("kiro-deepseek-3.2", "deepseek-3.2"),
            ("kiro-glm-5", "glm-5"),
        ],
    )
    def test_syntactic_reverse(self, alias: str, canonical: str) -> None:
        assert resolve_model_alias(alias, {}) == canonical

    def test_passthrough_canonical_and_unknown(self) -> None:
        assert resolve_model_alias("claude-opus-5", {}) == "claude-opus-5"
        assert resolve_model_alias("mystery-model", {}) == "mystery-model"


class TestModelAliasesConfigAlignment:
    """Static config must stay aligned with generate_model_alias rules."""

    def test_model_aliases_derived_from_fallback(self) -> None:
        expected: dict[str, str] = {}
        for entry in FALLBACK_MODELS:
            model_id = entry["modelId"]
            alias = generate_model_alias(model_id)
            if alias:
                expected[alias] = model_id
        assert MODEL_ALIASES == expected

    def test_fallback_includes_opus_5_excludes_retired(self) -> None:
        fallback_ids = {entry["modelId"] for entry in FALLBACK_MODELS}
        assert "claude-opus-5" in fallback_ids
        assert "kiro-o-5" in MODEL_ALIASES
        assert MODEL_ALIASES["kiro-o-5"] == "claude-opus-5"
        retired = {"claude-opus-4.7", "claude-opus-4.8", "claude-sonnet-4.6"}
        assert retired.isdisjoint(fallback_ids)
        assert retired.issubset(set(HIDDEN_FROM_LIST))
        assert "kiro-o-4.7" not in MODEL_ALIASES
        assert "kiro-o-4.8" not in MODEL_ALIASES
        assert "kiro-s-4.6" not in MODEL_ALIASES


class TestResolverDynamicAliases:
    """ModelResolver lists and resolves aliases from currently available models."""

    def test_lists_kiro_o_5_when_opus_5_available_not_stale_48(self) -> None:
        cache = ModelInfoCache()
        cache._cache = {
            "claude-opus-5": {"modelId": "claude-opus-5"},
            "claude-opus-4.8": {"modelId": "claude-opus-4.8"},
            "auto": {"modelId": "auto"},
        }
        resolver = ModelResolver(
            cache=cache,
            aliases=MODEL_ALIASES,
            hidden_from_list=["claude-opus-4.8"],
        )
        models = resolver.get_available_models()
        assert "claude-opus-5" in models
        assert "kiro-o-5" in models
        assert "claude-opus-4.8" not in models
        assert "kiro-o-4.8" not in models
        assert "auto" in models
        assert "auto-kiro" not in models

    def test_resolve_kiro_o_5_to_claude_opus_5(self) -> None:
        cache = ModelInfoCache()
        cache._cache = {"claude-opus-5": {"modelId": "claude-opus-5"}}
        resolver = ModelResolver(cache=cache, aliases=MODEL_ALIASES)
        result = resolver.resolve("kiro-o-5")
        assert result.internal_id == "claude-opus-5"
        assert result.source == "cache"
        assert result.is_verified is True

    def test_resolve_stale_alias_still_passthrough(self) -> None:
        """Clients may still request retired aliases; Kiro is the final arbiter."""
        cache = ModelInfoCache()
        cache._cache = {"claude-opus-5": {"modelId": "claude-opus-5"}}
        resolver = ModelResolver(cache=cache, aliases={})
        result = resolver.resolve("kiro-o-4.8")
        assert result.internal_id == "claude-opus-4.8"
        assert result.source == "passthrough"

    def test_get_model_id_for_kiro_resolves_syntactic_alias(self) -> None:
        assert get_model_id_for_kiro("kiro-o-5", {}) == "claude-opus-5"
        assert get_model_id_for_kiro("kiro-deepseek-3.2", {}) == "deepseek-3.2"

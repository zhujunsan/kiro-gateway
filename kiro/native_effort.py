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
Native reasoning effort support (Kiro IDE parity).

Kiro IDE does not tune reasoning through prompt tags. It sends a top-level
``additionalModelRequestFields`` document in the GenerateAssistantResponse
request body, next to ``conversationState`` and ``profileArn``:

- Claude:  ``{"output_config": {"effort": "high"}, "thinking": {"type": "adaptive"}}``
- GPT/Kimi: ``{"reasoning": {"effort": "high"}}``

Which keys and values are legal is declared per model by ListAvailableModels
``models[].additionalModelRequestFieldsSchema`` (a JSON schema with
``additionalProperties: false``). The upstream validates strictly: an unknown
key or enum value is rejected with HTTP 400 ``REQUEST_BODY_INVALID``, and
models without a schema reject the field entirely. This module therefore only
emits values that the model's schema allows.

Capability sources, in priority order:

1. Live schemas registered from the last successful ListAvailableModels call
   (:func:`register_model_schemas`).
2. :data:`STATIC_CAPABILITIES`, a snapshot captured on 2026-10-22, so effort
   works before the first model-list request.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from loguru import logger

# Effort scale shared by OpenAI / Anthropic / Kiro, ordered from cheapest to
# deepest. Used to map a requested level onto the nearest level a model allows.
EFFORT_ORDER: Tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_RANK: Dict[str, int] = {level: rank for rank, level in enumerate(EFFORT_ORDER)}

# Levels Kiro IDE refuses to combine with ``thinking.type=disabled``.
_THINKING_ONLY_LEVELS = frozenset({"xhigh", "max"})

# Schema paths in the order Kiro IDE probes them.
_EFFORT_PATHS: Tuple[str, ...] = ("output_config", "reasoning")


@dataclass(frozen=True)
class NativeReasoningCapability:
    """Reasoning controls a model accepts in ``additionalModelRequestFields``.

    Attributes:
        effort_path: Top-level key holding ``effort`` (``output_config`` or
            ``reasoning``), or None if the model has no effort control.
        effort_levels: Allowed effort values, as declared by the schema.
        default_effort: Schema default effort (informational).
        thinking_types: Allowed ``thinking.type`` values (Claude only).
        thinking_displays: Allowed ``thinking.display`` values (Claude only).
    """

    effort_path: Optional[str] = None
    effort_levels: Tuple[str, ...] = ()
    default_effort: Optional[str] = None
    thinking_types: Tuple[str, ...] = ()
    thinking_displays: Tuple[str, ...] = ()

    @property
    def supports_effort(self) -> bool:
        return bool(self.effort_path and self.effort_levels)


def _enum(node: Any) -> Tuple[str, ...]:
    """Return the string enum of a schema node, or an empty tuple."""
    if not isinstance(node, Mapping):
        return ()
    values = node.get("enum")
    if not isinstance(values, list):
        return ()
    return tuple(v for v in values if isinstance(v, str) and v)


def _child(node: Any, *path: str) -> Any:
    """Walk ``properties`` of a JSON schema along ``path``."""
    for key in path:
        if not isinstance(node, Mapping):
            return None
        props = node.get("properties")
        if not isinstance(props, Mapping):
            return None
        node = props.get(key)
    return node


def capability_from_schema(schema: Any) -> Optional[NativeReasoningCapability]:
    """Parse ``additionalModelRequestFieldsSchema`` like Kiro IDE does.

    Args:
        schema: The raw schema object from ListAvailableModels.

    Returns:
        Parsed capability, or None when the schema declares no reasoning
        controls (the model then must not receive the field at all).
    """
    if not isinstance(schema, Mapping):
        return None

    effort_path: Optional[str] = None
    effort_levels: Tuple[str, ...] = ()
    default_effort: Optional[str] = None
    for path in _EFFORT_PATHS:
        node = _child(schema, path, "effort")
        levels = _enum(node)
        if levels:
            effort_path = path
            effort_levels = levels
            default = node.get("default") if isinstance(node, Mapping) else None
            default_effort = default if isinstance(default, str) else None
            break

    thinking_types = _enum(_child(schema, "thinking", "type"))
    thinking_displays = _enum(_child(schema, "thinking", "display"))

    if not effort_levels and not thinking_types:
        return None
    return NativeReasoningCapability(
        effort_path=effort_path,
        effort_levels=effort_levels,
        default_effort=default_effort,
        thinking_types=thinking_types,
        thinking_displays=thinking_displays,
    )


def _claude(levels: Iterable[str], default: str, thinking: Iterable[str]) -> NativeReasoningCapability:
    return NativeReasoningCapability(
        effort_path="output_config",
        effort_levels=tuple(levels),
        default_effort=default,
        thinking_types=tuple(thinking),
        thinking_displays=("summarized", "omitted"),
    )


def _reasoning(levels: Iterable[str], default: str) -> NativeReasoningCapability:
    return NativeReasoningCapability(
        effort_path="reasoning", effort_levels=tuple(levels), default_effort=default
    )


_CLAUDE_5_LEVELS = ("low", "medium", "high", "xhigh", "max")
_CLAUDE_46_LEVELS = ("low", "medium", "high", "max")
_TOGGLEABLE = ("adaptive", "disabled")

# Snapshot of ListAvailableModels schemas (us-east-1, 2026-10-22). Only models
# observed with a schema are listed; anything else gets no native fields until
# a live schema is registered. Keep in sync when Kiro ships new models.
STATIC_CAPABILITIES: Dict[str, NativeReasoningCapability] = {
    "claude-opus-5.5": _claude(_CLAUDE_5_LEVELS, "medium", ("adaptive",)),
    "claude-sonnet-5.5": _claude(_CLAUDE_5_LEVELS, "high", ("adaptive", "between_tools")),
    "claude-opus-5": _claude(_CLAUDE_5_LEVELS, "high", _TOGGLEABLE),
    "claude-sonnet-5": _claude(_CLAUDE_5_LEVELS, "high", _TOGGLEABLE),
    "claude-opus-4.8": _claude(_CLAUDE_5_LEVELS, "high", _TOGGLEABLE),
    "claude-opus-4.7": _claude(_CLAUDE_5_LEVELS, "xhigh", _TOGGLEABLE),
    "claude-opus-4.6": _claude(_CLAUDE_46_LEVELS, "high", _TOGGLEABLE),
    "claude-sonnet-4.6": _claude(_CLAUDE_46_LEVELS, "high", _TOGGLEABLE),
    "gpt-5.6-terra": _reasoning(("none", "low", "medium", "high", "xhigh", "max"), "high"),
    "gpt-5.6-luna": _reasoning(("none", "low", "medium", "high", "xhigh", "max"), "high"),
    "kimi-k3": _reasoning(("low", "high", "max"), "max"),
}

# modelId -> capability from the latest ListAvailableModels. A None value means
# "listed live without a schema" and overrides the static snapshot.
_live_capabilities: Dict[str, Optional[NativeReasoningCapability]] = {}
_live_lock = threading.Lock()


def register_model_schemas(models: Iterable[Mapping[str, Any]]) -> None:
    """Record per-model reasoning capabilities from ListAvailableModels.

    Args:
        models: Raw ``models`` entries (dicts with ``modelId`` and optional
            ``additionalModelRequestFieldsSchema``).
    """
    parsed: Dict[str, Optional[NativeReasoningCapability]] = {}
    for model in models:
        if not isinstance(model, Mapping):
            continue
        model_id = model.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            continue
        parsed[model_id] = capability_from_schema(model.get("additionalModelRequestFieldsSchema"))
    if not parsed:
        return
    with _live_lock:
        _live_capabilities.update(parsed)
    supported = sorted(m for m, cap in parsed.items() if cap is not None)
    logger.debug(f"Native effort schemas registered for: {', '.join(supported) or '(none)'}")


def clear_live_capabilities() -> None:
    """Forget registered live schemas (used by tests)."""
    with _live_lock:
        _live_capabilities.clear()


def get_native_capability(model_id: Optional[str]) -> Optional[NativeReasoningCapability]:
    """Return reasoning capability for a Kiro model ID, live schema first."""
    if not model_id:
        return None
    with _live_lock:
        if model_id in _live_capabilities:
            return _live_capabilities[model_id]
    return STATIC_CAPABILITIES.get(model_id)


def _is_known_model(model_id: str) -> bool:
    with _live_lock:
        if model_id in _live_capabilities:
            return True
    return model_id in STATIC_CAPABILITIES


def normalize_effort(value: Any) -> Optional[str]:
    """Lower-case a client effort value; None when it is not a known level."""
    if not isinstance(value, str):
        return None
    effort = value.strip().lower()
    return effort if effort in _EFFORT_RANK else None


_EFFORT_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)-(?P<effort>" + "|".join(EFFORT_ORDER) + r")$", re.IGNORECASE
)


def split_effort_suffix(model: str) -> Tuple[str, Optional[str]]:
    """Split a ``<model>-<effort>`` name into base model and effort.

    Lets clients that cannot send effort parameters (e.g. Cursor custom
    models) pick a level through the model name: ``kiro-o-5-high``,
    ``gpt-5.6-luna-low``. A name that is itself a known model is left alone.

    Args:
        model: Model name as sent by the client.

    Returns:
        ``(base_model, effort)``; effort is None when there is no suffix.
    """
    if not isinstance(model, str):
        return model, None
    match = _EFFORT_SUFFIX_RE.match(model.strip())
    if not match or _is_known_model(model.strip()):
        return model, None
    return match.group("base"), match.group("effort").lower()


def _nearest_level(effort: str, levels: Iterable[str]) -> Optional[str]:
    """Map ``effort`` to the closest allowed level (ties go to the cheaper one)."""
    target = _EFFORT_RANK.get(effort)
    ranked = sorted((lv for lv in levels if lv in _EFFORT_RANK), key=_EFFORT_RANK.__getitem__)
    if target is None or not ranked:
        return None
    return min(ranked, key=lambda lv: (abs(_EFFORT_RANK[lv] - target), _EFFORT_RANK[lv]))


def build_additional_model_request_fields(
    model_id: Optional[str],
    effort: Optional[str] = None,
    thinking: Optional[str] = None,
    thinking_display: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the native ``additionalModelRequestFields`` document.

    Args:
        model_id: Kiro model ID (after alias resolution).
        effort: Requested effort (none/minimal/low/medium/high/xhigh/max).
            Unsupported levels map to the nearest allowed level.
        thinking: ``"disabled"`` to turn thinking off, ``"enabled"`` to turn
            it on (sent as ``adaptive``), or None for no preference.
        thinking_display: Optional ``thinking.display`` passthrough.

    Returns:
        The document to send, or None when the client expressed no reasoning
        preference or the model has no native reasoning controls (callers then
        keep the legacy fake reasoning behavior).
    """
    from kiro.config import NATIVE_EFFORT_ENABLED

    if not NATIVE_EFFORT_ENABLED:
        return None

    effort = normalize_effort(effort)
    if thinking not in ("disabled", "enabled"):
        thinking = None
    if effort == "none" and thinking is None:
        thinking = "disabled"
    if effort is None and thinking is None:
        return None

    capability = get_native_capability(model_id)
    if capability is None:
        return None

    fields: Dict[str, Any] = {}
    levels = capability.effort_levels
    level: Optional[str] = None

    if thinking == "disabled":
        if "disabled" in capability.thinking_types:
            fields["thinking"] = {"type": "disabled"}
            if effort and effort != "none" and capability.supports_effort:
                # Mirror Kiro IDE: xhigh/max are not valid without thinking.
                allowed = [lv for lv in levels if lv not in _THINKING_ONLY_LEVELS] or list(levels)
                level = _nearest_level(effort, allowed)
        elif capability.supports_effort:
            # No thinking switch: "none" if offered, otherwise the cheapest level.
            level = "none" if "none" in levels else _nearest_level("none", levels)
    else:
        if "adaptive" in capability.thinking_types:
            thinking_obj: Dict[str, Any] = {"type": "adaptive"}
            if thinking_display and thinking_display in capability.thinking_displays:
                thinking_obj["display"] = thinking_display
            fields["thinking"] = thinking_obj
        if effort and capability.supports_effort:
            level = _nearest_level(effort, levels)

    if level and capability.effort_path:
        fields[capability.effort_path] = {"effort": level}

    if not fields:
        return None
    if effort and level and level != effort:
        logger.debug(f"Native effort '{effort}' mapped to '{level}' for model {model_id}")
    return fields

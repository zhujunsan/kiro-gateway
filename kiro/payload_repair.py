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
Self-healing for optional reasoning fields rejected by Kiro (HTTP 400).

The gateway adds two optional, best-effort things to GenerateAssistantResponse:

1. ``additionalModelRequestFields`` (native effort / thinking / max_tokens),
   validated strictly against a per-model schema that can drift from our
   snapshot or differ between accounts.
2. ``reasoningContent`` on history assistant messages (replayed thinking +
   signature), which some models reject and which fails when a signature is
   invalid.

Neither is required for a correct answer, so a 400 caused by them must never
reach the user. :func:`repair_payload` inspects the error, removes or narrows
the offending part in place, and tells the HTTP client to retry once more.
What was learned is fed back into :mod:`kiro.native_effort` so later requests
do not hit the same 400.

Each rule fires at most once per request, so a repair loop is bounded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from kiro import native_effort

ADDITIONAL_FIELDS_KEY = "additionalModelRequestFields"
REQUEST_BODY_INVALID = "REQUEST_BODY_INVALID"
THINKING_SIGNATURE_INVALID = "THINKING_SIGNATURE_INVALID"

# Upper bound on repairs for one request; there are only four rules.
MAX_REPAIRS_PER_REQUEST = 4

_ENUM_RE = re.compile(r"enumeration\s*(\[[^\]]*\])", re.IGNORECASE)
_MAXIMUM_RE = re.compile(r"maximum value of\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class KiroRejection:
    """Parsed Kiro 400 error body.

    Attributes:
        reason: Kiro ``reason`` code (e.g. ``REQUEST_BODY_INVALID``), or "".
        message: Human-readable ``message``, or "".
    """

    reason: str
    message: str


def parse_rejection(body: bytes) -> KiroRejection:
    """Parse a Kiro error body into reason + message.

    Args:
        body: Raw response body.

    Returns:
        Parsed rejection; empty strings when the body is not a JSON object.
    """
    try:
        data = json.loads(body.decode("utf-8", errors="replace")) if body else None
    except (json.JSONDecodeError, ValueError):
        data = None
    if not isinstance(data, dict):
        return KiroRejection(reason="", message="")
    reason = data.get("reason")
    message = data.get("message")
    return KiroRejection(
        reason=reason if isinstance(reason, str) else "",
        message=message if isinstance(message, str) else "",
    )


def _model_id(payload: Dict[str, Any]) -> Optional[str]:
    """Return the current message model ID of a Kiro payload, if present."""
    try:
        model_id = payload["conversationState"]["currentMessage"]["userInputMessage"]["modelId"]
    except (KeyError, TypeError):
        return None
    return model_id if isinstance(model_id, str) else None


def _history_with_reasoning(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return history assistant messages that carry ``reasoningContent``."""
    state = payload.get("conversationState")
    history = state.get("history") if isinstance(state, dict) else None
    if not isinstance(history, list):
        return []
    found = []
    for entry in history:
        message = entry.get("assistantResponseMessage") if isinstance(entry, dict) else None
        if isinstance(message, dict) and "reasoningContent" in message:
            found.append(message)
    return found


def _mentions_additional_fields(rejection: KiroRejection) -> bool:
    return ADDITIONAL_FIELDS_KEY in rejection.message


# --------------------------------------------------------------------------------------------------
# Rules. Each returns a short description when it changed the payload, else None.
# --------------------------------------------------------------------------------------------------

def _strip_history_reasoning(payload: Dict[str, Any], rejection: KiroRejection) -> Optional[str]:
    """Drop replayed reasoning from history (Kiro IDE does the same on retry)."""
    messages = _history_with_reasoning(payload)
    if not messages:
        return None
    if rejection.reason == THINKING_SIGNATURE_INVALID:
        pass
    elif rejection.reason == REQUEST_BODY_INVALID and not _mentions_additional_fields(rejection):
        # Generic "Improperly formed request": replayed reasoning is the most
        # likely gateway-added culprit (e.g. models that do not accept it).
        pass
    else:
        return None
    for message in messages:
        message.pop("reasoningContent", None)
    return f"removed replayed reasoning from {len(messages)} history message(s)"


def _narrow_max_tokens(payload: Dict[str, Any], rejection: KiroRejection) -> Optional[str]:
    """Clamp or drop ``max_tokens`` when the schema maximum/property was violated."""
    fields = payload.get(ADDITIONAL_FIELDS_KEY)
    if not isinstance(fields, dict) or "max_tokens" not in fields:
        return None
    if rejection.reason != REQUEST_BODY_INVALID or not _mentions_additional_fields(rejection):
        return None
    message = rejection.message
    maximum_match = _MAXIMUM_RE.search(message)
    if maximum_match:
        maximum = int(float(maximum_match.group(1)))
        if 0 < maximum < fields["max_tokens"]:
            fields["max_tokens"] = maximum
            native_effort.learn_max_output_tokens(_model_id(payload), maximum)
            return f"clamped max_tokens to schema maximum {maximum}"
    if "max_tokens" in message or maximum_match:
        fields.pop("max_tokens", None)
        native_effort.learn_max_output_tokens(_model_id(payload), None)
        if not fields:
            payload.pop(ADDITIONAL_FIELDS_KEY, None)
        return "removed max_tokens"
    return None


def _remap_effort(payload: Dict[str, Any], rejection: KiroRejection) -> Optional[str]:
    """Map an effort the schema enum rejected onto the nearest allowed level."""
    fields = payload.get(ADDITIONAL_FIELDS_KEY)
    if not isinstance(fields, dict):
        return None
    if rejection.reason != REQUEST_BODY_INVALID or not _mentions_additional_fields(rejection):
        return None
    enum_match = _ENUM_RE.search(rejection.message)
    if not enum_match:
        return None
    try:
        allowed = json.loads(enum_match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(allowed, list) or not allowed:
        return None
    allowed_levels = [v for v in allowed if isinstance(v, str) and native_effort.normalize_effort(v)]
    if len(allowed_levels) != len(allowed):
        return None  # Not an effort enum (e.g. thinking.type); let the next rule drop it.
    for path in ("output_config", "reasoning"):
        node = fields.get(path)
        current = node.get("effort") if isinstance(node, dict) else None
        if not isinstance(current, str) or current in allowed_levels:
            continue
        mapped = native_effort.nearest_level(current, allowed_levels)
        if mapped is None:
            return None
        node["effort"] = mapped
        native_effort.learn_effort_levels(_model_id(payload), tuple(allowed_levels))
        if mapped != "max" and "max_tokens" in fields:
            # max_tokens is only added for the max tier.
            fields.pop("max_tokens", None)
        return f"remapped effort '{current}' to '{mapped}'"
    return None


def _drop_additional_fields(payload: Dict[str, Any], rejection: KiroRejection) -> Optional[str]:
    """Last resort: send the request without native reasoning fields."""
    if ADDITIONAL_FIELDS_KEY not in payload:
        return None
    if rejection.reason != REQUEST_BODY_INVALID or not _mentions_additional_fields(rejection):
        return None
    payload.pop(ADDITIONAL_FIELDS_KEY, None)
    if "not supported" in rejection.message.lower():
        # The model rejects the field outright: stop sending it until the
        # model list is refreshed. Other validation errors stay per-request
        # so one odd combination does not disable effort for the model.
        native_effort.learn_unsupported(_model_id(payload))
    return "removed additionalModelRequestFields"


# Ordered from most to least specific: narrow before dropping everything.
_RULES: List[Callable[[Dict[str, Any], KiroRejection], Optional[str]]] = [
    _strip_history_reasoning,
    _narrow_max_tokens,
    _remap_effort,
    _drop_additional_fields,
]


class PayloadRepairer:
    """Applies each repair rule at most once for a single request.

    Example:
        >>> repairer = PayloadRepairer()
        >>> if repairer.repair(payload, body):
        ...     resend(payload)
    """

    def __init__(self) -> None:
        self._applied: Set[str] = set()

    def repair(self, payload: Optional[Dict[str, Any]], body: bytes) -> bool:
        """Repair ``payload`` in place for a 400 response body.

        Args:
            payload: Kiro request body that was rejected (mutated in place so
                later first-token retries reuse the repaired version).
            body: Raw 400 response body.

        Returns:
            True when the payload changed and the request should be resent.
        """
        if not isinstance(payload, dict) or len(self._applied) >= MAX_REPAIRS_PER_REQUEST:
            return False
        rejection = parse_rejection(body)
        if rejection.reason not in (REQUEST_BODY_INVALID, THINKING_SIGNATURE_INVALID):
            return False
        for rule in _RULES:
            name = rule.__name__
            if name in self._applied:
                continue
            description = rule(payload, rejection)
            if description:
                self._applied.add(name)
                logger.warning(
                    f"Kiro rejected optional reasoning fields ({rejection.reason}: "
                    f"{rejection.message[:200]}); {description} and retrying. "
                    f"The answer is unaffected; reasoning depth may differ for this request."
                )
                return True
        return False

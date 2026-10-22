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
Native reasoning signature round-trip (Kiro IDE parity).

When native thinking is enabled, Kiro streams ``reasoningContentEvent`` text
deltas followed by a ``{"signature": ...}`` frame. Kiro IDE stores both and
replays them on later turns as::

    {"assistantResponseMessage": {
        "content": "...",
        "reasoningContent": {"reasoningText": {"text": "...", "signature": "..."}}
    }}

so the model sees its own earlier reasoning (important inside tool loops).
Upstream verifies the signature: a forged one fails with HTTP 400
``THINKING_SIGNATURE_INVALID``.

This module owns the client-facing carriers of that pair:

- Anthropic: the ``signature`` of a ``thinking`` content block.
- Responses: an opaque ``encrypted_content`` on the ``reasoning`` output item
  (``kgw1:`` + base64 JSON), which Codex echoes back verbatim.
- OpenAI Chat Completions has no carrier, so reasoning is not replayed there.

Gateway placeholder signatures (fake reasoning has no real signature) are
recognised and never replayed.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from typing import Any, Dict, Optional

from loguru import logger

# Placeholder signature emitted for fake reasoning thinking blocks. Clients
# replay it like a real one, so it must be recognisable on the way back in.
PLACEHOLDER_SIGNATURE_PREFIX = "sig_"
_PLACEHOLDER_SIGNATURE_RE = re.compile(r"^sig_[0-9a-f]{32}$")

# Versioned prefix for Responses ``encrypted_content`` produced by the gateway.
RESPONSES_ENCRYPTED_PREFIX = "kgw1:"

# Upper bound for a decoded Responses carrier. Real reasoning plus signature
# is far below this; the cap stops a hostile client from making us decode and
# forward arbitrarily large blobs.
_MAX_ENCRYPTED_CONTENT_CHARS = 4 * 1024 * 1024


def generate_placeholder_signature() -> str:
    """Generate the placeholder signature used for fake reasoning blocks.

    Returns:
        A ``sig_<32 hex>`` string that :func:`is_placeholder_signature` accepts.
    """
    return f"{PLACEHOLDER_SIGNATURE_PREFIX}{uuid.uuid4().hex}"


def is_placeholder_signature(signature: Any) -> bool:
    """Whether ``signature`` is a gateway placeholder (not replayable).

    Args:
        signature: Signature value from a client request.

    Returns:
        True for gateway placeholders, False otherwise (including non-strings).
    """
    return isinstance(signature, str) and bool(_PLACEHOLDER_SIGNATURE_RE.match(signature))


def make_reasoning(text: Any, signature: Any) -> Optional[Dict[str, str]]:
    """Build a unified reasoning record if it can be replayed upstream.

    Args:
        text: Reasoning text (may be empty; Kiro accepts an empty text).
        signature: Upstream signature.

    Returns:
        ``{"text": str, "signature": str}``, or None when the signature is
        missing, not a string, or a gateway placeholder.
    """
    if not isinstance(signature, str) or not signature.strip():
        return None
    if is_placeholder_signature(signature):
        return None
    return {"text": text if isinstance(text, str) else "", "signature": signature}


def to_history_reasoning_content(reasoning: Dict[str, str]) -> Dict[str, Any]:
    """Convert a unified reasoning record to Kiro's history shape.

    Args:
        reasoning: Record produced by :func:`make_reasoning`.

    Returns:
        ``{"reasoningText": {"text": ..., "signature": ...}}``.
    """
    return {"reasoningText": {"text": reasoning["text"], "signature": reasoning["signature"]}}


def encode_responses_encrypted_content(text: str, signature: Optional[str]) -> Optional[str]:
    """Pack reasoning text + signature into a Responses ``encrypted_content``.

    Args:
        text: Full reasoning text of the turn.
        signature: Upstream signature, or None.

    Returns:
        Opaque carrier string, or None when there is nothing replayable.
    """
    reasoning = make_reasoning(text, signature)
    if reasoning is None:
        return None
    raw = json.dumps({"t": reasoning["text"], "s": reasoning["signature"]}, ensure_ascii=False)
    return RESPONSES_ENCRYPTED_PREFIX + base64.b64encode(raw.encode("utf-8")).decode("ascii")


def decode_responses_encrypted_content(value: Any) -> Optional[Dict[str, str]]:
    """Unpack a gateway ``encrypted_content`` back into a reasoning record.

    Foreign values (real OpenAI blobs, compaction stubs, garbage) are ignored
    rather than rejected: replay is best-effort and must never fail a request.

    Args:
        value: ``encrypted_content`` from a Responses input reasoning item.

    Returns:
        Reasoning record, or None when the value is not a valid gateway carrier.
    """
    if not isinstance(value, str) or not value.startswith(RESPONSES_ENCRYPTED_PREFIX):
        return None
    if len(value) > _MAX_ENCRYPTED_CONTENT_CHARS:
        logger.warning("Ignoring oversized reasoning encrypted_content in Responses input")
        return None
    try:
        raw = base64.b64decode(value[len(RESPONSES_ENCRYPTED_PREFIX):], validate=True)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        logger.debug(f"Ignoring undecodable reasoning encrypted_content: {e}")
        return None
    if not isinstance(data, dict):
        return None
    return make_reasoning(data.get("t"), data.get("s"))

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
Local Responses history compaction (POST /v1/responses/compact).

OpenAI's compact endpoint returns an encrypted compaction item produced by
their models. This gateway has no such backend, so we approximate by:
1. Normalizing / merging adjacent same-role message items
2. Trimming oldest items until the window fits under a byte budget
   (same spirit as ``payload_guards.trim_payload_to_limit`` / AUTO_TRIM)
3. Folding dropped content into a single ``type: compaction`` stub whose
   ``encrypted_content`` is plaintext (opaque to clients; round-tripped as
   a summary user message by converters_responses)

Return shape matches CompactedResponse: ``object=response.compaction`` with
an ``output`` array suitable as the next ``/v1/responses`` ``input``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from kiro.config import KIRO_MAX_PAYLOAD_BYTES
from kiro.streaming_responses import generate_response_id


class CompactRequest(BaseModel):
    """Request body for ``POST /v1/responses/compact``."""

    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[str] = None
    # Accepted for wire compatibility; store chaining is owned elsewhere.
    previous_response_id: Optional[str] = None

    model_config = {"extra": "allow"}


@dataclass
class CompactStats:
    """Stats from a local compaction pass."""

    original_items: int
    final_items: int
    original_bytes: int
    final_bytes: int
    trimmed: bool
    dropped_items: int


def _items_bytes(items: List[Any]) -> int:
    return len(json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _normalize_input_items(input_data: Union[str, List[Any], None]) -> List[Dict[str, Any]]:
    if input_data is None:
        raise ValueError("input is required for compaction (previous_response_id store not used here)")
    if isinstance(input_data, str):
        text = input_data.strip()
        if not text:
            raise ValueError("input string must not be empty")
        return [{"type": "message", "role": "user", "content": text}]
    if not isinstance(input_data, list):
        raise ValueError(
            f"input must be a string or array of items, got {type(input_data).__name__}"
        )
    if len(input_data) == 0:
        raise ValueError("input array must not be empty")

    items: List[Dict[str, Any]] = []
    for i, raw in enumerate(input_data):
        if isinstance(raw, str):
            items.append({"type": "message", "role": "user", "content": raw})
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                f"input[{i}]: expected object or string, got {type(raw).__name__}"
            )
        # Drop transient compaction triggers; keep everything else for trimming.
        if raw.get("type") == "compaction_trigger":
            continue
        items.append(dict(raw))

    if not items:
        raise ValueError("input has no compactable items after filtering")
    return items


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _item_preview(item: Dict[str, Any], limit: int = 400) -> str:
    item_type = item.get("type") or ("message" if "role" in item else "unknown")
    if item_type == "message" or (item_type is None and "role" in item):
        role = item.get("role") or "user"
        text = _message_text(item.get("content")).strip().replace("\n", " ")
        if len(text) > limit:
            text = text[: limit - 3] + "..."
        return f"[{role}] {text}"
    if item_type == "function_call":
        name = item.get("name") or "?"
        args = str(item.get("arguments") or "")
        if len(args) > 120:
            args = args[:117] + "..."
        return f"[function_call {name}] {args}"
    if item_type == "function_call_output":
        out = _message_text(item.get("output")).strip().replace("\n", " ")
        if len(out) > limit:
            out = out[: limit - 3] + "..."
        return f"[function_call_output] {out}"
    if item_type == "compaction":
        enc = str(item.get("encrypted_content") or "")
        if len(enc) > limit:
            enc = enc[: limit - 3] + "..."
        return f"[compaction] {enc}"
    if item_type == "reasoning":
        return "[reasoning]"
    return f"[{item_type}]"


def merge_adjacent_message_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive ``message`` items that share the same role.

    Mirrors ``converters_core.merge_adjacent_messages`` at the Responses
    item layer so compaction sees fewer, denser turns.
    """
    if not items:
        return []

    merged: List[Dict[str, Any]] = []
    for item in items:
        item_type = item.get("type")
        if item_type is None and "role" in item:
            item_type = "message"

        last_type = merged[-1].get("type") if merged else None
        last_is_message = bool(merged) and (
            last_type == "message"
            or (last_type is None and "role" in merged[-1])
        )
        if (
            last_is_message
            and item_type == "message"
            and (merged[-1].get("role") or "user") == (item.get("role") or "user")
        ):
            last = merged[-1]
            left = _message_text(last.get("content")).rstrip()
            right = _message_text(item.get("content")).lstrip()
            combined = f"{left}\n{right}".strip() if left and right else (left or right)
            last["type"] = "message"
            last["content"] = combined
            continue

        merged.append(dict(item))

    return merged


def _summarize_dropped(dropped: List[Dict[str, Any]]) -> str:
    lines = [
        "Prior conversation context compacted by kiro-gateway (local trim; "
        "encrypted_content is a plaintext stub, not OpenAI encryption).",
        f"Dropped {len(dropped)} item(s):",
    ]
    for item in dropped:
        lines.append(f"- {_item_preview(item)}")
    return "\n".join(lines)


def compact_responses_input(
    input_data: Union[str, List[Any], None],
    *,
    max_bytes: Optional[int] = None,
    instructions: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], CompactStats]:
    """
    Compact Responses ``input`` into a shorter OpenAI-ish output window.

    Returns:
        (output_items, stats) where output_items is suitable as the next
        ``/v1/responses`` ``input`` (and as CompactedResponse.output).
    """
    budget = max_bytes if max_bytes is not None else KIRO_MAX_PAYLOAD_BYTES
    items = _normalize_input_items(input_data)
    items = merge_adjacent_message_items(items)

    # Optional instructions are folded into a leading system message so the
    # compaction window remains self-contained for the next turn.
    if instructions and str(instructions).strip():
        items = [
            {
                "type": "message",
                "role": "system",
                "content": str(instructions).strip(),
            }
        ] + items

    original_bytes = _items_bytes(items)
    original_count = len(items)

    retained = list(items)
    dropped: List[Dict[str, Any]] = []

    # Trim oldest items while over budget, always keep at least one.
    # Prefer dropping non-user messages first when the head is not a user turn,
    # then drop pairs/items from the front (AUTO_TRIM spirit).
    while len(retained) > 1 and _items_bytes(retained) > budget:
        dropped.append(retained.pop(0))

    # If still over budget with a single huge item, keep it (cannot trim further).
    trimmed = bool(dropped)
    if trimmed:
        # Replace any prior compaction stubs in the dropped set with text only.
        summary = _summarize_dropped(dropped)
        compaction_item = {
            "type": "compaction",
            "id": f"cmp_{uuid.uuid4().hex}",
            "encrypted_content": summary,
        }
        # OpenAI docs: user messages then a compaction item. We keep the
        # retained tail (includes recent user/assistant/tool items) and append
        # the compaction stub so clients can pass output through as-is.
        output = list(retained) + [compaction_item]
    else:
        output = list(retained)

    final_bytes = _items_bytes(output)
    stats = CompactStats(
        original_items=original_count,
        final_items=len(output),
        original_bytes=original_bytes,
        final_bytes=final_bytes,
        trimmed=trimmed,
        dropped_items=len(dropped),
    )
    return output, stats


def build_compacted_response(
    *,
    model: str,
    input_data: Union[str, List[Any], None],
    instructions: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a CompactedResponse-compatible JSON dict."""
    output, stats = compact_responses_input(
        input_data,
        max_bytes=max_bytes,
        instructions=instructions,
    )
    # Rough token accounting from UTF-8 bytes (same heuristic as some proxies).
    input_tokens = max(1, stats.original_bytes // 4)
    output_tokens = max(1, stats.final_bytes // 4)
    return {
        "id": generate_response_id(),
        "object": "response.compaction",
        "created_at": int(time.time()),
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "metadata": {
            "compacted": stats.trimmed,
            "original_items": stats.original_items,
            "final_items": stats.final_items,
            "dropped_items": stats.dropped_items,
            "original_bytes": stats.original_bytes,
            "final_bytes": stats.final_bytes,
        },
    }

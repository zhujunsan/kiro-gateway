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
Cursor-safe model alias generation and resolution.

Cursor sniffs provider-branded names like ``claude-opus-*`` and routes them
outside the custom OpenAI endpoint. Aliases such as ``kiro-o-5`` avoid that
while still mapping back to the real Kiro model ID.

This module has no dependency on ``kiro.config`` so config can build the static
``MODEL_ALIASES`` table from ``generate_model_alias`` without circular imports.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

# Claude family letter codes used in Cursor-safe aliases.
_CLAUDE_FAMILY_TO_CODE: Dict[str, str] = {
    "opus": "o",
    "sonnet": "s",
    "haiku": "h",
}
_CLAUDE_CODE_TO_FAMILY: Dict[str, str] = {
    code: family for family, code in _CLAUDE_FAMILY_TO_CODE.items()
}

# claude-{opus|sonnet|haiku}-{version} where version may include dots (4.5, 5).
_CLAUDE_CANONICAL_RE = re.compile(
    r"^claude-(opus|sonnet|haiku)-(.+)$",
    re.IGNORECASE,
)
# kiro-{o|s|h}-{version}
_KIRO_CLAUDE_ALIAS_RE = re.compile(
    r"^kiro-([osh])-(.+)$",
    re.IGNORECASE,
)


def generate_model_alias(model_id: str) -> Optional[str]:
    """
    Generate a Cursor-safe alias for a canonical model ID.

    Rules:
    - ``auto`` and ``gpt-*`` → ``None`` (native IDs, no alias)
    - ``claude-{opus|sonnet|haiku}-{ver}`` → ``kiro-{o|s|h}-{ver}``
    - other non-empty IDs → ``kiro-{id}``

    Args:
        model_id: Canonical model ID (e.g. ``claude-opus-5``, ``deepseek-3.2``).

    Returns:
        Alias string, or ``None`` when the model should be used under its
        real ID with no synthetic alias.

    Examples:
        >>> generate_model_alias("claude-opus-5")
        'kiro-o-5'
        >>> generate_model_alias("claude-sonnet-5")
        'kiro-s-5'
        >>> generate_model_alias("deepseek-3.2")
        'kiro-deepseek-3.2'
        >>> generate_model_alias("auto")
        >>> generate_model_alias("gpt-5.6-sol")
    """
    if not model_id:
        return None

    name = model_id.strip()
    if not name:
        return None

    lower = name.lower()
    if lower == "auto" or lower.startswith("gpt-"):
        return None

    claude_match = _CLAUDE_CANONICAL_RE.match(lower)
    if claude_match:
        family = claude_match.group(1).lower()
        version = claude_match.group(2)
        code = _CLAUDE_FAMILY_TO_CODE[family]
        return f"kiro-{code}-{version}"

    return f"kiro-{lower}"


def resolve_model_alias(
    name: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Resolve an alias (or passthrough name) to a canonical model ID.

    Resolution order:
    1. Explicit ``aliases`` table (static config / custom mappings)
    2. Syntactic reverse of ``kiro-o|s|h-*`` → ``claude-{opus|sonnet|haiku}-*``
    3. Syntactic reverse of ``kiro-*`` → strip the ``kiro-`` prefix
    4. Return ``name`` unchanged

    Args:
        name: Model name from the client (alias or real ID).
        aliases: Optional explicit alias → real ID mapping.

    Returns:
        Canonical (or best-effort) model ID to continue resolution with.

    Examples:
        >>> resolve_model_alias("kiro-o-5", {})
        'claude-opus-5'
        >>> resolve_model_alias("kiro-deepseek-3.2", {})
        'deepseek-3.2'
        >>> resolve_model_alias("my-auto", {"my-auto": "auto"})
        'auto'
        >>> resolve_model_alias("claude-opus-5", {})
        'claude-opus-5'
    """
    if not name:
        return name

    table = aliases or {}
    if name in table:
        return table[name]

    claude_alias = _KIRO_CLAUDE_ALIAS_RE.match(name)
    if claude_alias:
        code = claude_alias.group(1).lower()
        version = claude_alias.group(2)
        family = _CLAUDE_CODE_TO_FAMILY.get(code)
        if family:
            return f"claude-{family}-{version}"

    lower = name.lower()
    if lower.startswith("kiro-") and len(name) > 5:
        return name[5:]

    return name

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

"""Proxy resolution for outbound HTTP clients.

httpx only accepts the proxy schemes ``http``, ``https``, ``socks5`` and
``socks5h``. Many proxy clients (Clash, v2ray, curl, ...) export the generic
``socks://`` form in ``ALL_PROXY`` / ``HTTPS_PROXY``. Passing that straight to
httpx raises ``ValueError: Unknown scheme for proxy URL`` at client
construction, which crashes the gateway on startup.

We resolve the proxy ourselves and normalize ``socks://`` to ``socks5h://``
(the ``h`` variant resolves DNS through the proxy — the right choice when the
proxy is the only route to the upstream, e.g. behind a GFW-style network).
Callers pass the result as httpx's explicit ``proxy=`` argument.
"""
from __future__ import annotations

import os

# Ordered by httpx's own precedence for an https:// target URL: a scheme
# specific proxy wins over the catch-all ALL_PROXY.
_PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


def normalize_proxy_url(url: str | None) -> str | None:
    """Return an httpx-acceptable proxy URL, or None when there's nothing usable.

    ``socks://`` (and the odd ``socks4://``, which httpx can't do either) are
    rewritten to ``socks5h://`` so DNS is resolved proxy-side. Already-valid
    schemes pass through untouched.
    """
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    scheme, sep, rest = url.partition("://")
    if not sep:
        # No scheme: match the VPN_PROXY_URL convention in main.py and assume
        # http:// rather than guessing SOCKS.
        return f"http://{url}"
    scheme_lower = scheme.lower()
    if scheme_lower in ("socks", "socks4"):
        # socks:// / socks4:// aren't httpx schemes. Rewrite to socks5h:// so
        # DNS is resolved proxy-side (right for GFW-style networks where the
        # proxy is the only route out).
        return f"socks5h://{rest}"
    return url


def resolve_proxy() -> str | None:
    """Resolve the outbound proxy from the environment, normalized for httpx.

    Returns the first set proxy env var (httpx precedence order) with its scheme
    normalized, or None when no proxy is configured. NO_PROXY is intentionally
    not honoured here because every caller targets a remote upstream, never
    localhost.
    """
    for var in _PROXY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return normalize_proxy_url(val)
    return None

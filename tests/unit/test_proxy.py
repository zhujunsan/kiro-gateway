# -*- coding: utf-8 -*-

"""Unit tests for kiro.proxy proxy resolution/normalization.

The key regression these guard: a socks:// proxy in the environment must be
rewritten to a scheme httpx accepts (socks5h://) instead of crashing the
gateway at httpx.AsyncClient construction with "Unknown scheme for proxy URL".
"""

import httpx
import pytest

from kiro.proxy import normalize_proxy_url, resolve_proxy

_PROXY_VARS = ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
               "HTTP_PROXY", "http_proxy"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("socks://127.0.0.1:7891", "socks5h://127.0.0.1:7891"),
        ("socks4://127.0.0.1:7891", "socks5h://127.0.0.1:7891"),
        ("SOCKS://host:1080", "socks5h://host:1080"),
        ("socks://user:pass@host:1080", "socks5h://user:pass@host:1080"),
        ("socks5://127.0.0.1:7891", "socks5://127.0.0.1:7891"),
        ("socks5h://127.0.0.1:7891", "socks5h://127.0.0.1:7891"),
        ("http://127.0.0.1:7890", "http://127.0.0.1:7890"),
        ("https://proxy:8080", "https://proxy:8080"),
        ("127.0.0.1:7890", "http://127.0.0.1:7890"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_proxy_url(raw, expected):
    assert normalize_proxy_url(raw) == expected


@pytest.mark.parametrize(
    "normalized",
    ["socks5h://127.0.0.1:7891", "http://127.0.0.1:7890", "https://proxy:8080"],
)
def test_normalized_url_is_accepted_by_httpx(normalized):
    # The whole point: the normalized form must not raise at construction.
    client = httpx.Client(proxy=normalized)
    client.close()


def test_socks_scheme_would_crash_httpx_without_normalization():
    # Documents the underlying httpx limitation we work around.
    with pytest.raises(ValueError):
        httpx.Client(proxy="socks://127.0.0.1:7891")


def test_resolve_proxy_none_when_unset(monkeypatch):
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    assert resolve_proxy() is None


def test_resolve_proxy_normalizes_env(monkeypatch):
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7891")
    assert resolve_proxy() == "socks5h://127.0.0.1:7891"


def test_resolve_proxy_https_takes_precedence(monkeypatch):
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://all:1080")
    monkeypatch.setenv("HTTPS_PROXY", "http://https:8080")
    assert resolve_proxy() == "http://https:8080"

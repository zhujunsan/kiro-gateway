# -*- coding: utf-8 -*-
"""Classification of Kiro credential failures vs transport failures.

Kiro credentials expire when the user signs out of Kiro IDE / kiro-cli, or when
a re-login invalidates the stored refresh token. The AWS SSO OIDC token endpoint
answers those with HTTP 400 ``invalid_grant``, and Kiro Desktop auth answers with
400/401. Those are *user state*, not outages: retrying cannot fix them, and every
retry produces another identical failure.

Before this module the ``/usage`` probe treated any ``httpx.HTTPError`` — status
errors included — as "upstream unreachable". A signed-out user therefore polled
forever and escalated to Sentry every cooldown window, producing thousands of
events that no code change could fix (Sentry KIRO-GATEWAY-TRAY-D).

This module is the single place that answers "is this a credential problem?" so
``/usage``, account initialization, and health reporting all agree.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

# Stable machine-readable codes clients switch on. The tray uses these to stop
# polling and to prompt for re-login instead of showing a network error.
USAGE_AUTH_REQUIRED_CODE = "usage_auth_required"
ACCOUNT_AUTH_REQUIRED_CODE = "account_auth_required"
NO_CREDENTIALS_CODE = "account_not_configured"

# HTTP statuses the IdP uses for "these credentials are no longer valid".
# 400 is what AWS SSO OIDC returns for invalid_grant / invalid_client.
_AUTH_FAILURE_STATUSES = frozenset({400, 401, 403})

# Substrings that mark an exception message as credential-related even when no
# HTTP response is attached (e.g. the ValueError raised by get_access_token()
# after refresh exhaustion).
_AUTH_FAILURE_MARKERS = (
    "invalid_grant",
    "invalid_client",
    "refresh token is not set",
    "token expired and refresh failed",
    "failed to obtain access token",
    "does not contain accesstoken",
    "client id is not set",
    "client secret is not set",
)


def is_auth_failure(exc: BaseException) -> bool:
    """Return whether ``exc`` means "Kiro credentials are invalid / absent".

    A credential failure is permanent until the user re-authenticates, so callers
    must stop retrying and stop reporting it as an outage.

    Args:
        exc: Exception raised while obtaining a token or calling upstream.

    Returns:
        True when the failure is caused by invalid, expired, or missing Kiro
        credentials; False for transport errors (connect/timeout/DNS/TLS) and
        for upstream capacity errors (429/5xx).

    Examples:
        >>> is_auth_failure(httpx.ConnectError("proxy down"))
        False
        >>> is_auth_failure(ValueError("Failed to obtain access token"))
        True
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _AUTH_FAILURE_STATUSES

    # ValueError is what KiroAuthManager raises once refresh is unrecoverable.
    if isinstance(exc, ValueError):
        return True

    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_FAILURE_MARKERS)


def auth_failure_reason(exc: BaseException) -> str:
    """Return a short, stable reason tag for logs, tags, and fingerprints.

    Args:
        exc: Credential-related exception (see :func:`is_auth_failure`).

    Returns:
        One of ``"invalid_grant"``, ``"unauthorized"``, ``"forbidden"``,
        ``"missing_credentials"``, or ``"refresh_failed"``.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 400:
            return "invalid_grant"
        if status == 401:
            return "unauthorized"
        if status == 403:
            return "forbidden"

    text = str(exc).lower()
    if "is not set" in text:
        return "missing_credentials"
    return "refresh_failed"


def transport_failure_reason(exc: BaseException) -> str:
    """Return a coarse cause tag used to fingerprint genuine outages.

    Grouping every ``/usage`` failure under one fingerprint made DNS failures,
    TLS handshake failures, and proxy timeouts indistinguishable in Sentry. This
    splits them so each cause can be triaged on its own.

    Args:
        exc: Transport or upstream-status error after retries were exhausted.

    Returns:
        Short tag such as ``"dns"``, ``"tls"``, ``"timeout"``, ``"connect"``,
        ``"upstream_status"``, or ``"other"``.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return "upstream_status"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"

    text = str(exc).lower()
    if "ssl" in text or "handshake" in text or "certificate" in text:
        return "tls"
    if (
        "nodename nor servname" in text
        or "name or service not known" in text
        or "temporary failure in name resolution" in text
        or "getaddrinfo" in text
    ):
        return "dns"
    if isinstance(exc, httpx.ConnectError):
        return "connect"
    return "other"


def build_auth_required_payload(
    *,
    code: str,
    message: str,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Build the error body shared by ``/usage`` and account-level auth errors.

    Args:
        code: Stable client-facing code (see module constants).
        message: Actionable, human-readable instruction.
        reason: Optional short cause tag from :func:`auth_failure_reason`.

    Returns:
        Dict shaped like the gateway's other error bodies, plus
        ``login_required: True`` so clients need not match on ``code``.
    """
    error: dict[str, Any] = {
        "message": message,
        "type": code,
        "code": code,
        "login_required": True,
    }
    if reason:
        error["reason"] = reason
    return {"error": error}

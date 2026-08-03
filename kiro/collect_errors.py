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
Failure classification for the non-streaming (collect) request path.

The streaming path classifies transport failures via
``debug_logger.classify_streaming_exception`` so a stalled or truncated upstream
response is reported as a network incident (502/504). The non-streaming path used
to funnel every exception into a generic ``500 / source="gateway"`` bucket, which
mislabels upstream problems as gateway bugs and hides them from network dashboards:

* ``FirstTokenTimeoutError``   -> 504 ``first_token_timeout``  (upstream never spoke)
* ``httpx.RemoteProtocolError`` -> 502 ``incomplete_upstream_response`` (stream cut)
* ``httpx.TimeoutException``   -> 504 ``timeout``
* other ``httpx.TransportError`` -> 502 ``connection_error``
* anything else                -> 500 ``gateway`` (a real gateway bug)

This module owns that mapping for every non-streaming caller (OpenAI, Anthropic,
Responses) so the two paths stay consistent and new transports only need one edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass(frozen=True)
class CollectFailure:
    """Classification of a non-streaming collect failure.

    Attributes:
        status_code: HTTP status the client should receive.
        source: Incident source bucket (``network`` / ``gateway``).
        code: Stable incident code, aligned with the streaming classifier.
        phase: Request phase in which the failure happened.
        message: User-facing message describing the failure.
        is_upstream: Whether the failure originates upstream rather than in the
            gateway itself. Callers use it to pick log severity / failover.
    """

    status_code: int
    source: str
    code: str
    phase: str
    message: str
    is_upstream: bool


def _exception_detail(exc: BaseException) -> str:
    """Describe ``exc`` for a client message, never returning an empty string.

    httpx routinely raises transport errors with no message (e.g.
    ``ConnectError("")``); falling back to the class name keeps the response
    body meaningful instead of trailing off after a colon.
    """
    text = str(exc).strip()
    return text or type(exc).__name__


def _first_token_timeout_message(exc: BaseException) -> str:
    """Prefer the retry-exhaustion text, else build an actionable default."""
    text = str(exc).strip()
    if "attempt" in text.lower():
        return text
    return (
        f"{text or 'Model did not respond in time'}. "
        "The model accepted the request but produced no output. Please try again."
    )


def classify_collect_exception(exc: BaseException) -> CollectFailure:
    """Map a non-streaming collect exception to a client-facing failure.

    Args:
        exc: Exception raised while requesting or collecting the Kiro response.

    Returns:
        A :class:`CollectFailure` with status code, incident tags and message.
    """
    # Imported lazily: streaming_core imports config/parsers, and importing it at
    # module scope would create a cycle for callers that import this first.
    from kiro.streaming_core import FirstTokenTimeoutError

    if isinstance(exc, FirstTokenTimeoutError):
        return CollectFailure(
            status_code=504,
            source="network",
            code="first_token_timeout",
            phase="first_token",
            message=_first_token_timeout_message(exc),
            is_upstream=True,
        )

    if isinstance(exc, httpx.RemoteProtocolError):
        return CollectFailure(
            status_code=502,
            source="network",
            code="incomplete_upstream_response",
            phase="streaming",
            message=(
                "Upstream closed the connection before the response was complete: "
                f"{_exception_detail(exc)}. Please try again."
            ),
            is_upstream=True,
        )

    if isinstance(exc, httpx.TimeoutException):
        return CollectFailure(
            status_code=504,
            source="network",
            code="timeout",
            phase="streaming",
            message=(
                "Upstream timed out while sending the response: "
                f"{_exception_detail(exc)}. Please try again."
            ),
            is_upstream=True,
        )

    if isinstance(exc, httpx.TransportError):
        return CollectFailure(
            status_code=502,
            source="network",
            code="connection_error",
            phase="connect",
            message=(
                "Connection to upstream failed: "
                f"{_exception_detail(exc)}. Check your network and try again."
            ),
            is_upstream=True,
        )

    return CollectFailure(
        status_code=500,
        source="gateway",
        code=type(exc).__name__,
        phase="unknown",
        message=f"Internal Server Error: {exc}",
        is_upstream=False,
    )


def collect_failure_or_none(exc: BaseException) -> Optional[CollectFailure]:
    """Return a classification only for recognized upstream transport failures.

    Args:
        exc: Exception to inspect.

    Returns:
        The :class:`CollectFailure` when the error is an upstream transport
        problem, or ``None`` when it should be treated as a gateway bug.
    """
    failure = classify_collect_exception(exc)
    return failure if failure.is_upstream else None


__all__ = [
    "CollectFailure",
    "classify_collect_exception",
    "collect_failure_or_none",
]

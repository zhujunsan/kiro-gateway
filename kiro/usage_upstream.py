# -*- coding: utf-8 -*-
"""Amazon Q usage probe helpers with retry and rate-limited outage reporting.

``GET /usage`` is polled by the tray menu and credit sampler. Occasional proxy
or upstream ConnectError — including AWS SSO OIDC token refresh — must not spam
Sentry or surface as hard 500s. Callers already treat non-200 as a soft miss;
this module soft-fails transport errors, retries getUsageLimits with backoff,
and only escalates when failures look sustained (outage-like).
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi.responses import JSONResponse
from loguru import logger

# Per-request transport retries (not counting the first attempt).
USAGE_UPSTREAM_MAX_ATTEMPTS = 3
USAGE_UPSTREAM_BASE_DELAY_S = 0.4
USAGE_UPSTREAM_MAX_DELAY_S = 2.5

# Escalate to Sentry only after a sustained failure streak, then cooldown.
USAGE_OUTAGE_CONSECUTIVE_THRESHOLD = 5
USAGE_OUTAGE_DURATION_S = 120.0
USAGE_OUTAGE_REPORT_COOLDOWN_S = 900.0

# Upstream HTTP statuses worth retrying (capacity / gateway blips).
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


@dataclass
class UsageFetchResult:
    """Outcome of a probed getUsageLimits call after retries."""

    response: Optional[httpx.Response]
    error: Optional[BaseException]
    attempts: int
    reported_outage: bool


class UsageUpstreamMonitor:
    """Track consecutive /usage upstream failures and gate Sentry reports."""

    def __init__(self) -> None:
        self._consecutive_failures = 0
        self._first_failure_at: Optional[float] = None
        self._last_report_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def consecutive_failures(self) -> int:
        """Return the current consecutive soft-failure count."""
        return self._consecutive_failures

    async def note_success(self) -> None:
        """Reset the failure streak after a successful upstream read."""
        async with self._lock:
            self._consecutive_failures = 0
            self._first_failure_at = None

    async def note_failure(self, *, now: Optional[float] = None) -> bool:
        """Record a soft failure and decide whether to report an outage.

        Args:
            now: Optional monotonic-ish timestamp (``time.time()``); injectable
                for tests.

        Returns:
            ``True`` when this failure should trigger a Sentry/outage report.
        """
        ts = time.time() if now is None else now
        async with self._lock:
            if self._consecutive_failures == 0:
                self._first_failure_at = ts
            self._consecutive_failures += 1
            duration = 0.0 if self._first_failure_at is None else ts - self._first_failure_at
            sustained = (
                self._consecutive_failures >= USAGE_OUTAGE_CONSECUTIVE_THRESHOLD
                or duration >= USAGE_OUTAGE_DURATION_S
            )
            if not sustained:
                return False
            if (
                self._last_report_at is not None
                and ts - self._last_report_at < USAGE_OUTAGE_REPORT_COOLDOWN_S
            ):
                return False
            self._last_report_at = ts
            return True


# Process-wide monitor so tray polling builds a real failure streak.
usage_upstream_monitor = UsageUpstreamMonitor()


def build_usage_network_error_response(exc: BaseException) -> JSONResponse:
    """Return a soft 503 JSON body for unreachable usage upstream.

    Args:
        exc: Last transport/HTTP error after retries were exhausted.

    Returns:
        JSONResponse with a stable ``usage_upstream_unreachable`` code.
    """
    message = (
        "Unable to reach Amazon Q usage API (network/proxy). "
        "Quota display will refresh when connectivity recovers."
    )
    logger.debug(
        "GET /usage returning soft 503 after retries: {}: {}",
        type(exc).__name__,
        exc,
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": message,
                "type": "usage_upstream_unreachable",
                "code": "usage_upstream_unreachable",
            }
        },
    )


def build_usage_account_unavailable_response() -> JSONResponse:
    """Return a soft 503 when no Kiro account is initialized yet."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "No initialized Kiro account available yet. Retry shortly.",
                "type": "usage_account_unavailable",
                "code": "usage_account_unavailable",
            }
        },
    )


async def finalize_usage_transport_failure(
    exc: BaseException,
    *,
    attempts: int = 1,
    monitor: Optional[UsageUpstreamMonitor] = None,
) -> JSONResponse:
    """Record a transport soft-failure and return the shared 503 JSON.

    Used for both AWS SSO token refresh failures and getUsageLimits transport
    exhaustion so /usage never bubbles ConnectError/ReadTimeout as HTTP 500.

    Args:
        exc: Transport or HTTP error that ended the attempt.
        attempts: How many tries were made for this request (token path is 1).
        monitor: Failure-streak monitor; defaults to the process singleton.

    Returns:
        Soft 503 JSONResponse with ``usage_upstream_unreachable``.
    """
    mon = monitor if monitor is not None else usage_upstream_monitor
    should_report = await mon.note_failure()
    if should_report:
        report_usage_outage(
            exc,
            consecutive=mon.consecutive_failures,
            attempts=attempts,
        )
    return build_usage_network_error_response(exc)


async def obtain_usage_access_token(
    auth: Any,
    *,
    monitor: Optional[UsageUpstreamMonitor] = None,
) -> tuple[Optional[str], Optional[JSONResponse]]:
    """Resolve a bearer token for GET /usage without raising transport errors.

    ``auth.get_access_token()`` may refresh via AWS SSO OIDC. Proxy/IdP blips
    raise ``httpx.ConnectError`` / ``httpx.ReadTimeout`` (subclasses of
    ``httpx.HTTPError``). Those must soft-fail like getUsageLimits, not 500.

    Args:
        auth: Initialized ``KiroAuthManager`` (duck-typed: ``get_access_token``).
        monitor: Optional outage monitor override for tests.

    Returns:
        ``(token, None)`` on success, or ``(None, soft_503_response)`` on
        ``httpx.HTTPError``. Non-HTTP errors (e.g. ``ValueError``) propagate.
    """
    try:
        token = await auth.get_access_token()
    except httpx.HTTPError as exc:
        logger.debug(
            "GET /usage token refresh transport error: {}: {}",
            type(exc).__name__,
            exc,
        )
        return None, await finalize_usage_transport_failure(
            exc,
            attempts=1,
            monitor=monitor,
        )
    return token, None


def _retry_delay_seconds(attempt_index: int) -> float:
    """Exponential backoff with jitter for attempt ``attempt_index`` (0-based)."""
    base = min(
        USAGE_UPSTREAM_MAX_DELAY_S,
        USAGE_UPSTREAM_BASE_DELAY_S * (2 ** attempt_index),
    )
    return base + random.uniform(0.0, 0.2)


def _should_retry_status(status_code: int) -> bool:
    """Return whether an upstream HTTP status warrants another attempt."""
    return status_code in _RETRYABLE_STATUS


def report_usage_outage(
    exc: BaseException,
    *,
    consecutive: int,
    attempts: int,
) -> None:
    """Best-effort Sentry/log escalation for sustained /usage outages.

    Occasional failures stay at DEBUG/soft-503 only. When the monitor says the
    streak looks like an outage, emit one warning log and one Sentry event
    (if the SDK is initialized). Never raises.
    """
    logger.warning(
        "GET /usage upstream outage after {} consecutive soft failures "
        "(last request attempts={}): {}: {}",
        consecutive,
        attempts,
        type(exc).__name__,
        exc,
    )
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("subsystem", "usage_upstream")
            scope.set_tag("usage_outage", "true")
            scope.set_extra("consecutive_failures", consecutive)
            scope.set_extra("request_attempts", attempts)
            scope.fingerprint = ["kiro-usage-upstream-outage"]
            sentry_sdk.capture_message(
                f"GET /usage upstream unreachable (consecutive={consecutive}): "
                f"{type(exc).__name__}: {exc}",
                level="error",
            )
    except Exception as report_exc:  # noqa: BLE001 — never break /usage on reporter failure
        logger.debug("GET /usage outage report skipped: {}", report_exc)


async def fetch_usage_limits(
    *,
    url: str,
    headers: dict[str, str],
    proxy: Optional[str],
    timeout: float = 60.0,
    monitor: Optional[UsageUpstreamMonitor] = None,
    sleep: Any = asyncio.sleep,
    transport_get: Any = None,
) -> UsageFetchResult:
    """GET getUsageLimits with bounded backoff retries and outage gating.

    Args:
        url: Fully built Amazon Q usage URL.
        headers: Request headers including bearer token.
        proxy: Optional HTTP(S) proxy URL for httpx.
        timeout: Per-attempt timeout seconds.
        monitor: Failure-streak monitor; defaults to the process singleton.
        sleep: Awaitable sleep (injectable for tests).
        transport_get: Optional async ``(url, headers) -> Response`` override
            used by unit tests instead of creating a real httpx client.

    Returns:
        ``UsageFetchResult`` with either a response or the last error.
    """
    mon = monitor if monitor is not None else usage_upstream_monitor
    last_error: Optional[BaseException] = None
    attempts = 0

    for attempt in range(USAGE_UPSTREAM_MAX_ATTEMPTS):
        attempts = attempt + 1
        try:
            if transport_get is not None:
                resp = await transport_get(url, headers)
            else:
                async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                    resp = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            last_error = exc
            logger.debug(
                "GET /usage attempt {}/{} transport error: {}: {}",
                attempts,
                USAGE_UPSTREAM_MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
            )
            if attempt + 1 >= USAGE_UPSTREAM_MAX_ATTEMPTS:
                break
            await sleep(_retry_delay_seconds(attempt))
            continue

        if resp.status_code == 200 or not _should_retry_status(resp.status_code):
            await mon.note_success()
            return UsageFetchResult(
                response=resp,
                error=None,
                attempts=attempts,
                reported_outage=False,
            )

        last_error = httpx.HTTPStatusError(
            f"upstream status {resp.status_code}",
            request=resp.request,
            response=resp,
        )
        logger.debug(
            "GET /usage attempt {}/{} retryable status {}",
            attempts,
            USAGE_UPSTREAM_MAX_ATTEMPTS,
            resp.status_code,
        )
        if attempt + 1 >= USAGE_UPSTREAM_MAX_ATTEMPTS:
            # Return the last response so callers can forward status/body.
            should_report = await mon.note_failure()
            if should_report:
                report_usage_outage(
                    last_error,
                    consecutive=mon.consecutive_failures,
                    attempts=attempts,
                )
            return UsageFetchResult(
                response=resp,
                error=last_error,
                attempts=attempts,
                reported_outage=should_report,
            )
        await sleep(_retry_delay_seconds(attempt))

    assert last_error is not None
    should_report = await mon.note_failure()
    if should_report:
        report_usage_outage(
            last_error,
            consecutive=mon.consecutive_failures,
            attempts=attempts,
        )
    return UsageFetchResult(
        response=None,
        error=last_error,
        attempts=attempts,
        reported_outage=should_report,
    )

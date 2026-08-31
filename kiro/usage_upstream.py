# -*- coding: utf-8 -*-
"""Amazon Q usage probe helpers with retry and rate-limited outage reporting.

``GET /usage`` is polled by the tray menu and credit sampler. Occasional proxy
or upstream ConnectError — including AWS SSO OIDC token refresh — must not spam
Sentry or surface as hard 500s. Callers already treat non-200 as a soft miss;
this module soft-fails transport errors, retries getUsageLimits with backoff,
and only escalates when failures look sustained (outage-like).

Credential failures are handled separately from transport failures. When the
user is signed out of Kiro, the OIDC token endpoint answers 400 forever; no
amount of retrying or reporting helps. Those return
``usage_auth_required`` (see :mod:`kiro.auth_state`) so the tray can stop polling
and prompt for re-login, and they are reported to Sentry at most once per
process instead of every cooldown window.
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

from kiro.auth_state import (
    USAGE_AUTH_REQUIRED_CODE,
    auth_failure_reason,
    build_auth_required_payload,
    is_auth_failure,
    transport_failure_reason,
)

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

# Upstream statuses meaning "this bearer token is not accepted" — a credential
# problem the user must fix by signing in again, never a transient blip.
_AUTH_REJECT_STATUS = frozenset({401, 403})


@dataclass
class UsageFetchResult:
    """Outcome of a probed getUsageLimits call after retries.

    ``auth_required`` marks the upstream rejecting the bearer token (401/403).
    Callers must surface that as ``usage_auth_required`` rather than forwarding
    the raw status, so clients stop polling instead of retrying forever.
    """

    response: Optional[httpx.Response]
    error: Optional[BaseException]
    attempts: int
    reported_outage: bool
    auth_required: bool = False


class UsageUpstreamMonitor:
    """Track consecutive /usage upstream failures and gate Sentry reports."""

    def __init__(self) -> None:
        self._consecutive_failures = 0
        self._first_failure_at: Optional[float] = None
        self._last_report_at: Optional[float] = None
        self._auth_reported = False
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
            self._auth_reported = False

    async def note_auth_failure(self) -> bool:
        """Record a credential failure and decide whether to report it once.

        Credential failures repeat identically until the user re-authenticates,
        so streak counting and cooldowns are the wrong model: they still produce
        an event every window forever. Report at most once per process, and reset
        only after a success (see :meth:`note_success`).

        Returns:
            ``True`` only for the first credential failure since the last success.
        """
        async with self._lock:
            if self._auth_reported:
                return False
            self._auth_reported = True
            return True

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


def build_usage_login_required_response(*, code: str, message: str) -> JSONResponse:
    """Return 401 for a known "no usable credentials" account state.

    Used when the account pool itself reports that credentials are missing or
    expired, so ``/usage`` never had a token to try. Shares the response shape
    with :func:`build_usage_auth_required_response` so clients need one branch.

    Args:
        code: ``account_auth_required`` or ``account_not_configured``.
        message: Actionable instruction from ``AccountManager.describe_init_failure``.

    Returns:
        401 JSONResponse carrying ``login_required``.
    """
    return JSONResponse(
        status_code=401,
        content=build_auth_required_payload(code=code, message=message),
    )


def build_usage_auth_required_response(exc: BaseException) -> JSONResponse:
    """Return 401 ``usage_auth_required`` for expired / missing Kiro credentials.

    401 (not 503) is deliberate: 503 means "try again later", which is exactly
    what made clients poll a signed-out account forever. 401 tells the caller the
    request cannot succeed until the user re-authenticates.

    Args:
        exc: Credential failure classified by ``auth_state.is_auth_failure``.

    Returns:
        JSONResponse with ``login_required`` and an actionable message.
    """
    reason = auth_failure_reason(exc)
    logger.warning(
        "GET /usage requires Kiro re-login ({}): {}: {}",
        reason,
        type(exc).__name__,
        exc,
    )
    return JSONResponse(
        status_code=401,
        content=build_auth_required_payload(
            code=USAGE_AUTH_REQUIRED_CODE,
            message=(
                "Kiro credentials are expired or missing. Open Kiro (or run "
                "'kiro-cli login') and sign in again, then retry."
            ),
            reason=reason,
        ),
    )


async def finalize_usage_transport_failure(
    exc: BaseException,
    *,
    attempts: int = 1,
    monitor: Optional[UsageUpstreamMonitor] = None,
) -> JSONResponse:
    """Record a soft failure and return the appropriate error JSON.

    Used for both AWS SSO token refresh failures and getUsageLimits transport
    exhaustion so /usage never bubbles ConnectError/ReadTimeout as HTTP 500.

    Credential failures short-circuit here: they return 401
    ``usage_auth_required`` and never touch the outage streak, so a signed-out
    user cannot look like an Amazon Q outage.

    Args:
        exc: Transport, credential, or HTTP error that ended the attempt.
        attempts: How many tries were made for this request (token path is 1).
        monitor: Failure-streak monitor; defaults to the process singleton.

    Returns:
        401 ``usage_auth_required`` for credential failures, otherwise a soft 503
        ``usage_upstream_unreachable``.
    """
    mon = monitor if monitor is not None else usage_upstream_monitor
    if is_auth_failure(exc):
        if await mon.note_auth_failure():
            report_usage_auth_required(exc)
        return build_usage_auth_required_response(exc)

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
    """Resolve a bearer token for GET /usage without raising.

    ``auth.get_access_token()`` may refresh via AWS SSO OIDC and can fail in two
    fundamentally different ways:

    * Transport blips (``httpx.ConnectError`` / ``ReadTimeout``) — retryable, so
      they soft-fail as 503 and feed the outage streak.
    * Credential failures (``HTTPStatusError`` 400 ``invalid_grant`` after a Kiro
      sign-out, or ``ValueError`` once refresh is unrecoverable) — permanent until
      the user signs in again, so they return 401 ``usage_auth_required``.

    ``ValueError`` used to propagate and surface as HTTP 500; it is a signed-out
    user, not a gateway bug.

    Args:
        auth: Initialized ``KiroAuthManager`` (duck-typed: ``get_access_token``).
        monitor: Optional outage monitor override for tests.

    Returns:
        ``(token, None)`` on success, or ``(None, error_response)`` where the
        response is 401 for credential failures and 503 for transport failures.
    """
    try:
        token = await auth.get_access_token()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug(
            "GET /usage token refresh failed: {}: {}",
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


def report_usage_auth_required(exc: BaseException) -> None:
    """Log a signed-out account at WARNING and tag it for Sentry-side filtering.

    This is deliberately *not* an error event. A signed-out user is not an
    application fault, and reporting it as one buried the real outages: a single
    account reached ``consecutive=6883``, one Sentry event per cooldown window.
    Clients are told via ``usage_auth_required``; here we only leave a local
    breadcrumb, plus a tagged Sentry breadcrumb when the SDK is present so the
    event that *does* matter (a later genuine outage) carries the context.

    Never raises.
    """
    reason = auth_failure_reason(exc)
    logger.warning(
        "GET /usage stopped: Kiro credentials require re-login ({}). "
        "Open Kiro and sign in again. Underlying error: {}: {}",
        reason,
        type(exc).__name__,
        exc,
    )
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        sentry_sdk.add_breadcrumb(
            category="usage",
            level="warning",
            message="GET /usage requires Kiro re-login",
            data={"reason": reason, "error": type(exc).__name__},
        )
    except Exception as report_exc:  # noqa: BLE001 — never break /usage on reporter failure
        logger.debug("GET /usage auth breadcrumb skipped: {}", report_exc)


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

    The fingerprint includes the failure cause so DNS, TLS, timeout, and proxy
    failures form separate Issues; a single shared fingerprint previously made
    them impossible to triage apart.
    """
    cause = transport_failure_reason(exc)
    logger.warning(
        "GET /usage upstream outage after {} consecutive soft failures "
        "(cause={}, last request attempts={}): {}: {}",
        consecutive,
        cause,
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
            scope.set_tag("usage_outage_cause", cause)
            scope.set_extra("consecutive_failures", consecutive)
            scope.set_extra("request_attempts", attempts)
            scope.fingerprint = ["kiro-usage-upstream-outage", cause]
            sentry_sdk.capture_message(
                f"GET /usage upstream unreachable (cause={cause}, "
                f"consecutive={consecutive}): {type(exc).__name__}: {exc}",
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

        if resp.status_code == 200:
            await mon.note_success()
            return UsageFetchResult(
                response=resp,
                error=None,
                attempts=attempts,
                reported_outage=False,
            )

        if not _should_retry_status(resp.status_code):
            # Amazon Q rejecting the bearer token is a credential problem, not a
            # transient one: report it once and tell the caller to stop polling.
            if resp.status_code in _AUTH_REJECT_STATUS:
                auth_error = httpx.HTTPStatusError(
                    f"upstream rejected credentials with status {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                if await mon.note_auth_failure():
                    report_usage_auth_required(auth_error)
                return UsageFetchResult(
                    response=resp,
                    error=auth_error,
                    attempts=attempts,
                    reported_outage=False,
                    auth_required=True,
                )
            # Other non-retryable statuses (e.g. 400 validation) are forwarded
            # as-is; they are upstream feedback, not connectivity failures.
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

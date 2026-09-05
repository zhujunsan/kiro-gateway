"""Unit tests for /usage upstream retry and outage reporting."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kiro.auth_state import (
    ACCOUNT_AUTH_REQUIRED_CODE,
    NO_CREDENTIALS_CODE,
    USAGE_AUTH_REQUIRED_CODE,
    auth_failure_reason,
    build_auth_required_payload,
    is_auth_failure,
    transport_failure_reason,
)
from kiro.usage_upstream import (
    USAGE_OUTAGE_CONSECUTIVE_THRESHOLD,
    USAGE_OUTAGE_DURATION_S,
    USAGE_OUTAGE_REPORT_COOLDOWN_S,
    USAGE_UPSTREAM_MAX_ATTEMPTS,
    UsageUpstreamMonitor,
    build_usage_auth_required_response,
    build_usage_login_required_response,
    build_usage_network_error_response,
    fetch_usage_limits,
    finalize_usage_transport_failure,
    obtain_usage_access_token,
    report_usage_auth_required,
    report_usage_outage,
)


# The tray polls GET /usage about once a minute; escalation gates are tuned to it.
_TRAY_POLL_INTERVAL_S = 60.0

# Failed polls needed to clear the count gate AND the duration gate at 60s/poll.
_POLLS_TO_ESCALATE = max(
    USAGE_OUTAGE_CONSECUTIVE_THRESHOLD,
    int(USAGE_OUTAGE_DURATION_S // _TRAY_POLL_INTERVAL_S) + 1,
)


def _polling_clock(interval: float = _TRAY_POLL_INTERVAL_S) -> Callable[[], float]:
    """Build a clock advancing one tray poll interval per call.

    Needed for paths like ``finalize_usage_transport_failure`` and
    ``fetch_usage_limits`` that cannot forward ``now`` to the monitor.

    Args:
        interval: Simulated seconds between consecutive polls.

    Returns:
        Zero-arg callable returning increasing timestamps.
    """
    state = SimpleNamespace(now=1_000.0 - interval)

    def clock() -> float:
        state.now += interval
        return state.now

    return clock


def _status_error(status: int, url: str = "https://oidc.test/token") -> httpx.HTTPStatusError:
    """Build an HTTPStatusError carrying a real response, like httpx does."""
    request = httpx.Request("POST", url)
    response = httpx.Response(status, request=request, text="body")
    return httpx.HTTPStatusError(f"status {status}", request=request, response=response)


class TestIsAuthFailure:
    """Credential failures must be told apart from transport failures."""

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_idp_rejection_statuses_are_auth_failures(self, status: int) -> None:
        assert is_auth_failure(_status_error(status)) is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_capacity_statuses_are_not_auth_failures(self, status: int) -> None:
        assert is_auth_failure(_status_error(status)) is False

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("All connection attempts failed"),
            httpx.ReadTimeout("slow"),
            httpx.ConnectTimeout("slow connect"),
            httpx.ConnectError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert"),
            httpx.ConnectError("[Errno 8] nodename nor servname provided"),
        ],
    )
    def test_transport_errors_are_not_auth_failures(self, exc: Exception) -> None:
        assert is_auth_failure(exc) is False

    @pytest.mark.parametrize(
        "message",
        [
            "Token expired and refresh failed. Please run 'kiro-cli login'",
            "Failed to obtain access token",
            "Refresh token is not set",
            "Client ID is not set (required for AWS SSO OIDC)",
        ],
    )
    def test_value_errors_from_auth_manager_are_auth_failures(self, message: str) -> None:
        assert is_auth_failure(ValueError(message)) is True

    def test_marker_match_without_response_object(self) -> None:
        assert is_auth_failure(RuntimeError("upstream said invalid_grant")) is True

    def test_unrelated_runtime_error_is_not_auth_failure(self) -> None:
        assert is_auth_failure(RuntimeError("disk full")) is False


class TestFailureReasonTags:
    """Reason tags drive log text, Sentry tags, and fingerprints."""

    def test_400_maps_to_invalid_grant(self) -> None:
        assert auth_failure_reason(_status_error(400)) == "invalid_grant"

    def test_401_and_403_map_distinctly(self) -> None:
        assert auth_failure_reason(_status_error(401)) == "unauthorized"
        assert auth_failure_reason(_status_error(403)) == "forbidden"

    def test_missing_credentials_detected_from_message(self) -> None:
        assert auth_failure_reason(ValueError("Client secret is not set")) == (
            "missing_credentials"
        )

    def test_generic_auth_failure_falls_back_to_refresh_failed(self) -> None:
        assert auth_failure_reason(ValueError("Failed to obtain access token")) == (
            "refresh_failed"
        )

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (httpx.ReadTimeout("x"), "timeout"),
            (httpx.ConnectTimeout("x"), "timeout"),
            (httpx.ConnectError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE]"), "tls"),
            (httpx.ConnectError("[Errno 8] nodename nor servname provided"), "dns"),
            (httpx.ConnectError("All connection attempts failed"), "connect"),
            (httpx.HTTPError("weird"), "other"),
        ],
    )
    def test_transport_causes_are_split(self, exc: Exception, expected: str) -> None:
        assert transport_failure_reason(exc) == expected

    def test_status_error_is_upstream_status(self) -> None:
        assert transport_failure_reason(_status_error(503)) == "upstream_status"


class TestBuildAuthRequiredPayload:
    """Response bodies must be switchable without string matching."""

    def test_payload_carries_code_and_login_flag(self) -> None:
        payload = build_auth_required_payload(
            code=USAGE_AUTH_REQUIRED_CODE, message="sign in", reason="invalid_grant"
        )
        assert payload["error"]["code"] == USAGE_AUTH_REQUIRED_CODE
        assert payload["error"]["type"] == USAGE_AUTH_REQUIRED_CODE
        assert payload["error"]["login_required"] is True
        assert payload["error"]["reason"] == "invalid_grant"

    def test_reason_is_omitted_when_absent(self) -> None:
        payload = build_auth_required_payload(code="x", message="y")
        assert "reason" not in payload["error"]


class TestAuthRequiredResponses:
    """401, not 503: 503 is what made clients poll a signed-out account."""

    def test_auth_required_response_is_401_with_actionable_message(self) -> None:
        response = build_usage_auth_required_response(_status_error(400))
        assert response.status_code == 401
        assert USAGE_AUTH_REQUIRED_CODE.encode() in response.body
        assert b"login_required" in response.body
        assert b"Open Kiro" in response.body

    @pytest.mark.parametrize(
        "code", [ACCOUNT_AUTH_REQUIRED_CODE, NO_CREDENTIALS_CODE]
    )
    def test_login_required_response_carries_account_code(self, code: str) -> None:
        response = build_usage_login_required_response(code=code, message="do this")
        assert response.status_code == 401
        assert code.encode() in response.body
        assert b"login_required" in response.body


@pytest.mark.asyncio
class TestMonitorAuthGating:
    """A signed-out account must produce one report, not one per window."""

    async def test_auth_failure_reports_only_once(self) -> None:
        monitor = UsageUpstreamMonitor()
        assert await monitor.note_auth_failure() is True
        for _ in range(50):
            assert await monitor.note_auth_failure() is False

    async def test_success_rearms_auth_reporting(self) -> None:
        monitor = UsageUpstreamMonitor()
        assert await monitor.note_auth_failure() is True
        await monitor.note_success()
        assert await monitor.note_auth_failure() is True

    async def test_auth_failures_do_not_advance_outage_streak(self) -> None:
        monitor = UsageUpstreamMonitor()
        for _ in range(20):
            await monitor.note_auth_failure()
        assert monitor.consecutive_failures == 0


@pytest.mark.asyncio
class TestFinalizeSplitsAuthFromTransport:
    """finalize_usage_transport_failure is the shared soft-fail entry point."""

    async def test_auth_failure_returns_401_and_never_reports_outage(self) -> None:
        monitor = UsageUpstreamMonitor()
        fake_sdk = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD * 3):
                response = await finalize_usage_transport_failure(
                    _status_error(400), attempts=1, monitor=monitor
                )
                assert response.status_code == 401
        fake_sdk.capture_message.assert_not_called()
        assert monitor.consecutive_failures == 0

    async def test_transport_failure_still_escalates(self) -> None:
        """A real sustained outage must still reach Sentry exactly once.

        Uses an injected 60s-per-poll clock (the tray cadence) because the
        duration gate is now ANDed with the count gate; the previous version of
        this test escalated on wall-clock-instant failures, which the fix
        deliberately no longer does.
        """
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        fake_sdk = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            for _ in range(_POLLS_TO_ESCALATE):
                response = await finalize_usage_transport_failure(
                    httpx.ConnectError("down"), attempts=1, monitor=monitor
                )
                assert response.status_code == 503
        fake_sdk.capture_message.assert_called_once()

    async def test_two_failed_polls_no_longer_escalate(self) -> None:
        """Regression for KIRO-GATEWAY-TRAY-20 / -22 (both fired at consecutive=2).

        Two failed 60s-apart polls used to satisfy the duration gate on its own
        and open a Sentry Issue. They must now stay a silent soft 503.
        """
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        fake_sdk = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            for exc in (httpx.ConnectError("blip"), httpx.ReadTimeout("slow")):
                response = await finalize_usage_transport_failure(
                    exc, attempts=3, monitor=monitor
                )
                assert response.status_code == 503
        fake_sdk.capture_message.assert_not_called()
        assert monitor.consecutive_failures == 2

    async def test_auth_then_transport_are_tracked_independently(self) -> None:
        monitor = UsageUpstreamMonitor()
        await finalize_usage_transport_failure(
            _status_error(401), attempts=1, monitor=monitor
        )
        assert monitor.consecutive_failures == 0
        await finalize_usage_transport_failure(
            httpx.ReadTimeout("slow"), attempts=1, monitor=monitor
        )
        assert monitor.consecutive_failures == 1


@pytest.mark.asyncio
class TestFetchUsageLimitsAuthRejection:
    """Amazon Q rejecting our bearer token is a credential problem."""

    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_reject_status_sets_auth_required(self, status: int) -> None:
        async def rejecting(url: str, headers: dict[str, str]):
            request = httpx.Request("GET", url)
            return httpx.Response(status, request=request, text="denied")

        monitor = UsageUpstreamMonitor()
        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=monitor,
            transport_get=rejecting,
        )
        assert result.auth_required is True
        assert result.attempts == 1  # never retried
        assert result.reported_outage is False
        assert monitor.consecutive_failures == 0

    async def test_non_auth_non_retryable_status_is_forwarded(self) -> None:
        async def bad_request(url: str, headers: dict[str, str]):
            request = httpx.Request("GET", url)
            return httpx.Response(400, request=request, text="bad param")

        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=UsageUpstreamMonitor(),
            transport_get=bad_request,
        )
        assert result.auth_required is False
        assert result.response is not None
        assert result.response.status_code == 400

    async def test_success_leaves_auth_required_false(self) -> None:
        async def ok(url: str, headers: dict[str, str]):
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"ok": True})

        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=UsageUpstreamMonitor(),
            transport_get=ok,
        )
        assert result.auth_required is False


class TestReportUsageAuthRequired:
    """Signed-out users leave a breadcrumb, never an error event."""

    def test_does_not_capture_an_error_event(self) -> None:
        fake_sdk = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            report_usage_auth_required(_status_error(400))
        fake_sdk.capture_message.assert_not_called()
        fake_sdk.capture_exception.assert_not_called()
        fake_sdk.add_breadcrumb.assert_called_once()

    def test_survives_missing_sentry_sdk(self) -> None:
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            report_usage_auth_required(_status_error(400))  # must not raise

    def test_survives_breadcrumb_failure(self) -> None:
        fake_sdk = MagicMock()
        fake_sdk.add_breadcrumb.side_effect = RuntimeError("sdk broken")
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            report_usage_auth_required(_status_error(400))  # must not raise


class TestBuildUsageNetworkErrorResponse:
    """Soft 503 must not raise HTTPException."""

    def test_connect_error_returns_503_with_stable_code(self) -> None:
        response = build_usage_network_error_response(httpx.ConnectError("proxy tls failed"))
        assert response.status_code == 503
        assert b"usage_upstream_unreachable" in response.body


@pytest.mark.asyncio
class TestObtainUsageAccessToken:
    """Token refresh transport errors must soft-fail like getUsageLimits."""

    async def test_success_returns_token(self) -> None:
        auth = MagicMock()
        auth.get_access_token = AsyncMock(return_value="tok-ok")
        token, error = await obtain_usage_access_token(
            auth,
            monitor=UsageUpstreamMonitor(),
        )
        assert token == "tok-ok"
        assert error is None

    async def test_connect_error_returns_soft_503_and_notes_monitor(self) -> None:
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=httpx.ConnectError("oidc unreachable")
        )
        monitor = UsageUpstreamMonitor()
        token, error = await obtain_usage_access_token(auth, monitor=monitor)
        assert token is None
        assert error is not None
        assert error.status_code == 503
        assert b"usage_upstream_unreachable" in error.body
        assert monitor.consecutive_failures == 1

    async def test_read_timeout_returns_soft_503(self) -> None:
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=httpx.ReadTimeout("oidc timed out")
        )
        token, error = await obtain_usage_access_token(
            auth,
            monitor=UsageUpstreamMonitor(),
        )
        assert token is None
        assert error is not None
        assert error.status_code == 503
        assert b"usage_upstream_unreachable" in error.body

    async def test_value_error_returns_auth_required_not_500(self) -> None:
        """Unrecoverable refresh is a signed-out user, not a gateway bug.

        ValueError used to propagate and surface as HTTP 500.
        """
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=ValueError("Token expired and refresh failed")
        )
        monitor = UsageUpstreamMonitor()
        token, error = await obtain_usage_access_token(auth, monitor=monitor)
        assert token is None
        assert error is not None
        assert error.status_code == 401
        assert USAGE_AUTH_REQUIRED_CODE.encode() in error.body
        assert b"login_required" in error.body
        # Credential failures must not look like an outage streak.
        assert monitor.consecutive_failures == 0

    async def test_oidc_400_returns_auth_required_and_skips_outage_streak(self) -> None:
        """The KIRO-GATEWAY-TRAY-D root cause: invalid_grant is not an outage."""
        request = httpx.Request("POST", "https://oidc.ap-northeast-1.amazonaws.com/token")
        response = httpx.Response(400, request=request, text="invalid_grant")
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Client error '400 Bad Request'", request=request, response=response
            )
        )
        monitor = UsageUpstreamMonitor()
        token, error = await obtain_usage_access_token(auth, monitor=monitor)
        assert token is None
        assert error is not None
        assert error.status_code == 401
        assert USAGE_AUTH_REQUIRED_CODE.encode() in error.body
        assert monitor.consecutive_failures == 0


@pytest.mark.asyncio
async def test_finalize_usage_transport_failure_escalates_after_streak() -> None:
    """Shared soft-fail path gates Sentry like getUsageLimits exhaustion.

    Driven by a simulated 60s tray poll clock: escalation now requires both the
    count and duration gates, so every earlier poll must stay silent.
    """
    monitor = UsageUpstreamMonitor(clock=_polling_clock())
    fake_sdk = MagicMock()
    scope = MagicMock()
    fake_sdk.new_scope.return_value.__enter__.return_value = scope
    with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
        for _ in range(_POLLS_TO_ESCALATE - 1):
            resp = await finalize_usage_transport_failure(
                httpx.ConnectError("blip"),
                attempts=1,
                monitor=monitor,
            )
            assert resp.status_code == 503
            fake_sdk.capture_message.assert_not_called()
        resp = await finalize_usage_transport_failure(
            httpx.ConnectError("sustained"),
            attempts=1,
            monitor=monitor,
        )
        assert resp.status_code == 503
        fake_sdk.capture_message.assert_called_once()
        assert scope.set_extra.call_args_list[0].args == (
            "consecutive_failures",
            _POLLS_TO_ESCALATE,
        )


@pytest.mark.asyncio
class TestUsageUpstreamMonitor:
    """Both gates must hold: enough failures AND a long enough window.

    The count and duration thresholds are ANDed. Previously they were ORed,
    which let the duration gate fire on the second consecutive failure because
    the tray polls ``/usage`` about once a minute (production Issues
    ``KIRO-GATEWAY-TRAY-20`` / ``-22`` on release 0.4.44, both ``consecutive=2``).
    """

    async def test_isolated_failures_do_not_report(self) -> None:
        monitor = UsageUpstreamMonitor()
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=1_000.0) is False

    async def test_enough_failures_but_window_too_short_does_not_report(self) -> None:
        """A burst of failures within seconds is a blip, not an outage."""
        monitor = UsageUpstreamMonitor()
        now = 1_000.0
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD * 4):
            assert await monitor.note_failure(now=now) is False
            now += 0.5
        # Still short of the duration gate even after many failures.
        assert now - 1_000.0 < USAGE_OUTAGE_DURATION_S

    async def test_long_window_but_too_few_failures_does_not_report(self) -> None:
        """Direct regression for the OR bug: 2 failures 60s apart must stay quiet.

        Replaces the old ``test_duration_threshold_reports``, which asserted the
        OR behaviour (duration alone escalating on failure #2). That assertion
        locked in the bug now fixed, so it is superseded rather than adjusted.
        """
        monitor = UsageUpstreamMonitor()
        assert await monitor.note_failure(now=1_000.0) is False
        # One tray poll interval later the duration gate alone is not enough.
        assert await monitor.note_failure(now=1_060.0) is False
        # Even hours later, two failures never escalate.
        assert await monitor.note_failure(now=1_000.0 + 3_600.0) is False
        assert monitor.consecutive_failures == 3
        assert USAGE_OUTAGE_CONSECUTIVE_THRESHOLD > 3

    async def test_both_gates_met_reports_once_then_cools_down(self) -> None:
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        step = USAGE_OUTAGE_DURATION_S / (USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1)
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=start + index * step) is False
        reported_at = start + USAGE_OUTAGE_DURATION_S
        assert await monitor.note_failure(now=reported_at) is True
        # Inside the cooldown window: no second event.
        assert await monitor.note_failure(now=reported_at + 10.0) is False
        assert (
            await monitor.note_failure(
                now=reported_at + USAGE_OUTAGE_REPORT_COOLDOWN_S - 0.001
            )
            is False
        )

    async def test_report_resumes_after_cooldown_expires(self) -> None:
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=start + index)
        reported_at = start + USAGE_OUTAGE_DURATION_S
        assert await monitor.note_failure(now=reported_at) is True
        assert (
            await monitor.note_failure(
                now=reported_at + USAGE_OUTAGE_REPORT_COOLDOWN_S
            )
            is True
        )

    async def test_exact_thresholds_are_inclusive(self) -> None:
        """``>=`` semantics: exactly N failures spanning exactly D seconds report."""
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=start) is False
        assert (
            await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S) is True
        )

    async def test_one_failure_short_of_count_gate_stays_quiet(self) -> None:
        """Duration satisfied, count one short: still no report."""
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            reported = await monitor.note_failure(
                now=start + (index + 1) * USAGE_OUTAGE_DURATION_S
            )
            assert reported is False

    async def test_one_second_short_of_duration_gate_stays_quiet(self) -> None:
        """Count satisfied, duration one second short: still no report."""
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=start) is False
        assert (
            await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S - 1.0)
            is False
        )

    async def test_success_resets_streak(self) -> None:
        monitor = UsageUpstreamMonitor()
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=1_000.0)
        await monitor.note_success()
        assert monitor.consecutive_failures == 0
        assert await monitor.note_failure(now=2_000.0) is False

    async def test_success_resets_first_failure_timestamp(self) -> None:
        """Both streak and window must restart, or the old start time leaks.

        Without resetting ``_first_failure_at``, a single post-success failure
        would inherit an hours-old window and clear the duration gate instantly.
        """
        monitor = UsageUpstreamMonitor()
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=1_000.0 + index)
        await monitor.note_success()
        # A fresh streak far in the future must re-accumulate from scratch.
        later = 1_000.0 + 10 * USAGE_OUTAGE_DURATION_S
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD * 2):
            assert await monitor.note_failure(now=later) is False

    async def test_success_midway_prevents_escalation(self) -> None:
        """A recovery in the middle of a long window resets the clock."""
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=start + index * 60.0)
        await monitor.note_success()
        assert await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S) is False
        assert monitor.consecutive_failures == 1

    async def test_backwards_clock_does_not_raise_or_report(self) -> None:
        """NTP corrections / suspend-resume can move the clock backwards."""
        monitor = UsageUpstreamMonitor()
        assert await monitor.note_failure(now=10_000.0) is False
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD * 2):
            assert await monitor.note_failure(now=9_000.0) is False

    async def test_backwards_clock_then_forward_still_escalates(self) -> None:
        """After a backwards jump the window restarts, never breaks escalation."""
        monitor = UsageUpstreamMonitor()
        await monitor.note_failure(now=10_000.0)
        await monitor.note_failure(now=9_000.0)
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD):
            reported = await monitor.note_failure(
                now=10_000.0 + USAGE_OUTAGE_DURATION_S
            )
            if reported:
                break
        else:  # pragma: no cover — guards against a permanently muted monitor
            pytest.fail("monitor stayed silent after the clock recovered")

    async def test_injected_clock_is_used_when_now_is_omitted(self) -> None:
        """Callers behind finalize/fetch cannot pass ``now``; the clock must apply."""
        ticks = iter([1_000.0, 1_060.0, 1_120.0, 1_180.0, 1_300.0])
        monitor = UsageUpstreamMonitor(clock=lambda: next(ticks))
        results = [await monitor.note_failure() for _ in range(5)]
        assert results == [False, False, False, False, True]


@pytest.mark.asyncio
class TestFetchUsageLimits:
    """Bounded retries with backoff; soft failure after exhaustion."""

    async def test_retries_transport_errors_then_fails(self) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        async def boom(url: str, headers: dict[str, str]):
            calls["n"] += 1
            raise httpx.ConnectError("proxy down")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monitor = UsageUpstreamMonitor()
        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=monitor,
            sleep=fake_sleep,
            transport_get=boom,
        )
        assert result.response is None
        assert isinstance(result.error, httpx.ConnectError)
        assert result.attempts == USAGE_UPSTREAM_MAX_ATTEMPTS
        assert calls["n"] == USAGE_UPSTREAM_MAX_ATTEMPTS
        assert len(sleeps) == USAGE_UPSTREAM_MAX_ATTEMPTS - 1
        assert monitor.consecutive_failures == 1

    async def test_success_after_retry(self) -> None:
        calls = {"n": 0}

        async def flaky(url: str, headers: dict[str, str]):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("blip")
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"ok": True})

        async def fake_sleep(delay: float) -> None:
            return None

        monitor = UsageUpstreamMonitor()
        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=monitor,
            sleep=fake_sleep,
            transport_get=flaky,
        )
        assert result.response is not None
        assert result.response.status_code == 200
        assert result.error is None
        assert calls["n"] == 2
        assert monitor.consecutive_failures == 0

    async def test_retries_retryable_status(self) -> None:
        calls = {"n": 0}

        async def status_flaky(url: str, headers: dict[str, str]):
            calls["n"] += 1
            request = httpx.Request("GET", url)
            if calls["n"] < 3:
                return httpx.Response(503, request=request, text="busy")
            return httpx.Response(200, request=request, json={"ok": True})

        async def fake_sleep(delay: float) -> None:
            return None

        result = await fetch_usage_limits(
            url="https://example.test/usage",
            headers={},
            proxy=None,
            monitor=UsageUpstreamMonitor(),
            sleep=fake_sleep,
            transport_get=status_flaky,
        )
        assert result.response is not None
        assert result.response.status_code == 200
        assert calls["n"] == 3

    async def test_single_exhausted_request_never_reports_outage(self) -> None:
        """One request burning all 3 attempts is still a single soft failure.

        Retries happen within seconds, so neither gate can be satisfied; the OR
        logic would also have stayed quiet here only because the streak was 1.
        """

        async def boom(url: str, headers: dict[str, str]):
            raise httpx.ReadTimeout("upstream slow")

        async def fake_sleep(delay: float) -> None:
            return None

        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        fake_sdk = MagicMock()
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            result = await fetch_usage_limits(
                url="https://example.test/usage",
                headers={},
                proxy=None,
                monitor=monitor,
                sleep=fake_sleep,
                transport_get=boom,
            )
        assert result.reported_outage is False
        assert monitor.consecutive_failures == 1
        fake_sdk.capture_message.assert_not_called()

    async def test_retryable_status_exhaustion_escalates_only_when_sustained(
        self,
    ) -> None:
        """503-forever upstream reports once the streak spans the duration gate."""

        async def busy(url: str, headers: dict[str, str]):
            request = httpx.Request("GET", url)
            return httpx.Response(503, request=request, text="busy")

        async def fake_sleep(delay: float) -> None:
            return None

        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        fake_sdk = MagicMock()
        reports: list[bool] = []
        with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
            for _ in range(_POLLS_TO_ESCALATE):
                result = await fetch_usage_limits(
                    url="https://example.test/usage",
                    headers={},
                    proxy=None,
                    monitor=monitor,
                    sleep=fake_sleep,
                    transport_get=busy,
                )
                assert result.response is not None
                assert result.response.status_code == 503
                reports.append(result.reported_outage)
        assert reports == [False] * (_POLLS_TO_ESCALATE - 1) + [True]
        fake_sdk.capture_message.assert_called_once()


def test_report_usage_outage_captures_sentry_message() -> None:
    """Outage reporter should call Sentry when SDK is importable."""
    fake_sdk = MagicMock()
    scope = MagicMock()
    fake_sdk.new_scope.return_value.__enter__.return_value = scope
    with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
        report_usage_outage(httpx.ConnectError("down"), consecutive=5, attempts=3)
    fake_sdk.capture_message.assert_called_once()
    scope.set_tag.assert_any_call("usage_outage", "true")


@pytest.mark.parametrize(
    "exc,expected_cause",
    [
        (httpx.ConnectError("[Errno 8] nodename nor servname provided"), "dns"),
        (httpx.ConnectError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE]"), "tls"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("All connection attempts failed"), "connect"),
    ],
)
def test_report_usage_outage_fingerprints_by_cause(
    exc: Exception, expected_cause: str
) -> None:
    """One shared fingerprint made DNS/TLS/timeout failures un-triageable."""
    fake_sdk = MagicMock()
    scope = MagicMock()
    fake_sdk.new_scope.return_value.__enter__.return_value = scope
    with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
        report_usage_outage(exc, consecutive=5, attempts=3)
    assert scope.fingerprint == ["kiro-usage-upstream-outage", expected_cause]
    scope.set_tag.assert_any_call("usage_outage_cause", expected_cause)


def test_report_usage_outage_survives_missing_sentry_sdk() -> None:
    """Reporter must never break /usage when the SDK is absent."""
    with patch.dict("sys.modules", {"sentry_sdk": None}):
        report_usage_outage(httpx.ConnectError("down"), consecutive=5, attempts=3)

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


async def _no_sleep(_delay: float) -> None:
    """Skip real backoff in retry tests."""
    return None


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
                # 5xx from the refresh endpoint: cause=upstream_status, which is
                # not an external cause, so it may open one Issue.
                response = await finalize_usage_transport_failure(
                    _status_error(500), attempts=1, monitor=monitor
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
        token, error = await obtain_usage_access_token(
            auth, monitor=monitor, sleep=_no_sleep
        )
        assert token is None
        assert error is not None
        assert error.status_code == 503
        assert b"usage_upstream_unreachable" in error.body
        # All retries of one poll count as a single streak failure.
        assert monitor.consecutive_failures == 1
        assert auth.get_access_token.await_count == USAGE_UPSTREAM_MAX_ATTEMPTS

    async def test_read_timeout_returns_soft_503(self) -> None:
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=httpx.ReadTimeout("oidc timed out")
        )
        token, error = await obtain_usage_access_token(
            auth,
            monitor=UsageUpstreamMonitor(),
            sleep=_no_sleep,
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
                _status_error(502),
                attempts=1,
                monitor=monitor,
            )
            assert resp.status_code == 503
            fake_sdk.capture_message.assert_not_called()
        resp = await finalize_usage_transport_failure(
            _status_error(503),
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

    async def test_both_gates_met_opens_episode_exactly_once(self) -> None:
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        step = USAGE_OUTAGE_DURATION_S / (USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1)
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=start + index * step) is False
        reported_at = start + USAGE_OUTAGE_DURATION_S
        assert await monitor.note_failure(now=reported_at) is True
        assert monitor.outage_open is True
        assert await monitor.note_failure(now=reported_at + 10.0) is False

    async def test_no_periodic_re_report_while_outage_persists(self) -> None:
        """Regression for TRAY-2B/-22: 640+ failures produced 40+ events.

        The old 15-minute cooldown re-reported forever. An episode must report
        once, however long it lasts.
        """
        monitor = UsageUpstreamMonitor()
        start = 1_000.0
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=start + index)
        assert await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S) is True
        later = [
            await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S + 60.0 * i)
            for i in range(1, 700)
        ]
        assert not any(later)
        assert monitor.consecutive_failures == USAGE_OUTAGE_CONSECUTIVE_THRESHOLD + 699

    async def test_new_episode_reports_again_after_recovery(self) -> None:
        """A success closes the episode; the next sustained streak is a new state change."""
        monitor = UsageUpstreamMonitor()
        for start in (1_000.0, 50_000.0):
            for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
                assert await monitor.note_failure(now=start + index) is False
            assert await monitor.note_failure(now=start + USAGE_OUTAGE_DURATION_S) is True
            recovery = await monitor.note_success(now=start + USAGE_OUTAGE_DURATION_S + 1.0)
            assert recovery is not None
            assert monitor.outage_open is False

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


# =============================================================================
# Raw ssl.SSLError / OSError, token retry, TLS hint, JSON decode classification
# (Sentry KIRO-GATEWAY-TRAY-2D, -2E, and the 200+HTML misclassification)
# =============================================================================

import json as _json
import ssl as _ssl

from kiro.auth import RefreshCredentialsSuspectError
from kiro.usage_upstream import USAGE_TLS_HANDSHAKE_REASON

_BAD_MAC = "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac"
_HANDSHAKE = "[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure"


def _auth_raising(*effects: object) -> MagicMock:
    """Auth double whose get_access_token follows ``effects`` in order."""
    auth = MagicMock()
    auth.get_access_token = AsyncMock(side_effect=list(effects))
    return auth


class TestObtainUsageAccessTokenRawErrors:
    """Second line of defense: raw OSError must never escape as HTTP 500."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [
            _ssl.SSLError(1, _BAD_MAC),
            _ssl.SSLError(1, "bad record mac"),  # no "ssl" marker in text
            _ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
            ConnectionResetError(54, "Connection reset by peer"),
            OSError("socket closed"),
        ],
    )
    async def test_raw_os_errors_soft_fail_as_503(self, raw: OSError) -> None:
        monitor = UsageUpstreamMonitor()
        auth = _auth_raising(*([raw] * USAGE_UPSTREAM_MAX_ATTEMPTS))
        token, error = await obtain_usage_access_token(auth, monitor=monitor, sleep=_no_sleep)
        assert token is None
        assert error is not None
        assert error.status_code == 503
        assert b"usage_upstream_unreachable" in error.body
        assert monitor.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_raw_ssl_error_is_classified_tls_for_reporting(self) -> None:
        """The normalized error keeps SSLError as cause, so cause=tls."""
        captured: list[BaseException] = []
        auth = _auth_raising(*([_ssl.SSLError(1, "bad record mac")] * USAGE_UPSTREAM_MAX_ATTEMPTS))

        async def spy(exc: BaseException, **_: object):
            captured.append(exc)
            return build_usage_network_error_response(exc)

        with patch("kiro.usage_upstream.finalize_usage_transport_failure", side_effect=spy):
            await obtain_usage_access_token(auth, monitor=UsageUpstreamMonitor(), sleep=_no_sleep)
        assert len(captured) == 1
        assert isinstance(captured[0], httpx.TransportError)
        assert transport_failure_reason(captured[0]) == "tls"

    @pytest.mark.asyncio
    async def test_non_os_runtime_error_still_propagates(self) -> None:
        """Genuine bugs must reach Sentry as unhandled, not be masked as 503."""
        auth = _auth_raising(RuntimeError("bug"))
        with pytest.raises(RuntimeError):
            await obtain_usage_access_token(auth, monitor=UsageUpstreamMonitor(), sleep=_no_sleep)


class TestObtainUsageAccessTokenRetry:
    """Token refresh retries transport/5xx only, with the getUsageLimits budget."""

    @pytest.mark.asyncio
    async def test_transient_transport_error_recovers_within_one_poll(self) -> None:
        monitor = UsageUpstreamMonitor()
        auth = _auth_raising(httpx.ConnectError("blip"), "tok")
        sleeps: list[float] = []

        async def record(delay: float) -> None:
            sleeps.append(delay)

        token, error = await obtain_usage_access_token(auth, monitor=monitor, sleep=record)
        assert token == "tok"
        assert error is None
        assert len(sleeps) == 1
        assert monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_raw_ssl_error_is_retried(self) -> None:
        auth = _auth_raising(_ssl.SSLError(1, _BAD_MAC), "tok")
        token, error = await obtain_usage_access_token(
            auth, monitor=UsageUpstreamMonitor(), sleep=_no_sleep
        )
        assert token == "tok"
        assert error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    async def test_5xx_is_retried_up_to_budget(self, status: int) -> None:
        auth = _auth_raising(*([_status_error(status)] * USAGE_UPSTREAM_MAX_ATTEMPTS))
        sleeps: list[float] = []

        async def record(delay: float) -> None:
            sleeps.append(delay)

        _, error = await obtain_usage_access_token(auth, monitor=UsageUpstreamMonitor(), sleep=record)
        assert error is not None and error.status_code == 503
        assert auth.get_access_token.await_count == USAGE_UPSTREAM_MAX_ATTEMPTS
        # No sleep after the final attempt.
        assert len(sleeps) == USAGE_UPSTREAM_MAX_ATTEMPTS - 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 429])
    async def test_4xx_is_not_retried(self, status: int) -> None:
        """Credential/client errors repeat identically; 429 is not in scope."""
        auth = _auth_raising(_status_error(status), "never")
        await obtain_usage_access_token(auth, monitor=UsageUpstreamMonitor(), sleep=_no_sleep)
        assert auth.get_access_token.await_count == 1

    @pytest.mark.asyncio
    async def test_value_error_is_not_retried(self) -> None:
        auth = _auth_raising(ValueError("Token expired and refresh failed"), "never")
        _, error = await obtain_usage_access_token(
            auth, monitor=UsageUpstreamMonitor(), sleep=_no_sleep
        )
        assert auth.get_access_token.await_count == 1
        assert error is not None and error.status_code == 401

    @pytest.mark.asyncio
    async def test_decoding_error_is_not_retried_and_is_soft_503(self) -> None:
        """200+HTML: not a credential failure, not worth an in-poll retry."""
        auth = _auth_raising(httpx.DecodingError("not json"), "never")
        monitor = UsageUpstreamMonitor()
        _, error = await obtain_usage_access_token(auth, monitor=monitor, sleep=_no_sleep)
        assert auth.get_access_token.await_count == 1
        assert error is not None and error.status_code == 503
        assert monitor.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_retry_then_credential_failure_stops_and_returns_401(self) -> None:
        """A 5xx followed by 400 stops immediately at the credential failure."""
        auth = _auth_raising(_status_error(500), _status_error(400), "never")
        monitor = UsageUpstreamMonitor()
        _, error = await obtain_usage_access_token(auth, monitor=monitor, sleep=_no_sleep)
        assert auth.get_access_token.await_count == 2
        assert error is not None and error.status_code == 401
        assert monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_suspected_credentials_return_login_required(self) -> None:
        """2B: long 5xx streak from Kiro Desktop is surfaced as re-login."""
        auth = _auth_raising(RefreshCredentialsSuspectError(500, 4000.0), "never")
        monitor = UsageUpstreamMonitor()
        _, error = await obtain_usage_access_token(auth, monitor=monitor, sleep=_no_sleep)
        assert auth.get_access_token.await_count == 1
        assert error is not None and error.status_code == 401
        body = _json.loads(error.body)
        assert body["error"]["login_required"] is True
        assert body["error"]["code"] == USAGE_AUTH_REQUIRED_CODE
        assert monitor.consecutive_failures == 0


class TestFetchUsageLimitsRawErrors:
    """getUsageLimits transport path must normalize raw OSError too."""

    @pytest.mark.asyncio
    async def test_raw_ssl_error_is_retried_and_soft_fails(self) -> None:
        calls = 0

        async def transport(_url: str, _headers: dict) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise _ssl.SSLError(1, _BAD_MAC)

        result = await fetch_usage_limits(
            url="https://q.us-east-1.amazonaws.com/getUsageLimits",
            headers={},
            proxy=None,
            monitor=UsageUpstreamMonitor(),
            sleep=_no_sleep,
            transport_get=transport,
        )
        assert calls == USAGE_UPSTREAM_MAX_ATTEMPTS
        assert result.response is None
        assert isinstance(result.error, httpx.TransportError)
        assert transport_failure_reason(result.error) == "tls"

    @pytest.mark.asyncio
    async def test_raw_ssl_error_then_success(self) -> None:
        request = httpx.Request("GET", "https://q.us-east-1.amazonaws.com/getUsageLimits")
        effects: list[object] = [_ssl.SSLError(1, _BAD_MAC), httpx.Response(200, request=request, json={})]

        async def transport(_url: str, _headers: dict) -> httpx.Response:
            effect = effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect  # type: ignore[return-value]

        result = await fetch_usage_limits(
            url=str(request.url), headers={}, proxy=None,
            monitor=UsageUpstreamMonitor(), sleep=_no_sleep, transport_get=transport,
        )
        assert result.response is not None and result.response.status_code == 200

    @pytest.mark.asyncio
    async def test_runtime_error_propagates(self) -> None:
        async def transport(_url: str, _headers: dict) -> httpx.Response:
            raise RuntimeError("bug")

        with pytest.raises(RuntimeError):
            await fetch_usage_limits(
                url="u", headers={}, proxy=None,
                monitor=UsageUpstreamMonitor(), sleep=_no_sleep, transport_get=transport,
            )


class TestUsageTlsHandshakeHint:
    """Handshake failures carry an actionable proxy/network hint in /usage 503."""

    def test_handshake_failure_body_has_hint_and_reason(self) -> None:
        response = build_usage_network_error_response(httpx.ConnectError(_HANDSHAKE))
        body = _json.loads(response.body)
        assert response.status_code == 503
        assert body["error"]["code"] == "usage_upstream_unreachable"
        assert body["error"]["reason"] == USAGE_TLS_HANDSHAKE_REASON
        assert "proxy or network" in body["error"]["message"]

    def test_record_layer_error_has_no_hint(self) -> None:
        normalized = httpx.ReadError(_BAD_MAC)
        normalized.__cause__ = _ssl.SSLError(1, _BAD_MAC)
        body = _json.loads(build_usage_network_error_response(normalized).body)
        assert "reason" not in body["error"]

    def test_plain_connect_error_has_no_hint(self) -> None:
        body = _json.loads(
            build_usage_network_error_response(httpx.ConnectError("All connection attempts failed")).body
        )
        assert "reason" not in body["error"]
        assert "login_required" not in body["error"]


class TestTransportFailureReasonSslType:
    """cause=tls is decided by type, not only by message text."""

    def test_raw_ssl_error_without_marker_is_tls(self) -> None:
        assert transport_failure_reason(_ssl.SSLError(1, "bad record mac")) == "tls"

    def test_wrapped_ssl_error_without_marker_is_tls(self) -> None:
        wrapped = httpx.ReadError("read failed")
        wrapped.__cause__ = _ssl.SSLError(1, "bad record mac")
        assert transport_failure_reason(wrapped) == "tls"

    def test_status_error_still_upstream_status(self) -> None:
        """Type checks for status/timeout still win over the chain check."""
        assert transport_failure_reason(_status_error(500)) == "upstream_status"

    def test_timeout_with_ssl_cause_stays_timeout(self) -> None:
        timeout = httpx.ReadTimeout("slow")
        timeout.__cause__ = _ssl.SSLError(1, "x")
        assert transport_failure_reason(timeout) == "timeout"


class TestJsonDecodeIsNotAuthFailure:
    """200+HTML bodies must not prompt a re-login."""

    def test_json_decode_error_is_not_auth_failure(self) -> None:
        with pytest.raises(_json.JSONDecodeError) as exc_info:
            _json.loads("<html>captive portal</html>")
        assert is_auth_failure(exc_info.value) is False

    def test_plain_value_error_is_still_auth_failure(self) -> None:
        """The established ValueError contract is unchanged."""
        assert is_auth_failure(ValueError("Failed to obtain access token")) is True

    def test_suspect_error_is_auth_failure(self) -> None:
        assert is_auth_failure(RefreshCredentialsSuspectError(500, 3600.0)) is True


class TestUsageEndpointSslError:
    """End-to-end: GET /usage must not return 500 for a raw SSLError (2D)."""

    def test_raw_ssl_error_in_token_refresh_returns_503(self, clean_app) -> None:
        from fastapi.testclient import TestClient
        from kiro.routes_openai import verify_api_key

        auth = MagicMock()
        auth.get_access_token = AsyncMock(side_effect=_ssl.SSLError(1, _BAD_MAC))
        account = SimpleNamespace(auth_manager=auth)
        clean_app.dependency_overrides[verify_api_key] = lambda: True
        original = getattr(clean_app.state, "account_manager", None)
        clean_app.state.account_manager = SimpleNamespace(_accounts={"a": account})
        try:
            # The sleep default is bound at definition time; zero the delay instead.
            with patch("kiro.usage_upstream._retry_delay_seconds", return_value=0.0), \
                 patch("kiro.usage_upstream.usage_upstream_monitor", UsageUpstreamMonitor()):
                client = TestClient(clean_app, raise_server_exceptions=False)
                response = client.get("/usage")
        finally:
            clean_app.state.account_manager = original
            clean_app.dependency_overrides = {}

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "usage_upstream_unreachable"


# =============================================================================
# State-change reporting: one Issue per episode, external causes breadcrumb-only
# (Sentry KIRO-GATEWAY-TRAY-2B, -22, -20, -2E)
# =============================================================================

from kiro.usage_upstream import (
    USAGE_OUTAGE_EXTERNAL_CAUSES,
    UsageOutageRecovery,
    is_external_outage_cause,
    note_usage_failure,
    note_usage_success,
)

_EXTERNAL_ERRORS = {
    "tls": httpx.ConnectError(_HANDSHAKE),
    "dns": httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known"),
    "timeout": httpx.ConnectTimeout(""),
    "connect": httpx.ConnectError("All connection attempts failed"),
}


def _fake_sentry() -> tuple[MagicMock, MagicMock]:
    """Return (sdk, scope) doubles wired like sentry_sdk.new_scope()."""
    sdk = MagicMock()
    scope = MagicMock()
    sdk.new_scope.return_value.__enter__.return_value = scope
    return sdk, scope


def _breadcrumbs(sdk: MagicMock) -> list[dict]:
    """Collect kwargs of every add_breadcrumb call."""
    return [c.kwargs for c in sdk.add_breadcrumb.call_args_list]


async def _drive(exc: BaseException, monitor: UsageUpstreamMonitor, polls: int) -> list[bool]:
    """Feed ``polls`` identical failures through the shared entry point."""
    return [await note_usage_failure(exc, attempts=3, monitor=monitor) for _ in range(polls)]


class TestExternalCauseClassification:
    """The external set is exactly what the user decided."""

    def test_external_set_is_exact(self) -> None:
        assert USAGE_OUTAGE_EXTERNAL_CAUSES == frozenset({"tls", "dns", "timeout", "connect"})

    @pytest.mark.parametrize("cause", ["upstream_status", "other", "", "TLS"])
    def test_non_external_causes(self, cause: str) -> None:
        """Case-sensitive: tags are produced lowercase by transport_failure_reason."""
        assert is_external_outage_cause(cause) is False

    @pytest.mark.parametrize("cause,exc", list(_EXTERNAL_ERRORS.items()))
    def test_fixtures_map_to_their_cause(self, cause: str, exc: BaseException) -> None:
        assert transport_failure_reason(exc) == cause


@pytest.mark.asyncio
class TestEpisodeReporting:
    """Issue once at episode start for gateway-relevant causes, never for external."""

    @pytest.mark.parametrize("cause", sorted(_EXTERNAL_ERRORS))
    async def test_external_cause_never_creates_event(self, cause: str) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            opened = await _drive(_EXTERNAL_ERRORS[cause], monitor, _POLLS_TO_ESCALATE + 20)
        sdk.capture_message.assert_not_called()
        sdk.capture_exception.assert_not_called()
        assert not any(opened)
        assert monitor.outage_open is True
        assert monitor.outage_issue_reported is False
        crumbs = _breadcrumbs(sdk)
        # One crumb per failure from the episode start on: "started" then "ongoing".
        assert len(crumbs) == 21
        assert crumbs[0]["message"] == "GET /usage upstream outage started"
        assert all(c["message"] == "GET /usage upstream outage ongoing" for c in crumbs[1:])
        assert all(c["data"]["cause"] == cause and c["data"]["external"] for c in crumbs)

    @pytest.mark.parametrize("exc", [_status_error(500), _status_error(503), httpx.DecodingError("x")])
    async def test_gateway_relevant_cause_reports_once_at_start(self, exc: BaseException) -> None:
        sdk, scope = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            opened = await _drive(exc, monitor, _POLLS_TO_ESCALATE + 50)
        assert opened.count(True) == 1
        assert opened.index(True) == _POLLS_TO_ESCALATE - 1
        sdk.capture_message.assert_called_once()
        assert scope.fingerprint[1] in {"upstream_status", "other"}
        # Every later failure is a breadcrumb, not an event.
        assert len(_breadcrumbs(sdk)) == 50
        assert all(c["message"].endswith("ongoing") for c in _breadcrumbs(sdk))

    async def test_failures_before_threshold_emit_nothing(self) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_status_error(500), monitor, _POLLS_TO_ESCALATE - 1)
        sdk.capture_message.assert_not_called()
        sdk.add_breadcrumb.assert_not_called()
        assert monitor.outage_open is False

    async def test_opening_cause_decides_episode(self) -> None:
        """External start then upstream_status later: still no Issue in that episode."""
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_EXTERNAL_ERRORS["connect"], monitor, _POLLS_TO_ESCALATE)
            await _drive(_status_error(500), monitor, 10)
        sdk.capture_message.assert_not_called()
        assert monitor.outage_cause == "connect"

    async def test_upstream_start_then_external_does_not_report_again(self) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_status_error(500), monitor, _POLLS_TO_ESCALATE)
            await _drive(_EXTERNAL_ERRORS["tls"], monitor, 10)
        sdk.capture_message.assert_called_once()

    async def test_external_logs_stay_below_warning(self) -> None:
        """Tray ships WARNING+ loguru records to Sentry Logs; external must stay INFO."""
        from loguru import logger as _logger

        levels: list[str] = []
        sink_id = _logger.add(lambda m: levels.append(m.record["level"].name), level="DEBUG")
        try:
            monitor = UsageUpstreamMonitor(clock=_polling_clock())
            with patch.dict("sys.modules", {"sentry_sdk": None}):
                await _drive(_EXTERNAL_ERRORS["dns"], monitor, _POLLS_TO_ESCALATE + 3)
                await note_usage_success(monitor)
        finally:
            _logger.remove(sink_id)
        assert "WARNING" not in levels and "ERROR" not in levels
        assert "INFO" in levels

    async def test_missing_sdk_never_breaks_reporting(self) -> None:
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            opened = await _drive(_status_error(500), monitor, _POLLS_TO_ESCALATE + 2)
            await note_usage_success(monitor)
        assert opened.count(True) == 1

    async def test_breadcrumb_failure_is_swallowed(self) -> None:
        sdk, _ = _fake_sentry()
        sdk.add_breadcrumb.side_effect = RuntimeError("sdk broken")
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_EXTERNAL_ERRORS["timeout"], monitor, _POLLS_TO_ESCALATE + 1)
            await note_usage_success(monitor)


@pytest.mark.asyncio
class TestEpisodeRecovery:
    """Recovery closes the episode with an info breadcrumb, never an event."""

    @pytest.mark.parametrize(
        "exc,external,issue",
        [
            (_EXTERNAL_ERRORS["tls"], True, False),
            (_status_error(500), False, True),
        ],
    )
    async def test_recovery_records_info_breadcrumb(
        self, exc: BaseException, external: bool, issue: bool
    ) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(exc, monitor, _POLLS_TO_ESCALATE + 2)
            events_before = sdk.capture_message.call_count
            await note_usage_success(monitor)
        assert sdk.capture_message.call_count == events_before
        last = _breadcrumbs(sdk)[-1]
        assert last["level"] == "info"
        assert last["message"] == "GET /usage upstream recovered"
        assert last["data"]["external"] is external
        assert last["data"]["issue_reported"] is issue
        assert last["data"]["consecutive_failures"] == _POLLS_TO_ESCALATE + 2
        assert monitor.outage_open is False
        assert monitor.consecutive_failures == 0

    async def test_success_without_episode_is_silent(self) -> None:
        """Recovering from a sub-threshold blip is not a state change."""
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_EXTERNAL_ERRORS["connect"], monitor, 2)
            await note_usage_success(monitor)
        sdk.add_breadcrumb.assert_not_called()

    async def test_repeated_success_reports_recovery_once(self) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_status_error(500), monitor, _POLLS_TO_ESCALATE)
            for _ in range(3):
                await note_usage_success(monitor)
        recovered = [c for c in _breadcrumbs(sdk) if c["message"].endswith("recovered")]
        assert len(recovered) == 1

    async def test_recovery_snapshot_duration_and_backwards_clock(self) -> None:
        monitor = UsageUpstreamMonitor()
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD):
            await monitor.note_failure(now=1_000.0 + index * 100.0, cause="other")
        assert monitor.outage_open is True
        recovery = await monitor.note_success(now=1_000.0 + 1_000.0)
        assert recovery == UsageOutageRecovery(
            cause="other", issue_reported=True,
            consecutive_failures=USAGE_OUTAGE_CONSECUTIVE_THRESHOLD, duration_s=1_000.0,
        )
        for index in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD):
            await monitor.note_failure(now=5_000.0 + index * 100.0, cause="dns")
        backwards = await monitor.note_success(now=1.0)
        assert backwards is not None and backwards.duration_s == 0.0
        assert backwards.issue_reported is False

    async def test_auth_failure_does_not_close_episode(self) -> None:
        """A 401 mid-outage is not a recovery."""
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_EXTERNAL_ERRORS["connect"], monitor, _POLLS_TO_ESCALATE)
            await finalize_usage_transport_failure(_status_error(401), monitor=monitor)
        assert monitor.outage_open is True


@pytest.mark.asyncio
class TestFetchUsageLimitsEpisodeWiring:
    """getUsageLimits paths go through the same state-change policy."""

    async def test_200_after_outage_reports_recovery(self) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())
        request = httpx.Request("GET", "https://q.test/getUsageLimits")

        async def ok(_url: str, _headers: dict) -> httpx.Response:
            return httpx.Response(200, request=request, json={})

        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            await _drive(_EXTERNAL_ERRORS["timeout"], monitor, _POLLS_TO_ESCALATE)
            await fetch_usage_limits(
                url=str(request.url), headers={}, proxy=None,
                monitor=monitor, sleep=_no_sleep, transport_get=ok,
            )
        assert _breadcrumbs(sdk)[-1]["message"] == "GET /usage upstream recovered"
        sdk.capture_message.assert_not_called()

    async def test_transport_exhaustion_external_never_reports(self) -> None:
        sdk, _ = _fake_sentry()
        monitor = UsageUpstreamMonitor(clock=_polling_clock())

        async def down(_url: str, _headers: dict) -> httpx.Response:
            raise httpx.ConnectError("All connection attempts failed")

        reported: list[bool] = []
        with patch.dict("sys.modules", {"sentry_sdk": sdk}):
            for _ in range(_POLLS_TO_ESCALATE + 5):
                result = await fetch_usage_limits(
                    url="u", headers={}, proxy=None,
                    monitor=monitor, sleep=_no_sleep, transport_get=down,
                )
                reported.append(result.reported_outage)
        assert not any(reported)
        sdk.capture_message.assert_not_called()

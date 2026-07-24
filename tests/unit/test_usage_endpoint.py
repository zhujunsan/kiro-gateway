"""Unit tests for /usage upstream retry and outage reporting."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kiro.usage_upstream import (
    USAGE_OUTAGE_CONSECUTIVE_THRESHOLD,
    USAGE_OUTAGE_DURATION_S,
    USAGE_UPSTREAM_MAX_ATTEMPTS,
    UsageUpstreamMonitor,
    build_usage_network_error_response,
    fetch_usage_limits,
    finalize_usage_transport_failure,
    obtain_usage_access_token,
    report_usage_outage,
)


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

    async def test_value_error_propagates(self) -> None:
        auth = MagicMock()
        auth.get_access_token = AsyncMock(
            side_effect=ValueError("Token expired and refresh failed")
        )
        with pytest.raises(ValueError, match="Token expired"):
            await obtain_usage_access_token(auth, monitor=UsageUpstreamMonitor())


@pytest.mark.asyncio
async def test_finalize_usage_transport_failure_escalates_after_streak() -> None:
    """Shared soft-fail path gates Sentry like getUsageLimits exhaustion."""
    monitor = UsageUpstreamMonitor()
    fake_sdk = MagicMock()
    scope = MagicMock()
    fake_sdk.new_scope.return_value.__enter__.return_value = scope
    with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
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


@pytest.mark.asyncio
class TestUsageUpstreamMonitor:
    """Failure streak gates Sentry escalation."""

    async def test_isolated_failures_do_not_report(self) -> None:
        monitor = UsageUpstreamMonitor()
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=1_000.0) is False

    async def test_consecutive_threshold_reports_once_then_cools_down(self) -> None:
        monitor = UsageUpstreamMonitor()
        now = 1_000.0
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            assert await monitor.note_failure(now=now) is False
            now += 1
        assert await monitor.note_failure(now=now) is True
        # Still within cooldown.
        assert await monitor.note_failure(now=now + 10) is False

    async def test_duration_threshold_reports(self) -> None:
        monitor = UsageUpstreamMonitor()
        assert await monitor.note_failure(now=1_000.0) is False
        assert await monitor.note_failure(now=1_000.0 + USAGE_OUTAGE_DURATION_S) is True

    async def test_success_resets_streak(self) -> None:
        monitor = UsageUpstreamMonitor()
        for _ in range(USAGE_OUTAGE_CONSECUTIVE_THRESHOLD - 1):
            await monitor.note_failure(now=1_000.0)
        await monitor.note_success()
        assert monitor.consecutive_failures == 0
        assert await monitor.note_failure(now=2_000.0) is False


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


def test_report_usage_outage_captures_sentry_message() -> None:
    """Outage reporter should call Sentry when SDK is importable."""
    fake_sdk = MagicMock()
    scope = MagicMock()
    fake_sdk.new_scope.return_value.__enter__.return_value = scope
    with patch.dict("sys.modules", {"sentry_sdk": fake_sdk}):
        report_usage_outage(httpx.ConnectError("down"), consecutive=5, attempts=3)
    fake_sdk.capture_message.assert_called_once()
    scope.set_tag.assert_any_call("usage_outage", "true")

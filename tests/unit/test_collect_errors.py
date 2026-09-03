# -*- coding: utf-8 -*-

"""
Unit tests for kiro.collect_errors.

The non-streaming (collect) path used to report every failure as a gateway 500,
so upstream stalls and cut connections were logged as gateway bugs
(Sentry KIRO-GATEWAY-TRAY-1C / -1R). These tests pin the mapping between an
exception and the client-facing status / incident tags, and keep it aligned with
the streaming classifier in kiro.debug_logger.
"""

import httpx
import pytest

from kiro.collect_errors import (
    CollectFailure,
    classify_collect_exception,
    collect_failure_or_none,
    immediate_collect_response,
)
from kiro.debug_logger import classify_streaming_exception
from kiro.streaming_core import FirstTokenTimeoutError


class TestClassifyFirstTokenTimeout:
    """A silent upstream is a 504 network incident, never a gateway 500."""

    def test_maps_to_504_network_incident(self):
        """
        What it does: Classifies FirstTokenTimeoutError.
        Purpose: Non-streaming stalls must be reported like streaming ones.
        """
        failure = classify_collect_exception(
            FirstTokenTimeoutError("No response within 30.0 seconds")
        )
        assert failure.status_code == 504
        assert failure.source == "network"
        assert failure.code == "first_token_timeout"
        assert failure.phase == "first_token"
        assert failure.is_upstream is True

    def test_keeps_retry_exhaustion_message(self):
        """
        What it does: Preserves the "after N attempts" wording.
        Purpose: The retry-exhaustion text is already actionable; don't replace it.
        """
        failure = classify_collect_exception(
            FirstTokenTimeoutError(
                "Model did not respond within 30.0s after 3 attempts. Please try again."
            )
        )
        assert "after 3 attempts" in failure.message

    def test_augments_bare_timeout_message(self):
        """
        What it does: Adds guidance to a bare timeout message.
        Purpose: Error messages must tell the user what to do next.
        """
        failure = classify_collect_exception(
            FirstTokenTimeoutError("No response within 30.0 seconds")
        )
        assert "No response within 30.0 seconds" in failure.message
        assert "try again" in failure.message.lower()

    def test_empty_message_still_produces_guidance(self):
        """
        What it does: Handles an empty exception message.
        Purpose: Never surface a blank error body to the client.
        """
        failure = classify_collect_exception(FirstTokenTimeoutError(""))
        assert failure.message.strip()
        assert "try again" in failure.message.lower()

    def test_agrees_with_streaming_classifier(self):
        """
        What it does: Cross-checks against classify_streaming_exception.
        Purpose: Both paths must bucket the same failure identically.
        """
        exc = FirstTokenTimeoutError("No response within 30.0 seconds")
        src, code, phase, status = classify_streaming_exception(exc)
        failure = classify_collect_exception(exc)
        assert (failure.source, failure.code, failure.phase, failure.status_code) == (
            src, code, phase, status,
        )


class TestClassifyTransportErrors:
    """httpx transport failures are upstream problems, not gateway bugs."""

    def test_remote_protocol_error_maps_to_502(self):
        """
        What it does: Classifies a truncated upstream response.
        Purpose: TRAY-1R reported these as 500 gateway errors.
        """
        failure = classify_collect_exception(
            httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )
        )
        assert failure.status_code == 502
        assert failure.code == "incomplete_upstream_response"
        assert failure.source == "network"
        assert failure.is_upstream is True

    def test_remote_protocol_error_agrees_with_streaming_classifier(self):
        """
        What it does: Cross-checks the cut-stream mapping.
        Purpose: Keep streaming / non-streaming dashboards consistent.
        """
        exc = httpx.RemoteProtocolError("incomplete chunked read")
        src, code, phase, status = classify_streaming_exception(exc)
        failure = classify_collect_exception(exc)
        assert (failure.source, failure.code, failure.phase, failure.status_code) == (
            src, code, phase, status,
        )

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ReadTimeout("read timed out"),
            httpx.ConnectTimeout("connect timed out"),
            httpx.WriteTimeout("write timed out"),
        ],
    )
    def test_timeouts_map_to_504(self, exc):
        """
        What it does: Classifies httpx read/connect/write timeouts.
        Purpose: Upstream slowness is 504. PoolTimeout is classified separately.
        """
        failure = classify_collect_exception(exc)
        assert failure.status_code == 504
        assert failure.code == "timeout"
        assert failure.is_upstream is True

    def test_pool_timeout_maps_to_503_not_504(self):
        """
        What it does: Classifies PoolTimeout as local capacity, not a read timeout.
        Purpose: TRAY-1B retried PoolTimeout as 504 and froze the machine.
        """
        failure = classify_collect_exception(httpx.PoolTimeout("pool timed out"))
        assert failure.status_code == 503
        assert failure.code == "pool_exhausted"
        assert failure.source == "network"
        assert failure.is_upstream is True
        assert failure.login_required is False

    def test_pool_timeout_agrees_with_streaming_classifier(self):
        """
        What it does: Cross-checks PoolTimeout mapping on both classifiers.
        Purpose: Streaming and collect dashboards must share pool_exhausted.
        """
        exc = httpx.PoolTimeout("No available connections")
        src, code, phase, status = classify_streaming_exception(exc)
        failure = classify_collect_exception(exc)
        assert (failure.source, failure.code, failure.phase, failure.status_code) == (
            src, code, phase, status,
        )

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadError("read failed"),
            httpx.WriteError("write failed"),
            httpx.CloseError("close failed"),
        ],
    )
    def test_transport_errors_map_to_502(self, exc):
        """
        What it does: Classifies non-timeout transport failures.
        Purpose: Connection-level failures are 502 upstream incidents.
        """
        failure = classify_collect_exception(exc)
        assert failure.status_code == 502
        assert failure.code == "connection_error"
        assert failure.phase == "connect"
        assert failure.is_upstream is True

    def test_remote_protocol_error_wins_over_generic_transport(self):
        """
        What it does: Verifies branch ordering.
        Purpose: RemoteProtocolError is a TransportError subclass; the more
                 specific "incomplete response" bucket must be chosen.
        """
        failure = classify_collect_exception(httpx.RemoteProtocolError("boom"))
        assert failure.code == "incomplete_upstream_response"

    def test_timeout_wins_over_generic_transport(self):
        """
        What it does: Verifies timeouts are not swallowed by the transport branch.
        Purpose: TimeoutException is also a TransportError subclass.
        """
        failure = classify_collect_exception(httpx.ReadTimeout("slow"))
        assert failure.code == "timeout"
        assert failure.status_code == 504

    def test_empty_transport_message_still_actionable(self):
        """
        What it does: Handles httpx errors with no message.
        Purpose: httpx frequently raises ConnectError(""), which previously
                 produced empty client-facing errors.
        """
        failure = classify_collect_exception(httpx.ConnectError(""))
        assert "ConnectError" in failure.message
        assert failure.message.strip()


class TestClassifyGatewayBugs:
    """Genuine gateway defects must stay 500 so they remain visible."""

    @pytest.mark.parametrize(
        "exc",
        [
            KeyError("missing"),
            TypeError("wrong type"),
            RuntimeError("boom"),
        ],
    )
    def test_non_transport_exceptions_map_to_500(self, exc):
        """
        What it does: Classifies ordinary programming errors.
        Purpose: Real bugs must not be relabelled as upstream problems.
        """
        failure = classify_collect_exception(exc)
        assert failure.status_code == 500
        assert failure.source == "gateway"
        assert failure.code == type(exc).__name__
        assert failure.phase == "unknown"
        assert failure.is_upstream is False

    def test_message_is_prefixed_for_clients(self):
        """
        What it does: Verifies the 500 message shape.
        Purpose: Keep the existing client-facing wording for gateway errors.
        """
        failure = classify_collect_exception(RuntimeError("oops"))
        assert failure.message == "Internal Server Error: oops"

    def test_http_status_error_429_is_not_auth_or_transport(self):
        """
        What it does: Ensures capacity HTTPStatusError is not 401 or 502.
        Purpose: 429 from a status error must not look like login_required
                 and must not be treated as a transport failure.
        """
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(429, request=request)
        failure = classify_collect_exception(
            httpx.HTTPStatusError("busy", request=request, response=response)
        )
        assert failure.status_code == 500
        assert failure.login_required is False
        assert failure.is_upstream is False

    def test_oidc_400_invalid_grant_maps_to_401(self):
        """
        What it does: Maps AWS SSO OIDC 400 to 401 login_required.
        Purpose: TRAY-1F/1Z: chat/messages returned 500 for expired creds.
        """
        request = httpx.Request("POST", "https://oidc.us-east-1.amazonaws.com/token")
        response = httpx.Response(
            400, request=request, json={"error": "invalid_grant"}
        )
        exc = httpx.HTTPStatusError(
            "Client error '400 Bad Request'", request=request, response=response
        )
        failure = classify_collect_exception(exc)
        assert failure.status_code == 401
        assert failure.code == "login_required"
        assert failure.login_required is True
        assert failure.source == "auth"
        assert failure.is_upstream is False
        assert "kiro-cli login" in failure.message.lower() or "sign in" in failure.message.lower()
        assert failure.reason == "invalid_grant"

    def test_oidc_400_agrees_with_streaming_classifier(self):
        """
        What it does: Cross-checks OIDC 400 on both classifiers.
        Purpose: Streaming and collect must both expose 401 login_required.
        """
        request = httpx.Request("POST", "https://oidc.test/token")
        response = httpx.Response(400, request=request)
        exc = httpx.HTTPStatusError("bad", request=request, response=response)
        src, code, phase, status = classify_streaming_exception(exc)
        failure = classify_collect_exception(exc)
        assert (failure.source, failure.code, failure.phase, failure.status_code) == (
            src, code, phase, status,
        )

    def test_access_token_value_error_maps_to_401(self):
        """
        What it does: Maps get_access_token() ValueError to login_required.
        Purpose: SQLite-mode refresh exhaustion raises ValueError, not HTTPStatusError.
        """
        failure = classify_collect_exception(
            ValueError("Token expired and refresh failed. Please run 'kiro-cli login'.")
        )
        assert failure.status_code == 401
        assert failure.login_required is True
        assert failure.code == "login_required"

    def test_generic_value_error_is_auth_because_is_auth_failure_says_so(self):
        """
        What it does: Any ValueError is classified as credentials, matching /usage.
        Purpose: is_auth_failure(ValueError) is True for the whole class; do not
                 keep a 500 mapping that contradicts that contract.
        """
        failure = classify_collect_exception(ValueError("oops"))
        assert failure.status_code == 401
        assert failure.login_required is True


class TestCollectFailureOrNone:
    """Helper used by routes to decide between failover and hard failure."""

    def test_returns_failure_for_upstream_error(self):
        """
        What it does: Returns a classification for upstream failures.
        Purpose: Routes use it to trigger account failover.
        """
        failure = collect_failure_or_none(httpx.ConnectError("refused"))
        assert isinstance(failure, CollectFailure)
        assert failure.is_upstream is True

    def test_returns_failure_for_pool_timeout(self):
        """
        What it does: PoolTimeout is tagged as an upstream-shaped incident.
        Purpose: SSE error bodies reuse collect_failure_or_none.
        """
        failure = collect_failure_or_none(httpx.PoolTimeout("full"))
        assert failure is not None
        assert failure.code == "pool_exhausted"

    def test_returns_none_for_oidc_400(self):
        """
        What it does: Auth failures are not transport failures.
        Purpose: Must not trigger account failover as a network outage.
        """
        request = httpx.Request("POST", "https://oidc.test/token")
        response = httpx.Response(400, request=request)
        assert collect_failure_or_none(
            httpx.HTTPStatusError("bad", request=request, response=response)
        ) is None

    def test_returns_none_for_gateway_bug(self):
        """
        What it does: Returns None for gateway bugs.
        Purpose: A gateway defect must not trigger failover retries.
        """
        assert collect_failure_or_none(RuntimeError("boom")) is None


class TestCollectFailureImmutability:
    """The classification is a value object; callers must not mutate it."""

    def test_is_frozen(self):
        """
        What it does: Verifies CollectFailure is immutable.
        Purpose: Prevent one route from corrupting shared classification data.
        """
        failure = classify_collect_exception(httpx.ConnectError("x"))
        with pytest.raises(Exception):
            failure.status_code = 500


class TestImmediateCollectResponse:
    """401/503 must leave the route as HTTP JSON, not a gateway 500."""

    def test_oidc_400_returns_401_with_login_required(self):
        """
        What it does: Builds the chat 401 envelope from an OIDC 400.
        Purpose: Clients must see login_required, not Internal Server Error.
        """
        request = httpx.Request("POST", "https://oidc.test/token")
        response = httpx.Response(400, request=request)
        failure = classify_collect_exception(
            httpx.HTTPStatusError("bad", request=request, response=response)
        )
        openai_resp = immediate_collect_response(failure, "openai")
        anthropic_resp = immediate_collect_response(failure, "anthropic")
        assert openai_resp is not None and openai_resp.status_code == 401
        assert anthropic_resp is not None and anthropic_resp.status_code == 401
        openai_body = openai_resp.body.decode()
        anthropic_body = anthropic_resp.body.decode()
        assert "login_required" in openai_body
        assert "login_required" in anthropic_body
        assert "authentication_error" in anthropic_body

    def test_pool_timeout_returns_503(self):
        """
        What it does: Builds a 503 pool_exhausted body.
        Purpose: Fail fast with a stable code instead of retrying as 504.
        """
        failure = classify_collect_exception(httpx.PoolTimeout("full"))
        resp = immediate_collect_response(failure, "openai")
        assert resp is not None
        assert resp.status_code == 503
        assert b"pool_exhausted" in resp.body

    def test_connect_error_returns_none_so_caller_can_failover(self):
        """
        What it does: Leaves transport failures to the route.
        Purpose: DNS/connect errors still fail over; do not steal that path.
        """
        failure = classify_collect_exception(httpx.ConnectError("refused"))
        assert immediate_collect_response(failure, "openai") is None

    def test_runtime_error_returns_none_for_gateway_500(self):
        """
        What it does: Leaves genuine bugs unmapped.
        Purpose: 500 remains the signal for unexpected gateway defects.
        """
        failure = classify_collect_exception(RuntimeError("boom"))
        assert immediate_collect_response(failure, "openai") is None

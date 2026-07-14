# -*- coding: utf-8 -*-

"""
Unit tests for OpenAI Responses API endpoints (routes_responses.py).

Covers:
- Authentication on POST /v1/responses
- HTTP 400 for unsupported input items
- POST /v1/responses/compact → 501
- Mocked KiroHttpClient streaming success (200)
"""

from unittest.mock import AsyncMock, patch

from kiro.config import PROXY_API_KEY


class TestResponsesAuth:
    """Authentication tests for /v1/responses."""

    def test_responses_requires_authentication(self, test_client):
        response = test_client.post(
            "/v1/responses",
            json={"model": "claude-sonnet-4-5", "input": "Hello"},
        )
        assert response.status_code == 401

    def test_responses_rejects_invalid_key(self, test_client, invalid_proxy_api_key):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"},
            json={"model": "claude-sonnet-4-5", "input": "Hello"},
        )
        assert response.status_code == 401

    def test_responses_accepts_valid_key_past_auth(
        self, test_client, valid_proxy_api_key
    ):
        """Valid key should not get 401 (may fail later on network/mock)."""
        with patch(
            "kiro.routes_responses.KiroHttpClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.request_with_retry = AsyncMock(
                side_effect=Exception("Network blocked")
            )
            mock_instance.close = AsyncMock()
            mock_cls.return_value = mock_instance

            response = test_client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={"model": "claude-sonnet-4-5", "input": "Hello"},
            )

        assert response.status_code != 401


class TestResponsesValidation:
    """Validation → HTTP 400 for unsupported Responses items."""

    def test_unknown_input_item_returns_400(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": [
                    {"type": "web_search_call", "id": "ws_1"},
                ],
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Unsupported" in detail or "web_search_call" in detail


class TestResponsesCompact:
    """POST /v1/responses/compact is intentionally unimplemented."""

    def test_compact_requires_auth(self, test_client):
        response = test_client.post("/v1/responses/compact", json={})
        assert response.status_code == 401

    def test_compact_returns_501(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses/compact",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={},
        )
        assert response.status_code == 501
        assert "not implemented" in response.json().get("detail", "").lower()


class TestResponsesStreaming:
    """Mocked KiroHttpClient streaming path returns 200."""

    @patch("kiro.routes_responses.stream_with_first_token_retry")
    @patch("kiro.routes_responses.KiroHttpClient")
    def test_streaming_returns_200(
        self,
        mock_kiro_http_client_class,
        mock_stream_retry,
        test_client,
        valid_proxy_api_key,
    ):
        async def mock_stream(*args, **kwargs):
            yield (
                'event: response.created\n'
                'data: {"type":"response.created","response":{"id":"resp_test"}}\n\n'
            )
            yield (
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
            )
            yield (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"id":"resp_test","status":"completed"}}\n\n'
            )

        mock_stream_retry.side_effect = mock_stream

        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_instance.client = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_instance

        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = response.text
        assert "response.created" in body
        assert "response.completed" in body
        mock_kiro_http_client_class.assert_called()
        call_kwargs = mock_kiro_http_client_class.call_args[1]
        assert call_kwargs.get("shared_client") is None

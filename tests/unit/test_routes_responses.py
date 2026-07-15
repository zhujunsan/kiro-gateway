# -*- coding: utf-8 -*-

"""
Unit tests for OpenAI Responses API endpoints (routes_responses.py).

Covers:
- Authentication on POST /v1/responses
- HTTP 400 for unsupported input items
- POST /v1/responses/compact → local compaction (200)
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
    """Validation → HTTP 400 / 422 for unsupported Responses items."""

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

    def test_hosted_only_tools_returns_422(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "tools": [{"type": "web_search"}, {"type": "file_search"}],
            },
        )
        assert response.status_code == 422
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("code") == "hosted_tools_not_supported"
        else:
            assert "hosted" in str(detail).lower()

    def test_tool_choice_hosted_type_returns_422(
        self, test_client, valid_proxy_api_key
    ):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "Read", "parameters": {}},
                    {"type": "web_search"},
                ],
                "tool_choice": {"type": "web_search"},
            },
        )
        assert response.status_code == 422
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("code") == "hosted_tools_not_supported"

    def test_tool_choice_unknown_function_returns_400(
        self, test_client, valid_proxy_api_key
    ):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "Read", "parameters": {}},
                ],
                "tool_choice": {"type": "function", "name": "missing_tool"},
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail", {})
        detail_text = detail.get("message", detail) if isinstance(detail, dict) else detail
        assert "not found" in str(detail_text).lower() or "missing_tool" in str(detail_text)

    def test_namespace_and_web_search_tools_do_not_400(
        self, test_client, valid_proxy_api_key
    ):
        """Codex sends namespace wrappers + web_search; must not reject the request."""
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
                json={
                    "model": "claude-sonnet-4-5",
                    "input": "Hello",
                    "tools": [
                        {
                            "type": "function",
                            "name": "exec_command",
                            "parameters": {"type": "object"},
                        },
                        {
                            "type": "namespace",
                            "name": "multi_agent_v1",
                            "description": "sub-agents",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "spawn_agent",
                                    "parameters": {"type": "object"},
                                }
                            ],
                        },
                        {"type": "web_search", "external_web_access": False},
                    ],
                },
            )

        # Must not be 400/422 — mixed hosted+function is allowed (hosted stripped)
        assert response.status_code not in (400, 422)
        detail = str(response.json().get("detail", ""))
        assert "Unsupported tool type" not in detail
        assert "hosted_tools_not_supported" not in detail

    def test_explicit_temperature_returns_400_sampling_not_supported(
        self, test_client, valid_proxy_api_key
    ):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "temperature": 0.7,
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            assert detail.get("code") == "sampling_not_supported"
        else:
            assert "sampling" in str(detail).lower() or "temperature" in str(detail)

    def test_explicit_top_p_returns_400_sampling_not_supported(
        self, test_client, valid_proxy_api_key
    ):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "top_p": 0.95,
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            assert detail.get("code") == "sampling_not_supported"

    def test_null_temperature_does_not_400_on_validation(
        self, test_client, valid_proxy_api_key
    ):
        """Omit/null sampling params must not fail validation (may fail later on mock)."""
        with patch("kiro.routes_responses.KiroHttpClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.request_with_retry = AsyncMock(
                side_effect=Exception("Network blocked")
            )
            mock_instance.close = AsyncMock()
            mock_cls.return_value = mock_instance

            response = test_client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "input": "Hello",
                    "temperature": None,
                    "top_p": None,
                },
            )

        assert response.status_code != 400
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            assert detail.get("code") != "sampling_not_supported"



class TestResponsesCompact:
    """POST /v1/responses/compact performs local history compaction."""

    def test_compact_requires_auth(self, test_client):
        response = test_client.post("/v1/responses/compact", json={})
        assert response.status_code == 401

    def test_compact_requires_model(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses/compact",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={},
        )
        assert response.status_code == 422

    def test_compact_requires_input(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses/compact",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4-5"},
        )
        assert response.status_code == 400
        assert "input is required" in str(response.json().get("detail", "")).lower()

    def test_compact_returns_compaction_object(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses/compact",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": [
                    {"type": "message", "role": "user", "content": "Hello"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi"}],
                    },
                    {"type": "message", "role": "user", "content": "Again"},
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("object") == "response.compaction"
        assert body.get("model") == "claude-sonnet-4-5"
        assert isinstance(body.get("output"), list)
        assert len(body["output"]) >= 1


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

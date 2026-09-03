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

import pytest
from fastapi import HTTPException

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

    @patch("kiro.routes_responses.stream_with_first_token_retry")
    @patch("kiro.routes_responses.KiroHttpClient")
    def test_first_token_timeout_emits_sse_error_without_re_raise(
        self,
        mock_kiro_http_client_class,
        mock_stream_retry,
        test_client,
        valid_proxy_api_key,
    ):
        """
        What it does: After a streamed chunk, first-token exhaustion raises
        HTTPException(504); route must emit SSE error and end the stream
        without re-raising (TRAY-M / Starlette RuntimeError).
        """
        async def mock_stream(*args, **kwargs):
            yield (
                'event: response.created\n'
                'data: {"type":"response.created","response":{"id":"resp_test"}}\n\n'
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    "Model did not respond within 30s after 3 attempts. "
                    "Please try again."
                ),
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
        body = response.text
        assert "response.created" in body
        assert "event: error" in body
        assert "did not respond within" in body
        assert "upstream_timeout" in body
        assert '"type": "server_error"' in body or '"type":"server_error"' in body
        assert "RuntimeError" not in body
        assert "Unexpected message" not in body


class TestResponsesModelsCompatibility:
    """Responses model listing shares OpenAI's async discovery service."""

    def test_openai_and_responses_model_routes_share_async_source(
        self, test_client, valid_proxy_api_key
    ):
        manager = test_client.app.state.account_manager
        model_ids = ["auto", "gpt-5.6-sol", "kiro-s-4.6", "claude-sonnet-4.6"]
        headers = {"Authorization": f"Bearer {valid_proxy_api_key}"}

        with patch.object(
            manager,
            "get_all_available_models",
            new=AsyncMock(return_value=model_ids),
        ) as get_models:
            openai_response = test_client.get("/v1/models", headers=headers)
            responses_response = test_client.get(
                "/v1/responses/models", headers=headers
            )

        assert openai_response.status_code == 200
        assert responses_response.status_code == 200
        # OpenAI data keeps aliases for Cursor; Codex models omit them
        assert [item["id"] for item in openai_response.json()["data"]] == model_ids
        codex_slugs = ["auto", "gpt-5.6-sol", "claude-sonnet-4.6"]
        assert [item["slug"] for item in openai_response.json()["models"]] == codex_slugs
        assert [item["slug"] for item in responses_response.json()["models"]] == codex_slugs
        assert get_models.await_count == 2


class TestInvalidModelExpectedResponse:
    """Responses API keeps model availability errors actionable in both modes."""

    @pytest.mark.parametrize("stream", [False, True])
    def test_invalid_model_returns_protocol_400_not_service_failure(
        self, test_client, valid_proxy_api_key, stream
    ):
        account = test_client.app.state.account_manager.get_first_account()
        manager = test_client.app.state.account_manager
        upstream = AsyncMock()
        upstream.status_code = 400
        upstream.aread = AsyncMock(return_value=(
            b'{"message":"Invalid model ID.",'
            b'"reason":"INVALID_MODEL_ID"}'
        ))
        client = AsyncMock()
        client.request_with_retry = AsyncMock(return_value=upstream)
        client.close = AsyncMock()
        client.client = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = True
        try:
            with patch.object(
                manager, "get_next_account", new=AsyncMock(return_value=account)
            ), patch.object(
                manager, "report_failure", new=AsyncMock()
            ), patch("kiro.routes_responses.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={
                        "model": "future-model-from-client",
                        "input": "hello",
                        "stream": stream,
                    },
                )
        finally:
            test_client.app.state.account_system = original_mode

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "model_not_available"
        assert error["param"] == "model"
        assert "GET /v1/models" in error["message"]
        assert client.request_with_retry.await_count == 1


class TestResponsesNetworkErrors:
    """Responses uses OpenAI network error bodies before and after SSE starts."""

    def test_legacy_502_returns_top_level_openai_error(
        self, test_client, valid_proxy_api_key
    ):
        """Avoid FastAPI detail for a typed network exception."""
        from kiro.network_errors import NetworkHTTPException

        client = AsyncMock()
        client.request_with_retry = AsyncMock(
            side_effect=NetworkHTTPException(
                status_code=502,
                error_code="proxy_error",
                user_message="Kiro Gateway could not connect to the Kiro upstream service through the proxy. Check proxy settings.",
            )
        )
        client.close = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = False
        try:
            with patch("kiro.routes_responses.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={"model": "claude-sonnet-4-5", "input": "hello"},
                )
        finally:
            test_client.app.state.account_system = original_mode

        assert response.status_code == 502
        assert "detail" not in response.json()
        assert response.json()["error"]["type"] == "server_error"
        assert response.json()["error"]["code"] == "proxy_error"

    @pytest.mark.parametrize("stream", [False, True])
    def test_oidc_400_returns_401_login_required(
        self, test_client, valid_proxy_api_key, stream
    ):
        """OIDC invalid_grant on /v1/responses must be 401, not 500."""
        import httpx

        request = httpx.Request("POST", "https://oidc.us-east-1.amazonaws.com/token")
        oidc_response = httpx.Response(400, request=request, json={"error": "invalid_grant"})
        client = AsyncMock()
        client.request_with_retry = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "400 invalid_grant", request=request, response=oidc_response
            )
        )
        client.close = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = False
        try:
            with patch("kiro.routes_responses.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={
                        "model": "claude-sonnet-4-5",
                        "input": "hello",
                        "stream": stream,
                    },
                )
        finally:
            test_client.app.state.account_system = original_mode

        assert response.status_code == 401
        body = response.json()
        assert body["error"]["login_required"] is True
        assert body["error"]["code"] == "login_required"
        assert "Internal Server Error" not in response.text

    def test_pool_exhausted_returns_503(self, test_client, valid_proxy_api_key):
        """PoolTimeout on /v1/responses is 503 pool_exhausted."""
        from kiro.network_errors import NetworkHTTPException

        client = AsyncMock()
        client.request_with_retry = AsyncMock(
            side_effect=NetworkHTTPException(
                status_code=503,
                error_code="pool_exhausted",
                user_message="Kiro Gateway could not connect to the Kiro upstream service: Connection pool exhausted. Retry shortly.",
            )
        )
        client.close = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = False
        try:
            with patch("kiro.routes_responses.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/responses",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={"model": "claude-sonnet-4-5", "input": "hello"},
                )
        finally:
            test_client.app.state.account_system = original_mode

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "pool_exhausted"

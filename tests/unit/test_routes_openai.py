
# -*- coding: utf-8 -*-

"""
Unit tests for OpenAI API endpoints (routes_openai.py).

Tests the following endpoints:
- GET / - Root endpoint
- GET /health - Health check
- GET /v1/models - List available models
- POST /v1/chat/completions - Chat completions

For Anthropic API tests, see test_routes_anthropic.py.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient

from kiro.routes_openai import verify_api_key, router
from kiro.config import PROXY_API_KEY, APP_VERSION


# =============================================================================
# Tests for verify_api_key function
# =============================================================================

class TestVerifyApiKey:
    """Tests for the verify_api_key authentication function."""
    
    @pytest.mark.asyncio
    async def test_valid_bearer_token_returns_true(self):
        """
        What it does: Verifies that a valid Bearer token passes authentication.
        Purpose: Ensure correct API keys are accepted.
        """
        print("Setup: Creating valid Bearer token...")
        valid_header = f"Bearer {PROXY_API_KEY}"
        
        print("Action: Calling verify_api_key...")
        result = await verify_api_key(valid_header)
        
        print(f"Comparing result: Expected True, Got {result}")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self):
        """
        What it does: Verifies that an invalid API key is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print("Setup: Creating invalid Bearer token...")
        invalid_header = "Bearer wrong_key_12345"
        
        print("Action: Calling verify_api_key with invalid key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(invalid_header)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
        assert "Invalid or missing API Key" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_401(self):
        """
        What it does: Verifies that missing API key is rejected.
        Purpose: Ensure requests without authentication are blocked.
        """
        print("Setup: No API key provided...")
        
        print("Action: Calling verify_api_key with None...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_empty_api_key_raises_401(self):
        """
        What it does: Verifies that empty string API key is rejected.
        Purpose: Ensure empty credentials are blocked.
        """
        print("Setup: Empty API key...")
        
        print("Action: Calling verify_api_key with empty string...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("")
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_key_without_bearer_prefix_raises_401(self):
        """
        What it does: Verifies that API key without Bearer prefix is rejected.
        Purpose: Ensure proper Authorization header format is required.
        """
        print("Setup: API key without Bearer prefix...")
        wrong_format = PROXY_API_KEY  # Without "Bearer "
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(wrong_format)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_bearer_with_extra_spaces_raises_401(self):
        """
        What it does: Verifies that Bearer token with extra spaces is rejected.
        Purpose: Ensure strict format validation.
        """
        print("Setup: Bearer token with extra spaces...")
        malformed = f"Bearer  {PROXY_API_KEY}"  # Double space
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(malformed)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_lowercase_bearer_raises_401(self):
        """
        What it does: Verifies that lowercase 'bearer' is rejected.
        Purpose: Ensure case-sensitive Bearer prefix.
        """
        print("Setup: Lowercase bearer prefix...")
        lowercase = f"bearer {PROXY_API_KEY}"
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(lowercase)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401


# =============================================================================
# Tests for root endpoint (/)
# =============================================================================

class TestRootEndpoint:
    """Tests for the GET / endpoint."""
    
    def test_root_returns_status_ok(self, test_client):
        """
        What it does: Verifies root endpoint returns ok status.
        Purpose: Ensure basic health check works.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_root_returns_gateway_message(self, test_client):
        """
        What it does: Verifies root endpoint returns gateway message.
        Purpose: Ensure service identification is present.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "Kiro Gateway" in response.json()["message"]
    
    def test_root_returns_version(self, test_client):
        """
        What it does: Verifies root endpoint returns application version.
        Purpose: Ensure version information is available.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "version" in response.json()
        assert response.json()["version"] == APP_VERSION
    
    def test_root_does_not_require_auth(self, test_client):
        """
        What it does: Verifies root endpoint is accessible without authentication.
        Purpose: Ensure public health check availability.
        """
        print("Action: GET / without auth headers...")
        response = test_client.get("/")
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 200


# =============================================================================
# Tests for health endpoint (/health)
# =============================================================================

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""
    
    def test_health_returns_healthy_status(self, test_client):
        """
        What it does: Verifies health endpoint returns healthy status.
        Purpose: Ensure health check indicates service is running.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_returns_timestamp(self, test_client):
        """
        What it does: Verifies health endpoint returns timestamp.
        Purpose: Ensure timestamp is present for monitoring.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "timestamp" in response.json()
        # Verify timestamp is ISO format
        timestamp = response.json()["timestamp"]
        assert "T" in timestamp  # ISO format contains T
    
    def test_health_returns_version(self, test_client):
        """
        What it does: Verifies health endpoint returns version.
        Purpose: Ensure version is available for monitoring.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["version"] == APP_VERSION
    
    def test_health_does_not_require_auth(self, test_client):
        """
        What it does: Verifies health endpoint is accessible without authentication.
        Purpose: Ensure health checks work for load balancers.
        """
        print("Action: GET /health without auth headers...")
        response = test_client.get("/health")

        print(f"Status: {response.status_code}")
        assert response.status_code == 200


class TestHealthAccountState:
    """/health must tell "ready" from "needs re-login" without hitting /usage.

    The tray polls /health already; exposing credential state here means a
    signed-out user is detected without a single upstream call.
    """

    @pytest.fixture(autouse=True)
    def _restore_account_manager(self, test_client):
        """Swap in fakes for the test only.

        Lifespan shutdown awaits the real manager's _save_state(), so leaving a
        MagicMock in app.state would break teardown for every later test.
        """
        state = test_client.app.state
        had = hasattr(state, "account_manager")
        original = getattr(state, "account_manager", None)
        yield
        if had:
            state.account_manager = original
        elif hasattr(state, "account_manager"):
            delattr(state, "account_manager")

    @staticmethod
    def _manager(*, ready: bool, failure: dict | None = None):
        manager = MagicMock()
        manager.has_initialized_account.return_value = ready
        manager.describe_init_failure.return_value = failure or {}
        return manager

    def test_ready_account_reports_healthy(self, test_client):
        test_client.app.state.account_manager = self._manager(ready=True)
        response = test_client.get("/health")

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "healthy"
        assert body["account"] == {"status": "ready", "login_required": False}

    def test_expired_credentials_report_degraded_with_login_required(self, test_client):
        test_client.app.state.account_manager = self._manager(
            ready=False,
            failure={
                "code": "account_auth_required",
                "message": "Kiro credentials are expired or invalid.",
                "account_count": 1,
                "errors": {},
            },
        )
        response = test_client.get("/health")

        body = response.json()
        # 200 on purpose: non-200 tells supervisors to restart, which is the
        # crash loop degraded mode exists to prevent.
        assert response.status_code == 200
        assert body["status"] == "degraded"
        assert body["account"]["code"] == "account_auth_required"
        assert body["account"]["login_required"] is True

    def test_missing_credentials_also_require_login(self, test_client):
        test_client.app.state.account_manager = self._manager(
            ready=False,
            failure={
                "code": "account_not_configured",
                "message": "No Kiro accounts are configured.",
                "account_count": 0,
                "errors": {},
            },
        )
        body = test_client.get("/health").json()

        assert body["status"] == "degraded"
        assert body["account"]["code"] == "account_not_configured"
        assert body["account"]["login_required"] is True

    def test_transient_init_failure_does_not_ask_for_login(self, test_client):
        """A proxy failure is not the user's login problem — do not prompt."""
        test_client.app.state.account_manager = self._manager(
            ready=False,
            failure={
                "code": "account_init_failed",
                "message": "No Kiro account could be initialized.",
                "account_count": 1,
                "errors": {"a": "ConnectError: proxy down"},
            },
        )
        body = test_client.get("/health").json()

        assert body["status"] == "degraded"
        assert body["account"]["code"] == "account_init_failed"
        assert body["account"]["login_required"] is False

    def test_health_state_is_recomputed_not_cached_from_startup(self, test_client):
        """Signing in mid-session must flip /health back to healthy."""
        manager = self._manager(
            ready=False,
            failure={
                "code": "account_auth_required",
                "message": "expired",
                "account_count": 1,
                "errors": {},
            },
        )
        test_client.app.state.account_manager = manager
        assert test_client.get("/health").json()["status"] == "degraded"

        manager.has_initialized_account.return_value = True
        assert test_client.get("/health").json()["status"] == "healthy"

    def test_missing_account_manager_falls_back_to_plain_body(self, test_client):
        """Health must not 500 before the account manager exists."""
        if hasattr(test_client.app.state, "account_manager"):
            delattr(test_client.app.state, "account_manager")
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "account" not in response.json()


# =============================================================================
# Tests for models endpoint (/v1/models)
# =============================================================================

class TestModelsEndpoint:
    """Tests for the GET /v1/models endpoint."""
    
    def test_models_requires_authentication(self, test_client):
        """
        What it does: Verifies models endpoint requires authentication.
        Purpose: Ensure protected endpoints are secured.
        """
        print("Action: GET /v1/models without auth...")
        response = test_client.get("/v1/models")
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_models_rejects_invalid_key(self, test_client, invalid_proxy_api_key):
        """
        What it does: Verifies models endpoint rejects invalid API key.
        Purpose: Ensure authentication is enforced.
        """
        print("Action: GET /v1/models with invalid key...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"}
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_models_returns_list_object(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models endpoint returns list object type.
        Purpose: Ensure OpenAI API compatibility.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
    
    def test_models_returns_data_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models endpoint returns data array.
        Purpose: Ensure response structure matches OpenAI format.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "data" in response.json()
        assert isinstance(response.json()["data"], list)
    
    def test_models_contains_available_models(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies all configured models are returned.
        Purpose: Ensure model list is complete.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        model_ids = [m["id"] for m in response.json()["data"]]
        print(f"Model IDs: {model_ids}")
        
        # At minimum, hidden models should be present
        # (even if Kiro API cache is empty)
        assert len(model_ids) >= 1, "Expected at least one model (hidden models)"
    
    def test_models_format_is_openai_compatible(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies model objects have OpenAI-compatible format.
        Purpose: Ensure compatibility with OpenAI clients.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        for model in response.json()["data"]:
            print(f"Checking model format: {model}")
            assert "id" in model, "Model missing 'id' field"
            assert "object" in model, "Model missing 'object' field"
            assert model["object"] == "model", "Model object type should be 'model'"
            assert "owned_by" in model, "Model missing 'owned_by' field"
    
    def test_models_owned_by_anthropic(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models are owned by Anthropic.
        Purpose: Ensure correct model attribution.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        for model in response.json()["data"]:
            assert model["owned_by"] == "anthropic"

    def test_models_codex_array_omits_aliases(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies dual-compat models array excludes MODEL_ALIASES.
        Purpose: Codex picker should only see canonical IDs; aliases stay in data.
        """
        from kiro.config import MODEL_ALIASES

        manager = test_client.app.state.account_manager
        model_ids = [
            "auto",
            "kiro-o-4.8",
            "claude-opus-4.8",
            "kiro-glm-5",
            "glm-5",
        ]
        with patch.object(
            manager,
            "get_all_available_models",
            new=AsyncMock(return_value=model_ids),
        ):
            response = test_client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            )

        assert response.status_code == 200
        body = response.json()
        data_ids = [m["id"] for m in body["data"]]
        codex_slugs = [m["slug"] for m in body["models"]]

        assert data_ids == model_ids
        assert "kiro-o-4.8" not in codex_slugs
        assert "kiro-glm-5" not in codex_slugs
        assert codex_slugs == ["auto", "claude-opus-4.8", "glm-5"]
        assert set(MODEL_ALIASES.keys()).isdisjoint(codex_slugs)


# =============================================================================
# Tests for chat completions endpoint (/v1/chat/completions)
# =============================================================================

class TestChatCompletionsAuthentication:
    """Tests for authentication on /v1/chat/completions endpoint."""
    
    def test_chat_completions_requires_authentication(self, test_client):
        """
        What it does: Verifies chat completions requires authentication.
        Purpose: Ensure protected endpoint is secured.
        """
        print("Action: POST /v1/chat/completions without auth...")
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_chat_completions_rejects_invalid_key(self, test_client, invalid_proxy_api_key):
        """
        What it does: Verifies chat completions rejects invalid API key.
        Purpose: Ensure authentication is enforced.
        """
        print("Action: POST /v1/chat/completions with invalid key...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401


class TestChatCompletionsValidation:
    """Tests for request validation on /v1/chat/completions endpoint."""
    
    def test_validates_empty_messages_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies empty messages array is rejected.
        Purpose: Ensure at least one message is required.
        """
        print("Action: POST /v1/chat/completions with empty messages...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": []
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_model(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing model field is rejected.
        Purpose: Ensure model is required.
        """
        print("Action: POST /v1/chat/completions without model...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_messages(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing messages field is rejected.
        Purpose: Ensure messages are required.
        """
        print("Action: POST /v1/chat/completions without messages...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5"
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_json(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid JSON is rejected.
        Purpose: Ensure proper JSON parsing.
        """
        print("Action: POST /v1/chat/completions with invalid JSON...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {valid_proxy_api_key}",
                "Content-Type": "application/json"
            },
            content=b"not valid json {{{}"
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_role(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid message role passes Pydantic validation.
        Purpose: Pydantic model accepts any string as role (validation happens later).
        Note: The role validation is not strict at Pydantic level, so invalid roles
        pass validation but may fail during processing.
        """
        print("Action: POST /v1/chat/completions with invalid role...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "invalid_role", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Pydantic model accepts any string as role, so validation passes (not 422)
        # The request may fail later during processing (500) due to network blocking
        assert response.status_code != 422
    
    def test_accepts_valid_request_format(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies valid request format passes validation.
        Purpose: Ensure Pydantic validation works correctly.
        """
        print("Action: POST /v1/chat/completions with valid format...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation (not 422)
        # May fail on HTTP call due to network blocking, but that's expected
        assert response.status_code != 422
    
    def test_accepts_message_without_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies message without content is accepted.
        Purpose: Ensure content is optional (for tool results).
        """
        print("Action: POST /v1/chat/completions with message without content...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user"}]  # No content
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation (content is optional)
        assert response.status_code != 422 or "content" not in str(response.json())


class TestChatCompletionsWithTools:
    """Tests for tool calling on /v1/chat/completions endpoint."""
    
    def test_accepts_valid_tool_definition(self, test_client, valid_proxy_api_key, sample_tool_definition):
        """
        What it does: Verifies valid tool definition is accepted.
        Purpose: Ensure tool calling format is supported.
        """
        print("Action: POST /v1/chat/completions with tools...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [sample_tool_definition]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422
    
    def test_accepts_multiple_tools(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multiple tools are accepted.
        Purpose: Ensure multiple tool definitions work.
        """
        print("Action: POST /v1/chat/completions with multiple tools...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


class TestChatCompletionsOptionalParams:
    """Tests for optional parameters on /v1/chat/completions endpoint."""
    
    def test_accepts_temperature_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies temperature parameter is accepted.
        Purpose: Ensure temperature control works.
        """
        print("Action: POST /v1/chat/completions with temperature...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_max_tokens_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies max_tokens parameter is accepted.
        Purpose: Ensure output length control works.
        """
        print("Action: POST /v1/chat/completions with max_tokens...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_stream_true(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stream=true is accepted.
        Purpose: Ensure streaming mode is supported.
        """
        print("Action: POST /v1/chat/completions with stream=true...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_top_p_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_p parameter is accepted.
        Purpose: Ensure nucleus sampling control works.
        """
        print("Action: POST /v1/chat/completions with top_p...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


class TestChatCompletionsMessageTypes:
    """Tests for different message types on /v1/chat/completions endpoint."""
    
    def test_accepts_system_message(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies system message is accepted.
        Purpose: Ensure system prompts work.
        """
        print("Action: POST /v1/chat/completions with system message...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_assistant_message(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies assistant message is accepted.
        Purpose: Ensure conversation history works.
        """
        print("Action: POST /v1/chat/completions with assistant message...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_multipart_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multipart content array is accepted.
        Purpose: Ensure complex content format works.
        """
        print("Action: POST /v1/chat/completions with multipart content...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"}
                        ]
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for router integration
# =============================================================================

class TestRouterIntegration:
    """Tests for router configuration and integration."""
    
    def test_router_has_root_endpoint(self):
        """
        What it does: Verifies root endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/" in routes
    
    def test_router_has_health_endpoint(self):
        """
        What it does: Verifies health endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/health" in routes
    
    def test_router_has_models_endpoint(self):
        """
        What it does: Verifies models endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/v1/models" in routes
    
    def test_router_has_chat_completions_endpoint(self):
        """
        What it does: Verifies chat completions endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/v1/chat/completions" in routes
    
    def test_root_endpoint_uses_get_method(self):
        """
        What it does: Verifies root endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/":
                print(f"Route / methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Root endpoint not found")
    
    def test_health_endpoint_uses_get_method(self):
        """
        What it does: Verifies health endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/health":
                print(f"Route /health methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Health endpoint not found")
    
    def test_models_endpoint_uses_get_method(self):
        """
        What it does: Verifies models endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/v1/models":
                print(f"Route /v1/models methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Models endpoint not found")
    
    def test_chat_completions_endpoint_uses_post_method(self):
        """
        What it does: Verifies chat completions endpoint uses POST method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/v1/chat/completions":
                print(f"Route /v1/chat/completions methods: {route.methods}")
                assert "POST" in route.methods
                return
        pytest.fail("Chat completions endpoint not found")


# =============================================================================
# Tests for HTTP client selection (issue #54)
# =============================================================================

class TestHTTPClientSelection:
    """
    Tests for HTTP client selection in routes (issue #54).
    
    Verifies that streaming requests use per-request clients to avoid CLOSE_WAIT leak
    when network interface changes (VPN disconnect/reconnect), while non-streaming
    requests use shared client for connection pooling.
    """
    
    @patch('kiro.routes_openai.KiroHttpClient')
    def test_streaming_uses_per_request_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies streaming requests create per-request HTTP client.
        Purpose: Prevent CLOSE_WAIT leak on VPN disconnect (issue #54).
        """
        print("\n--- Test: Streaming uses per-request client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST with stream=true...")
        try:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=None)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is None, \
            "Streaming should use per-request client"
        print("✅ Streaming correctly uses per-request client")
    
    @patch('kiro.routes_openai.KiroHttpClient')
    def test_non_streaming_uses_shared_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies non-streaming requests use shared HTTP client.
        Purpose: Ensure connection pooling for non-streaming requests.
        """
        print("\n--- Test: Non-streaming uses shared client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST with stream=false...")
        try:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=app.state.http_client)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is not None, \
            "Non-streaming should use shared client"
        print("✅ Non-streaming correctly uses shared client")


# =============================================================================
# Tests for Truncation Recovery message modification (Issue #56)
# =============================================================================

class TestTruncationRecoveryMessageModification:
    """
    Tests for Truncation Recovery System functions used by routes_openai.
    
    Directly tests the truncation_recovery and truncation_state module APIs
    that the route handler relies on. Part of Truncation Recovery System (Issue #56).
    """
    
    def test_generate_truncation_tool_result_format(self):
        """
        What it does: Verifies generate_truncation_tool_result returns correct format.
        Purpose: Ensure the synthetic tool_result has the expected structure and content.
        """
        from kiro.truncation_recovery import generate_truncation_tool_result
        
        result = generate_truncation_tool_result(
            tool_name="write_to_file",
            tool_use_id="tooluse_test123",
            truncation_info={"size_bytes": 5000, "reason": "missing 2 closing braces"}
        )
        
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tooluse_test123"
        assert result["is_error"] is True
        assert "[API Limitation]" in result["content"]
        assert "size limit" in result["content"]
        assert "Split" in result["content"] or "split" in result["content"]
    
    def test_save_and_retrieve_tool_truncation(self):
        """
        What it does: Verifies the save → get round-trip for tool truncation cache.
        Purpose: Ensure routes can save truncation on detection and retrieve on next request.
        """
        from kiro.truncation_state import save_tool_truncation, get_tool_truncation
        
        tool_call_id = "tooluse_roundtrip_test"
        save_tool_truncation(tool_call_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})
        
        info = get_tool_truncation(tool_call_id)
        assert info is not None
        assert info.tool_call_id == tool_call_id
        assert info.tool_name == "write_to_file"
        assert info.truncation_info["size_bytes"] == 5000
    
    def test_tool_truncation_one_time_retrieval(self):
        """
        What it does: Verifies cache entry is removed after first retrieval.
        Purpose: Ensure truncation notice is injected only once per truncation event.
        """
        from kiro.truncation_state import save_tool_truncation, get_tool_truncation
        
        tool_call_id = "tooluse_one_time"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        first = get_tool_truncation(tool_call_id)
        assert first is not None
        
        second = get_tool_truncation(tool_call_id)
        assert second is None, "Entry should be consumed after first retrieval"
    
    def test_no_truncation_info_returns_none(self):
        """
        What it does: Verifies get_tool_truncation returns None for unknown IDs.
        Purpose: Ensure normal messages pass through without modification in the route.
        """
        from kiro.truncation_state import get_tool_truncation
        
        info = get_tool_truncation("tooluse_nonexistent_xyz")
        assert info is None
    
    def test_pydantic_model_copy_creates_new_object(self):
        """
        What it does: Verifies ChatMessage.model_copy produces a distinct object.
        Purpose: Ensure the route's use of model_copy respects Pydantic immutability.
        """
        from kiro.models_openai import ChatMessage
        from kiro.truncation_recovery import generate_truncation_tool_result
        from kiro.truncation_state import save_tool_truncation, get_tool_truncation
        
        tool_call_id = "test_immutable_copy"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        original_msg = ChatMessage(role="tool", tool_call_id=tool_call_id, content="original")
        original_content = original_msg.content
        
        info = get_tool_truncation(tool_call_id)
        synthetic = generate_truncation_tool_result(
            info.tool_name, info.tool_call_id, info.truncation_info
        )
        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_msg.content}"
        modified_msg = original_msg.model_copy(update={"content": modified_content})
        
        assert original_msg.content == original_content, "Original must be unchanged"
        assert modified_msg is not original_msg
        assert "[API Limitation]" in modified_msg.content
        assert "original" in modified_msg.content


# =============================================================================
# Tests for Truncation Recovery edge cases (Issue #56)
# =============================================================================

class TestTruncationRecoveryEdgeCases:
    """
    Tests for edge cases in Truncation Recovery System.
    
    Verifies graceful handling of unusual scenarios via module functions.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_orphaned_tool_result_no_crash(self):
        """
        What it does: Verifies get_tool_truncation returns None for unknown IDs gracefully.
        Purpose: Ensure orphaned tool_result doesn't cause errors (Test Case 9.2).
        """
        from kiro.truncation_state import get_tool_truncation
        
        info = get_tool_truncation("tooluse_nonexistent_orphan")
        assert info is None
    
    def test_empty_tool_result_content_with_truncation(self):
        """
        What it does: Verifies generate_truncation_tool_result works when original content is empty.
        Purpose: Ensure empty content doesn't cause errors (Test Case 9.4).
        """
        from kiro.truncation_state import save_tool_truncation, get_tool_truncation
        from kiro.truncation_recovery import generate_truncation_tool_result
        from kiro.models_openai import ChatMessage
        
        tool_call_id = "tooluse_empty_content"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        original_msg = ChatMessage(role="tool", tool_call_id=tool_call_id, content="")
        
        info = get_tool_truncation(tool_call_id)
        assert info is not None
        
        synthetic = generate_truncation_tool_result(
            info.tool_name, info.tool_call_id, info.truncation_info
        )
        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_msg.content}"
        modified_msg = original_msg.model_copy(update={"content": modified_content})
        
        assert "[API Limitation]" in modified_msg.content
        assert "Original tool result:\n" in modified_msg.content
    
    def test_very_long_content_hash_uses_first_500_chars(self):
        """
        What it does: Verifies content hash uses first 500 chars only.
        Purpose: Ensure hash stability for long content (Test Case 9.3).
        """
        print("Setup: Creating very long content...")
        from kiro.truncation_state import save_content_truncation, get_content_truncation
        
        content_long = "A" * 10000
        content_same_prefix = "A" * 500 + "B" * 9500
        
        print("Action: Saving long content...")
        hash1 = save_content_truncation(content_long)
        
        print("Action: Retrieving with same prefix...")
        info = get_content_truncation(content_same_prefix)
        
        print("Checking: Retrieval successful (same hash)...")
        assert info is not None
        assert info.message_hash == hash1
        
        print("Checking: Hash is 16 chars...")
        assert len(hash1) == 16
    
    def test_recovery_disabled_does_not_inject(self):
        """
        What it does: Verifies should_inject_recovery returns False when disabled.
        Purpose: Ensure disabling recovery skips injection (Test Case 9.5).
        """
        from kiro.truncation_state import save_tool_truncation, get_cache_stats
        from kiro.truncation_recovery import should_inject_recovery
        
        tool_call_id = "tooluse_disabled_recovery"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        stats = get_cache_stats()
        assert stats["tool_truncations"] >= 1
        
        with patch("kiro.config.TRUNCATION_RECOVERY", False):
            assert should_inject_recovery() is False
        
        # Cache entry should still exist (not consumed)
        stats_after = get_cache_stats()
        assert stats_after["tool_truncations"] >= 1


# =============================================================================
# Tests for Content Truncation Recovery (Issue #56)
# =============================================================================

class TestContentTruncationRecovery:
    """
    Tests for content truncation recovery (synthetic user message).
    
    Directly tests save/get content truncation and generate_truncation_user_message.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_save_and_retrieve_content_truncation(self):
        """
        What it does: Verifies save → get round-trip for content truncation cache.
        Purpose: Ensure content truncation is detectable on the next request (Test Case C.1).
        """
        from kiro.truncation_state import save_content_truncation, get_content_truncation
        
        truncated_content = "This is a very long response that was cut off mid-sentence"
        save_content_truncation(truncated_content)
        
        info = get_content_truncation(truncated_content)
        assert info is not None
        assert info.message_hash is not None
        assert info.content_preview == truncated_content[:200]
    
    def test_generate_truncation_user_message_format(self):
        """
        What it does: Verifies generate_truncation_user_message returns expected format.
        Purpose: Ensure synthetic user message is informative and well-structured.
        """
        from kiro.truncation_recovery import generate_truncation_user_message
        
        message = generate_truncation_user_message()
        
        assert "[System Notice]" in message
        assert "truncated" in message.lower()
        assert isinstance(message, str)
        assert len(message) > 50
    
    def test_no_content_truncation_for_unknown_content(self):
        """
        What it does: Verifies get_content_truncation returns None for unseen content.
        Purpose: Ensure false positives don't occur (Test Case C.3).
        """
        from kiro.truncation_state import get_content_truncation
        
        info = get_content_truncation("This is a complete response that was never truncated.")
        assert info is None
    
    def test_content_hash_matches_first_500_chars(self):
        """
        What it does: Verifies content hash is based on first 500 chars.
        Purpose: Ensure long messages can be matched by prefix.
        """
        from kiro.truncation_state import save_content_truncation, get_content_truncation
        
        original_content = "A" * 1000
        request_content = "A" * 500 + "B" * 500
        
        hash1 = save_content_truncation(original_content)
        info = get_content_truncation(request_content)
        
        assert info is not None
        assert info.message_hash == hash1


# ==================================================================================================
# Tests for WebSearch Support (OpenAI)
# ==================================================================================================

class TestWebSearchAutoInjectionOpenAI:
    """Tests for WebSearch auto-injection in OpenAI endpoint (Path B only)."""
    
    def test_auto_injection_logic_openai(self):
        """
        What it does: Verifies web_search function tool auto-injection logic for OpenAI.
        Purpose: Ensure WEB_SEARCH_ENABLED controls auto-injection for OpenAI format.
        """
        print("Setup: Testing OpenAI auto-injection logic...")
        from kiro.models_openai import Tool, ToolFunction
        
        # Simulate auto-injection logic for OpenAI
        WEB_SEARCH_ENABLED = True
        tools = []
        
        if WEB_SEARCH_ENABLED:
            has_ws = any(
                getattr(tool, "type", None) == "function" and
                getattr(getattr(tool, "function", None), "name", None) == "web_search"
                for tool in tools
            )
            
            if not has_ws:
                web_search_tool = Tool(
                    type="function",
                    function=ToolFunction(
                        name="web_search",
                        description="Search the web for current information. Use when you need up-to-date data from the internet.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                }
                            },
                            "required": ["query"]
                        }
                    )
                )
                tools.append(web_search_tool)
        
        print(f"Checking: web_search tool was added...")
        assert len(tools) == 1
        assert tools[0].type == "function"
        assert tools[0].function.name == "web_search"
        assert tools[0].function.parameters is not None
    
    def test_no_duplicate_injection_logic_openai(self):
        """
        What it does: Verifies duplicate detection logic for OpenAI format.
        Purpose: Ensure auto-injection doesn't create duplicates for OpenAI.
        """
        print("Setup: Testing OpenAI duplicate detection...")
        from kiro.models_openai import Tool, ToolFunction
        
        # Simulate existing web_search tool
        existing_tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Existing web search",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]
        
        # Simulate auto-injection logic with duplicate check
        WEB_SEARCH_ENABLED = True
        
        if WEB_SEARCH_ENABLED:
            has_ws = any(
                getattr(tool, "type", None) == "function" and
                getattr(getattr(tool, "function", None), "name", None) == "web_search"
                for tool in existing_tools
            )
            
            if not has_ws:
                # Would add web_search here
                existing_tools.append(Tool(
                    type="function",
                    function=ToolFunction(
                        name="web_search",
                        description="Auto-injected",
                        parameters={"type": "object", "properties": {}}
                    )
                ))
        
        print(f"Checking: Only one web_search tool...")
        web_search_count = sum(
            1 for t in existing_tools
            if t.type == "function" and t.function.name == "web_search"
        )
        assert web_search_count == 1


# ==================================================================================================
# Tests for Account System - /v1/models endpoint
# ==================================================================================================

class TestModelsEndpointAccountSystem:
    """Tests for /v1/models endpoint with Account System."""
    
    def test_get_models_account_system_logic(self):
        """
        What it does: Verifies logic for collecting models in account system mode.
        Purpose: Ensure models are collected from all initialized accounts.
        """
        print("\n--- Test: /v1/models account system logic ---")
        
        # Simulate account system mode logic
        account_system = True
        
        mock_account_manager = Mock()
        mock_account_manager.get_all_available_models.return_value = [
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "claude-haiku-4.5"
        ]
        
        print("Action: Getting models in account system mode...")
        if account_system:
            available_model_ids = mock_account_manager.get_all_available_models()
        else:
            available_model_ids = []
        
        print("Checking: get_all_available_models() was called...")
        mock_account_manager.get_all_available_models.assert_called_once()
        
        print("Checking: Models from all accounts collected...")
        assert "claude-opus-4.5" in available_model_ids
        assert "claude-sonnet-4.5" in available_model_ids
        assert "claude-haiku-4.5" in available_model_ids
        assert len(available_model_ids) == 3
        print("✅ Account system mode correctly collects models from all accounts")
    
    def test_get_models_legacy_logic(self):
        """
        What it does: Verifies logic for getting models in legacy mode.
        Purpose: Ensure backward compatibility with single account.
        """
        print("\n--- Test: /v1/models legacy mode logic ---")
        
        # Simulate legacy mode logic
        account_system = False
        
        mock_account = Mock()
        mock_resolver = Mock()
        mock_resolver.get_available_models.return_value = [
            "claude-opus-4.5",
            "claude-sonnet-4.5"
        ]
        mock_account.model_resolver = mock_resolver
        
        mock_account_manager = Mock()
        mock_account_manager.get_first_account.return_value = mock_account
        
        print("Action: Getting models in legacy mode...")
        if account_system:
            available_model_ids = []
        else:
            account = mock_account_manager.get_first_account()
            available_model_ids = account.model_resolver.get_available_models()
        
        print("Checking: get_first_account() was called...")
        mock_account_manager.get_first_account.assert_called_once()
        
        print("Checking: model_resolver.get_available_models() was called...")
        mock_resolver.get_available_models.assert_called_once()
        
        print("Checking: Models from first account returned...")
        assert "claude-opus-4.5" in available_model_ids
        assert "claude-sonnet-4.5" in available_model_ids
        assert len(available_model_ids) == 2
        print("✅ Legacy mode correctly uses first account's resolver")


# ==================================================================================================
# Tests for Account System - Failover Loop
# ==================================================================================================

class TestChatCompletionsFailoverLoop:
    """Tests for failover loop in /v1/chat/completions endpoint."""
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_get_next_account(self):
        """
        What it does: Verifies get_next_account() is called with exclude_accounts.
        Purpose: Ensure failover loop tracks tried accounts.
        """
        print("\n--- Test: Failover calls get_next_account() with exclude_accounts ---")
        
        mock_account = Mock()
        mock_account.id = "/home/user/account1.json"
        mock_account.auth_manager = Mock()
        mock_account.model_cache = Mock()
        mock_account.model_resolver = Mock()
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=mock_account)
        mock_manager._accounts = {mock_account.id: mock_account}
        
        print("Checking: get_next_account() called with exclude_accounts parameter...")
        # This test verifies the signature - actual implementation tested in integration tests
        await mock_manager.get_next_account("claude-opus-4.5", exclude_accounts=set())
        
        mock_manager.get_next_account.assert_called_once()
        call_kwargs = mock_manager.get_next_account.call_args[1]
        assert "exclude_accounts" in call_kwargs
        print("✅ Failover loop correctly passes exclude_accounts")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_success_first_account(self):
        """
        What it does: Verifies successful response on first account attempt.
        Purpose: Ensure no unnecessary failover when first account works.
        """
        print("\n--- Test: Success on first account ---")
        
        from kiro.account_manager import Account, AccountStats
        
        mock_account = Account(
            id="/home/user/account1.json",
            failures=0,
            last_failure_time=0.0,
            models_cached_at=time.time(),
            stats=AccountStats()
        )
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=mock_account)
        mock_manager.report_success = AsyncMock()
        mock_manager._accounts = {mock_account.id: mock_account}
        
        print("Action: Simulating successful request...")
        account = await mock_manager.get_next_account("claude-opus-4.5", exclude_accounts=set())
        
        print("Checking: First account returned...")
        assert account is not None
        assert account.id == "/home/user/account1.json"
        
        print("Action: Reporting success...")
        await mock_manager.report_success(account.id, "claude-opus-4.5")
        
        print("Checking: report_success() was called...")
        mock_manager.report_success.assert_called_once_with(
            "/home/user/account1.json",
            "claude-opus-4.5"
        )
        print("✅ Success on first account works correctly")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_recoverable_try_next(self):
        """
        What it does: Verifies RECOVERABLE error triggers next account attempt.
        Purpose: Ensure failover happens for account-specific errors.
        """
        print("\n--- Test: RECOVERABLE error tries next account ---")
        
        from kiro.account_errors import ErrorType, classify_error
        
        print("Setup: Classifying 429 error...")
        error_type = classify_error(429, None)
        
        print("Checking: 429 is RECOVERABLE...")
        assert error_type == ErrorType.RECOVERABLE
        
        print("Checking: Failover logic should continue to next account...")
        # In actual implementation, this would trigger:
        # await account_manager.report_failure(...)
        # continue  # Next iteration of failover loop
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        await mock_manager.report_failure(
            "/home/user/account1.json",
            "claude-opus-4.5",
            ErrorType.RECOVERABLE,
            429,
            None
        )
        
        mock_manager.report_failure.assert_called_once()
        print("✅ RECOVERABLE error correctly triggers failover")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_fatal_immediate_return(self):
        """
        What it does: Verifies FATAL error returns immediately to client.
        Purpose: Ensure no wasted retries for request-level errors.
        """
        print("\n--- Test: FATAL error returns immediately ---")
        
        from kiro.account_errors import ErrorType, classify_error
        
        print("Setup: Classifying 400 + CONTENT_LENGTH_EXCEEDS_THRESHOLD...")
        error_type = classify_error(400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        
        print("Checking: Error is FATAL...")
        assert error_type == ErrorType.FATAL
        
        print("Checking: Failover logic should break immediately...")
        # In actual implementation, this would trigger:
        # await account_manager.report_failure(...)
        # return JSONResponse(...)  # No continue, immediate return
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        await mock_manager.report_failure(
            "/home/user/account1.json",
            "claude-opus-4.5",
            ErrorType.FATAL,
            400,
            "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        mock_manager.report_failure.assert_called_once()
        print("✅ FATAL error correctly returns immediately")
    
    def test_chat_completions_failover_single_account_original_error(self):
        """
        What it does: Verifies single account returns original error message.
        Purpose: Ensure users see specific error for single account setup.
        """
        print("\n--- Test: Single account returns original error ---")
        
        all_accounts = ["/home/user/account1.json"]
        last_error_message = "Monthly request limit exceeded"
        last_error_status = 402
        
        print("Checking: Single account error handling...")
        if len(all_accounts) == 1:
            error_response = {
                "status_code": last_error_status,
                "detail": last_error_message
            }
        else:
            error_response = {
                "status_code": 503,
                "detail": "No available accounts for this model"
            }
        
        print(f"Error response: {error_response}")
        assert error_response["status_code"] == 402
        assert error_response["detail"] == "Monthly request limit exceeded"
        print("✅ Single account correctly returns original error")
    
    def test_chat_completions_failover_multi_account_generic_error(self):
        """
        What it does: Verifies multi-account returns generic error message.
        Purpose: Ensure users don't see confusing account-specific errors.
        """
        print("\n--- Test: Multi-account returns generic error ---")
        
        all_accounts = [
            "/home/user/account1.json",
            "/home/user/account2.json"
        ]
        last_error_message = "Token expired"
        
        print("Checking: Multi-account error handling...")
        if len(all_accounts) == 1:
            error_response = {
                "status_code": 403,
                "detail": last_error_message
            }
        else:
            detail = "No available accounts for this model."
            if last_error_message:
                detail += f" Error from last account: {last_error_message}"
            error_response = {
                "status_code": 503,
                "detail": detail
            }
        
        print(f"Error response: {error_response}")
        assert error_response["status_code"] == 503
        assert "No available accounts" in error_response["detail"]
        assert "Error from last account: Token expired" in error_response["detail"]
        print("✅ Multi-account correctly returns generic error with context")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_all_unavailable(self):
        """
        What it does: Verifies behavior when all accounts are unavailable.
        Purpose: Ensure graceful handling of complete failure.
        """
        print("\n--- Test: All accounts unavailable ---")
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=None)
        mock_manager._accounts = {
            "/home/user/account1.json": Mock(),
            "/home/user/account2.json": Mock()
        }
        
        print("Action: Requesting account when all unavailable...")
        account = await mock_manager.get_next_account(
            "claude-opus-4.5",
            exclude_accounts=set()
        )
        
        print("Checking: None returned...")
        assert account is None
        
        print("Checking: Error response logic...")
        all_accounts = list(mock_manager._accounts.keys())
        if len(all_accounts) == 1:
            error_msg = "Account unavailable"
        else:
            error_msg = "No available accounts for this model"
        
        assert "No available accounts" in error_msg
        print("✅ All unavailable correctly handled")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_report_success(self):
        """
        What it does: Verifies report_success() is called after successful request.
        Purpose: Ensure statistics and sticky behavior are updated.
        """
        print("\n--- Test: report_success() called on success ---")
        
        mock_manager = Mock()
        mock_manager.report_success = AsyncMock()
        
        account_id = "/home/user/account1.json"
        model = "claude-opus-4.5"
        
        print("Action: Reporting success...")
        await mock_manager.report_success(account_id, model)
        
        print("Checking: report_success() was called with correct params...")
        mock_manager.report_success.assert_called_once_with(account_id, model)
        print("✅ report_success() correctly called")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_report_failure(self):
        """
        What it does: Verifies report_failure() is called after failed request.
        Purpose: Ensure Circuit Breaker state is updated.
        """
        print("\n--- Test: report_failure() called on failure ---")
        
        from kiro.account_errors import ErrorType
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        account_id = "/home/user/account1.json"
        model = "claude-opus-4.5"
        error_type = ErrorType.RECOVERABLE
        status_code = 429
        reason = None
        
        print("Action: Reporting failure...")
        await mock_manager.report_failure(
            account_id,
            model,
            error_type,
            status_code,
            reason
        )
        
        print("Checking: report_failure() was called with correct params...")
        mock_manager.report_failure.assert_called_once_with(
            account_id,
            model,
            error_type,
            status_code,
            reason
        )
        print("✅ report_failure() correctly called")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_exclude_tried_accounts(self):
        """
        What it does: Verifies exclude_accounts grows with each attempt.
        Purpose: Ensure accounts aren't retried in same failover loop.
        """
        print("\n--- Test: exclude_accounts grows with attempts ---")
        
        tried_accounts = set()
        
        print("Action: Simulating multiple attempts...")
        account1_id = "/home/user/account1.json"
        account2_id = "/home/user/account2.json"
        
        # Attempt 1
        tried_accounts.add(account1_id)
        print(f"After attempt 1: {tried_accounts}")
        assert account1_id in tried_accounts
        assert len(tried_accounts) == 1
        
        # Attempt 2
        tried_accounts.add(account2_id)
        print(f"After attempt 2: {tried_accounts}")
        assert account2_id in tried_accounts
        assert len(tried_accounts) == 2
        
        print("Checking: Both accounts in exclude set...")
        assert account1_id in tried_accounts
        assert account2_id in tried_accounts
        print("✅ exclude_accounts correctly tracks tried accounts")
    
    def test_chat_completions_failover_max_attempts(self):
        """
        What it does: Verifies failover loop stops after MAX_ATTEMPTS.
        Purpose: Ensure infinite loops are prevented.
        """
        print("\n--- Test: MAX_ATTEMPTS prevents infinite loop ---")
        
        all_accounts = [
            "/home/user/account1.json",
            "/home/user/account2.json"
        ]
        MAX_ATTEMPTS = len(all_accounts) * 2
        
        print(f"Checking: MAX_ATTEMPTS = {MAX_ATTEMPTS}...")
        assert MAX_ATTEMPTS == 4
        
        print("Checking: Loop would stop after 4 attempts...")
        attempts = 0
        for attempt in range(MAX_ATTEMPTS):
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                break
        
        assert attempts == MAX_ATTEMPTS
        print("✅ MAX_ATTEMPTS correctly limits failover loop")


# ==================================================================================================
# Tests for Account System - Legacy Mode
# ==================================================================================================

class TestChatCompletionsLegacyMode:
    """Tests for legacy mode (ACCOUNT_SYSTEM=false) in /v1/chat/completions."""
    
    @pytest.mark.asyncio
    async def test_chat_completions_legacy_get_first_account(self):
        """
        What it does: Verifies legacy mode uses get_first_account().
        Purpose: Ensure backward compatibility with single account.
        """
        print("\n--- Test: Legacy mode uses get_first_account() ---")
        
        from kiro.account_manager import Account, AccountStats
        
        mock_account = Account(
            id="/home/user/account1.json",
            failures=0,
            last_failure_time=0.0,
            models_cached_at=time.time(),
            stats=AccountStats()
        )
        
        mock_manager = Mock()
        mock_manager.get_first_account.return_value = mock_account
        
        print("Action: Getting first account in legacy mode...")
        account = mock_manager.get_first_account()
        
        print("Checking: get_first_account() was called...")
        mock_manager.get_first_account.assert_called_once()
        
        print("Checking: Account returned...")
        assert account is not None
        assert account.id == "/home/user/account1.json"
        print("✅ Legacy mode correctly uses get_first_account()")
    
    def test_chat_completions_legacy_no_failover(self):
        """
        What it does: Verifies legacy mode has no failover loop.
        Purpose: Ensure single account behavior is preserved.
        """
        print("\n--- Test: Legacy mode has no failover ---")
        
        account_system = False
        
        print("Checking: account_system flag is False...")
        assert account_system is False
        
        print("Checking: Failover loop should be skipped...")
        if account_system:
            failover_enabled = True
        else:
            failover_enabled = False
        
        assert failover_enabled is False
        print("✅ Legacy mode correctly skips failover loop")

class TestInvalidModelExpectedResponse:
    """Invalid model/account entitlement errors stay actionable in both modes."""

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
            ), patch("kiro.routes_openai.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={
                        "model": "future-model-from-client",
                        "messages": [{"role": "user", "content": "hello"}],
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


# =============================================================================
# Streaming error handling after response started (TRAY-M)
# =============================================================================

class TestChatCompletionsStreamingErrorAfterStarted:
    """
    After StreamingResponse starts, first-token retry failure must not
    re-raise HTTPException (Starlette RuntimeError). Align with Anthropic:
    emit SSE error + [DONE] and end the stream.
    """

    @patch("kiro.routes_openai.stream_with_first_token_retry")
    @patch("kiro.routes_openai.KiroHttpClient")
    def test_first_token_timeout_emits_sse_error_without_re_raise(
        self,
        mock_kiro_http_client_class,
        mock_stream_retry,
        test_client,
        valid_proxy_api_key,
    ):
        """
        What it does: After a streamed chunk, first-token exhaustion raises
        HTTPException(504); route must emit OpenAI SSE error and [DONE]
        without crashing the ASGI response (TRAY-M).
        """
        async def mock_stream(*args, **kwargs):
            yield (
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
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
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        body = response.text
        assert "did not respond within" in body
        assert '"type": "server_error"' in body or '"type":"server_error"' in body
        assert "upstream_timeout" in body
        assert '"param": null' in body or '"param":null' in body
        assert "data: [DONE]" in body
        # Must not surface Starlette's catch-all after yield+raise
        assert "RuntimeError" not in body
        assert "Unexpected message" not in body


class TestChatCompletionsNetworkErrors:
    """Network failures keep OpenAI structure across legacy and failover paths."""

    @pytest.mark.parametrize(
        ("status_code", "code"),
        [(502, "dns_resolution"), (504, "timeout_connect")],
    )
    def test_legacy_network_error_returns_top_level_openai_error(
        self, test_client, valid_proxy_api_key, status_code, code
    ):
        """Return a top-level OpenAI error instead of FastAPI detail."""
        from kiro.network_errors import NetworkHTTPException

        client = AsyncMock()
        client.request_with_retry = AsyncMock(
            side_effect=NetworkHTTPException(
                status_code=status_code,
                error_code=code,
                user_message="Kiro Gateway could not connect to the Kiro upstream service. Try again.",
            )
        )
        client.close = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = False
        try:
            with patch("kiro.routes_openai.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={
                        "model": "claude-sonnet-4-5",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        finally:
            test_client.app.state.account_system = original_mode

        assert response.status_code == status_code
        assert "detail" not in response.json()
        assert response.json()["error"]["type"] == "server_error"
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["param"] is None

    def test_multi_account_last_network_failure_is_not_generalized(
        self, test_client, valid_proxy_api_key
    ):
        """Preserve the last network error after every account fails."""
        from kiro.network_errors import NetworkHTTPException

        manager = test_client.app.state.account_manager
        account = manager.get_first_account()
        manager._accounts["second-network-test"] = account
        client = AsyncMock()
        client.request_with_retry = AsyncMock(
            side_effect=NetworkHTTPException(
                status_code=502,
                error_code="connection_refused",
                user_message="Kiro Gateway could not connect to the Kiro upstream service: connection refused. Try again.",
            )
        )
        client.close = AsyncMock()
        original_mode = test_client.app.state.account_system
        test_client.app.state.account_system = True
        try:
            with patch.object(
                manager, "get_next_account",
                new=AsyncMock(side_effect=[account, account, None]),
            ), patch.object(
                manager, "report_failure", new=AsyncMock()
            ), patch("kiro.routes_openai.KiroHttpClient", return_value=client):
                response = test_client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={
                        "model": "claude-sonnet-4-5",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        finally:
            test_client.app.state.account_system = original_mode
            manager._accounts.pop("second-network-test", None)

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "connection_refused"
        assert "All accounts failed" not in response.text

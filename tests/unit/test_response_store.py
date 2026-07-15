# -*- coding: utf-8 -*-

"""
Unit tests for Responses API in-process store and CRUD routes.
"""

import time
from unittest.mock import AsyncMock, patch

from kiro.response_store import (
    ResponseStore,
    chain_input_with_previous,
    get_response_store,
    normalize_input_items,
    reset_response_store,
    should_store_response,
)


class TestShouldStoreResponse:
    def test_omit_defaults_to_store(self):
        assert should_store_response(None) is True

    def test_true_stores(self):
        assert should_store_response(True) is True

    def test_false_skips(self):
        assert should_store_response(False) is False


class TestResponseStoreTTLLru:
    def test_put_get_delete(self):
        store = ResponseStore(ttl_seconds=3600, max_size=10)
        store.put(
            "resp_1",
            {"id": "resp_1", "output": [{"type": "message", "role": "assistant"}]},
            "hello",
        )
        got = store.get("resp_1")
        assert got is not None
        assert got.response_id == "resp_1"
        assert got.input[0]["content"] == "hello"
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None

    def test_ttl_expiry(self):
        store = ResponseStore(ttl_seconds=1, max_size=10)
        store.put("resp_ttl", {"id": "resp_ttl", "output": []}, "x")
        assert store.get("resp_ttl") is not None
        # get() returns a copy — expire the entry still held in the store
        with store._lock:
            store._data["resp_ttl"].created_at = time.time() - 5
        assert store.get("resp_ttl") is None

    def test_lru_eviction(self):
        store = ResponseStore(ttl_seconds=3600, max_size=2)
        store.put("a", {"id": "a", "output": []}, "1")
        store.put("b", {"id": "b", "output": []}, "2")
        store.put("c", {"id": "c", "output": []}, "3")
        assert store.get("a") is None
        assert store.get("b") is not None
        assert store.get("c") is not None

    def test_chain_input_with_previous(self):
        store = ResponseStore(ttl_seconds=3600, max_size=10)
        store.put(
            "resp_prev",
            {
                "id": "resp_prev",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi"}],
                    }
                ],
            },
            "Hello",
        )
        prior = store.get("resp_prev")
        chained = chain_input_with_previous(prior, "Again")
        assert chained[0]["content"] == "Hello"
        assert chained[1]["role"] == "assistant"
        assert chained[2]["content"] == "Again"

    def test_normalize_input_items(self):
        assert normalize_input_items("hi")[0]["role"] == "user"
        assert normalize_input_items([{"type": "message", "role": "user"}])[0]["type"] == "message"


class TestResponsesStoreRoutes:
    def setup_method(self):
        reset_response_store(ttl_seconds=3600, max_size=100)

    def test_background_true_returns_400(self, test_client, valid_proxy_api_key):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "background": True,
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("code") == "not_supported"

    def test_missing_previous_response_id_returns_400(
        self, test_client, valid_proxy_api_key
    ):
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "previous_response_id": "resp_missing",
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("code") == "previous_response_not_found"

    def test_get_delete_and_cancel(self, test_client, valid_proxy_api_key):
        store = get_response_store()
        store.put(
            "resp_abc",
            {"id": "resp_abc", "object": "response", "status": "completed", "output": []},
            "hi",
        )

        got = test_client.get(
            "/v1/responses/resp_abc",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert got.status_code == 200
        assert got.json()["id"] == "resp_abc"

        missing = test_client.get(
            "/v1/responses/resp_nope",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert missing.status_code == 404

        cancel = test_client.post(
            "/v1/responses/resp_abc/cancel",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert cancel.status_code == 501
        detail = cancel.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("code") == "not_supported"

        deleted = test_client.delete(
            "/v1/responses/resp_abc",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        gone = test_client.get(
            "/v1/responses/resp_abc",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert gone.status_code == 404

    @patch("kiro.routes_responses.collect_stream_response")
    @patch("kiro.routes_responses.KiroHttpClient")
    def test_non_stream_stores_by_default(
        self,
        mock_kiro_http_client_class,
        mock_collect,
        test_client,
        valid_proxy_api_key,
    ):
        mock_collect.return_value = {
            "id": "resp_stored",
            "object": "response",
            "status": "completed",
            "model": "claude-sonnet-4-5",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                }
            ],
        }

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
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "resp_stored"

        stored = get_response_store().get("resp_stored")
        assert stored is not None
        assert stored.input[0]["content"] == "Hello"

        got = test_client.get(
            "/v1/responses/resp_stored",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
        )
        assert got.status_code == 200

    @patch("kiro.routes_responses.collect_stream_response")
    @patch("kiro.routes_responses.KiroHttpClient")
    def test_store_false_skips_and_previous_chains(
        self,
        mock_kiro_http_client_class,
        mock_collect,
        test_client,
        valid_proxy_api_key,
    ):
        # Seed a previous response
        get_response_store().put(
            "resp_prev",
            {
                "id": "resp_prev",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Prev"}],
                    }
                ],
            },
            "First",
        )

        def _collect(*args, **kwargs):
            return {
                "id": "resp_next",
                "object": "response",
                "status": "completed",
                "output": [],
            }

        mock_collect.side_effect = _collect

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_instance.client = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_instance

        # store=false → not persisted
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Second",
                "stream": False,
                "store": False,
                "previous_response_id": "resp_prev",
            },
        )
        assert response.status_code == 200
        assert get_response_store().get("resp_next") is None

        # Verify chaining was applied before convert by inspecting build payload path:
        # re-run with store true and check stored input includes prior turns.
        mock_collect.side_effect = None
        mock_collect.return_value = {
            "id": "resp_chained",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": []}],
        }
        response2 = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "input": "Third",
                "stream": False,
                "previous_response_id": "resp_prev",
            },
        )
        assert response2.status_code == 200
        stored = get_response_store().get("resp_chained")
        assert stored is not None
        assert any(
            isinstance(item, dict) and item.get("content") == "First"
            for item in stored.input
        )
        assert any(
            isinstance(item, dict) and item.get("content") == "Third"
            for item in stored.input
        )
        assert any(
            isinstance(item, dict) and item.get("role") == "assistant"
            for item in stored.input
        )

    @patch("kiro.routes_responses.stream_with_first_token_retry")
    @patch("kiro.routes_responses.KiroHttpClient")
    def test_streaming_stores_completed(
        self,
        mock_kiro_http_client_class,
        mock_stream_retry,
        test_client,
        valid_proxy_api_key,
    ):
        async def mock_stream(*args, **kwargs):
            yield (
                'event: response.created\n'
                'data: {"type":"response.created","response":{"id":"resp_stream"}}\n\n'
            )
            yield (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":'
                '{"id":"resp_stream","status":"completed","output":[]}}\n\n'
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
                "input": "Hello stream",
                "stream": True,
            },
        )
        assert response.status_code == 200
        # Consume body so stream_wrapper finally runs
        assert "response.completed" in response.text

        stored = get_response_store().get("resp_stream")
        assert stored is not None
        assert stored.input[0]["content"] == "Hello stream"

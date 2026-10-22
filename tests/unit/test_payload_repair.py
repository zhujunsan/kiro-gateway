# -*- coding: utf-8 -*-

"""
Unit tests for self-healing of optional reasoning fields (``kiro.payload_repair``).

Error bodies below are verbatim Kiro responses captured on 2026-10-22.

Covers:
- Each repair rule (history reasoning, max_tokens, effort enum, drop all)
- Rule ordering and the once-per-rule bound
- Learning: repaired capabilities feed back into kiro.native_effort
- Errors that must NOT be repaired (unrelated 400s, malformed bodies)
- KiroHttpClient integration: a repairable 400 is resent and succeeds
"""

import copy
import json
from unittest.mock import AsyncMock, patch

import pytest

from kiro import native_effort
from kiro.http_client import KiroHttpClient
from kiro.native_effort import build_additional_model_request_fields, get_native_capability
from kiro.payload_repair import MAX_REPAIRS_PER_REQUEST, PayloadRepairer, parse_rejection

NOT_SUPPORTED = (
    b'{"message":"additionalModelRequestFields is not supported for this model",'
    b'"reason":"REQUEST_BODY_INVALID"}'
)
ENUM_46 = (
    b'{"message":"Invalid additionalModelRequestFields: does not have a value in the enumeration '
    b'[\\"low\\", \\"medium\\", \\"high\\", \\"max\\"]","reason":"REQUEST_BODY_INVALID"}'
)
MAX_64K = (
    b'{"message":"Invalid additionalModelRequestFields: must have a maximum value of 64000.0",'
    b'"reason":"REQUEST_BODY_INVALID"}'
)
NO_MAX_TOKENS_PROPERTY = (
    b'{"message":"Invalid additionalModelRequestFields: property \'max_tokens\' is not defined in '
    b'the schema and the schema does not allow additional properties","reason":"REQUEST_BODY_INVALID"}'
)
SIGNATURE_INVALID = (
    b'{"message":"Bedrock error message: The model returned the following errors: messages.1.content.0: '
    b'Invalid `signature` in `thinking` block","reason":"THINKING_SIGNATURE_INVALID"}'
)
IMPROPERLY_FORMED = b'{"message":"Improperly formed request.","reason":"REQUEST_BODY_INVALID"}'
INVALID_MODEL = b'{"message":"Invalid model ID.","reason":"INVALID_MODEL_ID"}'
UNKNOWN_ADDITIONAL = (
    b'{"message":"Invalid additionalModelRequestFields: something new","reason":"REQUEST_BODY_INVALID"}'
)


@pytest.fixture(autouse=True)
def isolated_registry():
    native_effort.clear_live_capabilities()
    yield
    native_effort.clear_live_capabilities()


def _payload(model="claude-sonnet-4.6", fields=None, history_reasoning=False):
    history = [{"userInputMessage": {"content": "q", "modelId": model, "origin": "AI_EDITOR"}}]
    assistant = {"content": "a"}
    if history_reasoning:
        assistant["reasoningContent"] = {"reasoningText": {"text": "r", "signature": "S"}}
    history.append({"assistantResponseMessage": assistant})
    payload = {
        "conversationState": {
            "history": history,
            "currentMessage": {"userInputMessage": {"content": "q2", "modelId": model, "origin": "AI_EDITOR"}},
        }
    }
    if fields is not None:
        payload["additionalModelRequestFields"] = fields
    return payload


def _history_assistant(payload):
    return payload["conversationState"]["history"][1]["assistantResponseMessage"]


# ==================================================================================================
# Parsing
# ==================================================================================================

class TestParseRejection:
    @pytest.mark.parametrize("body", [b"", b"not json", b"[]", b"null", b'{"reason": 5, "message": []}'])
    def test_malformed_bodies(self, body):
        rejection = parse_rejection(body)
        assert rejection.reason == "" and rejection.message == ""

    def test_real_body(self):
        rejection = parse_rejection(ENUM_46)
        assert rejection.reason == "REQUEST_BODY_INVALID"
        assert "enumeration" in rejection.message


# ==================================================================================================
# Individual rules
# ==================================================================================================

class TestRepairRules:
    def test_signature_invalid_strips_history_reasoning(self):
        payload = _payload(fields={"thinking": {"type": "adaptive"}}, history_reasoning=True)
        assert PayloadRepairer().repair(payload, SIGNATURE_INVALID)
        assert "reasoningContent" not in _history_assistant(payload)
        # Native effort is untouched: it was not the problem.
        assert payload["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}

    def test_generic_400_strips_history_reasoning(self):
        payload = _payload(model="deepseek-3.2", history_reasoning=True)
        assert PayloadRepairer().repair(payload, IMPROPERLY_FORMED)
        assert "reasoningContent" not in _history_assistant(payload)

    def test_generic_400_without_gateway_additions_is_not_repaired(self):
        payload = _payload()
        before = copy.deepcopy(payload)
        assert not PayloadRepairer().repair(payload, IMPROPERLY_FORMED)
        assert payload == before

    def test_enum_violation_remaps_effort_and_learns(self):
        payload = _payload(fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}})
        assert PayloadRepairer().repair(payload, ENUM_46)
        assert payload["additionalModelRequestFields"]["output_config"] == {"effort": "high"}
        assert get_native_capability("claude-sonnet-4.6").effort_levels == ("low", "medium", "high", "max")

    def test_enum_remap_on_gpt_reasoning_path(self):
        body = (b'{"message":"Invalid additionalModelRequestFields: does not have a value in the enumeration '
                b'[\\"low\\", \\"high\\"]","reason":"REQUEST_BODY_INVALID"}')
        payload = _payload(model="gpt-5.6-luna", fields={"reasoning": {"effort": "medium"}})
        assert PayloadRepairer().repair(payload, body)
        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}}

    def test_enum_remap_away_from_max_drops_max_tokens(self):
        body = (b'{"message":"Invalid additionalModelRequestFields: does not have a value in the enumeration '
                b'[\\"low\\", \\"medium\\", \\"high\\"]","reason":"REQUEST_BODY_INVALID"}')
        payload = _payload(fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
                                   "max_tokens": 64000})
        assert PayloadRepairer().repair(payload, body)
        assert payload["additionalModelRequestFields"] == {
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}
        }

    def test_non_effort_enum_falls_through_to_drop(self):
        body = (b'{"message":"Invalid additionalModelRequestFields: does not have a value in the enumeration '
                b'[\\"adaptive\\"]","reason":"REQUEST_BODY_INVALID"}')
        payload = _payload(fields={"thinking": {"type": "disabled"}})
        assert PayloadRepairer().repair(payload, body)
        assert "additionalModelRequestFields" not in payload

    def test_maximum_violation_clamps_and_learns(self):
        payload = _payload(fields={"output_config": {"effort": "max"}, "max_tokens": 128000})
        assert PayloadRepairer().repair(payload, MAX_64K)
        assert payload["additionalModelRequestFields"]["max_tokens"] == 64000
        assert get_native_capability("claude-sonnet-4.6").max_output_tokens == 64000

    def test_undefined_max_tokens_property_removed_and_learned(self):
        payload = _payload(model="gpt-5.6-luna", fields={"reasoning": {"effort": "max"}, "max_tokens": 4096})
        assert PayloadRepairer().repair(payload, NO_MAX_TOKENS_PROPERTY)
        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": "max"}}
        assert get_native_capability("gpt-5.6-luna").max_output_tokens is None

    def test_not_supported_drops_fields_and_disables_model(self):
        payload = _payload(model="claude-opus-4.6", fields={"output_config": {"effort": "high"}})
        assert PayloadRepairer().repair(payload, NOT_SUPPORTED)
        assert "additionalModelRequestFields" not in payload
        assert get_native_capability("claude-opus-4.6") is None
        # Subsequent requests skip native fields entirely.
        assert build_additional_model_request_fields("claude-opus-4.6", effort="high") is None

    def test_unknown_validation_error_drops_fields_without_disabling_model(self):
        payload = _payload(fields={"output_config": {"effort": "high"}})
        assert PayloadRepairer().repair(payload, UNKNOWN_ADDITIONAL)
        assert "additionalModelRequestFields" not in payload
        assert get_native_capability("claude-sonnet-4.6") is not None

    @pytest.mark.parametrize("body", [INVALID_MODEL, b"", b"garbage", b'{"reason":"THROTTLING"}'])
    def test_unrelated_errors_are_not_repaired(self, body):
        payload = _payload(fields={"output_config": {"effort": "high"}}, history_reasoning=True)
        before = copy.deepcopy(payload)
        assert not PayloadRepairer().repair(payload, body)
        assert payload == before

    @pytest.mark.parametrize("payload", [None, [], "x"])
    def test_non_dict_payload_is_not_repaired(self, payload):
        assert not PayloadRepairer().repair(payload, NOT_SUPPORTED)


class TestRepairBounds:
    def test_each_rule_runs_once_then_gives_up(self):
        repairer = PayloadRepairer()
        payload = _payload(fields={"output_config": {"effort": "high"}})
        assert repairer.repair(payload, UNKNOWN_ADDITIONAL)  # drops fields
        # Same error again: nothing left to remove → give up (no infinite loop).
        payload["additionalModelRequestFields"] = {"output_config": {"effort": "high"}}
        assert not repairer.repair(payload, UNKNOWN_ADDITIONAL)

    def test_total_repairs_are_bounded(self):
        repairer = PayloadRepairer()
        attempts = 0
        payload = _payload(fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"},
                                   "max_tokens": 128000}, history_reasoning=True)
        for body in [SIGNATURE_INVALID, MAX_64K, ENUM_46, UNKNOWN_ADDITIONAL, UNKNOWN_ADDITIONAL]:
            if repairer.repair(payload, body):
                attempts += 1
        assert attempts <= MAX_REPAIRS_PER_REQUEST

    def test_independent_requests_have_independent_budgets(self):
        for _ in range(2):
            payload = _payload(fields={"output_config": {"effort": "high"}})
            assert PayloadRepairer().repair(payload, UNKNOWN_ADDITIONAL)


# ==================================================================================================
# KiroHttpClient integration
# ==================================================================================================

def _response(status, body=b""):
    response = AsyncMock()
    response.status_code = status
    response.aread = AsyncMock(return_value=body)
    response.aclose = AsyncMock()
    return response


class TestHttpClientRepairsAndResends:
    async def _send(self, responses, payload, stream=False):
        auth = AsyncMock()
        auth.get_access_token = AsyncMock(return_value="t")
        http_client = KiroHttpClient(auth)
        mock_client = AsyncMock()
        mock_client.is_closed = False
        sent_bodies = []

        async def record(*args, **kwargs):
            sent_bodies.append(json.loads(kwargs["content"]))
            return responses.pop(0)

        mock_client.request = AsyncMock(side_effect=record)
        mock_client.send = AsyncMock(side_effect=lambda req, stream: responses.pop(0))
        mock_client.build_request = lambda method, url, **kw: (sent_bodies.append(json.loads(kw["content"])), None)[1]
        with patch.object(http_client, "_get_client", return_value=mock_client):
            with patch("kiro.http_client.get_kiro_headers", return_value={}):
                result = await http_client.request_with_retry("POST", "https://x", payload, stream=stream)
        return result, sent_bodies

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_effort_rejection_is_healed(self, stream):
        payload = _payload(fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}})
        result, sent = await self._send([_response(400, ENUM_46), _response(200)], payload, stream=stream)
        assert result.status_code == 200
        assert [b["additionalModelRequestFields"]["output_config"]["effort"] for b in sent] == ["xhigh", "high"]

    @pytest.mark.asyncio
    async def test_cascading_repairs_then_success(self):
        payload = _payload(fields={"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
                                   "max_tokens": 128000}, history_reasoning=True)
        responses = [_response(400, SIGNATURE_INVALID), _response(400, MAX_64K),
                     _response(400, NOT_SUPPORTED), _response(200)]
        result, sent = await self._send(responses, payload)
        assert result.status_code == 200
        assert len(sent) == 4
        assert "reasoningContent" not in sent[1]["conversationState"]["history"][1]["assistantResponseMessage"]
        assert sent[2]["additionalModelRequestFields"]["max_tokens"] == 64000
        assert "additionalModelRequestFields" not in sent[3]

    @pytest.mark.asyncio
    async def test_unrepairable_400_returned_without_resend(self):
        payload = _payload()
        result, sent = await self._send([_response(400, IMPROPERLY_FORMED)], payload)
        assert result.status_code == 400
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_repairs_do_not_consume_invalid_model_retries(self):
        payload = _payload(fields={"output_config": {"effort": "xhigh"}})
        responses = [_response(400, ENUM_46), _response(400, INVALID_MODEL), _response(200)]
        with patch("kiro.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, sent = await self._send(responses, payload)
        assert result.status_code == 200 and len(sent) == 3

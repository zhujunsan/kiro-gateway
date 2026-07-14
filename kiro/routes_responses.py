# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
FastAPI routes for OpenAI Responses API (Codex wire_api=responses).

Endpoints:
- POST /v1/responses — create a response (stream or JSON)
- POST /v1/responses/compact — not implemented (501)
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from kiro.config import PROFILE_ARN
from kiro.models_responses import ResponsesRequest
from kiro.converters_responses import build_kiro_payload_from_responses
from kiro.streaming_responses import (
    stream_with_first_token_retry,
    collect_stream_response,
)
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id
from kiro.routes_openai import verify_api_key

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


router = APIRouter(tags=["OpenAI Responses API"])


def _tokenizer_inputs(request_data: ResponsesRequest):
    """Best-effort inputs for fallback token counting (optional)."""
    if isinstance(request_data.input, str):
        messages = [{"role": "user", "content": request_data.input}]
    elif isinstance(request_data.input, list):
        messages = request_data.input
    else:
        messages = None

    tools = (
        [t.model_dump() if hasattr(t, "model_dump") else t for t in request_data.tools]
        if request_data.tools
        else None
    )
    return messages, tools


def _log_kiro_payload(kiro_payload: dict) -> None:
    try:
        body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")


async def _handle_success_response(
    *,
    request: Request,
    request_data: ResponsesRequest,
    http_client: KiroHttpClient,
    response,
    url: str,
    kiro_payload: dict,
    model_cache,
    auth_manager,
) -> StreamingResponse | JSONResponse:
    """Stream or collect a successful Kiro generateAssistantResponse."""
    messages_for_tokenizer, tools_for_tokenizer = _tokenizer_inputs(request_data)

    if request_data.stream:

        async def stream_wrapper():
            streaming_error = None
            client_disconnected = False
            try:

                async def make_retry_request():
                    return await http_client.request_with_retry(
                        "POST", url, kiro_payload, stream=True
                    )

                async for chunk in stream_with_first_token_retry(
                    make_request=make_retry_request,
                    client=http_client.client,
                    model=request_data.model,
                    model_cache=model_cache,
                    auth_manager=auth_manager,
                    initial_response=response,
                    request_messages=messages_for_tokenizer,
                    request_tools=tools_for_tokenizer,
                ):
                    yield chunk
            except GeneratorExit:
                client_disconnected = True
                logger.debug("Client disconnected during Responses streaming (GeneratorExit)")
            except Exception as e:
                streaming_error = e
                try:
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass
                raise
            finally:
                await http_client.close()
                if streaming_error:
                    error_type = type(streaming_error).__name__
                    error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                    logger.error(
                        f"HTTP 500 - POST /v1/responses (streaming) - [{error_type}] {error_msg[:100]}"
                    )
                elif client_disconnected:
                    logger.info("HTTP 200 - POST /v1/responses (streaming) - client disconnected")
                else:
                    logger.info("HTTP 200 - POST /v1/responses (streaming) - completed")
                if debug_logger:
                    if streaming_error:
                        from kiro.debug_logger import classify_streaming_exception

                        src, code, phase, st = classify_streaming_exception(streaming_error)
                        debug_logger.flush_on_error(
                            st, str(streaming_error),
                            source=src, code=code, phase=phase,
                        )
                    elif client_disconnected:
                        debug_logger.flush_on_disconnect()
                    else:
                        debug_logger.discard_buffers()

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    responses_json = await collect_stream_response(
        http_client.client,
        response,
        request_data.model,
        model_cache,
        auth_manager,
        request_messages=messages_for_tokenizer,
        request_tools=tools_for_tokenizer,
    )
    await http_client.close()
    logger.info("HTTP 200 - POST /v1/responses (non-streaming) - completed")
    if debug_logger:
        debug_logger.discard_buffers()
    return JSONResponse(content=responses_json)


def _error_json_response(status_code: int, message: str, error_info=None) -> JSONResponse:
    if error_info is not None:
        from kiro.kiro_errors import build_openai_error_response

        resp_status, resp_body = build_openai_error_response(error_info, status_code)
        return JSONResponse(status_code=resp_status, content=resp_body)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "kiro_api_error",
                "code": status_code,
            }
        },
    )


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def create_response(request: Request, request_data: ResponsesRequest):
    """
    OpenAI Responses API endpoint (Codex-compatible).

    Accepts Responses-format requests and translates them to Kiro
    generateAssistantResponse. Supports streaming SSE and non-streaming JSON.
    """
    logger.info(
        f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})"
    )

    if request.app.state.account_system:
        from kiro.account_errors import classify_error, ErrorType

        account_manager = request.app.state.account_manager
        all_accounts = list(account_manager._accounts.keys())
        max_attempts = len(all_accounts) * 2

        last_error_message = None
        last_error_status = None
        tried_accounts = set()

        for _attempt in range(max_attempts):
            account = await account_manager.get_next_account(
                request_data.model,
                exclude_accounts=tried_accounts,
            )

            if account is None:
                if len(all_accounts) == 1:
                    raise HTTPException(
                        status_code=last_error_status or 503,
                        detail=last_error_message or "Account unavailable",
                    )
                detail = "No available accounts for this model."
                if last_error_message:
                    detail += f" Error from last account: {last_error_message}"
                raise HTTPException(status_code=503, detail=detail)

            tried_accounts.add(account.id)
            auth_manager = account.auth_manager
            model_cache = account.model_cache

            conversation_id = generate_conversation_id()
            profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

            try:
                kiro_payload = build_kiro_payload_from_responses(
                    request_data,
                    conversation_id,
                    profile_arn_for_payload,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            _log_kiro_payload(kiro_payload)

            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.debug(f"Kiro API URL: {url} (account: {account.id})")

            if request_data.stream:
                http_client = KiroHttpClient(auth_manager, shared_client=None)
            else:
                http_client = KiroHttpClient(
                    auth_manager, shared_client=request.app.state.http_client
                )

            try:
                response = await http_client.request_with_retry(
                    "POST", url, kiro_payload, stream=True
                )

                if response.status_code == 200:
                    await account_manager.report_success(account.id, request_data.model)
                    return await _handle_success_response(
                        request=request,
                        request_data=request_data,
                        http_client=http_client,
                        response=response,
                        url=url,
                        kiro_payload=kiro_payload,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                    )

                try:
                    error_content = await response.aread()
                except Exception:
                    error_content = b"Unknown error"

                await http_client.close()
                error_text = error_content.decode("utf-8", errors="replace")

                error_reason = None
                error_info = None
                try:
                    error_json = json.loads(error_text)
                    from kiro.kiro_errors import enhance_kiro_error

                    error_info = enhance_kiro_error(error_json)
                    error_reason = error_info.reason
                    last_error_message = error_info.user_message
                    last_error_status = response.status_code
                except (json.JSONDecodeError, KeyError):
                    last_error_message = error_text
                    last_error_status = response.status_code

                error_type = classify_error(response.status_code, error_reason)

                if error_type == ErrorType.FATAL:
                    await account_manager.report_failure(
                        account.id, request_data.model, error_type,
                        response.status_code, error_reason,
                    )
                    logger.warning(
                        f"HTTP {response.status_code} - POST /v1/responses - "
                        f"{(last_error_message or '')[:100]}"
                    )
                    if debug_logger:
                        debug_logger.flush_on_error(
                            response.status_code, last_error_message,
                            source="kiro_upstream",
                            code=(error_reason or f"http_{response.status_code}"),
                            phase="response_parse",
                            upstream_status=response.status_code,
                        )
                    return _error_json_response(
                        response.status_code, last_error_message, error_info
                    )

                await account_manager.report_failure(
                    account.id, request_data.model, error_type,
                    response.status_code, error_reason,
                )
                if len(all_accounts) == 1:
                    break
                continue

            except HTTPException as e:
                await http_client.close()
                if e.status_code in (502, 504):
                    await account_manager.report_failure(
                        account.id, request_data.model, ErrorType.RECOVERABLE,
                        e.status_code, None,
                    )
                    last_error_message = str(e.detail)
                    last_error_status = e.status_code
                    if len(all_accounts) == 1:
                        break
                    logger.warning(
                        f"Network error on account {account.id}, trying next account"
                    )
                    continue

                logger.error(f"HTTP {e.status_code} - POST /v1/responses - {e.detail}")
                if debug_logger:
                    debug_logger.flush_on_error(
                        e.status_code, str(e.detail),
                        source=("network" if e.status_code in (502, 504) else "client_request"),
                        code=(
                            "timeout" if e.status_code == 504
                            else ("bad_gateway" if e.status_code == 502 else f"http_{e.status_code}")
                        ),
                        phase=("connect" if e.status_code in (502, 504) else "validation"),
                    )
                raise
            except Exception as e:
                await http_client.close()
                logger.error(f"Internal error: {e}", exc_info=True)
                logger.error(f"HTTP 500 - POST /v1/responses - {str(e)[:100]}")
                if debug_logger:
                    debug_logger.flush_on_error(
                        500, str(e),
                        source="gateway",
                        code=type(e).__name__,
                        phase="unknown",
                    )
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

        if len(all_accounts) == 1:
            raise HTTPException(status_code=last_error_status, detail=last_error_message)
        detail = "All accounts failed after full circle."
        if last_error_message:
            detail += f" Error from last account: {last_error_message}"
        raise HTTPException(status_code=503, detail=detail)

    # --- Legacy mode: single account, no failover ---
    account = request.app.state.account_manager.get_first_account()
    if not account.auth_manager:
        logger.error("No initialized accounts available (legacy mode)")
        raise HTTPException(503, "No initialized accounts available")

    auth_manager = account.auth_manager
    model_cache = account.model_cache

    conversation_id = generate_conversation_id()
    profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

    try:
        kiro_payload = build_kiro_payload_from_responses(
            request_data,
            conversation_id,
            profile_arn_for_payload,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _log_kiro_payload(kiro_payload)

    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")

    if request_data.stream:
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        http_client = KiroHttpClient(
            auth_manager, shared_client=request.app.state.http_client
        )

    try:
        response = await http_client.request_with_retry(
            "POST", url, kiro_payload, stream=True
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            error_message = error_text
            error_info = None
            try:
                error_json = json.loads(error_text)
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
            except (json.JSONDecodeError, KeyError):
                pass

            logger.warning(
                f"HTTP {response.status_code} - POST /v1/responses - {error_message[:100]}"
            )
            if debug_logger:
                debug_logger.flush_on_error(
                    response.status_code, error_message,
                    source="kiro_upstream",
                    code=f"http_{response.status_code}",
                    phase="response_parse",
                    upstream_status=response.status_code,
                )
            return _error_json_response(response.status_code, error_message, error_info)

        return await _handle_success_response(
            request=request,
            request_data=request_data,
            http_client=http_client,
            response=response,
            url=url,
            kiro_payload=kiro_payload,
            model_cache=model_cache,
            auth_manager=auth_manager,
        )

    except HTTPException as e:
        await http_client.close()
        if e.status_code in (502, 504):
            logger.warning("Network error (legacy mode, no failover available)")
        logger.error(f"HTTP {e.status_code} - POST /v1/responses - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(
                e.status_code, str(e.detail),
                source=("network" if e.status_code in (502, 504) else "client_request"),
                code=(
                    "timeout" if e.status_code == 504
                    else ("bad_gateway" if e.status_code == 502 else f"http_{e.status_code}")
                ),
                phase=("connect" if e.status_code in (502, 504) else "validation"),
            )
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST /v1/responses - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(
                500, str(e),
                source="gateway",
                code=type(e).__name__,
                phase="unknown",
            )
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/v1/responses/compact", dependencies=[Depends(verify_api_key)])
async def compact_response():
    """Responses compact endpoint — not implemented for Kiro proxy."""
    raise HTTPException(
        status_code=501,
        detail="POST /v1/responses/compact is not implemented",
    )

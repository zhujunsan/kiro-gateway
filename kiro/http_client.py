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
HTTP client for Kiro API with retry logic support.

Handles:
- 403: automatic token refresh and retry
- 429: exponential backoff
- 5xx: exponential backoff
- 400 + INVALID_MODEL_ID: short same-account backoff (intermittent upstream)
- Timeouts: exponential backoff
- PoolTimeout: fail fast (local capacity; retrying starves the pool)

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
import json
import random
from typing import Optional, Tuple

import httpx
from loguru import logger

from kiro.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
    INVALID_MODEL_ID_MAX_RETRIES,
    INVALID_MODEL_ID_RETRY_DELAY,
)
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from kiro.network_errors import (
    NetworkErrorInfo,
    NetworkHTTPException,
    classify_network_error,
    get_short_error_message,
    network_http_exception,
)
from kiro.proxy import resolve_proxy
from kiro.kiro_errors import INSUFFICIENT_MODEL_CAPACITY_REASON


_CAPACITY_JITTER_RATIO = 0.25
_MAX_RETRY_DELAY = 30.0


def _http_retry_delay(attempt: int, reason: Optional[str] = None) -> float:
    """Calculate bounded exponential backoff, adding jitter for capacity errors.

    Args:
        attempt: Zero-based retry attempt index.
        reason: Optional Kiro reason code parsed from the response.

    Returns:
        Delay in seconds, capped to avoid unbounded waits.
    """
    base_delay = min(BASE_RETRY_DELAY * (2 ** attempt), _MAX_RETRY_DELAY)
    if reason != INSUFFICIENT_MODEL_CAPACITY_REASON:
        return base_delay
    factor = random.uniform(1.0 - _CAPACITY_JITTER_RATIO, 1.0 + _CAPACITY_JITTER_RATIO)
    return min(base_delay * factor, _MAX_RETRY_DELAY)


async def _read_response_body(response: httpx.Response) -> bytes:
    """Read response body once; safe for stream and non-stream responses."""
    try:
        body = await response.aread()
        return bytes(body) if isinstance(body, (bytes, bytearray)) else b""
    except Exception:
        try:
            body = response.content
            return bytes(body) if isinstance(body, (bytes, bytearray)) else b""
        except Exception:
            return b""


def _peek_kiro_reason(body: bytes) -> Optional[str]:
    """Extract Kiro error ``reason`` from a response body, if present."""
    if not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    reason = data.get("reason")
    return reason if isinstance(reason, str) and reason else None


async def _drain_error_response(
    response: httpx.Response,
    stream: bool,
) -> Tuple[httpx.Response, bytes, Optional[str]]:
    """
    Drain an error response body and optionally close a streamed connection.

    Returns the same response (body cached via aread), raw bytes, and Kiro reason.
    """
    body = await _read_response_body(response)
    if stream:
        try:
            await response.aclose()
        except Exception as e:
            logger.debug(f"Error closing streamed error response: {e}")
    return response, body, _peek_kiro_reason(body)


class KiroHttpClient:
    """
    HTTP client for Kiro API with retry logic support.
    
    Automatically handles errors and retries requests:
    - 403: refreshes token and retries
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - 400 + INVALID_MODEL_ID: linear same-account backoff (bounded)
    - Timeouts: waits with exponential backoff
    
    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)
    
    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.
    
    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)
    
    Example:
        >>> # Per-request client (legacy mode)
        >>> client = KiroHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)
        
        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = KiroHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """
    
    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.
        
        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client
        # Dedicated client used when stream=True so streamed Kiro calls never
        # borrow the shared pool (CLOSE_WAIT leaks + TRAY-1B pool avalanche).
        self._stream_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.

        ``stream=True`` always creates (or reuses) a dedicated client and never
        borrows ``shared_client``. Non-streaming calls still use the shared
        pool when one was provided at initialization.

        
        httpx timeouts:
        - connect: TCP handshake (DNS + TCP SYN/ACK)
        - read: waiting for data from server between chunks
        - write: sending data to server
        - pool: waiting for free connection from pool
        
        IMPORTANT: FIRST_TOKEN_TIMEOUT is NOT used here!
        It is applied in streaming_openai.py via asyncio.wait_for() to control
        the wait time for the first token from the model (retry business logic).
        
        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)
        
        Returns:
            Active HTTP client
        """
        # Streaming Kiro calls (AWS event stream) must never sit on the shared
        # pool. Non-streaming chat/messages still use stream=True against Kiro;
        # borrowing keepalive slots is what turned one slow upstream into a
        # machine-wide PoolTimeout/504 avalanche (TRAY-1B).
        if stream:
            if self._stream_client is not None and not self._stream_client.is_closed:
                self.client = self._stream_client
                return self._stream_client
            timeout_config = httpx.Timeout(
                connect=30.0,
                read=STREAMING_READ_TIMEOUT,
                write=30.0,
                pool=30.0,
            )
            logger.debug(
                f"Creating dedicated streaming HTTP client "
                f"(read_timeout={STREAMING_READ_TIMEOUT}s); ignoring shared pool"
            )
            self._stream_client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True, proxy=resolve_proxy()
            )
            self.client = self._stream_client
            return self._stream_client

        # If using shared client, return it directly
        # Shared client should be pre-configured with appropriate timeouts
        if self._shared_client is not None:
            return self._shared_client
        
        # Create new client if needed (per-request non-streaming mode)
        if self.client is None or self.client.is_closed:
            timeout_config = httpx.Timeout(timeout=300.0)
            logger.debug("Creating non-streaming HTTP client (timeout=300s)")
            self.client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True, proxy=resolve_proxy()
            )
        return self.client
    
    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.
        
        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.
        
        Uses graceful exception handling to prevent errors during cleanup
        from masking the original exception in finally blocks.
        """
        stream_client = self._stream_client
        if stream_client is not None:
            self._stream_client = None
            if not stream_client.is_closed:
                try:
                    await stream_client.aclose()
                except Exception as e:
                    logger.warning(f"Error closing dedicated streaming HTTP client: {e}")
            if self.client is stream_client:
                self.client = self._shared_client

        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return
        
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup code
                # Propagating here could mask the original exception
                logger.warning(f"Error closing HTTP client: {e}")
    
    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.
        
        Automatically handles various error types:
        - 403: refreshes token via auth_manager.force_refresh() and retries
        - 429: waits with exponential backoff (1s, 2s, 4s)
        - 5xx: waits with exponential backoff
        - 400 + INVALID_MODEL_ID: linear same-account backoff (0.5s..2.5s by default)
        - Timeouts: waits with exponential backoff
        - PoolTimeout: fails immediately with 503 ``pool_exhausted`` (not retried)
        
        For streaming, STREAMING_READ_TIMEOUT is used for waiting between chunks.
        First token timeout is controlled separately in streaming_openai.py via asyncio.wait_for().
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Optional JSON body (for POST/PUT/PATCH)
            params: Optional query parameters (for GET)
            stream: Use streaming (default False)
        
        Returns:
            httpx.Response with successful response
        
        Raises:
            NetworkHTTPException: On network failure after all retry attempts.
        """
        # Determine the number of retry attempts
        # FIRST_TOKEN_TIMEOUT is used in streaming_openai.py, not here
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        # INVALID_MODEL_ID has its own budget; ensure the loop can cover it.
        loop_limit = max(max_retries, 1 + INVALID_MODEL_ID_MAX_RETRIES)
        
        client = await self._get_client(stream=stream)
        last_error = None
        last_error_info: Optional[NetworkErrorInfo] = None
        last_response: Optional[httpx.Response] = None  # Для сохранения последнего 429/5xx
        invalid_model_retries_done = 0
        
        for attempt in range(loop_limit):
            try:
                # Get current token
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(self.auth_manager, token)
                
                # Build request kwargs based on parameters
                request_kwargs = {"headers": headers}
                
                if json_data is not None:
                    request_kwargs["content"] = json.dumps(json_data).encode()
                
                if params is not None:
                    request_kwargs["params"] = params
                
                if stream:
                    # Prevent CLOSE_WAIT connection leak (issue #38)
                    headers["Connection"] = "close"
                    req = client.build_request(method, url, **request_kwargs)
                    logger.debug("Sending request to Kiro API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Kiro API...")
                    response = await client.request(method, url, **request_kwargs)
                
                # Check status
                if response.status_code == 200:
                    if invalid_model_retries_done > 0:
                        logger.debug(
                            f"INVALID_MODEL_ID recovered after "
                            f"{invalid_model_retries_done} same-account retry(ies)"
                        )
                    return response
                
                # 403 - token expired, refresh and retry
                if response.status_code == 403:
                    if attempt >= max_retries - 1:
                        return response
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{MAX_RETRIES})")
                    await self.auth_manager.force_refresh()
                    continue
                
                # 429 - bounded exponential backoff. Capacity errors add jitter
                # to prevent synchronized retries across gateway instances.
                if response.status_code == 429:
                    response, _body, reason = await _drain_error_response(response, stream)
                    last_response = response
                    if attempt >= max_retries - 1:
                        break
                    delay = _http_retry_delay(attempt, reason)
                    reason_suffix = f" ({reason})" if reason else ""
                    logger.warning(
                        f"Received 429{reason_suffix}, waiting {delay:.3f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # 5xx - server error, wait and retry
                if 500 <= response.status_code < 600:
                    last_response = response  # Сохраняем для возврата после exhaustion
                    if attempt >= max_retries - 1:
                        break
                    delay = _http_retry_delay(attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # 400 - only INVALID_MODEL_ID is eligible for same-account retry
                if response.status_code == 400:
                    response, _body, reason = await _drain_error_response(response, stream)
                    if (
                        reason == "INVALID_MODEL_ID"
                        and invalid_model_retries_done < INVALID_MODEL_ID_MAX_RETRIES
                    ):
                        invalid_model_retries_done += 1
                        delay = INVALID_MODEL_ID_RETRY_DELAY * invalid_model_retries_done
                        logger.warning(
                            f"Received 400 INVALID_MODEL_ID, waiting {delay}s "
                            f"(attempt {invalid_model_retries_done}/"
                            f"{INVALID_MODEL_ID_MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    return response
                
                # Other errors - return as is
                return response
                
            except httpx.PoolTimeout as e:
                # Local pool wait is not an upstream timeout. Retrying holds the
                # waiter for another 30s×N and starves every other request.
                last_error = e
                error_info = classify_network_error(e)
                last_error_info = error_info
                logger.error(
                    "{} - failing fast without retry (attempt {}/{})",
                    get_short_error_message(error_info),
                    attempt + 1,
                    max_retries,
                )
                break

            except httpx.TimeoutException as e:
                last_error = e
                
                # Classify timeout error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = _http_retry_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    break
                
            except httpx.RequestError as e:
                last_error = e
                
                # Classify the error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = _http_retry_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    break
        
        # If we have a last_response (429/5xx retry exhausted), return it
        # This allows the caller to see the real status code and error body
        if last_response is not None:
            logger.warning(
                f"Retries exhausted for HTTP {last_response.status_code}, "
                f"returning response to caller for classification"
            )
            return last_response
        
        # All attempts exhausted. Keep technical details in logs only and raise a
        # typed exception so every API route can preserve the protocol shape.
        if last_error_info is not None:
            logger.error(
                "Network retries exhausted [{}]: {}",
                last_error_info.category.value,
                last_error_info.technical_details,
            )
            raise network_http_exception(last_error_info)

        # Defensive fallback: the retry loop should always capture a classified
        # error before reaching here. Keep it typed and client-safe regardless.
        status_code = 504 if stream else 502
        error_code = "timeout" if stream else "connection_error"
        raise NetworkHTTPException(
            status_code=status_code,
            error_code=error_code,
            user_message=(
                "Kiro Gateway could not connect to the Kiro upstream service "
                f"after {max_retries} attempts. Check the gateway network and "
                "proxy settings, then try again."
            ),
        )
    
    async def __aenter__(self) -> "KiroHttpClient":
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()
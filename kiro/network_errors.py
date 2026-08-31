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
Network error classification and user-friendly message formatting.

This module provides a centralized system for classifying network errors
and converting them into actionable, user-friendly messages with troubleshooting steps.

Architecture:
- ErrorCategory: Enum of all possible network error types
- NetworkErrorInfo: Structured information about an error
- classify_network_error(): Analyzes exceptions and returns NetworkErrorInfo
- format_error_for_user(): Formats errors for API responses (OpenAI/Anthropic)
"""

import socket
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger


class ErrorCategory(str, Enum):
    """
    Categories of network errors.
    
    Each category represents a distinct type of network failure
    with specific troubleshooting steps.
    """
    DNS_RESOLUTION = "dns_resolution"
    CONNECTION_REFUSED = "connection_refused"
    CONNECTION_RESET = "connection_reset"
    NETWORK_UNREACHABLE = "network_unreachable"
    TIMEOUT_CONNECT = "timeout_connect"
    TIMEOUT_READ = "timeout_read"
    SSL_ERROR = "ssl_error"
    PROXY_ERROR = "proxy_error"
    TOO_MANY_REDIRECTS = "too_many_redirects"
    UNKNOWN = "unknown"


@dataclass
class NetworkErrorInfo:
    """
    Structured information about a network error.
    
    Attributes:
        category: Error category for classification
        user_message: Clear, non-technical message for end users
        troubleshooting_steps: List of actionable steps to resolve the issue
        technical_details: Technical error details for logging and debugging
        is_retryable: Whether retrying the request might succeed
        suggested_http_code: Appropriate HTTP status code (502, 504, etc.)
    """
    category: ErrorCategory
    user_message: str
    troubleshooting_steps: List[str]
    technical_details: str
    is_retryable: bool
    suggested_http_code: int


ApiErrorFormat = Literal["openai", "anthropic"]


class NetworkHTTPException(HTTPException):
    """HTTP exception carrying a sanitized Kiro upstream network failure.

    Attributes:
        error_info: Original classified network information, when available.
        error_code: Stable machine-readable network failure code.
        user_message: Sanitized message safe to return to API clients.
    """

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        user_message: str,
        error_info: Optional[NetworkErrorInfo] = None,
    ) -> None:
        """Initialize a client-safe upstream network exception.

        Args:
            status_code: HTTP 502 for connection failures or 504 for timeouts.
            error_code: Stable machine-readable network failure code.
            user_message: Sanitized, actionable message for API clients.
            error_info: Optional detailed classification retained for internal use.
        """
        super().__init__(status_code=status_code, detail=user_message)
        self.error_info = error_info
        self.error_code = error_code
        self.user_message = user_message


def _client_network_message(error_info: NetworkErrorInfo) -> str:
    """Build a concise, actionable message without technical details.

    Args:
        error_info: Classified network failure.

    Returns:
        Client-safe message identifying the Kiro upstream connection failure.
    """
    action = (
        error_info.troubleshooting_steps[0]
        if error_info.troubleshooting_steps
        else "Check the gateway network and proxy settings, then try again"
    )
    return (
        "Kiro Gateway could not connect to the Kiro upstream service: "
        f"{error_info.user_message} {action}."
    )


def network_http_exception(error_info: NetworkErrorInfo) -> NetworkHTTPException:
    """Create a typed HTTP exception from classified network information.

    Args:
        error_info: Classified network failure retained for failover and logging.

    Returns:
        Typed exception containing only client-safe response text.
    """
    return NetworkHTTPException(
        status_code=error_info.suggested_http_code,
        error_code=error_info.category.value,
        user_message=_client_network_message(error_info),
        error_info=error_info,
    )


def upstream_network_exception(
    status_code: int,
    error_code: str,
    message: str,
) -> NetworkHTTPException:
    """Create a typed exception for an already-classified upstream failure.

    Args:
        status_code: HTTP 502 or 504 selected by the transport classifier.
        error_code: Stable transport failure code.
        message: Existing client-safe failure description.

    Returns:
        Typed exception suitable for shared JSON and SSE formatting.
    """
    prefix = "Kiro Gateway could not receive a response from the Kiro upstream service: "
    clean_message = message.strip() or "The upstream request failed. Please try again."
    if clean_message.startswith("Kiro Gateway could not"):
        user_message = clean_message
    else:
        user_message = f"{prefix}{clean_message}"
    return NetworkHTTPException(
        status_code=status_code,
        error_code=error_code,
        user_message=user_message,
    )


def network_exception_from_exception(
    error: BaseException,
) -> Optional[NetworkHTTPException]:
    """Extract or synthesize a typed network failure from an exception.

    Args:
        error: Exception raised by request, collect, or streaming code.

    Returns:
        A typed network exception for recognized upstream transport failures,
        otherwise ``None`` so ordinary exceptions retain their original type.
    """
    if isinstance(error, NetworkHTTPException):
        return error

    if isinstance(error, HTTPException) and error.status_code in (502, 504):
        code = "upstream_timeout" if error.status_code == 504 else "upstream_connection_error"
        return upstream_network_exception(error.status_code, code, str(error.detail))

    from kiro.collect_errors import collect_failure_or_none

    failure = collect_failure_or_none(error)
    if failure is None:
        return None
    return upstream_network_exception(
        failure.status_code,
        failure.code,
        failure.message,
    )


def build_network_error_body(
    error: NetworkHTTPException,
    format_type: ApiErrorFormat = "openai",
) -> Dict[str, Any]:
    """Build a standards-compatible API error body for a network failure.

    Args:
        error: Typed network exception to expose to the client.
        format_type: Target API protocol, ``openai`` or ``anthropic``.

    Returns:
        OpenAI- or Anthropic-compatible top-level error object.
    """
    if format_type == "anthropic":
        return {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": error.user_message,
            },
        }
    return {
        "error": {
            "message": error.user_message,
            "type": "server_error",
            "code": error.error_code,
            "param": None,
        }
    }


def build_stream_error_body(
    error: BaseException,
    format_type: ApiErrorFormat = "openai",
) -> Dict[str, Any]:
    """Build a protocol error body after an SSE response has started.

    Args:
        error: Exception raised while producing the stream.
        format_type: Target API protocol, ``openai`` or ``anthropic``.

    Returns:
        Protocol-compatible error body. Recognized transport failures retain
        their actionable message and stable code; other failures remain generic.
    """
    network_error = network_exception_from_exception(error)
    if network_error is not None:
        return build_network_error_body(network_error, format_type)

    if isinstance(error, HTTPException):
        message = str(error.detail)
        code: Any = error.status_code
    else:
        message = str(error).strip() or type(error).__name__
        code = type(error).__name__

    if format_type == "anthropic":
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "code": code,
            "param": None,
        }
    }


def build_network_error_response(
    error: NetworkHTTPException,
    format_type: ApiErrorFormat = "openai",
) -> JSONResponse:
    """Build an HTTP JSON response for a typed network failure.

    Args:
        error: Typed network exception to return.
        format_type: Target API protocol, ``openai`` or ``anthropic``.

    Returns:
        JSON response preserving the exception's 502 or 504 status.
    """
    return JSONResponse(
        status_code=error.status_code,
        content=build_network_error_body(error, format_type),
    )


def classify_network_error(error: Exception) -> NetworkErrorInfo:
    """
    Classifies a network error and returns structured information.
    
    Analyzes the exception type, error message, and underlying cause
    to determine the specific type of network failure and provide
    appropriate user-facing messages and troubleshooting steps.
    
    Args:
        error: The exception that occurred (typically httpx.RequestError)
    
    Returns:
        NetworkErrorInfo with classification and user-friendly details
    
    Example:
        >>> try:
        ...     response = await client.get("https://example.com")
        ... except httpx.RequestError as e:
        ...     error_info = classify_network_error(e)
        ...     logger.error(f"[{error_info.category}] {error_info.user_message}")
    """
    error_type = type(error).__name__
    error_str = str(error)
    
    # Extract technical details for logging
    technical_details = f"{error_type}: {error_str}"
    
    # Analyze httpx.ConnectError (connection establishment failures)
    if isinstance(error, httpx.ConnectError):
        return _classify_connect_error(error, technical_details)
    
    # Analyze httpx.TimeoutException (various timeout types)
    if isinstance(error, httpx.TimeoutException):
        return _classify_timeout_error(error, technical_details)
    
    # Analyze httpx.TooManyRedirects
    if isinstance(error, httpx.TooManyRedirects):
        return NetworkErrorInfo(
            category=ErrorCategory.TOO_MANY_REDIRECTS,
            user_message="Too many redirects - the server is redirecting in a loop.",
            troubleshooting_steps=[
                "This is likely a server-side configuration issue",
                "Try accessing the service directly without the gateway",
                "Contact the service provider if the issue persists"
            ],
            technical_details=technical_details,
            is_retryable=False,
            suggested_http_code=502
        )
    
    # Analyze httpx.ProxyError
    if isinstance(error, httpx.ProxyError):
        return NetworkErrorInfo(
            category=ErrorCategory.PROXY_ERROR,
            user_message="Proxy connection failed - cannot connect through the configured proxy.",
            troubleshooting_steps=[
                "Check proxy configuration (HTTP_PROXY, HTTPS_PROXY environment variables)",
                "Verify proxy server is accessible",
                "Try disabling proxy temporarily",
                "Check proxy authentication credentials if required"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Generic httpx.RequestError (catch-all)
    if isinstance(error, httpx.RequestError):
        return NetworkErrorInfo(
            category=ErrorCategory.UNKNOWN,
            user_message="Network request failed due to an unexpected error.",
            troubleshooting_steps=[
                "Check your internet connection",
                "Verify firewall/antivirus settings",
                "Try again in a few moments",
                "Check the debug logs for more details"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Non-httpx errors (shouldn't happen, but handle gracefully)
    return NetworkErrorInfo(
        category=ErrorCategory.UNKNOWN,
        user_message="An unexpected error occurred.",
        troubleshooting_steps=[
            "Check the debug logs for details",
            "Try again in a few moments",
            "Report this issue if it persists"
        ],
        technical_details=technical_details,
        is_retryable=True,
        suggested_http_code=500
    )


def _classify_connect_error(error: httpx.ConnectError, technical_details: str) -> NetworkErrorInfo:
    """
    Classifies httpx.ConnectError into specific subcategories.
    
    Args:
        error: The ConnectError exception
        technical_details: Technical error string for logging
    
    Returns:
        NetworkErrorInfo with specific classification
    """
    error_str = str(error)
    
    # Check underlying cause chain for more specific errors
    cause = error.__cause__
    
    # Check for DNS errors (socket.gaierror)
    if cause and isinstance(cause, socket.gaierror):
        # DNS resolution failed
        # Common errno values:
        # - 11001 (Windows): WSAHOST_NOT_FOUND
        # - -2, -3, -5 (Unix): EAI_NONAME, EAI_AGAIN, EAI_NODATA
        errno = getattr(cause, 'errno', None)
        
        return NetworkErrorInfo(
            category=ErrorCategory.DNS_RESOLUTION,
            user_message="DNS resolution failed - cannot resolve the provider's domain name.",
            troubleshooting_steps=[
                "Check your internet connection",
                "Try changing DNS servers to Google DNS (8.8.8.8, 8.8.4.4) or Cloudflare (1.1.1.1, 1.0.0.1)",
                "Temporarily disable VPN if you're using one",
                "Check if firewall/antivirus is blocking DNS requests",
                "Verify the domain name is correct and the service is operational"
            ],
            technical_details=f"{technical_details} (errno: {errno})",
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Check for connection refused
    if "Connection refused" in error_str or "ECONNREFUSED" in error_str:
        return NetworkErrorInfo(
            category=ErrorCategory.CONNECTION_REFUSED,
            user_message="Connection refused - the server is not accepting connections.",
            troubleshooting_steps=[
                "The service may be temporarily down",
                "Check if the service is running and accessible",
                "Verify firewall is not blocking the connection",
                "Try again in a few moments"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Check for connection reset
    if "Connection reset" in error_str or "ECONNRESET" in error_str:
        return NetworkErrorInfo(
            category=ErrorCategory.CONNECTION_RESET,
            user_message="Connection reset - the server closed the connection unexpectedly.",
            troubleshooting_steps=[
                "This is usually a temporary server issue",
                "Try again in a few moments",
                "Check if VPN/proxy is interfering with the connection",
                "Verify network stability"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Check for network unreachable
    if "Network is unreachable" in error_str or "No route to host" in error_str or "ENETUNREACH" in error_str:
        return NetworkErrorInfo(
            category=ErrorCategory.NETWORK_UNREACHABLE,
            user_message="Network unreachable - cannot reach the server's network.",
            troubleshooting_steps=[
                "Check your internet connection",
                "Verify network adapter is enabled and working",
                "Check routing table if using VPN",
                "Try disabling VPN temporarily",
                "Restart network adapter or router"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=502
        )
    
    # Check for SSL/TLS errors
    if "SSL" in error_str or "TLS" in error_str or "certificate" in error_str.lower():
        return NetworkErrorInfo(
            category=ErrorCategory.SSL_ERROR,
            user_message="SSL/TLS error - secure connection could not be established.",
            troubleshooting_steps=[
                "Check system date and time (incorrect time causes SSL errors)",
                "Update SSL certificates on your system",
                "Check if antivirus/firewall is intercepting HTTPS traffic",
                "Verify the server's SSL certificate is valid"
            ],
            technical_details=technical_details,
            is_retryable=False,
            suggested_http_code=502
        )
    
    # Generic connection error
    return NetworkErrorInfo(
        category=ErrorCategory.UNKNOWN,
        user_message="Connection failed - unable to establish connection to the server.",
        troubleshooting_steps=[
            "Check your internet connection",
            "Verify firewall/antivirus settings",
            "Try disabling VPN temporarily",
            "Check if the service is accessible from other devices"
        ],
        technical_details=technical_details,
        is_retryable=True,
        suggested_http_code=502
    )


def _classify_timeout_error(error: httpx.TimeoutException, technical_details: str) -> NetworkErrorInfo:
    """
    Classifies httpx.TimeoutException into specific subcategories.
    
    Args:
        error: The TimeoutException
        technical_details: Technical error string for logging
    
    Returns:
        NetworkErrorInfo with specific classification
    """
    # ConnectTimeout: TCP handshake timeout
    if isinstance(error, httpx.ConnectTimeout):
        return NetworkErrorInfo(
            category=ErrorCategory.TIMEOUT_CONNECT,
            user_message="Connection timeout - server did not respond to connection attempt.",
            troubleshooting_steps=[
                "Check your internet connection speed",
                "The server may be overloaded or slow to respond",
                "Try again in a few moments",
                "Check if firewall is delaying connections"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=504
        )
    
    # ReadTimeout: Server stopped sending data
    if isinstance(error, httpx.ReadTimeout):
        return NetworkErrorInfo(
            category=ErrorCategory.TIMEOUT_READ,
            user_message="Read timeout - server stopped responding during data transfer.",
            troubleshooting_steps=[
                "The server may be processing a complex request",
                "Check your internet connection stability",
                "Try again with a simpler request",
                "The service may be experiencing high load"
            ],
            technical_details=technical_details,
            is_retryable=True,
            suggested_http_code=504
        )
    
    # Generic timeout
    return NetworkErrorInfo(
        category=ErrorCategory.TIMEOUT_READ,
        user_message="Request timeout - operation took too long to complete.",
        troubleshooting_steps=[
            "Check your internet connection",
            "The server may be slow or overloaded",
            "Try again in a few moments"
        ],
        technical_details=technical_details,
        is_retryable=True,
        suggested_http_code=504
    )


def format_error_for_user(
    error_info: NetworkErrorInfo,
    format_type: str = "openai",
    include_troubleshooting: bool = True,
) -> Dict[str, Any]:
    """Format classified network information for an API response.

    Args:
        error_info: Classified network failure.
        format_type: ``openai``, ``anthropic``, or the generic fallback.
        include_troubleshooting: Retained for API compatibility. Client responses
            always use one concise actionable suggestion.

    Returns:
        Standards-compatible error body without technical details.
    """
    del include_troubleshooting
    error = network_http_exception(error_info)
    if format_type in ("openai", "anthropic"):
        return build_network_error_body(error, format_type)
    return {
        "error": {
            "message": error.user_message,
            "type": "server_error",
            "code": error.error_code,
            "param": None,
        }
    }

def get_short_error_message(error_info: NetworkErrorInfo) -> str:
    """
    Returns a short, single-line error message for logging.
    
    Args:
        error_info: The classified error information
    
    Returns:
        Short error message suitable for log files
    
    Example:
        >>> error_info = classify_network_error(exception)
        >>> logger.warning(get_short_error_message(error_info))
    """
    return error_info.user_message
